from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from ..agents import OpenAIChatClient, PlannerAgent, TraceAuditorAgent, TraceSynthesizerAgent
from ..common import sanitize_for_persistence, write_json, write_text
from ..config import save_runtime_snapshot
from ..renderers import export_trace_for_benchmark, render_trace_markdown
from ..schemas import InferenceStep, TracePackage
from ..storage import EvidenceLedger, SharedEvidenceCache, WorkspaceManager
from ..tools import ObservationExtractor, ToolRegistry
from ..tools.base import ToolExecutionContext
from .executor import PlanExecutor
from .preprocess import DenseCaptionPreprocessor


def _query_terms(task, audit_report=None) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", "%s %s" % (task.question, " ".join(task.options)))
    if audit_report is not None:
        tokens.extend(re.findall(r"[A-Za-z0-9]+", json.dumps(audit_report, ensure_ascii=False)))
    ordered = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen or len(lowered) < 3:
            continue
        seen.add(lowered)
        ordered.append(lowered)
    return ordered[:30]


def _compact_round_summary(round_index: int, plan, execution_records, audit_report) -> Dict[str, object]:
    return {
        "round": int(round_index),
        "strategy": getattr(plan, "strategy", ""),
        "tools": [item["tool_name"] for item in execution_records],
        "verdict": audit_report.verdict if audit_report is not None else None,
        "feedback": audit_report.feedback if audit_report is not None else "",
        "missing_information": list(audit_report.missing_information or []) if audit_report is not None else [],
    }


def _trace_from_initial_steps(task, steps: List[str]) -> TracePackage:
    inference_steps = [
        InferenceStep(step_id=index + 1, text=text, supporting_observation_ids=[], answer_relevance="medium")
        for index, text in enumerate(steps or [])
        if str(text or "").strip()
    ]
    return TracePackage(
        task_key=task.sample_key,
        mode="refine",
        evidence_entries=[],
        inference_steps=inference_steps,
        final_answer="",
        benchmark_renderings={},
        metadata={"source": "initial_trace_steps"},
    )


class PipelineRunner(object):
    def __init__(self, profile, models_config):
        self.profile = profile
        self.models_config = models_config
        self.workspace = WorkspaceManager(profile)
        self.llm_client = OpenAIChatClient(profile, models_config)
        self.tool_registry = ToolRegistry(self.workspace, profile, models_config, llm_client=self.llm_client)
        self.preprocessor = DenseCaptionPreprocessor(self.workspace, self.tool_registry, models_config)
        self.executor = PlanExecutor(
            tool_registry=self.tool_registry,
            evidence_cache=SharedEvidenceCache(self.workspace),
            extractor=ObservationExtractor(),
            models_config=models_config,
        )
        self.planner = PlannerAgent(self.llm_client, models_config.agents["planner"])
        self.synthesizer = TraceSynthesizerAgent(self.llm_client, models_config.agents["trace_synthesizer"])
        self.auditor = TraceAuditorAgent(self.llm_client, models_config.agents["trace_auditor"])

    def close(self):
        self.tool_registry.close()

    def _write_runtime_snapshot(self, run):
        payload = {
            "machine": self.profile.redacted_snapshot(),
            "agent_models": {
                name: {
                    "backend": config.backend,
                    "model": config.model,
                    "endpoint": config.endpoint,
                    "prompt_version": config.prompt_version,
                }
                for name, config in sorted(self.models_config.agents.items())
            },
            "tool_backends": {
                name: {
                    "backend": config.backend,
                    "model": config.model,
                    "endpoint": config.endpoint,
                    "prompt_version": config.prompt_version,
                }
                for name, config in sorted(self.models_config.tools.items())
            },
        }
        save_runtime_snapshot(str(run.snapshot_path), payload)

    def _persist_trace(self, run, task, trace_package):
        trace_dict = sanitize_for_persistence(trace_package.dict())
        write_json(run.trace_dir / "trace_package.json", trace_dict)
        write_text(run.trace_dir / "trace_readable.md", render_trace_markdown(trace_dict))
        export_payload = export_trace_for_benchmark(task.benchmark, task, trace_dict)
        write_json(run.results_dir / "benchmark_export.json", sanitize_for_persistence(export_payload))
        return export_payload

    def _build_evidence_summary(self, ledger: EvidenceLedger) -> Dict[str, object]:
        entries = ledger.entries()
        observations = ledger.observations()
        return {
            "evidence_entry_count": len(entries),
            "observation_count": len(observations),
            "evidence_entries": entries[-10:],
            "recent_observations": observations[-20:],
        }

    def run_task(
        self,
        task,
        mode: str = "generate",
        max_rounds: int = 2,
        clip_duration_s: float = 30.0,
        initial_trace_path: Optional[str] = None,
    ) -> Dict[str, object]:
        run = self.workspace.create_run(task)
        self._write_runtime_snapshot(run)
        preprocess_output = self.preprocessor.get_or_build(task, clip_duration_s=clip_duration_s)
        video_fingerprint = preprocess_output["video_fingerprint"]
        summary_text = preprocess_output["summary"]
        self.workspace.write_run_manifest(
            run,
            {
                "benchmark": task.benchmark,
                "sample_key": task.sample_key,
                "run_id": run.run_id,
                "mode": mode,
                "task": task.persistable_dict(),
                "preprocess_cache": preprocess_output["cache_dir"],
                "clip_duration_s": float(clip_duration_s),
            },
        )
        evidence_ledger = EvidenceLedger(run)
        execution_context = ToolExecutionContext(
            workspace=self.workspace,
            run=run,
            task=task,
            models_config=self.models_config,
            llm_client=self.llm_client,
            evidence_lookup=lambda ids: [
                item for item in evidence_ledger.observations() if item.get("observation_id") in set(ids or [])
            ],
            preprocess_bundle=preprocess_output,
        )

        current_trace = None
        if mode == "refine":
            if initial_trace_path:
                current_trace = TracePackage.parse_obj(json.loads(open(initial_trace_path, "r", encoding="utf-8").read()))
            elif task.initial_trace_steps:
                current_trace = _trace_from_initial_steps(task, task.initial_trace_steps)

        compact_rounds = []
        latest_audit = None

        if current_trace is not None:
            current_trace_dict = sanitize_for_persistence(current_trace.dict())
            write_json(run.trace_dir / "initial_trace_package.json", current_trace_dict)
            initial_summary = self._build_evidence_summary(evidence_ledger)
            audit_raw, latest_audit = self.auditor.audit(task, current_trace_dict, initial_summary)
            write_text(run.auditor_dir / "initial_raw.txt", audit_raw)
            write_json(run.auditor_dir / "initial_report.json", latest_audit.dict())
            if latest_audit.verdict == "PASS":
                export_payload = self._persist_trace(run, task, current_trace)
                final_payload = {
                    "run_dir": self.workspace.relative_path(run.run_dir),
                    "trace_package": sanitize_for_persistence(current_trace.dict()),
                    "audit_report": latest_audit.dict(),
                    "benchmark_export": export_payload,
                }
                write_json(run.results_dir / "final_result.json", final_payload)
                return final_payload

        rounds_executed = 0
        while rounds_executed < max(1, int(max_rounds)):
            planning_mode = "generate" if current_trace is None else "refine"
            use_summary = current_trace is None or latest_audit is None or bool(latest_audit.missing_information)
            retrieved_observations = evidence_ledger.retrieve(
                query_terms=_query_terms(task, latest_audit.dict() if latest_audit else None),
                limit=50,
            )
            planner_raw, plan = self.planner.plan(
                task=task,
                mode=planning_mode,
                summary_text=summary_text if use_summary else "",
                compact_rounds=compact_rounds,
                retrieved_observations=retrieved_observations,
                audit_feedback=latest_audit.dict() if latest_audit is not None else None,
                tool_names=self.tool_registry.adapters.keys(),
            )
            round_prefix = "round_%02d" % (rounds_executed + 1)
            write_text(run.planner_dir / ("%s_raw.txt" % round_prefix), planner_raw)
            write_json(run.planner_dir / ("%s_plan.json" % round_prefix), sanitize_for_persistence(plan.dict()))

            execution_records = self.executor.execute_plan(
                plan=plan,
                execution_context=execution_context,
                evidence_ledger=evidence_ledger,
                video_fingerprint=video_fingerprint,
            )

            evidence_entries = evidence_ledger.entries()
            observations = evidence_ledger.observations()
            synthesizer_raw, trace_package = self.synthesizer.synthesize(
                task=task,
                mode=planning_mode,
                evidence_entries=evidence_entries,
                observations=observations,
                current_trace=sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
                refinement_instructions=plan.refinement_instructions,
            )
            trace_dict = trace_package.dict()
            trace_dict["task_key"] = task.sample_key
            trace_dict["mode"] = planning_mode
            trace_dict["benchmark_renderings"][task.benchmark] = export_trace_for_benchmark(task.benchmark, task, trace_dict)
            trace_package = TracePackage.parse_obj(trace_dict)
            write_text(run.synthesizer_dir / ("%s_raw.txt" % round_prefix), synthesizer_raw)
            write_json(
                run.synthesizer_dir / ("%s_trace_package.json" % round_prefix),
                sanitize_for_persistence(trace_package.dict()),
            )

            evidence_summary = self._build_evidence_summary(evidence_ledger)
            audit_raw, latest_audit = self.auditor.audit(
                task=task,
                trace_package=sanitize_for_persistence(trace_package.dict()),
                evidence_summary=evidence_summary,
            )
            write_text(run.auditor_dir / ("%s_raw.txt" % round_prefix), audit_raw)
            write_json(run.auditor_dir / ("%s_report.json" % round_prefix), latest_audit.dict())

            current_trace = trace_package
            compact_rounds.append(_compact_round_summary(rounds_executed + 1, plan, execution_records, latest_audit))
            rounds_executed += 1
            if latest_audit.verdict == "PASS":
                break

        export_payload = self._persist_trace(run, task, current_trace)
        final_payload = {
            "run_dir": self.workspace.relative_path(run.run_dir),
            "trace_package": sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
            "audit_report": latest_audit.dict() if latest_audit is not None else None,
            "benchmark_export": sanitize_for_persistence(export_payload),
            "preprocess": sanitize_for_persistence(preprocess_output),
            "rounds_executed": rounds_executed,
        }
        write_json(run.results_dir / "final_result.json", final_payload)
        return final_payload
