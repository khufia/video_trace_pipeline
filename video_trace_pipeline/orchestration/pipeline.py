from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from ..agents import AtomicFactAgent, OpenAIChatClient, PlannerAgent, TraceAuditorAgent, TraceSynthesizerAgent
from ..common import sanitize_for_persistence, write_json, write_text
from ..config import save_runtime_snapshot
from ..renderers import export_trace_for_benchmark, render_trace_markdown, write_run_debug_bundle, write_run_readable_bundle
from ..schemas import InferenceStep, TracePackage
from ..storage import EvidenceLedger, SharedEvidenceCache, WorkspaceManager
from ..tools import ObservationExtractor, ToolRegistry
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation
from .executor import PlanExecutor
from .plan_normalizer import ExecutionPlanNormalizer
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


def _truncate_text(value: Any, max_len: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _compact_clip_ref(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    compact = {}
    for key in ("video_id", "start_s", "end_s", "confidence"):
        if payload.get(key) is not None:
            compact[key] = payload.get(key)
    return compact or None


def _compact_frame_ref(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    compact = {}
    for key in ("video_id", "timestamp_s", "frame_path"):
        if payload.get(key) is not None:
            compact[key] = payload.get(key)
    clip = _compact_clip_ref(payload.get("clip"))
    if clip:
        compact["clip"] = clip
    return compact or None


def _compact_request_summary(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ("query", "focus_query"):
        value = _truncate_text(request_payload.get(key), max_len=180)
        if value:
            summary[key] = value
    if isinstance(request_payload.get("time_hints"), list) and request_payload.get("time_hints"):
        summary["time_hints"] = [_truncate_text(item, max_len=80) for item in list(request_payload.get("time_hints") or [])[:4]]
    if isinstance(request_payload.get("text_contexts"), list) and request_payload.get("text_contexts"):
        summary["text_contexts"] = [_truncate_text(item, max_len=120) for item in list(request_payload.get("text_contexts") or [])[:3]]
    if isinstance(request_payload.get("evidence_ids"), list) and request_payload.get("evidence_ids"):
        summary["evidence_ids"] = list(request_payload.get("evidence_ids") or [])[:6]
    clip = _compact_clip_ref(request_payload.get("clip"))
    if clip is not None:
        summary["clip"] = clip
    clips = [_compact_clip_ref(item) for item in list(request_payload.get("clips") or [])[:3]]
    clips = [item for item in clips if item is not None]
    if clips:
        summary["clips"] = clips
    frame = _compact_frame_ref(request_payload.get("frame"))
    if frame is not None:
        summary["frame"] = frame
    frames = [_compact_frame_ref(item) for item in list(request_payload.get("frames") or [])[:3]]
    frames = [item for item in frames if item is not None]
    if frames:
        summary["frames"] = frames
    return summary


def _count_result_items(data: Dict[str, Any]) -> Dict[str, int]:
    counts = {}
    for key in ("clips", "frames", "reads", "segments", "captions", "transcripts", "regions"):
        value = data.get(key)
        if isinstance(value, list) and value:
            counts["%s_count" % key[:-1] if key.endswith("s") else "%s_count" % key] = len(value)
    return counts


def _compact_result_summary(result_payload: Dict[str, Any], evidence_entry: Dict[str, Any], observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary_text = (
        result_payload.get("summary")
        or ((result_payload.get("data") or {}).get("summary") if isinstance(result_payload.get("data"), dict) else None)
        or ((result_payload.get("data") or {}).get("overall_summary") if isinstance(result_payload.get("data"), dict) else None)
        or ((result_payload.get("data") or {}).get("text") if isinstance(result_payload.get("data"), dict) else None)
        or result_payload.get("raw_output_text")
    )
    if summary_text:
        summary["summary"] = _truncate_text(summary_text)
    for key in ("time_start_s", "time_end_s", "frame_ts_s"):
        if evidence_entry.get(key) is not None:
            summary[key] = evidence_entry.get(key)
    if evidence_entry.get("evidence_id"):
        summary["evidence_id"] = evidence_entry.get("evidence_id")
    if observations:
        summary["observation_count"] = len(observations)
    data = dict(result_payload.get("data") or {})
    summary.update(_count_result_items(data))
    returned_clips = [_compact_clip_ref(item) for item in list(data.get("clips") or [])[:3]]
    returned_clips = [item for item in returned_clips if item is not None]
    if returned_clips:
        summary["returned_clips"] = returned_clips
    returned_frames = [_compact_frame_ref(item) for item in list(data.get("frames") or [])[:3]]
    returned_frames = [item for item in returned_frames if item is not None]
    if returned_frames:
        summary["returned_frames"] = returned_frames
    return summary


def _compact_step_summary(execution_record: Dict[str, Any]) -> Dict[str, Any]:
    request_payload = dict(execution_record.get("request") or {})
    result_payload = dict(execution_record.get("result") or {})
    evidence_entry = dict(execution_record.get("evidence_entry") or {})
    observations = list(execution_record.get("observations") or [])
    return {
        "step_id": execution_record.get("step_id"),
        "tool_name": execution_record.get("tool_name"),
        "purpose": _truncate_text(execution_record.get("purpose"), max_len=180),
        "request": _compact_request_summary(request_payload),
        "result": _compact_result_summary(result_payload, evidence_entry, observations),
    }


def _compact_round_summary(round_index: int, plan, execution_records, audit_report) -> Dict[str, object]:
    return {
        "round": int(round_index),
        "strategy": getattr(plan, "strategy", ""),
        "use_summary": bool(getattr(plan, "use_summary", False)),
        "refinement_instructions": _truncate_text(getattr(plan, "refinement_instructions", ""), max_len=240),
        "tools": [item["tool_name"] for item in execution_records],
        "step_summaries": [_compact_step_summary(item) for item in list(execution_records or [])],
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


def _trace_cue_bank(trace_package: Optional[TracePackage]) -> List[str]:
    if trace_package is None:
        return []
    cues: List[str] = []
    final_answer = str(getattr(trace_package, "final_answer", "") or "").strip()
    if final_answer:
        cues.append("Prior final answer hypothesis: %s" % final_answer)
    for item in list(getattr(trace_package, "inference_steps", []) or []):
        text = str(getattr(item, "text", "") or "").strip()
        if text:
            cues.append(text)
    ordered: List[str] = []
    seen = set()
    for cue in cues:
        normalized = " ".join(cue.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered[:12]


_BLOCKING_FINDING_CATEGORIES = {
    "ANSWER_ERROR",
    "ATTRIBUTION_GAP",
    "COUNTING_GAP",
    "INCOMPLETE_TRACE",
    "INFERENCE_ERROR",
    "READING_GAP",
    "TEMPORAL_GAP",
}


def _is_blocking_finding(finding: Dict[str, Any]) -> bool:
    severity = str((finding or {}).get("severity") or "").strip().upper()
    if severity not in {"HIGH", "MEDIUM"}:
        return False
    category = str((finding or {}).get("category") or "").strip().upper()
    if category in _BLOCKING_FINDING_CATEGORIES:
        return True
    message = str((finding or {}).get("message") or "").strip().casefold()
    return any(
        marker in message
        for marker in (
            "ambiguous",
            "attribute",
            "count",
            "earliest",
            "entity",
            "handed",
            "label",
            "missing",
            "not grounded",
            "speaker",
            "temporal",
            "timestamp",
            "unsupported",
            "unresolved",
        )
    )


def _should_accept_audit(audit_report: Optional[Any]) -> bool:
    if audit_report is None or str(getattr(audit_report, "verdict", "") or "").strip().upper() != "PASS":
        return False
    if list(getattr(audit_report, "missing_information", []) or []):
        return False
    findings = [dict(item) for item in list(getattr(audit_report, "findings", []) or []) if isinstance(item, dict)]
    return not any(_is_blocking_finding(item) for item in findings)


def _trace_supported_observation_ids(trace_package: Optional[TracePackage]) -> set[str]:
    supported = set()
    for step in list(getattr(trace_package, "inference_steps", []) or []):
        for observation_id in list(getattr(step, "supporting_observation_ids", []) or []):
            text = str(observation_id or "").strip()
            if text:
                supported.add(text)
    return supported


def _evidence_status_updates(
    evidence_entries: List[Dict[str, Any]],
    trace_package: Optional[TracePackage],
    *,
    accepted: bool,
) -> Dict[str, str]:
    cited_observation_ids = _trace_supported_observation_ids(trace_package)
    if accepted and not cited_observation_ids:
        return {}
    updates: Dict[str, str] = {}
    for entry in list(evidence_entries or []):
        if not isinstance(entry, dict):
            continue
        evidence_id = str(entry.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        prior_status = str(entry.get("status") or "provisional").strip().lower() or "provisional"
        observation_ids = {
            str(observation_id or "").strip()
            for observation_id in list(entry.get("observation_ids") or [])
            if str(observation_id or "").strip()
        }
        next_status = prior_status
        if accepted and observation_ids.intersection(cited_observation_ids):
            next_status = "validated"
        elif accepted and prior_status == "validated":
            next_status = "superseded"
        elif prior_status not in {"validated", "superseded"}:
            next_status = "provisional"
        if next_status != prior_status:
            updates[evidence_id] = next_status
    return updates


class PipelineRunner(object):
    def __init__(
        self,
        profile,
        models_config,
        persist_tool_models: Optional[List[str]] = None,
        preload_persisted_models: bool = False,
    ):
        self.profile = profile
        self.models_config = models_config
        self.preload_persisted_models = bool(preload_persisted_models)
        self._persisted_models_preloaded = False
        self.workspace = WorkspaceManager(profile)
        self.llm_client = OpenAIChatClient(profile, models_config)
        self.tool_registry = ToolRegistry(
            self.workspace,
            profile,
            models_config,
            llm_client=self.llm_client,
            persist_tool_models=persist_tool_models,
        )
        self.preprocessor = DenseCaptionPreprocessor(self.workspace, self.tool_registry, models_config)
        self.executor = PlanExecutor(
            tool_registry=self.tool_registry,
            evidence_cache=SharedEvidenceCache(self.workspace),
            extractor=ObservationExtractor(
                atomicizer=AtomicFactAgent(self.llm_client, models_config.agents["atomicizer"])
                if models_config.agents.get("atomicizer") is not None
                else None
            ),
            models_config=models_config,
        )
        self.plan_normalizer = ExecutionPlanNormalizer(self.tool_registry)
        self.planner = PlannerAgent(self.llm_client, models_config.agents["planner"])
        self.synthesizer = TraceSynthesizerAgent(self.llm_client, models_config.agents["trace_synthesizer"])
        self.auditor = TraceAuditorAgent(self.llm_client, models_config.agents["trace_auditor"])

    def close(self):
        self.tool_registry.close()

    def preload_models(self, progress_reporter=None):
        if self._persisted_models_preloaded or not self.preload_persisted_models:
            return {
                "enabled": False,
                "requested_tools": self.tool_registry.persistent_tool_names(),
                "loaded_models": [],
                "parallel_workers": 0,
                "shared_tools": [],
            }
        requested_tools = self.tool_registry.persistent_tool_names()
        if progress_reporter is not None and hasattr(progress_reporter, "on_model_preload_start"):
            progress_reporter.on_model_preload_start(tool_names=requested_tools)
        payload = self.tool_registry.preload_persistent_models()
        self._persisted_models_preloaded = True
        if progress_reporter is not None and hasattr(progress_reporter, "on_model_preload_end"):
            progress_reporter.on_model_preload_end(preload_payload=sanitize_for_persistence(payload))
        return payload

    def _write_runtime_snapshot(self, run):
        payload = {
            "machine": self.profile.redacted_snapshot(),
            "runtime_environment": sanitize_for_persistence(
                {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                    "requested_nvidia_smi_gpu_indices": os.environ.get("VTP_NVIDIA_SMI_GPU_INDICES"),
                    "machine_profile_path": os.environ.get("VTP_MACHINE_PROFILE_PATH"),
                    "models_config_path": os.environ.get("VTP_MODELS_CONFIG_PATH"),
                }
            ),
            "agent_models": {
                name: {
                    "backend": config.backend,
                    "model": config.model,
                    "endpoint": config.endpoint,
                    "prompt_version": config.prompt_version,
                }
                for name, config in sorted(self.models_config.agents.items())
            },
            "tool_implementations": {
                name: {
                    "implementation": tool_implementation(name),
                    "model": config.model,
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
        subjects = {}
        predicates = {}
        statuses = {}
        for item in observations:
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            if subject:
                subjects[subject] = subjects.get(subject, 0) + 1
            if predicate:
                predicates[predicate] = predicates.get(predicate, 0) + 1
        for entry in entries:
            status = str(entry.get("status") or "provisional").strip().lower() or "provisional"
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "evidence_entry_count": len(entries),
            "observation_count": len(observations),
            "evidence_status_counts": statuses,
            "top_subjects": sorted(subjects.items(), key=lambda pair: (-pair[1], pair[0]))[:15],
            "top_predicates": sorted(predicates.items(), key=lambda pair: (-pair[1], pair[0]))[:15],
            "evidence_entries": entries[-10:],
            "recent_observations": observations[-20:],
        }

    def _refresh_run_readables(self, run) -> None:
        with contextlib.suppress(Exception):
            write_run_readable_bundle(run.run_dir)

    def run_task(
        self,
        task,
        mode: str = "generate",
        max_rounds: int = 3,
        clip_duration_s: Optional[float] = None,
        initial_trace_path: Optional[str] = None,
        results_name: Optional[str] = None,
        progress_reporter=None,
    ) -> Dict[str, object]:
        preprocess_settings = self.preprocessor.resolve_preprocess_settings(clip_duration_s)
        effective_clip_duration_s = float(preprocess_settings["clip_duration_s"])
        run = self.workspace.create_run(task)
        self._write_runtime_snapshot(run)
        manifest_payload = {
            "benchmark": task.benchmark,
            "sample_key": task.sample_key,
            "run_id": run.run_id,
            "mode": mode,
            "task": task.persistable_dict(),
            "preprocess_cache": None,
            "clip_duration_s": effective_clip_duration_s,
            "results_name": str(results_name or "").strip() or None,
        }
        self.workspace.write_run_manifest(run, manifest_payload)
        self._refresh_run_readables(run)
        if progress_reporter is not None and hasattr(progress_reporter, "on_run_start"):
            progress_reporter.on_run_start(
                task=task,
                run_dir=self.workspace.relative_path(run.run_dir),
                mode=mode,
                max_rounds=max_rounds,
                clip_duration_s=effective_clip_duration_s,
            )
        self.preload_models(progress_reporter=progress_reporter)
        if progress_reporter is not None and hasattr(progress_reporter, "on_preprocess_start"):
            progress_reporter.on_preprocess_start()
        preprocess_output = self.preprocessor.get_or_build(task, clip_duration_s=clip_duration_s)
        if progress_reporter is not None and hasattr(progress_reporter, "on_preprocess_end"):
            progress_reporter.on_preprocess_end(sanitize_for_persistence(preprocess_output))
        video_fingerprint = preprocess_output["video_fingerprint"]
        summary_text = preprocess_output["summary"]
        preprocess_planning_memory = dict(preprocess_output.get("planner_context") or {})
        manifest_payload["preprocess_cache"] = preprocess_output["cache_dir"]
        manifest_payload["clip_duration_s"] = effective_clip_duration_s
        self.workspace.write_run_manifest(run, manifest_payload)
        self._refresh_run_readables(run)
        evidence_ledger = EvidenceLedger(run)
        execution_context = ToolExecutionContext(
            workspace=self.workspace,
            run=run,
            task=task,
            models_config=self.models_config,
            llm_client=self.llm_client,
            evidence_lookup=evidence_ledger.lookup_records,
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
            self._refresh_run_readables(run)
            if progress_reporter is not None and hasattr(progress_reporter, "on_initial_audit"):
                progress_reporter.on_initial_audit(sanitize_for_persistence(latest_audit.dict()))
            evidence_updates = _evidence_status_updates(
                evidence_ledger.entries(),
                current_trace,
                accepted=_should_accept_audit(latest_audit),
            )
            if evidence_updates:
                evidence_ledger.update_entry_statuses(evidence_updates)
            if _should_accept_audit(latest_audit):
                export_payload = self._persist_trace(run, task, current_trace)
                export_target = self.workspace.export_run_target(run, task, results_name=results_name)
                final_payload = {
                    "run_dir": self.workspace.relative_path(run.run_dir),
                    "trace_package": sanitize_for_persistence(current_trace.dict()),
                    "audit_report": latest_audit.dict(),
                    "benchmark_export": export_payload,
                }
                if export_target is not None:
                    final_payload["exported_results_dir"] = str(export_target)
                write_json(run.results_dir / "final_result.json", final_payload)
                with contextlib.suppress(Exception):
                    write_run_debug_bundle(run.run_dir)
                self.workspace.export_run_bundle(run, task, results_name=results_name)
                if progress_reporter is not None and hasattr(progress_reporter, "on_complete"):
                    progress_reporter.on_complete(final_payload=sanitize_for_persistence(final_payload))
                return final_payload

        rounds_executed = 0
        while rounds_executed < max(1, int(max_rounds)):
            planning_mode = "generate" if current_trace is None else "refine"
            summary_context_supplied = bool(str(summary_text or "").strip()) and (
                current_trace is None or latest_audit is None or bool(latest_audit.missing_information)
            )
            existing_entries = evidence_ledger.entries()
            prefer_validated_reuse = any(str(item.get("status") or "").strip().lower() == "validated" for item in existing_entries)
            retrieved_observations = evidence_ledger.retrieve(
                query_terms=_query_terms(task, latest_audit.dict() if latest_audit else None),
                evidence_status="validated" if prefer_validated_reuse else None,
                limit=50,
            )
            if prefer_validated_reuse and not retrieved_observations:
                retrieved_observations = evidence_ledger.retrieve(
                    query_terms=_query_terms(task, latest_audit.dict() if latest_audit else None),
                    limit=50,
                )
            if progress_reporter is not None and hasattr(progress_reporter, "on_round_start"):
                progress_reporter.on_round_start(
                    round_index=rounds_executed + 1,
                    planning_mode=planning_mode,
                    use_summary=summary_context_supplied,
                    retrieved_count=len(list(retrieved_observations or [])),
                )
            planner_kwargs = dict(
                task=task,
                mode=planning_mode,
                summary_text=summary_text if summary_context_supplied else "",
                compact_rounds=compact_rounds,
                retrieved_observations=retrieved_observations,
                current_trace_cues=_trace_cue_bank(current_trace),
                preprocess_planning_memory=preprocess_planning_memory,
                audit_feedback=latest_audit.dict() if latest_audit is not None else None,
                tool_catalog=self.tool_registry.tool_catalog(),
            )
            planner_raw, plan = self.planner.plan(**planner_kwargs)
            plan = self.plan_normalizer.normalize(task, plan)
            round_prefix = "round_%02d" % (rounds_executed + 1)
            write_text(run.planner_dir / ("%s_raw.txt" % round_prefix), planner_raw)
            write_json(run.planner_dir / ("%s_plan.json" % round_prefix), sanitize_for_persistence(plan.dict()))
            self._refresh_run_readables(run)
            if progress_reporter is not None and hasattr(progress_reporter, "on_planner"):
                progress_reporter.on_planner(
                    round_index=rounds_executed + 1,
                    plan_payload=sanitize_for_persistence(plan.dict()),
                    planner_dir=self.workspace.relative_path(run.planner_dir),
                )

            execution_records = self.executor.execute_plan(
                plan=plan,
                execution_context=execution_context,
                evidence_ledger=evidence_ledger,
                video_fingerprint=video_fingerprint,
                progress_reporter=progress_reporter,
                round_index=rounds_executed + 1,
            )
            self._refresh_run_readables(run)

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
            self._refresh_run_readables(run)
            if progress_reporter is not None and hasattr(progress_reporter, "on_trace"):
                progress_reporter.on_trace(
                    round_index=rounds_executed + 1,
                    trace_payload=sanitize_for_persistence(trace_package.dict()),
                    trace_dir=self.workspace.relative_path(run.synthesizer_dir),
                )

            evidence_summary = self._build_evidence_summary(evidence_ledger)
            audit_raw, latest_audit = self.auditor.audit(
                task=task,
                trace_package=sanitize_for_persistence(trace_package.dict()),
                evidence_summary=evidence_summary,
            )
            write_text(run.auditor_dir / ("%s_raw.txt" % round_prefix), audit_raw)
            write_json(run.auditor_dir / ("%s_report.json" % round_prefix), latest_audit.dict())
            self._refresh_run_readables(run)
            if progress_reporter is not None and hasattr(progress_reporter, "on_audit"):
                progress_reporter.on_audit(
                    round_index=rounds_executed + 1,
                    audit_payload=sanitize_for_persistence(latest_audit.dict()),
                    auditor_dir=self.workspace.relative_path(run.auditor_dir),
                )

            current_trace = trace_package
            evidence_updates = _evidence_status_updates(
                evidence_ledger.entries(),
                current_trace,
                accepted=_should_accept_audit(latest_audit),
            )
            if evidence_updates:
                evidence_ledger.update_entry_statuses(evidence_updates)
            compact_rounds.append(_compact_round_summary(rounds_executed + 1, plan, execution_records, latest_audit))
            rounds_executed += 1
            if _should_accept_audit(latest_audit):
                break

        export_payload = self._persist_trace(run, task, current_trace)
        export_target = self.workspace.export_run_target(run, task, results_name=results_name)
        final_payload = {
            "run_dir": self.workspace.relative_path(run.run_dir),
            "trace_package": sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
            "audit_report": latest_audit.dict() if latest_audit is not None else None,
            "benchmark_export": sanitize_for_persistence(export_payload),
            "preprocess": sanitize_for_persistence(preprocess_output),
            "rounds_executed": rounds_executed,
        }
        if export_target is not None:
            final_payload["exported_results_dir"] = str(export_target)
        write_json(run.results_dir / "final_result.json", final_payload)
        with contextlib.suppress(Exception):
            write_run_debug_bundle(run.run_dir)
        self.workspace.export_run_bundle(run, task, results_name=results_name)
        if progress_reporter is not None and hasattr(progress_reporter, "on_complete"):
            progress_reporter.on_complete(final_payload=sanitize_for_persistence(final_payload))
        return final_payload
