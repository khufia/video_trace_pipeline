from __future__ import annotations

import contextlib
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ..agents import AtomicFactAgent, OpenAIChatClient, PlannerAgent, TraceAuditorAgent, TraceSynthesizerAgent
from ..common import sanitize_for_persistence, write_json
from ..config import save_runtime_snapshot
from ..renderers import export_trace_for_benchmark, write_run_debug_bundle
from ..schemas import InferenceStep, PlanStep, TracePackage
from ..storage import EvidenceLedger, WorkspaceManager
from ..tools import ObservationExtractor, ToolRegistry
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation
from .executor import PlanExecutor
from .preprocess import DenseCaptionPreprocessor


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
    if isinstance(evidence_entry.get("time_intervals"), list) and evidence_entry.get("time_intervals"):
        summary["time_intervals"] = list(evidence_entry.get("time_intervals") or [])
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


def _tool_output_history(execution_record: Dict[str, Any]) -> Dict[str, Any]:
    result_payload = dict(execution_record.get("result") or {})
    return sanitize_for_persistence(
        {
            "step_id": execution_record.get("step_id"),
            "tool_name": execution_record.get("tool_name"),
            "purpose": execution_record.get("purpose"),
            "request": execution_record.get("request") or {},
            "result": {
                "ok": result_payload.get("ok"),
                "data": result_payload.get("data") or {},
                "summary": result_payload.get("summary"),
                "raw_output_text": result_payload.get("raw_output_text"),
                "artifact_refs": result_payload.get("artifact_refs") or [],
                "metadata": result_payload.get("metadata") or {},
            },
            "evidence_entry": execution_record.get("evidence_entry") or {},
            "observations": execution_record.get("observations") or [],
        }
    )


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


def _dedupe_evidence_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for entry in list(entries or []):
        if not isinstance(entry, dict):
            continue
        evidence_id = str(entry.get("evidence_id") or "").strip()
        signature = evidence_id or json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(dict(entry))
    return ordered


def _dedupe_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for observation in list(observations or []):
        if not isinstance(observation, dict):
            continue
        observation_id = str(observation.get("observation_id") or "").strip()
        evidence_id = str(observation.get("evidence_id") or "").strip()
        text = str(observation.get("atomic_text") or observation.get("text") or "").strip()
        signature = observation_id or (evidence_id, text) or json.dumps(observation, sort_keys=True, ensure_ascii=False, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(dict(observation))
    return ordered


def _combined_preprocess_context(preprocess_output: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(preprocess_output, dict):
        return None
    planner_segments = preprocess_output.get("planner_segments")
    if not isinstance(planner_segments, list):
        return None
    context = {
        "source": "planner_segments.json",
        "cache_dir": preprocess_output.get("cache_dir"),
        "manifest": {
            "video_id": preprocess_output.get("video_id"),
            "video_fingerprint": preprocess_output.get("video_fingerprint"),
            "clip_duration_s": preprocess_output.get("clip_duration_s"),
            "preprocess_settings": preprocess_output.get("preprocess_settings"),
            "metrics": preprocess_output.get("metrics"),
        },
        "planner_segments": planner_segments,
    }
    return sanitize_for_persistence(context)


def _estimate_prompt_tokens_from_request(request: Dict[str, Any]) -> int:
    prompt_text = "\n\n".join(
        str(request.get(key) or "")
        for key in ("system_prompt", "user_prompt")
    )
    # Cheap estimator for warning/telemetry only. We intentionally do not block on it.
    return max(1, int(len(prompt_text) / 4))


def _observation_text(observation: Dict[str, Any]) -> str:
    text = (
        observation.get("atomic_text")
        or observation.get("text")
        or observation.get("support")
        or observation.get("value")
        or ""
    )
    return _truncate_text(text, max_len=900)


def _ledger_synthesis_context(ledger: EvidenceLedger) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return _dedupe_evidence_entries(ledger.entries()), _dedupe_observations(ledger.observations())


def _unresolved_trace(task, missing_information: List[str]) -> TracePackage:
    missing = [str(item or "").strip() for item in list(missing_information or []) if str(item or "").strip()]
    text = "The pipeline stopped before producing a supported answer."
    if missing:
        text += " Missing information: %s." % "; ".join(missing[:5])
    return TracePackage(
        task_key=task.sample_key,
        mode="unresolved",
        evidence_entries=[],
        inference_steps=[
            InferenceStep(
                step_id=1,
                text=text,
                supporting_observation_ids=[],
                answer_relevance="high",
            )
        ],
        final_answer="",
        benchmark_renderings={},
        metadata={"source": "planner_action_loop", "missing_information": missing},
    )


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
        prior_status = str(entry.get("status") or "candidate").strip().lower() or "candidate"
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
        elif prior_status not in {"validated", "superseded", "refuted", "irrelevant", "stale", "unknown"}:
            next_status = "candidate"
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
            extractor=ObservationExtractor(
                atomicizer=AtomicFactAgent(self.llm_client, models_config.agents["atomicizer"])
                if models_config.agents.get("atomicizer") is not None
                else None
            ),
            models_config=models_config,
        )
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
        write_json(run.trace_package_path, trace_dict)
        export_payload = export_trace_for_benchmark(task.benchmark, task, trace_dict)
        write_json(run.benchmark_export_path, sanitize_for_persistence(export_payload))
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
            status = str(entry.get("status") or "candidate").strip().lower() or "candidate"
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "evidence_entry_count": len(entries),
            "observation_count": len(observations),
            "evidence_status_counts": statuses,
            "top_subjects": sorted(subjects.items(), key=lambda pair: (-pair[1], pair[0]))[:15],
            "top_predicates": sorted(predicates.items(), key=lambda pair: (-pair[1], pair[0]))[:15],
            "evidence_entries": entries[-10:],
            "recent_observations": [
                {
                    **dict(item),
                    "text": _observation_text(dict(item)),
                }
                for item in observations[-20:]
            ],
        }

    def run_task(
        self,
        task,
        mode: str = "generate",
        max_rounds: int = 15,
        clip_duration_s: Optional[float] = None,
        initial_trace_path: Optional[str] = None,
        progress_reporter=None,
    ) -> Dict[str, object]:
        video_id = task.video_id or task.sample_key
        run = self.workspace.create_run(task)
        self.workspace.clear_video_artifacts(video_id)
        try:
            return self._run_task_impl(
                run=run,
                task=task,
                mode=mode,
                max_rounds=max_rounds,
                clip_duration_s=clip_duration_s,
                initial_trace_path=initial_trace_path,
                progress_reporter=progress_reporter,
            )
        finally:
            self.workspace.clear_video_artifacts(video_id)

    def _run_task_impl(
        self,
        run,
        task,
        mode: str = "generate",
        max_rounds: int = 15,
        clip_duration_s: Optional[float] = None,
        initial_trace_path: Optional[str] = None,
        progress_reporter=None,
    ) -> Dict[str, object]:
        preprocess_enabled = self.preprocessor.is_enabled()
        preprocess_settings = self.preprocessor.resolve_preprocess_settings(clip_duration_s) if preprocess_enabled else {}
        effective_clip_duration_s = (
            float(preprocess_settings["clip_duration_s"])
            if preprocess_enabled and preprocess_settings.get("clip_duration_s") is not None
            else None
        )
        self._write_runtime_snapshot(run)
        manifest_payload = {
            "benchmark": task.benchmark,
            "sample_key": task.sample_key,
            "video_id": task.video_id or task.sample_key,
            "run_id": run.run_id,
            "mode": mode,
            "task": task.persistable_dict(),
            "preprocess_enabled": bool(preprocess_enabled),
            "preprocess_cache": None,
            "clip_duration_s": effective_clip_duration_s,
            "max_rounds": int(max_rounds),
        }
        self.workspace.write_run_manifest(run, manifest_payload)
        if progress_reporter is not None and hasattr(progress_reporter, "on_run_start"):
            progress_reporter.on_run_start(
                task=task,
                run_dir=self.workspace.relative_path(run.run_dir),
                mode=mode,
                max_rounds=max_rounds,
                clip_duration_s=effective_clip_duration_s,
            )
        self.preload_models(progress_reporter=progress_reporter)
        preprocess_output = None
        if preprocess_enabled:
            if progress_reporter is not None and hasattr(progress_reporter, "on_preprocess_start"):
                progress_reporter.on_preprocess_start()
            preprocess_output = self.preprocessor.get_or_build(task, clip_duration_s=clip_duration_s)
            if progress_reporter is not None and hasattr(progress_reporter, "on_preprocess_end"):
                progress_reporter.on_preprocess_end(sanitize_for_persistence(preprocess_output))
            manifest_payload["preprocess_cache"] = preprocess_output.get("cache_dir")
            manifest_payload["clip_duration_s"] = effective_clip_duration_s
            self.workspace.write_run_manifest(run, manifest_payload)
        preprocess_context = _combined_preprocess_context(preprocess_output)
        if preprocess_enabled and preprocess_context is None:
            raise RuntimeError(
                "Preprocessing is enabled, but the preprocess bundle did not contain planner_segments from planner_segments.json."
            )
        video_fingerprint = (
            str(preprocess_output.get("video_fingerprint"))
            if isinstance(preprocess_output, dict) and preprocess_output.get("video_fingerprint")
            else self.workspace.video_fingerprint(task.video_path)
        )
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
        latest_audit = None

        if current_trace is not None:
            current_trace_dict = sanitize_for_persistence(current_trace.dict())
            initial_round_dir = run.round_dir(0)
            write_json(initial_round_dir / "initial_trace_package.json", current_trace_dict)
            initial_summary = self._build_evidence_summary(evidence_ledger)
            initial_audit_request = self.auditor.build_request(
                task,
                current_trace_dict,
                initial_summary,
                preprocess_context=preprocess_context,
            )
            write_json(initial_round_dir / "auditor_request.json", sanitize_for_persistence(initial_audit_request))
            _, latest_audit = self.auditor.complete_request(initial_audit_request)
            write_json(initial_round_dir / "auditor_report.json", latest_audit.dict())
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
                final_payload = {
                    "run_dir": self.workspace.relative_path(run.run_dir),
                    "trace_package": sanitize_for_persistence(current_trace.dict()),
                    "audit_report": latest_audit.dict(),
                    "benchmark_export": export_payload,
                    "preprocess": sanitize_for_persistence(preprocess_output) if preprocess_output is not None else None,
                }
                write_json(run.final_result_path, final_payload)
                with contextlib.suppress(Exception):
                    write_run_debug_bundle(run.run_dir)
                if progress_reporter is not None and hasattr(progress_reporter, "on_complete"):
                    progress_reporter.on_complete(final_payload=sanitize_for_persistence(final_payload))
                return final_payload

        rounds_executed = 0
        planner_turns = 0
        synthesis_attempts = 0
        max_synthesis_attempts = 3
        max_tool_rounds = max(1, int(max_rounds))
        max_planner_turns = max_tool_rounds + max_synthesis_attempts + 2
        action_history: List[Dict[str, Any]] = []
        stop_missing_information: List[str] = []
        while planner_turns < max_planner_turns:
            planning_mode = "generate" if current_trace is None else "refine"
            planner_turns += 1
            round_index = planner_turns
            round_dir = run.round_dir(round_index)
            tool_catalog = {
                name: spec
                for name, spec in self.tool_registry.tool_catalog().items()
                if name != "verifier"
            }
            if progress_reporter is not None and hasattr(progress_reporter, "on_round_start"):
                progress_reporter.on_round_start(
                    round_index=round_index,
                    planning_mode=planning_mode,
                    retrieved_count=0,
                )
            planner_evidence_summary = self._build_evidence_summary(evidence_ledger)
            planner_kwargs = dict(
                task=task,
                mode=planning_mode,
                audit_feedback=latest_audit.dict() if latest_audit is not None else None,
                tool_catalog=tool_catalog,
                evidence_summary=planner_evidence_summary,
                preprocess_context=preprocess_context,
                action_history=action_history,
                current_trace=sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
            )
            planner_request = self.planner.build_request(**planner_kwargs)
            estimated_prompt_tokens = _estimate_prompt_tokens_from_request(planner_request)
            if estimated_prompt_tokens >= 100000:
                warning_payload = {
                    "type": "large_planner_prompt",
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "message": "Planner prompt is large because full planner_segments preprocess context is included.",
                }
                manifest_payload.setdefault("warnings", []).append(warning_payload)
                self.workspace.write_run_manifest(run, manifest_payload)
                write_json(round_dir / "planner_prompt_warning.json", warning_payload)
            write_json(round_dir / "planner_request.json", sanitize_for_persistence(planner_request))
            planner_raw, action = self.planner.complete_request(planner_request)
            write_json(round_dir / "planner_raw.json", {"raw": planner_raw})
            action_payload = sanitize_for_persistence(action.dict())
            write_json(round_dir / "planner_action.json", action_payload)
            if progress_reporter is not None and hasattr(progress_reporter, "on_planner"):
                progress_reporter.on_planner(
                    round_index=round_index,
                    plan_payload=action_payload,
                    round_dir=self.workspace.relative_path(round_dir),
                )

            action_record: Dict[str, Any] = {
                "turn": round_index,
                "mode": planning_mode,
                "action": action_payload,
            }

            if action.action_type == "tool_call":
                if rounds_executed >= max_tool_rounds:
                    stop_missing_information = [
                        "Planner requested another tool call after the max_rounds tool budget was exhausted."
                    ]
                    action_record["error"] = "tool_budget_exhausted"
                    action_record["rounds_executed"] = rounds_executed
                    action_record["max_rounds"] = max_tool_rounds
                    action_history.append(action_record)
                    write_json(round_dir / "action_history.json", sanitize_for_persistence(action_history))
                    break
                tool_request = dict(action.tool_request or {})
                tool_request.setdefault("tool_name", action.tool_name)
                if action.tool_name == "verifier":
                    raise ValueError("Planner attempted to call disabled verifier tool.")
                plan = SimpleNamespace(
                    steps=[
                        PlanStep(
                            step_id=rounds_executed + 1,
                            tool_name=str(action.tool_name),
                            purpose=action.expected_observation or action.rationale,
                            inputs=tool_request,
                        )
                    ]
                )
                execution_records = self.executor.execute_plan(
                    plan=plan,
                    execution_context=execution_context,
                    evidence_ledger=evidence_ledger,
                    video_fingerprint=video_fingerprint,
                    progress_reporter=progress_reporter,
                    round_index=round_index,
                )
                action_record["tool_results"] = [
                    _compact_step_summary(record)
                    for record in execution_records
                ]
                action_record["tool_outputs"] = [
                    _tool_output_history(record)
                    for record in execution_records
                ]
                action_history.append(action_record)
                write_json(round_dir / "action_history.json", sanitize_for_persistence(action_history))
                rounds_executed += 1
                continue

            if action.action_type == "stop_unresolved":
                stop_missing_information = list(action.missing_information or [])
                action_history.append(action_record)
                write_json(round_dir / "action_history.json", sanitize_for_persistence(action_history))
                break

            synthesis_attempts += 1
            round_evidence_entries, round_observations = _ledger_synthesis_context(evidence_ledger)
            synthesizer_request = self.synthesizer.build_request(
                task=task,
                mode=planning_mode,
                round_evidence_entries=round_evidence_entries,
                round_observations=round_observations,
                current_trace=sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
                refinement_instructions=action.synthesis_instructions or action.rationale,
                audit_feedback=latest_audit.dict() if latest_audit is not None else None,
                preprocess_context=preprocess_context,
            )
            write_json(round_dir / "synthesizer_request.json", sanitize_for_persistence(synthesizer_request))
            _, trace_package = self.synthesizer.complete_request(synthesizer_request)
            trace_dict = trace_package.dict()
            trace_dict["task_key"] = task.sample_key
            trace_dict["mode"] = planning_mode
            trace_dict["benchmark_renderings"][task.benchmark] = export_trace_for_benchmark(task.benchmark, task, trace_dict)
            trace_package = TracePackage.parse_obj(trace_dict)
            write_json(round_dir / "synthesizer_trace_package.json", sanitize_for_persistence(trace_package.dict()))
            if progress_reporter is not None and hasattr(progress_reporter, "on_trace"):
                progress_reporter.on_trace(
                    round_index=round_index,
                    trace_payload=sanitize_for_persistence(trace_package.dict()),
                    round_dir=self.workspace.relative_path(round_dir),
                )

            evidence_summary = self._build_evidence_summary(evidence_ledger)
            auditor_request = self.auditor.build_request(
                task=task,
                trace_package=sanitize_for_persistence(trace_package.dict()),
                evidence_summary=evidence_summary,
                preprocess_context=preprocess_context,
            )
            write_json(round_dir / "auditor_request.json", sanitize_for_persistence(auditor_request))
            _, latest_audit = self.auditor.complete_request(auditor_request)
            write_json(round_dir / "auditor_report.json", latest_audit.dict())
            if progress_reporter is not None and hasattr(progress_reporter, "on_audit"):
                progress_reporter.on_audit(
                    round_index=round_index,
                    audit_payload=sanitize_for_persistence(latest_audit.dict()),
                    round_dir=self.workspace.relative_path(round_dir),
                )

            current_trace = trace_package
            action_record["trace_package"] = sanitize_for_persistence(trace_package.dict())
            action_record["audit_report"] = sanitize_for_persistence(latest_audit.dict())
            action_history.append(action_record)
            write_json(round_dir / "action_history.json", sanitize_for_persistence(action_history))
            evidence_updates = _evidence_status_updates(
                evidence_ledger.entries(),
                current_trace,
                accepted=_should_accept_audit(latest_audit),
            )
            if evidence_updates:
                evidence_ledger.update_entry_statuses(evidence_updates)
            if _should_accept_audit(latest_audit):
                break
            if synthesis_attempts >= max_synthesis_attempts:
                break

        if current_trace is None:
            current_trace = _unresolved_trace(task, stop_missing_information)
        export_payload = self._persist_trace(run, task, current_trace)
        final_payload = {
            "run_dir": self.workspace.relative_path(run.run_dir),
            "trace_package": sanitize_for_persistence(current_trace.dict()) if current_trace is not None else None,
            "audit_report": latest_audit.dict() if latest_audit is not None else None,
            "benchmark_export": sanitize_for_persistence(export_payload),
            "preprocess": sanitize_for_persistence(preprocess_output) if preprocess_output is not None else None,
            "rounds_executed": rounds_executed,
            "planner_turns": planner_turns,
            "synthesis_attempts": synthesis_attempts,
            "action_history": sanitize_for_persistence(action_history),
        }
        write_json(run.final_result_path, final_payload)
        with contextlib.suppress(Exception):
            write_run_debug_bundle(run.run_dir)
        if progress_reporter is not None and hasattr(progress_reporter, "on_complete"):
            progress_reporter.on_complete(final_payload=sanitize_for_persistence(final_payload))
        return final_payload
