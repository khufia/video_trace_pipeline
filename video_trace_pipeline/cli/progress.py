from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_scores(scores: Dict[str, Any]) -> str:
    if not isinstance(scores, dict) or not scores:
        return "none"
    parts = []
    for key in sorted(scores):
        value = scores.get(key)
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric.is_integer():
                parts.append("%s=%d" % (key, int(numeric)))
            else:
                parts.append("%s=%0.2f" % (key, numeric))
        else:
            parts.append("%s=%s" % (key, value))
    return ", ".join(parts)


def _format_seconds(value: Any) -> str:
    text = "%.3f" % float(value)
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _render_temporal_anchor(item: Dict[str, Any]) -> str:
    start_s = item.get("time_start_s")
    end_s = item.get("time_end_s")
    frame_ts_s = item.get("frame_ts_s")
    if start_s is not None or end_s is not None:
        if start_s is None:
            start_s = end_s
        if end_s is None:
            end_s = start_s
        if float(start_s) == float(end_s):
            return _format_seconds(start_s)
        return "%s to %s" % (_format_seconds(start_s), _format_seconds(end_s))
    if frame_ts_s is not None:
        return _format_seconds(frame_ts_s)
    return ""


def _inference_time_anchor(step_payload: Dict[str, Any], evidence_entries: List[Dict[str, Any]]) -> str:
    supporting_ids = {
        str(item).strip()
        for item in list(step_payload.get("supporting_observation_ids") or [])
        if str(item).strip()
    }
    if not supporting_ids:
        return ""
    anchors = []
    seen = set()
    for item in list(evidence_entries or []):
        observation_ids = {
            str(observation_id).strip()
            for observation_id in list(item.get("observation_ids") or [])
            if str(observation_id).strip()
        }
        if not supporting_ids.intersection(observation_ids):
            continue
        anchor = _render_temporal_anchor(dict(item or {}))
        if not anchor or anchor in seen:
            continue
        seen.add(anchor)
        anchors.append(anchor)
    if not anchors:
        return ""
    if len(anchors) == 1:
        return anchors[0]
    return "%s | %s" % (anchors[0], anchors[1])


def _format_confidence(result_payload: Dict[str, Any]) -> str:
    metadata = dict(result_payload.get("metadata") or {})
    confidence = metadata.get("confidence")
    if not isinstance(confidence, (int, float)):
        return ""
    parts = ["max=%0.4f" % float(confidence)]
    confidence_avg = metadata.get("confidence_avg")
    if isinstance(confidence_avg, (int, float)):
        parts.append("avg=%0.4f" % float(confidence_avg))
    confidence_count = metadata.get("confidence_count")
    if isinstance(confidence_count, int) and confidence_count > 1:
        parts.append("n=%d" % int(confidence_count))
    return " ".join(parts)


def _iter_tool_steps(plan_payload: Dict[str, Any]) -> Iterable[str]:
    for item in list(plan_payload.get("steps") or []):
        if not isinstance(item, dict):
            continue
        step_id = item.get("step_id")
        tool_name = str(item.get("tool_name") or "").strip()
        purpose = _short_text(item.get("purpose") or "", limit=140)
        depends_on = list(item.get("depends_on") or [])
        arguments = dict(item.get("arguments") or {})
        parts = ["%s. %s" % (step_id, tool_name or "tool")]
        if purpose:
            parts.append(purpose)
        if arguments:
            parts.append("args=%s" % _short_text(arguments, limit=180))
        if depends_on:
            parts.append("depends_on=%s" % ",".join(str(value) for value in depends_on))
        yield " | ".join(parts)


class LiveRunReporter(object):
    def __init__(self, console):
        self.console = console

    def on_run_start(self, *, task, run_dir: str, mode: str, max_rounds: int, clip_duration_s: float):
        self.console.print("")
        self.console.print(
            "[bold cyan]Run[/bold cyan] %s | mode=%s | max_rounds=%s | clip_duration=%ss"
            % (task.sample_key, mode, int(max_rounds), int(float(clip_duration_s)))
        )
        self.console.print("question: %s" % _short_text(task.question, limit=240))
        if list(task.options or []):
            self.console.print("options: %s" % " | ".join(str(item) for item in list(task.options or [])))
        self.console.print("run_dir: %s" % run_dir)

    def on_model_preload_start(self, *, tool_names: List[str]):
        if not list(tool_names or []):
            return
        self.console.print("")
        self.console.print("[bold cyan]Model Preload[/bold cyan]")
        self.console.print("tools: %s" % " | ".join(str(item) for item in list(tool_names or [])))

    def on_model_preload_end(self, *, preload_payload: Dict[str, Any]):
        if not bool(preload_payload.get("enabled")):
            return
        self.console.print(
            "loaded=%s | parallel_workers=%s"
            % (
                len(list(preload_payload.get("loaded_models") or [])),
                int(preload_payload.get("parallel_workers") or 0),
            )
        )
        for item in list(preload_payload.get("loaded_models") or []):
            tool_name = str(item.get("tool_name") or "").strip()
            device_label = str(item.get("device_label") or "").strip()
            resolved_model_path = str(item.get("resolved_model_path") or "").strip()
            if tool_name and device_label and resolved_model_path:
                self.console.print("  - %s on %s -> %s" % (tool_name, device_label, resolved_model_path))
        for item in list(preload_payload.get("shared_tools") or []):
            tool_name = str(item.get("tool_name") or "").strip()
            shared_with = str(item.get("shared_with") or "").strip()
            if tool_name and shared_with:
                self.console.print("  - %s shares the persisted runner with %s" % (tool_name, shared_with))

    def on_preprocess_start(self):
        self.console.print("")
        self.console.print("[bold cyan]Preprocess[/bold cyan] dense-caption summary")

    def on_preprocess_end(self, preprocess_output: Dict[str, Any]):
        self.console.print(
            "cache_hit=%s | cache_dir=%s"
            % (
                bool(preprocess_output.get("cache_hit")),
                str(preprocess_output.get("cache_dir") or ""),
            )
        )
        self.console.print("summary: %s" % _short_text(preprocess_output.get("summary") or "", limit=300))

    def on_initial_audit(self, audit_payload: Dict[str, Any]):
        self.console.print("")
        self.console.print("[bold cyan]Initial Audit[/bold cyan]")
        self.console.print(
            "verdict=%s | scores=%s"
            % (
                str(audit_payload.get("verdict") or ""),
                _format_scores(dict(audit_payload.get("scores") or {})),
            )
        )
        feedback = _short_text(audit_payload.get("feedback") or "", limit=260)
        if feedback:
            self.console.print("feedback: %s" % feedback)

    def on_round_start(
        self,
        *,
        round_index: int,
        planning_mode: str,
        use_summary: bool,
        retrieved_count: int,
    ):
        self.console.print("")
        self.console.print(
            "[bold green]Round %02d[/bold green] mode=%s | summary_context=%s | retrieved_observations=%s"
            % (int(round_index), planning_mode, bool(use_summary), int(retrieved_count))
        )

    def on_planner(self, *, round_index: int, plan_payload: Dict[str, Any], planner_dir: Optional[str] = None):
        self.console.print("[bold]Planner[/bold]")
        self.console.print(
            "strategy: %s"
            % _short_text(plan_payload.get("strategy") or "", limit=220)
        )
        self.console.print("plan_use_summary: %s" % bool(plan_payload.get("use_summary")))
        for line in _iter_tool_steps(plan_payload):
            self.console.print("  %s" % line)
        instructions = _short_text(plan_payload.get("refinement_instructions") or "", limit=260)
        if instructions:
            self.console.print("refinement: %s" % instructions)
        if planner_dir:
            self.console.print("planner_dir: %s" % planner_dir)

    def on_tool_start(
        self,
        *,
        round_index: int,
        step_id: int,
        tool_name: str,
        purpose: str,
        request_payload: Dict[str, Any],
    ):
        self.console.print("")
        self.console.print("[bold]Tool %02d[/bold] %s" % (int(step_id), tool_name))
        if purpose:
            self.console.print("purpose: %s" % _short_text(purpose, limit=220))
        self.console.print("request: %s" % _short_text(request_payload, limit=320))

    def on_tool_end(
        self,
        *,
        round_index: int,
        step_id: int,
        tool_name: str,
        result_payload: Dict[str, Any],
        observations: List[Dict[str, Any]],
        step_dir: Optional[str] = None,
    ):
        summary = _short_text(result_payload.get("summary") or "", limit=280)
        metadata = dict(result_payload.get("metadata") or {})
        parts = [
            "status=%s" % ("ok" if result_payload.get("ok", True) else "error"),
            "result_cache_hit=%s" % bool(result_payload.get("cache_hit")),
            "observations=%s" % len(list(observations or [])),
        ]
        if metadata.get("dense_frame_cache_hit") is not None:
            parts.append("frame_cache_hit=%s" % bool(metadata.get("dense_frame_cache_hit")))
        confidence = _format_confidence(result_payload)
        if confidence:
            parts.append("confidence=%s" % confidence)
        self.console.print(" | ".join(parts))
        if summary:
            self.console.print("output: %s" % summary)
        prefilter = dict((result_payload.get("metadata") or {}).get("prefilter") or {})
        if prefilter:
            total_windows = prefilter.get("total_windows")
            candidate_windows = prefilter.get("candidate_windows")
            dense_frame_count = prefilter.get("dense_frame_count")
            if total_windows is not None and candidate_windows is not None:
                extra = []
                if dense_frame_count is not None:
                    extra.append("dense_frames=%s" % dense_frame_count)
                if prefilter.get("dense_frame_cache_hit") is not None:
                    extra.append("frame_cache_hit=%s" % bool(prefilter.get("dense_frame_cache_hit")))
                suffix = " | " + " | ".join(extra) if extra else ""
                self.console.print(
                    "prefilter: enabled=%s | windows=%s/%s%s"
                    % (
                        bool(prefilter.get("enabled")),
                        int(candidate_windows),
                        int(total_windows),
                        suffix,
                    )
                )
        if any(
            key in metadata
            for key in ("dense_frame_count", "bounded_frame_count", "dense_frame_cache_hit", "embedding_cache_ready")
        ):
            parts = []
            if metadata.get("dense_frame_count") is not None:
                parts.append("dense_frames=%s" % int(metadata["dense_frame_count"]))
            if metadata.get("bounded_frame_count") is not None:
                parts.append("bounded_frames=%s" % int(metadata["bounded_frame_count"]))
            if metadata.get("dense_frame_cache_hit") is not None:
                parts.append("frame_cache_hit=%s" % bool(metadata["dense_frame_cache_hit"]))
            if metadata.get("embedding_cache_ready") is not None:
                parts.append("embedding_cache_ready=%s" % bool(metadata["embedding_cache_ready"]))
            if parts:
                self.console.print("frame_cache: %s" % " | ".join(parts))
        for item in list(observations or [])[:4]:
            atomic_text = _short_text(item.get("atomic_text") or "", limit=220)
            if atomic_text:
                prefix = ""
                if isinstance(item.get("confidence"), (int, float)):
                    prefix = "[%0.4f] " % float(item.get("confidence"))
                self.console.print("  - %s%s" % (prefix, atomic_text))
        if step_dir:
            self.console.print("step_dir: %s" % step_dir)
            self.console.print("request_file: %s" % str(Path(step_dir) / "request.json"))
            request_full_file = Path(step_dir) / "request_full.json"
            if request_full_file.exists():
                self.console.print("request_full_file: %s" % str(request_full_file))
            runtime_file = Path(step_dir) / "runtime.json"
            if runtime_file.exists():
                self.console.print("runtime_file: %s" % str(runtime_file))
                try:
                    runtime_payload = json.loads(runtime_file.read_text(encoding="utf-8"))
                except Exception:
                    runtime_payload = {}
                requested_model = str(runtime_payload.get("model_name") or "").strip()
                resolved_model_path = str(runtime_payload.get("resolved_model_path") or "").strip()
                if requested_model and resolved_model_path:
                    self.console.print("model: %s -> %s" % (requested_model, resolved_model_path))
                elif requested_model:
                    self.console.print("model: %s" % requested_model)
                elif resolved_model_path:
                    self.console.print("model_path: %s" % resolved_model_path)

    def on_trace(self, *, round_index: int, trace_payload: Dict[str, Any], trace_dir: Optional[str] = None):
        self.console.print("")
        self.console.print("[bold]Trace[/bold]")
        evidence_entries = [dict(item or {}) for item in list(trace_payload.get("evidence_entries") or []) if isinstance(item, dict)]
        for item in list(trace_payload.get("inference_steps") or [])[:8]:
            if not isinstance(item, dict):
                continue
            anchor = _inference_time_anchor(item, evidence_entries)
            line = "%s. %s" % (
                item.get("step_id"),
                _short_text(item.get("text") or "", limit=220),
            )
            if anchor:
                line = "%s [%s]" % (line, anchor)
            self.console.print(
                "  %s" % line
            )
        if trace_payload.get("final_answer") not in (None, ""):
            self.console.print("final_answer: %s" % _short_text(trace_payload.get("final_answer") or "", limit=220))
        if trace_dir:
            self.console.print("trace_dir: %s" % trace_dir)

    def on_audit(self, *, round_index: int, audit_payload: Dict[str, Any], auditor_dir: Optional[str] = None):
        self.console.print("")
        self.console.print("[bold]Auditor[/bold]")
        self.console.print(
            "verdict=%s | scores=%s"
            % (
                str(audit_payload.get("verdict") or ""),
                _format_scores(dict(audit_payload.get("scores") or {})),
            )
        )
        feedback = _short_text(audit_payload.get("feedback") or "", limit=320)
        if feedback:
            self.console.print("feedback: %s" % feedback)
        missing = [str(item).strip() for item in list(audit_payload.get("missing_information") or []) if str(item).strip()]
        for item in missing[:5]:
            self.console.print("  missing: %s" % _short_text(item, limit=220))
        if auditor_dir:
            self.console.print("auditor_dir: %s" % auditor_dir)

    def on_complete(self, *, final_payload: Dict[str, Any]):
        self.console.print("")
        self.console.print("[bold cyan]Complete[/bold cyan]")
        self.console.print("run_dir: %s" % str(final_payload.get("run_dir") or ""))
        if final_payload.get("exported_results_dir"):
            self.console.print("exported_results_dir: %s" % str(final_payload.get("exported_results_dir") or ""))
        audit_payload = dict(final_payload.get("audit_report") or {})
        if audit_payload:
            self.console.print("final_verdict: %s" % str(audit_payload.get("verdict") or ""))
