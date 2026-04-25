from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..common import ensure_dir, read_json, read_jsonl, write_json, write_text
from .exports import render_trace_markdown


_ROUND_RE = re.compile(r"round_(\d+)_")


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _truncate(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_seconds(value: Any) -> str:
    try:
        text = "%.3f" % float(value)
    except Exception:
        return str(value or "")
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _format_clip_range(item: Dict[str, Any]) -> str:
    start_s = item.get("start_s")
    end_s = item.get("end_s")
    if start_s is None and end_s is None:
        return ""
    if start_s is None:
        start_s = end_s
    if end_s is None:
        end_s = start_s
    return "%s-%s" % (_format_seconds(start_s), _format_seconds(end_s))


def _render_temporal_anchor(item: Dict[str, Any]) -> str:
    start_s = item.get("time_start_s")
    end_s = item.get("time_end_s")
    frame_ts_s = item.get("frame_ts_s")
    if start_s is not None or end_s is not None:
        if start_s is None:
            start_s = end_s
        if end_s is None:
            end_s = start_s
        try:
            if float(start_s) == float(end_s):
                return _format_seconds(start_s)
        except Exception:
            pass
        return "%s to %s" % (_format_seconds(start_s), _format_seconds(end_s))
    if frame_ts_s is not None:
        return _format_seconds(frame_ts_s)
    return ""


def _summarize_clips(clips: Iterable[Dict[str, Any]], limit: int = 4) -> str:
    rendered = [_format_clip_range(item) for item in list(clips or []) if _format_clip_range(item)]
    if not rendered:
        return ""
    if len(rendered) <= limit:
        return ", ".join(rendered)
    return "%s, +%d more" % (", ".join(rendered[:limit]), len(rendered) - limit)


def _summarize_frames(frames: Iterable[Dict[str, Any]], limit: int = 6) -> str:
    rendered = []
    for item in list(frames or []):
        timestamp_s = item.get("timestamp_s")
        if timestamp_s is None:
            continue
        rendered.append(_format_seconds(timestamp_s))
    if not rendered:
        return ""
    if len(rendered) <= limit:
        return ", ".join(rendered)
    return "%s, +%d more" % (", ".join(rendered[:limit]), len(rendered) - limit)


def _extract_query(request_payload: Dict[str, Any], result_payload: Dict[str, Any]) -> str:
    request_text = str((request_payload or {}).get("query") or "").strip()
    if request_text:
        return request_text
    data = dict((result_payload or {}).get("data") or {})
    return str(data.get("query") or "").strip()


def _summarize_result(result_payload: Dict[str, Any]) -> str:
    data = dict((result_payload or {}).get("data") or {})
    if data.get("answer"):
        return _truncate(data.get("answer"))
    if data.get("summary"):
        return _truncate(data.get("summary"))
    if data.get("response"):
        return _truncate(data.get("response"))
    if isinstance(data.get("clips"), list) and data.get("clips"):
        return "clips: %s" % _summarize_clips(data.get("clips") or [])
    if isinstance(data.get("frames"), list) and data.get("frames"):
        return "frames: %s" % _summarize_frames(data.get("frames") or [])
    summary_text = str((result_payload or {}).get("summary") or "").strip()
    if summary_text:
        return _truncate(summary_text)
    return ""


def _input_summary(request_payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    clips = request_payload.get("clips") or []
    frames = request_payload.get("frames") or []
    if request_payload.get("clip") and not clips:
        clips = [request_payload.get("clip")]
    if request_payload.get("frame") and not frames:
        frames = [request_payload.get("frame")]
    if clips:
        parts.append("clips=%s" % _summarize_clips(clips))
    if frames:
        parts.append("frames=%s" % _summarize_frames(frames))
    if request_payload.get("time_hints"):
        parts.append("time_hints=%d" % len(list(request_payload.get("time_hints") or [])))
    if request_payload.get("num_frames") is not None:
        parts.append("num_frames=%s" % request_payload.get("num_frames"))
    if request_payload.get("top_k") is not None:
        parts.append("top_k=%s" % request_payload.get("top_k"))
    return "; ".join(parts)


def _round_ids(run_path: Path) -> List[int]:
    round_numbers = set()
    for path in run_path.glob("planner/round_*_plan.json"):
        match = _ROUND_RE.search(path.name)
        if match:
            round_numbers.add(int(match.group(1)))
    for path in run_path.glob("synthesizer/round_*_trace_package.json"):
        match = _ROUND_RE.search(path.name)
        if match:
            round_numbers.add(int(match.group(1)))
    for path in run_path.glob("auditor/round_*_report.json"):
        match = _ROUND_RE.search(path.name)
        if match:
            round_numbers.add(int(match.group(1)))
    return sorted(round_numbers)


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _round_prefix(round_index: Any) -> str:
    return "round_%02d" % int(round_index or 0)


def _collect_round(run_path: Path, round_index: int) -> Dict[str, Any]:
    prefix = _round_prefix(round_index)
    plan_path = run_path / "planner" / ("%s_plan.json" % prefix)
    planner_raw_path = run_path / "planner" / ("%s_raw.txt" % prefix)
    trace_path = run_path / "synthesizer" / ("%s_trace_package.json" % prefix)
    synthesizer_raw_path = run_path / "synthesizer" / ("%s_raw.txt" % prefix)
    audit_path = run_path / "auditor" / ("%s_report.json" % prefix)
    auditor_raw_path = run_path / "auditor" / ("%s_raw.txt" % prefix)

    plan_payload = _read_json_if_exists(plan_path) or {}
    trace_payload = _read_json_if_exists(trace_path) or {}
    audit_payload = _read_json_if_exists(audit_path) or {}

    steps = []
    for item in list(plan_payload.get("steps") or []):
        if not isinstance(item, dict):
            continue
        steps.append(
            {
                "step_id": item.get("step_id"),
                "tool_name": str(item.get("tool_name") or "").strip(),
                "purpose": str(item.get("purpose") or "").strip(),
                "query": str(dict(item.get("arguments") or {}).get("query") or "").strip(),
            }
        )

    inference_steps = []
    for item in list(trace_payload.get("inference_steps") or []):
        if not isinstance(item, dict):
            continue
        inference_steps.append(
            {
                "step_id": item.get("step_id"),
                "text": str(item.get("text") or "").strip(),
                "time_start_s": item.get("time_start_s"),
                "time_end_s": item.get("time_end_s"),
                "frame_ts_s": item.get("frame_ts_s"),
            }
        )

    return {
        "round_index": int(round_index),
        "strategy": str(plan_payload.get("strategy") or "").strip(),
        "refinement_instructions": str(plan_payload.get("refinement_instructions") or "").strip(),
        "steps": steps,
        "trace_final_answer": str(trace_payload.get("final_answer") or "").strip(),
        "trace_evidence_count": len(list(trace_payload.get("evidence_entries") or [])),
        "trace_inference_count": len(inference_steps),
        "trace_inference_steps": inference_steps,
        "audit_verdict": str(audit_payload.get("verdict") or "").strip(),
        "audit_confidence": audit_payload.get("confidence"),
        "audit_feedback": str(audit_payload.get("feedback") or "").strip(),
        "audit_missing_information": [
            str(item).strip()
            for item in list(audit_payload.get("missing_information") or [])
            if str(item).strip()
        ],
        "audit_findings": [
            {
                "severity": str(item.get("severity") or "").strip(),
                "category": str(item.get("category") or "").strip(),
                "message": str(item.get("message") or "").strip(),
                "evidence_ids": [str(value).strip() for value in list(item.get("evidence_ids") or []) if str(value).strip()],
            }
            for item in list(audit_payload.get("findings") or [])
            if isinstance(item, dict)
        ],
        "planner_plan_path": _relative_path(plan_path, run_path) if plan_path.exists() else "",
        "planner_raw_path": _relative_path(planner_raw_path, run_path) if planner_raw_path.exists() else "",
        "trace_path": _relative_path(trace_path, run_path) if trace_path.exists() else "",
        "trace_raw_path": _relative_path(synthesizer_raw_path, run_path) if synthesizer_raw_path.exists() else "",
        "audit_path": _relative_path(audit_path, run_path) if audit_path.exists() else "",
        "audit_raw_path": _relative_path(auditor_raw_path, run_path) if auditor_raw_path.exists() else "",
    }


def _collect_tool_step(run_path: Path, tool_dir: Path) -> Dict[str, Any]:
    request_full_path = tool_dir / "request_full.json"
    request_path = tool_dir / "request.json"
    result_path = tool_dir / "result.json"
    timing_path = tool_dir / "timing.json"
    summary_path = tool_dir / "summary.md"
    artifact_refs_path = tool_dir / "artifact_refs.json"

    request_payload = _read_json_if_exists(request_full_path) or _read_json_if_exists(request_path) or {}
    result_payload = _read_json_if_exists(result_path) or {}
    timing_payload = _read_json_if_exists(timing_path) or {}
    if not timing_payload:
        timing_payload = dict(dict(result_payload.get("metadata") or {}).get("timing") or {})
    artifact_refs = list((_read_json_if_exists(artifact_refs_path) or {}).get("artifact_refs") or [])
    if not artifact_refs:
        artifact_refs = list(result_payload.get("artifact_refs") or [])
    observation_ids = list(dict(result_payload.get("metadata") or {}).get("observation_ids") or [])

    tool_name = str(result_payload.get("tool_name") or request_payload.get("tool_name") or tool_dir.name).strip()
    return {
        "tool_dir": tool_dir.name,
        "tool_name": tool_name,
        "query": _extract_query(request_payload, result_payload),
        "input_summary": _input_summary(request_payload),
        "result_summary": _summarize_result(result_payload),
        "summary_excerpt": _truncate(_read_text_if_exists(summary_path), 300),
        "ok": bool(result_payload.get("ok", True)),
        "cache_hit": bool(result_payload.get("cache_hit")),
        "execution_mode": str(timing_payload.get("execution_mode") or "").strip(),
        "duration_s": timing_payload.get("duration_s"),
        "observation_count": len(observation_ids),
        "artifact_count": len(artifact_refs),
        "request_path": _relative_path(request_full_path if request_full_path.exists() else request_path, run_path)
        if request_full_path.exists() or request_path.exists()
        else "",
        "result_path": _relative_path(result_path, run_path) if result_path.exists() else "",
        "summary_path": _relative_path(summary_path, run_path) if summary_path.exists() else "",
        "timing_path": _relative_path(timing_path, run_path) if timing_path.exists() else "",
    }


def build_run_debug_payload(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    manifest = _read_json_if_exists(run_path / "run_manifest.json") or {}
    final_result = _read_json_if_exists(run_path / "results" / "final_result.json") or {}
    trace_payload = _read_json_if_exists(run_path / "trace" / "trace_package.json") or {}
    evidence_entries = read_jsonl(run_path / "evidence" / "evidence_index.jsonl")
    observations = read_jsonl(run_path / "evidence" / "atomic_observations.jsonl")

    rounds = [_collect_round(run_path, round_index) for round_index in _round_ids(run_path)]
    tools_root = run_path / "tools"
    tool_steps = [
        _collect_tool_step(run_path, tool_dir)
        for tool_dir in sorted(tools_root.iterdir())
        if tool_dir.is_dir() and not tool_dir.name.startswith("_")
    ] if tools_root.exists() else []

    evidence_by_tool = Counter(str(item.get("tool_name") or "").strip() for item in evidence_entries if isinstance(item, dict))
    observations_by_tool = Counter(
        str(item.get("source_tool") or "").strip() for item in observations if isinstance(item, dict)
    )

    latest_round = rounds[-1] if rounds else {}
    task = dict(manifest.get("task") or {})
    return {
        "run": {
            "run_dir": str(run_path),
            "benchmark": str(manifest.get("benchmark") or "").strip(),
            "sample_key": str(manifest.get("sample_key") or "").strip(),
            "run_id": str(manifest.get("run_id") or "").strip(),
            "mode": str(manifest.get("mode") or "").strip(),
        },
        "task": {
            "question": str(task.get("question") or "").strip(),
            "options": [str(item).strip() for item in list(task.get("options") or []) if str(item).strip()],
            "gold_answer": str(task.get("gold_answer") or "").strip(),
            "video_id": str(task.get("video_id") or "").strip(),
            "question_id": str(task.get("question_id") or "").strip(),
        },
        "result": {
            "final_answer": str(trace_payload.get("final_answer") or "").strip(),
            "latest_audit_verdict": str(dict(final_result.get("audit_report") or {}).get("verdict") or latest_round.get("audit_verdict") or "").strip(),
            "latest_audit_feedback": str(dict(final_result.get("audit_report") or {}).get("feedback") or latest_round.get("audit_feedback") or "").strip(),
            "rounds_executed": final_result.get("rounds_executed"),
        },
        "rounds": rounds,
        "tool_steps": tool_steps,
        "evidence_summary": {
            "evidence_entry_count": len(evidence_entries),
            "observation_count": len(observations),
            "evidence_entries_by_tool": [
                {"tool_name": name, "count": count}
                for name, count in sorted(evidence_by_tool.items(), key=lambda pair: (-pair[1], pair[0]))
                if name
            ],
            "observations_by_tool": [
                {"tool_name": name, "count": count}
                for name, count in sorted(observations_by_tool.items(), key=lambda pair: (-pair[1], pair[0]))
                if name
            ],
        },
        "key_files": {
            "final_result": "results/final_result.json" if (run_path / "results" / "final_result.json").exists() else "",
            "trace_readable": "trace/trace_readable.md" if (run_path / "trace" / "trace_readable.md").exists() else "",
            "trace_package": "trace/trace_package.json" if (run_path / "trace" / "trace_package.json").exists() else "",
            "evidence_readable": "evidence/evidence_readable.md" if (run_path / "evidence" / "evidence_readable.md").exists() else "",
            "run_manifest": "run_manifest.json" if (run_path / "run_manifest.json").exists() else "",
            "runtime_snapshot": "runtime_snapshot.yaml" if (run_path / "runtime_snapshot.yaml").exists() else "",
        },
    }


def _md_link(label: str, target: str, link_prefix: str = "../") -> str:
    text = str(target or "").strip()
    if not text:
        return label
    return "[%s](%s%s)" % (label, str(link_prefix or ""), text)


def render_run_debug_markdown(
    payload: Dict[str, Any],
    *,
    link_prefix: str = "../",
    title: str = "Run Debug Report",
) -> str:
    lines = ["# %s" % str(title or "Run Debug Report").strip(), ""]

    run = dict(payload.get("run") or {})
    task = dict(payload.get("task") or {})
    result = dict(payload.get("result") or {})
    key_files = dict(payload.get("key_files") or {})
    rounds = list(payload.get("rounds") or [])
    latest_round = rounds[-1] if rounds else {}
    latest_round_prefix = _round_prefix(latest_round.get("round_index")) if latest_round else ""

    lines.append("## Start Here")
    lines.append("")
    lines.append(
        "1. %s"
        % _md_link(
            "Latest audit report",
            "auditor/%s_report_readable.md" % latest_round_prefix if latest_round_prefix else "",
            link_prefix,
        )
    )
    lines.append("2. %s" % _md_link("Final trace (readable)", key_files.get("trace_readable", ""), link_prefix))
    lines.append("3. %s" % _md_link("Evidence ledger (readable)", key_files.get("evidence_readable", ""), link_prefix))
    lines.append(
        "4. %s"
        % _md_link(
            "Final result summary",
            "results/final_result_readable.md" if key_files.get("final_result") else "",
            link_prefix,
        )
    )
    lines.append("")

    lines.append("## Task")
    lines.append("")
    lines.append("- Benchmark: %s" % run.get("benchmark", ""))
    lines.append("- Sample Key: %s" % run.get("sample_key", ""))
    lines.append("- Run ID: %s" % run.get("run_id", ""))
    lines.append("- Question: %s" % task.get("question", ""))
    if task.get("options"):
        lines.append("- Options: %s" % " | ".join(task.get("options") or []))
    if task.get("gold_answer"):
        lines.append("- Gold Answer: %s" % task.get("gold_answer"))
    lines.append("- Final Answer: %s" % (result.get("final_answer") or "<empty>"))
    lines.append("- Latest Audit Verdict: %s" % (result.get("latest_audit_verdict") or ""))
    if result.get("latest_audit_feedback"):
        lines.append("- Latest Audit Feedback: %s" % result.get("latest_audit_feedback"))
    lines.append("")

    lines.append("## Round Timeline")
    lines.append("")
    for round_payload in rounds:
        round_prefix = _round_prefix(round_payload.get("round_index"))
        lines.append("### Round %02d" % int(round_payload.get("round_index") or 0))
        lines.append("")
        if round_payload.get("strategy"):
            lines.append("- Strategy: %s" % round_payload.get("strategy"))
        if round_payload.get("refinement_instructions"):
            lines.append("- Refinement Instructions: %s" % _truncate(round_payload.get("refinement_instructions"), 400))
        files = [
            _md_link("plan summary", "planner/%s_summary.md" % round_prefix, link_prefix)
            if round_payload.get("planner_plan_path")
            else "",
            _md_link("plan", round_payload.get("planner_plan_path", ""), link_prefix),
            _md_link("planner raw", round_payload.get("planner_raw_path", ""), link_prefix),
            _md_link("trace readable", "synthesizer/%s_trace_readable.md" % round_prefix, link_prefix)
            if round_payload.get("trace_path")
            else "",
            _md_link("trace", round_payload.get("trace_path", ""), link_prefix),
            _md_link("trace raw", round_payload.get("trace_raw_path", ""), link_prefix),
            _md_link("audit readable", "auditor/%s_report_readable.md" % round_prefix, link_prefix)
            if round_payload.get("audit_path")
            else "",
            _md_link("audit", round_payload.get("audit_path", ""), link_prefix),
            _md_link("audit raw", round_payload.get("audit_raw_path", ""), link_prefix),
        ]
        lines.append("- Files: %s" % " | ".join(item for item in files if item))
        if round_payload.get("steps"):
            lines.append("- Planned Steps:")
            for item in round_payload.get("steps") or []:
                step_bits = ["%s. `%s`" % (item.get("step_id"), item.get("tool_name") or "tool")]
                if item.get("purpose"):
                    step_bits.append(item.get("purpose"))
                lines.append("  - %s" % " - ".join(step_bits))
                if item.get("query"):
                    lines.append("    Query: %s" % item.get("query"))
        lines.append("- Trace Final Answer: %s" % (round_payload.get("trace_final_answer") or "<empty>"))
        lines.append(
            "- Trace Size: evidence=%s, inference_steps=%s"
            % (round_payload.get("trace_evidence_count"), round_payload.get("trace_inference_count"))
        )
        if round_payload.get("trace_inference_steps"):
            lines.append("- Inference Steps:")
            for item in round_payload.get("trace_inference_steps") or []:
                line = "%s. %s" % (item.get("step_id"), item.get("text"))
                temporal_anchor = _render_temporal_anchor(item)
                if temporal_anchor:
                    line = "%s [%s]" % (line, temporal_anchor)
                lines.append("  - %s" % line)
        lines.append(
            "- Audit: verdict=%s, confidence=%s"
            % (round_payload.get("audit_verdict") or "", round_payload.get("audit_confidence"))
        )
        if round_payload.get("audit_feedback"):
            lines.append("  Feedback: %s" % round_payload.get("audit_feedback"))
        if round_payload.get("audit_missing_information"):
            lines.append("  Missing Information:")
            for item in round_payload.get("audit_missing_information") or []:
                lines.append("  - %s" % item)
        if round_payload.get("audit_findings"):
            lines.append("  Findings:")
            for item in round_payload.get("audit_findings") or []:
                finding_bits = [
                    str(item.get("severity") or "").strip(),
                    str(item.get("category") or "").strip(),
                ]
                finding_prefix = "/".join(bit for bit in finding_bits if bit)
                message = str(item.get("message") or "").strip()
                if finding_prefix and message:
                    lines.append("  - [%s] %s" % (finding_prefix, message))
                elif message:
                    lines.append("  - %s" % message)
        lines.append("")

    lines.append("## Tool Steps")
    lines.append("")
    for item in list(payload.get("tool_steps") or []):
        lines.append("### %s" % item.get("tool_dir", "tool"))
        lines.append("")
        lines.append("- Tool: %s" % item.get("tool_name", ""))
        lines.append("- Query: %s" % (item.get("query") or "<none>"))
        if item.get("input_summary"):
            lines.append("- Inputs: %s" % item.get("input_summary"))
        lines.append(
            "- Status: ok=%s, cache_hit=%s, execution_mode=%s, duration=%s"
            % (
                item.get("ok"),
                item.get("cache_hit"),
                item.get("execution_mode") or "",
                _format_seconds(item.get("duration_s")) if item.get("duration_s") is not None else "",
            )
        )
        lines.append(
            "- Evidence Surface: observations=%s, artifacts=%s"
            % (item.get("observation_count"), item.get("artifact_count"))
        )
        if item.get("result_summary"):
            lines.append("- Result Summary: %s" % item.get("result_summary"))
        if item.get("summary_excerpt"):
            lines.append("- Summary Excerpt: %s" % item.get("summary_excerpt"))
        file_links = [
            _md_link("request", item.get("request_path", ""), link_prefix),
            _md_link("result", item.get("result_path", ""), link_prefix),
            _md_link("summary", item.get("summary_path", ""), link_prefix),
            _md_link("timing", item.get("timing_path", ""), link_prefix),
        ]
        lines.append("- Files: %s" % " | ".join(link for link in file_links if link))
        lines.append("")

    evidence_summary = dict(payload.get("evidence_summary") or {})
    lines.append("## Evidence Coverage")
    lines.append("")
    lines.append(
        "- Totals: evidence_entries=%s, observations=%s"
        % (evidence_summary.get("evidence_entry_count"), evidence_summary.get("observation_count"))
    )
    if evidence_summary.get("evidence_entries_by_tool"):
        lines.append("- Evidence Entries By Tool:")
        for item in evidence_summary.get("evidence_entries_by_tool") or []:
            lines.append("  - %s: %s" % (item.get("tool_name"), item.get("count")))
    if evidence_summary.get("observations_by_tool"):
        lines.append("- Observations By Tool:")
        for item in evidence_summary.get("observations_by_tool") or []:
            lines.append("  - %s: %s" % (item.get("tool_name"), item.get("count")))
    lines.append("")

    lines.append("## Key Files")
    lines.append("")
    for label, target in (
        ("run_manifest", key_files.get("run_manifest", "")),
        ("runtime_snapshot", key_files.get("runtime_snapshot", "")),
        ("trace_package", key_files.get("trace_package", "")),
        ("trace_readable", key_files.get("trace_readable", "")),
        ("evidence_readable", key_files.get("evidence_readable", "")),
        ("final_result", key_files.get("final_result", "")),
    ):
        if target:
            lines.append("- %s" % _md_link(label, target, link_prefix))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_round_plan_markdown(round_payload: Dict[str, Any], link_prefix: str = "") -> str:
    round_index = int(round_payload.get("round_index") or 0)
    lines = ["# Round %02d Plan" % round_index, ""]
    if round_payload.get("strategy"):
        lines.append("## Strategy")
        lines.append("")
        lines.append(str(round_payload.get("strategy") or ""))
        lines.append("")
    if round_payload.get("refinement_instructions"):
        lines.append("## Refinement Instructions")
        lines.append("")
        lines.append(str(round_payload.get("refinement_instructions") or ""))
        lines.append("")
    lines.append("## Steps")
    lines.append("")
    steps = list(round_payload.get("steps") or [])
    if not steps:
        lines.append("No tool steps were planned in this round.")
        lines.append("")
    else:
        for item in steps:
            lines.append("%s. `%s` - %s" % (item.get("step_id"), item.get("tool_name") or "tool", item.get("purpose") or ""))
            if item.get("query"):
                lines.append("   Query: %s" % item.get("query"))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for label, target in (
        ("plan json", round_payload.get("planner_plan_path", "")),
        ("planner raw", round_payload.get("planner_raw_path", "")),
    ):
        if target:
            lines.append("- %s" % _md_link(label, target, link_prefix))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_round_audit_markdown(round_payload: Dict[str, Any], link_prefix: str = "") -> str:
    round_index = int(round_payload.get("round_index") or 0)
    lines = ["# Round %02d Audit" % round_index, ""]
    lines.append("- Verdict: %s" % (round_payload.get("audit_verdict") or ""))
    lines.append("- Confidence: %s" % (round_payload.get("audit_confidence") if round_payload.get("audit_confidence") is not None else ""))
    if round_payload.get("audit_feedback"):
        lines.append("- Feedback: %s" % round_payload.get("audit_feedback"))
    lines.append("")
    if round_payload.get("audit_missing_information"):
        lines.append("## Missing Information")
        lines.append("")
        for item in round_payload.get("audit_missing_information") or []:
            lines.append("- %s" % item)
        lines.append("")
    if round_payload.get("audit_findings"):
        lines.append("## Findings")
        lines.append("")
        for item in round_payload.get("audit_findings") or []:
            finding_bits = [
                str(item.get("severity") or "").strip(),
                str(item.get("category") or "").strip(),
            ]
            finding_prefix = "/".join(bit for bit in finding_bits if bit)
            message = str(item.get("message") or "").strip()
            evidence_ids = [str(value).strip() for value in list(item.get("evidence_ids") or []) if str(value).strip()]
            if finding_prefix and message:
                lines.append("- [%s] %s" % (finding_prefix, message))
            elif message:
                lines.append("- %s" % message)
            if evidence_ids:
                lines.append("  Evidence: %s" % ", ".join(evidence_ids))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for label, target in (
        ("audit json", round_payload.get("audit_path", "")),
        ("auditor raw", round_payload.get("audit_raw_path", "")),
    ):
        if target:
            lines.append("- %s" % _md_link(label, target, link_prefix))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_final_result_markdown(payload: Dict[str, Any], link_prefix: str = "") -> str:
    run = dict(payload.get("run") or {})
    task = dict(payload.get("task") or {})
    result = dict(payload.get("result") or {})
    lines = ["# Final Result", ""]
    lines.append("- Benchmark: %s" % run.get("benchmark", ""))
    lines.append("- Sample Key: %s" % run.get("sample_key", ""))
    lines.append("- Run ID: %s" % run.get("run_id", ""))
    lines.append("- Question: %s" % task.get("question", ""))
    if task.get("options"):
        lines.append("- Options: %s" % " | ".join(task.get("options") or []))
    lines.append("- Final Answer: %s" % (result.get("final_answer") or "<empty>"))
    lines.append("- Latest Audit Verdict: %s" % (result.get("latest_audit_verdict") or ""))
    if result.get("latest_audit_feedback"):
        lines.append("- Latest Audit Feedback: %s" % result.get("latest_audit_feedback"))
    if result.get("rounds_executed") is not None:
        lines.append("- Rounds Executed: %s" % result.get("rounds_executed"))
    lines.append("")
    lines.append("## Useful Files")
    lines.append("")
    for label, target in (
        ("run overview", "README.md"),
        ("trace readable", "trace/trace_readable.md"),
        ("evidence readable", "evidence/evidence_readable.md"),
        ("final_result json", "results/final_result.json"),
    ):
        lines.append("- %s" % _md_link(label, target, link_prefix))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_run_readable_bundle(run_dir: str | Path) -> Path:
    run_path = Path(run_dir).expanduser().resolve()
    payload = build_run_debug_payload(run_path)

    readme_path = run_path / "README.md"
    write_text(readme_path, render_run_debug_markdown(payload, link_prefix="", title="Run Overview"))

    for round_payload in list(payload.get("rounds") or []):
        round_prefix = _round_prefix(round_payload.get("round_index"))
        write_text(
            run_path / "planner" / ("%s_summary.md" % round_prefix),
            _render_round_plan_markdown(round_payload),
        )
        trace_payload = _read_json_if_exists(run_path / "synthesizer" / ("%s_trace_package.json" % round_prefix))
        if trace_payload:
            write_text(
                run_path / "synthesizer" / ("%s_trace_readable.md" % round_prefix),
                render_trace_markdown(trace_payload),
            )
        if round_payload.get("audit_path"):
            write_text(
                run_path / "auditor" / ("%s_report_readable.md" % round_prefix),
                _render_round_audit_markdown(round_payload),
            )

    if (run_path / "results" / "final_result.json").exists():
        write_text(run_path / "results" / "final_result_readable.md", _render_final_result_markdown(payload))
    return readme_path


def write_run_debug_bundle(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    payload = build_run_debug_payload(run_dir)
    run_path = Path(run_dir).expanduser().resolve()
    debug_dir = ensure_dir(Path(output_dir).expanduser().resolve()) if output_dir is not None else ensure_dir(run_path / "debug")

    write_json(debug_dir / "overview.json", payload)
    write_json(debug_dir / "tool_steps.json", {"tool_steps": payload.get("tool_steps") or []})
    write_json(debug_dir / "rounds.json", {"rounds": payload.get("rounds") or []})

    with (debug_dir / "tool_steps.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "tool_dir",
            "tool_name",
            "ok",
            "cache_hit",
            "execution_mode",
            "duration_s",
            "observation_count",
            "artifact_count",
            "query",
            "input_summary",
            "result_summary",
            "request_path",
            "result_path",
            "summary_path",
            "timing_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in list(payload.get("tool_steps") or []):
            writer.writerow({key: item.get(key) for key in fieldnames})

    report_path = debug_dir / "README.md"
    write_text(report_path, render_run_debug_markdown(payload))
    write_run_readable_bundle(run_path)
    return report_path
