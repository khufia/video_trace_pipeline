from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import ensure_dir, read_json, read_jsonl, write_json


_ROUND_RE = re.compile(r"round_(\d+)$")


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _round_ids(run_path: Path) -> List[int]:
    round_numbers = set()
    for path in run_path.iterdir():
        if not path.is_dir():
            continue
        match = _ROUND_RE.fullmatch(path.name)
        if match:
            round_numbers.add(int(match.group(1)))
    return sorted(round_numbers)


def _round_prefix(round_index: Any) -> str:
    return "round_%02d" % int(round_index or 0)


def _collect_round(run_path: Path, round_index: int) -> Dict[str, Any]:
    round_dir = run_path / _round_prefix(round_index)
    plan_path = round_dir / "planner_plan.json"
    plan_request_path = round_dir / "planner_request.json"
    trace_path = round_dir / "synthesizer_trace_package.json"
    trace_request_path = round_dir / "synthesizer_request.json"
    audit_path = round_dir / "auditor_report.json"
    audit_request_path = round_dir / "auditor_request.json"

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
                "inputs": dict(item.get("inputs") or {}),
                "input_refs": dict(item.get("input_refs") or {}),
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
                "time_intervals": list(item.get("time_intervals") or []),
            }
        )

    return {
        "round_index": int(round_index),
        "round_dir": _relative_path(round_dir, run_path),
        "planner_request_path": _relative_path(plan_request_path, run_path) if plan_request_path.exists() else "",
        "planner_plan_path": _relative_path(plan_path, run_path) if plan_path.exists() else "",
        "synthesizer_request_path": _relative_path(trace_request_path, run_path) if trace_request_path.exists() else "",
        "trace_path": _relative_path(trace_path, run_path) if trace_path.exists() else "",
        "auditor_request_path": _relative_path(audit_request_path, run_path) if audit_request_path.exists() else "",
        "audit_path": _relative_path(audit_path, run_path) if audit_path.exists() else "",
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
    }


def _collect_tool_step(run_path: Path, round_index: int, tool_dir: Path) -> Dict[str, Any]:
    request_path = tool_dir / "request.json"
    result_path = tool_dir / "result.json"
    timing_path = tool_dir / "timing.json"
    runtime_path = tool_dir / "runtime.json"
    observations_path = tool_dir / "observations.json"
    artifact_refs_path = tool_dir / "artifact_refs.json"

    request_payload = _read_json_if_exists(request_path) or {}
    result_payload = _read_json_if_exists(result_path) or {}
    timing_payload = _read_json_if_exists(timing_path) or {}
    runtime_payload = _read_json_if_exists(runtime_path) or {}
    artifact_refs_payload = read_json(artifact_refs_path) if artifact_refs_path.exists() else []
    observations_payload = read_json(observations_path) if observations_path.exists() else []

    return {
        "round_index": int(round_index),
        "tool_dir": _relative_path(tool_dir, run_path),
        "tool_name": str(result_payload.get("tool_name") or request_payload.get("tool_name") or tool_dir.name).strip(),
        "request": request_payload,
        "result": result_payload,
        "timing": timing_payload,
        "runtime": runtime_payload,
        "artifact_refs": artifact_refs_payload if isinstance(artifact_refs_payload, list) else [],
        "observations": observations_payload if isinstance(observations_payload, list) else [],
        "request_path": _relative_path(request_path, run_path) if request_path.exists() else "",
        "result_path": _relative_path(result_path, run_path) if result_path.exists() else "",
        "timing_path": _relative_path(timing_path, run_path) if timing_path.exists() else "",
        "runtime_path": _relative_path(runtime_path, run_path) if runtime_path.exists() else "",
        "observations_path": _relative_path(observations_path, run_path) if observations_path.exists() else "",
        "artifact_refs_path": _relative_path(artifact_refs_path, run_path) if artifact_refs_path.exists() else "",
    }


def build_run_debug_payload(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    manifest = _read_json_if_exists(run_path / "run_manifest.json") or {}
    final_result = _read_json_if_exists(run_path / "final_result.json") or {}
    trace_payload = _read_json_if_exists(run_path / "trace_package.json") or {}
    evidence_entries = read_jsonl(run_path / "evidence" / "evidence_index.jsonl")
    observations = read_jsonl(run_path / "evidence" / "atomic_observations.jsonl")

    rounds = [_collect_round(run_path, round_index) for round_index in _round_ids(run_path)]
    tool_steps: List[Dict[str, Any]] = []
    for round_index in _round_ids(run_path):
        tools_root = run_path / _round_prefix(round_index) / "tools"
        if not tools_root.exists():
            continue
        for tool_dir in sorted(tools_root.iterdir()):
            if tool_dir.is_dir() and not tool_dir.name.startswith("_"):
                tool_steps.append(_collect_tool_step(run_path, round_index, tool_dir))

    evidence_by_tool = Counter(str(item.get("tool_name") or "").strip() for item in evidence_entries if isinstance(item, dict))
    observations_by_tool = Counter(
        str(item.get("source_tool") or "").strip() for item in observations if isinstance(item, dict)
    )

    task = dict(manifest.get("task") or {})
    return {
        "run": {
            "run_dir": str(run_path),
            "benchmark": str(manifest.get("benchmark") or "").strip(),
            "sample_key": str(manifest.get("sample_key") or "").strip(),
            "video_id": str(manifest.get("video_id") or task.get("video_id") or "").strip(),
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
            "latest_audit_verdict": str(dict(final_result.get("audit_report") or {}).get("verdict") or "").strip(),
            "latest_audit_feedback": str(dict(final_result.get("audit_report") or {}).get("feedback") or "").strip(),
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
            "run_manifest": "run_manifest.json" if (run_path / "run_manifest.json").exists() else "",
            "runtime_snapshot": "runtime_snapshot.json" if (run_path / "runtime_snapshot.json").exists() else "",
            "trace_package": "trace_package.json" if (run_path / "trace_package.json").exists() else "",
            "benchmark_export": "benchmark_export.json" if (run_path / "benchmark_export.json").exists() else "",
            "final_result": "final_result.json" if (run_path / "final_result.json").exists() else "",
        },
    }


def write_run_debug_bundle(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    payload = build_run_debug_payload(run_dir)
    run_path = Path(run_dir).expanduser().resolve()
    debug_dir = ensure_dir(Path(output_dir).expanduser().resolve()) if output_dir is not None else ensure_dir(run_path / "debug")
    overview_path = debug_dir / "overview.json"
    write_json(overview_path, payload)
    write_json(debug_dir / "tool_steps.json", {"tool_steps": payload.get("tool_steps") or []})
    write_json(debug_dir / "rounds.json", {"rounds": payload.get("rounds") or []})
    return overview_path
