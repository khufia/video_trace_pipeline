from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ensure_dir, make_run_id, sanitize_path_component, write_json, write_text


def make_output_dir(output_dir: str | None, profile: dict[str, Any], task: dict[str, Any]) -> Path:
    if output_dir:
        run_dir = Path(output_dir).expanduser().resolve()
    else:
        root = Path(str(profile.get("workspace_root") or "workspace")).expanduser().resolve()
        run_dir = (
            root
            / "outputs"
            / sanitize_path_component(str(task.get("benchmark") or "adhoc"))
            / sanitize_path_component(str(task.get("sample_key") or task.get("video_id") or "sample"))
            / make_run_id()
        )
    return ensure_dir(run_dir)


def round_dir(run_dir: str | Path, round_idx: int) -> Path:
    return ensure_dir(Path(run_dir) / "rounds" / ("round_%02d" % int(round_idx)))


def step_dir(run_dir: str | Path, round_idx: int, step_index: int, tool_name: str) -> Path:
    return ensure_dir(round_dir(run_dir, round_idx) / "tools" / ("%02d_%s" % (int(step_index), tool_name)))


def build_final_result(
    task: dict[str, Any],
    run_dir: str | Path,
    preprocess_result: dict[str, Any],
    previous_steps: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    latest_trace: dict[str, Any] | None,
    latest_audit: dict[str, Any] | None,
    state: dict[str, Any] | None,
    rounds_executed: int,
    ok: bool = True,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = dict(latest_trace or {})
    audit = dict(latest_audit or {})
    tool_steps = []
    for record in previous_steps:
        step = dict(record.get("step") or {})
        result = dict(record.get("result") or {})
        tool_steps.append(
            {
                "round": record.get("round"),
                "id": step.get("id"),
                "tool": step.get("tool"),
                "ok": bool(result.get("ok")),
            }
        )
    return {
        "ok": bool(ok),
        "run_dir": str(Path(run_dir).resolve()),
        "benchmark": task.get("benchmark"),
        "sample_key": task.get("sample_key"),
        "final_answer": trace.get("answer") or trace.get("final_answer") or "",
        "confidence": trace.get("confidence"),
        "reason": trace.get("reasoning") or trace.get("reason") or "",
        "audit_verdict": audit.get("verdict"),
        "rounds_executed": int(rounds_executed or 0),
        "observations_count": len(observations or []),
        "tool_steps": tool_steps,
        "state": state,
        "preprocess_ok": bool((preprocess_result or {}).get("ok")),
        "error": error,
    }


def write_summary(run_dir: str | Path, final: dict[str, Any]) -> None:
    lines = [
        "# Simple Video Trace Pipeline Summary",
        "",
        "- ok: `%s`" % final.get("ok"),
        "- benchmark: `%s`" % (final.get("benchmark") or ""),
        "- sample_key: `%s`" % (final.get("sample_key") or ""),
        "- final_answer: `%s`" % (final.get("final_answer") or ""),
        "- audit_verdict: `%s`" % (final.get("audit_verdict") or ""),
        "- rounds_executed: `%s`" % final.get("rounds_executed"),
        "- observations_count: `%s`" % final.get("observations_count"),
    ]
    write_text(Path(run_dir) / "summary.md", "\n".join(lines) + "\n")
    write_json(Path(run_dir) / "final_result.json", final)
