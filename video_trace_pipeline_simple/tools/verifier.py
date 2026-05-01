from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..tool_io import ToolPayload, main


class Request(BaseModel):
    query: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)
    clips: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[dict[str, Any]] = Field(default_factory=list)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    transcripts: list[dict[str, Any]] = Field(default_factory=list)
    text_contexts: list[str] = Field(default_factory=list)
    ocr_results: list[dict[str, Any]] = Field(default_factory=list)
    verification_mode: str = "strict"


class Output(BaseModel):
    claim_results: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    summary: str = ""


class Result(BaseModel):
    ok: bool
    tool: str = "verifier"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "verifier"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "verifier"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.verifier_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def run(payload: ToolPayload, request: Request) -> Result:
    runner_request = request.model_dump(mode="json")
    runner_request["tool_name"] = "verifier"
    raw = _execute({"tool": "verifier", "task": payload.task, "request": runner_request, "runtime": _runtime(payload)})
    output = {
        "claim_results": [dict(item or {}) for item in list(raw.get("claim_results") or []) if isinstance(item, dict)],
        "unresolved_gaps": [str(item) for item in list(raw.get("unresolved_gaps") or []) if str(item).strip()],
        "summary": "Verifier checked %d claim(s)." % len(list(raw.get("claim_results") or [])),
    }
    return Result(ok=True, output=Output.model_validate(output), metadata={"raw_output": raw})


if __name__ == "__main__":
    main(run, request_model=Request)
