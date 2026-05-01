from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..tool_io import ToolPayload, main


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    segments: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class Result(BaseModel):
    ok: bool
    tool: str = "visual_temporal_grounder"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "visual_temporal_grounder"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "visual_temporal_grounder"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.timelens_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _segment_from_clip(item: dict[str, Any], index: int, query: str) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    return {
        "id": str(item.get("id") or "visual_seg_%03d" % index),
        "video_id": item.get("video_id"),
        "modality": "visual",
        "label": str(metadata.get("event_label") or item.get("label") or query or "visual evidence"),
        "start_s": item.get("start_s"),
        "end_s": item.get("end_s"),
        "confidence": item.get("confidence"),
        "summary": str(metadata.get("summary") or metadata.get("rationale") or "Model-grounded visual interval."),
        "metadata": metadata,
    }


def run(payload: ToolPayload, request: Request) -> Result:
    options = dict(request.options or {})
    runner_request = {
        "tool_name": "visual_temporal_grounder",
        "query": str(request.query or "").strip(),
        "top_k": int(options.get("top_k") or options.get("max_segments") or 5),
    }
    runtime = _runtime(payload)
    runtime["extra"]["use_embedding_prefilter"] = False
    raw = _execute(
        {
            "tool": "visual_temporal_grounder",
            "task": payload.task,
            "request": runner_request,
            "runtime": runtime,
        }
    )
    clips = [dict(item or {}) for item in list(raw.get("clips") or []) if isinstance(item, dict)]
    output = {
        "segments": [_segment_from_clip(item, index, runner_request["query"]) for index, item in enumerate(clips, start=1)],
        "summary": str(raw.get("summary") or ""),
    }
    return Result(
        ok=True,
        output=Output.model_validate(output),
        metadata={
            "backend": raw.get("retrieval_backend"),
            "prefilter": raw.get("prefilter"),
            "query_absent": raw.get("query_absent"),
            "video_duration": raw.get("video_duration"),
        },
    )


if __name__ == "__main__":
    main(run, request_model=Request)
