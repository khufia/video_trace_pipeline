from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..backends.media import clips_from_request, get_video_duration
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
    tool: str = "audio_temporal_grounder"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "audio_temporal_grounder"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "audio_temporal_grounder"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.spotsound_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _scope_has_clips(request: dict[str, Any]) -> bool:
    scope = dict(request.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def _full_clip(task: dict[str, Any]) -> dict[str, Any]:
    try:
        duration = float(get_video_duration(str(task.get("video_path") or "")) or 0.0)
    except Exception:
        duration = 0.0
    return {
        "video_id": task.get("video_id") or task.get("sample_key") or "video",
        "start_s": 0.0,
        "end_s": duration,
        "metadata": {"source": "full_video_scope"},
    }


def _segment_from_clip(item: dict[str, Any], index: int, query: str) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    return {
        "id": str(item.get("id") or "audio_seg_%03d" % index),
        "video_id": item.get("video_id"),
        "modality": "audio",
        "label": str(metadata.get("event_label") or item.get("label") or query or "audio evidence"),
        "start_s": item.get("start_s"),
        "end_s": item.get("end_s"),
        "confidence": item.get("confidence"),
        "summary": str(metadata.get("summary") or metadata.get("rationale") or metadata.get("event_label") or "Model-grounded audio interval."),
        "metadata": metadata,
    }


def run(payload: ToolPayload, request: Request) -> Result:
    request_payload = request.model_dump(mode="json")
    options = dict(request.options or {})
    clips = clips_from_request(request_payload, payload.task) if _scope_has_clips(request_payload) else [_full_clip(payload.task)]
    runtime = _runtime(payload)
    max_segments = int(options.get("top_k") or options.get("max_segments") or 5)
    segments: list[dict[str, Any]] = []
    summaries: list[str] = []
    for clip in clips:
        runner_request = {
            "tool_name": "audio_temporal_grounder",
            "query": str(request.query or "").strip(),
            "clips": [clip],
        }
        raw = _execute({"tool": "audio_temporal_grounder", "task": payload.task, "request": runner_request, "runtime": runtime})
        clips_out = [dict(item or {}) for item in list(raw.get("clips") or []) if isinstance(item, dict)]
        for item in clips_out:
            segments.append(_segment_from_clip(item, len(segments) + 1, runner_request["query"]))
        if raw.get("summary"):
            summaries.append(str(raw.get("summary")))
    return Result(ok=True, output=Output.model_validate({"segments": segments[:max_segments], "summary": "\n".join(summaries).strip()}))


if __name__ == "__main__":
    main(run, request_model=Request)
