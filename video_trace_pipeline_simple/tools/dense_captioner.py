from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..backends.media import clips_from_request
from ..tool_io import ToolPayload, main


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    captions: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class Result(BaseModel):
    ok: bool
    tool: str = "dense_captioner"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "dense_captioner"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "dense_captioner"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.timechat_dense_caption_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _relpath(path: str, runtime: dict[str, Any]) -> str | None:
    try:
        workspace = Path(str(runtime.get("workspace_root") or ".")).expanduser().resolve()
        return str(Path(path).expanduser().resolve().relative_to(workspace))
    except Exception:
        return None


def _scope_has_clips(request: dict[str, Any]) -> bool:
    scope = dict(request.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def run(payload: ToolPayload, request: Request) -> Result:
    request_payload = request.model_dump(mode="json")
    if not _scope_has_clips(request_payload):
        raise RuntimeError("dense_captioner requires request.temporal_scope.clips")
    runtime = _runtime(payload)
    options = dict(request.options or {})
    captions: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    caption_groups: list[dict[str, Any]] = []
    summaries: list[str] = []
    for clip in clips_from_request(request_payload, payload.task):
        runner_request = {
            "tool_name": "dense_captioner",
            "clips": [clip],
            "granularity": str(options.get("granularity") or "segment"),
            "focus_query": str(request.query or "").strip(),
        }
        raw = _execute({"tool": "dense_captioner", "task": payload.task, "request": runner_request, "runtime": runtime})
        raw_captions = [dict(item or {}) for item in list(raw.get("captions") or []) if isinstance(item, dict)]
        for index, item in enumerate(raw_captions, start=1):
            normalized = dict(item)
            normalized.setdefault("id", "caption_%03d" % (len(captions) + index))
            normalized.setdefault("start_s", item.get("start"))
            normalized.setdefault("end_s", item.get("end"))
            normalized.setdefault("caption", item.get("visual") or raw.get("overall_summary") or "")
            captions.append(normalized)
        sampled_frames = [dict(item or {}) for item in list(raw.get("sampled_frames") or []) if isinstance(item, dict)]
        for frame in sampled_frames:
            path = str(frame.get("frame_path") or "").strip()
            if path:
                frame.setdefault("relpath", _relpath(path, runtime))
            artifacts.append({"kind": "frame", **frame})
        caption_groups.append({"clips": [clip], "captions": raw_captions, "overall_summary": raw.get("overall_summary") or ""})
        if raw.get("overall_summary"):
            summaries.append(str(raw.get("overall_summary")))
    output = {"captions": captions, "artifacts": artifacts}
    return Result(
        ok=True,
        output=Output.model_validate(output),
        artifacts=artifacts,
        metadata={"overall_summary": "\n".join(summaries).strip(), "caption_groups": caption_groups},
    )


if __name__ == "__main__":
    main(run, request_model=Request)
