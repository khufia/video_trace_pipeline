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
    frames: list[dict[str, Any]] = Field(default_factory=list)


class Result(BaseModel):
    ok: bool
    tool: str = "frame_retriever"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "frame_retriever"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "frame_retriever"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.frame_retriever_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _scope_has_clips(request: dict[str, Any]) -> bool:
    scope = dict(request.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def _relpath(path: str, runtime: dict[str, Any]) -> str | None:
    try:
        workspace = Path(str(runtime.get("workspace_root") or ".")).expanduser().resolve()
        return str(Path(path).expanduser().resolve().relative_to(workspace))
    except Exception:
        return None


def _normalize_frame(item: dict[str, Any], payload: ToolPayload, runtime: dict[str, Any], clip: dict[str, Any] | None) -> dict[str, Any]:
    path = str(item.get("frame_path") or "").strip()
    metadata = dict(item.get("metadata") or {})
    if path:
        metadata.setdefault("source_path", path)
    if item.get("relevance_score") is not None:
        metadata.setdefault("relevance_score", item.get("relevance_score"))
    return {
        "video_id": item.get("video_id") or payload.task.get("video_id") or payload.task.get("sample_key"),
        "timestamp_s": item.get("timestamp_s", item.get("timestamp")),
        "frame_path": path or None,
        "relpath": item.get("relpath") or (_relpath(path, runtime) if path else None),
        "clip": item.get("clip") or clip,
        "rationale": item.get("rationale") or metadata.get("selection_reason"),
        "metadata": metadata,
    }


def _frame_sort_key(frame: dict[str, Any]) -> tuple[float, float, float]:
    metadata = dict(frame.get("metadata") or {})
    try:
        score = -float(metadata.get("relevance_score") or frame.get("relevance_score") or 0.0)
    except Exception:
        score = 0.0
    try:
        temporal = -float(metadata.get("temporal_score") or 0.0)
    except Exception:
        temporal = 0.0
    try:
        timestamp = float(frame.get("timestamp_s") or 0.0)
    except Exception:
        timestamp = 0.0
    return score, temporal, timestamp


def _runner_request(request: Request, task: dict[str, Any]) -> dict[str, Any]:
    request_payload = request.model_dump(mode="json")
    options = dict(request.options or {})
    removed_options = {
        "sequence_mode",
        "neighbor_radius_s",
        "include_anchor_neighbors",
        "sort_order",
    }.intersection(options)
    if removed_options:
        raise RuntimeError(
            "frame_retriever no longer accepts option(s): %s"
            % ", ".join(sorted(removed_options))
        )
    time_hints = [str(item).strip() for item in list(options.get("time_hints") or []) if str(item).strip()]
    clips = clips_from_request(request_payload, task) if _scope_has_clips(request_payload) else []
    if not clips and not time_hints:
        raise RuntimeError("frame_retriever requires request.temporal_scope.clips, temporal_scope.anchors, or options.time_hints")
    return {
        "tool_name": "frame_retriever",
        "query": str(request.query or "").strip() or None,
        "clips": clips,
        "time_hints": time_hints,
        "num_frames": int(options.get("num_frames") or 5),
    }


def run(payload: ToolPayload, request: Request) -> Result:
    runtime = _runtime(payload)
    runner_request = _runner_request(request, payload.task)
    clips = [dict(item or {}) for item in list(runner_request.get("clips") or []) if isinstance(item, dict)]
    time_hints = [str(item).strip() for item in list(runner_request.get("time_hints") or []) if str(item).strip()]
    subrequests: list[dict[str, Any]] = []
    if len(clips) > 1:
        for clip in clips:
            sub = dict(runner_request)
            sub["clips"] = [clip]
            subrequests.append(sub)
    elif not clips and len(time_hints) > 1:
        for time_hint in time_hints:
            sub = dict(runner_request)
            sub["time_hints"] = [time_hint]
            subrequests.append(sub)
    else:
        subrequests = [runner_request]

    frames: list[dict[str, Any]] = []
    frame_groups: list[dict[str, Any]] = []
    for subrequest in subrequests:
        raw = _execute({"tool": "frame_retriever", "task": payload.task, "request": subrequest, "runtime": runtime})
        clip = (list(subrequest.get("clips") or []) or [None])[0]
        group_frames = [
            _normalize_frame(dict(item or {}), payload, runtime, clip if isinstance(clip, dict) else None)
            for item in list(raw.get("frames") or [])
            if isinstance(item, dict)
        ]
        frames.extend(group_frames)
        frame_groups.append(
            {
                "clips": list(subrequest.get("clips") or []),
                "time_hints": list(subrequest.get("time_hints") or []),
                "frames": group_frames,
                "cache_metadata": raw.get("cache_metadata") or {},
                "rationale": raw.get("rationale") or "",
            }
        )

    limit = max(1, int(runner_request.get("num_frames") or 5))
    if len(subrequests) > 1 and len(frames) > limit:
        frames = sorted(frames, key=_frame_sort_key)[:limit]
    output = {"frames": frames}
    return Result(
        ok=True,
        output=Output.model_validate(output),
        artifacts=[{"kind": "frame", **frame} for frame in frames],
        metadata={"frame_groups": frame_groups},
    )


if __name__ == "__main__":
    main(run, request_model=Request)
