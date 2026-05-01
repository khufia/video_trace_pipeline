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
    media: dict[str, Any] = Field(default_factory=dict)
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    regions: list[dict[str, Any]] = Field(default_factory=list)
    spatial_description: str = ""


class Result(BaseModel):
    ok: bool
    tool: str = "spatial_grounder"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "spatial_grounder"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "spatial_grounder"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.spatial_grounder_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def run(payload: ToolPayload, request: Request) -> Result:
    media = dict(request.media or {})
    frames = [dict(item or {}) for item in list(media.get("frames") or []) if isinstance(item, dict)]
    if not frames:
        raise RuntimeError("spatial_grounder requires request.media.frames")
    runtime = _runtime(payload)
    regions: list[dict[str, Any]] = []
    descriptions: list[str] = []
    groundings: list[dict[str, Any]] = []
    for frame in frames:
        runner_request = {
            "tool_name": "spatial_grounder",
            "query": str(request.query or "").strip(),
            "frames": [frame],
        }
        raw = _execute({"tool": "spatial_grounder", "task": payload.task, "request": runner_request, "runtime": runtime})
        detections = [dict(item or {}) for item in list(raw.get("detections") or []) if isinstance(item, dict)]
        frame_regions = []
        for detection in detections:
            if detection.get("bbox") is None:
                continue
            region = {
                "label": detection.get("label"),
                "bbox": detection.get("bbox"),
                "frame": frame,
                "confidence": detection.get("confidence"),
                "metadata": dict(detection.get("metadata") or {}),
            }
            frame_regions.append(region)
            regions.append(region)
        description = str(raw.get("spatial_description") or "").strip()
        if description:
            descriptions.append(description)
        groundings.append({"frames": [frame], "detections": detections, "regions": frame_regions, "spatial_description": description})
    output = {"regions": regions, "spatial_description": "\n".join(descriptions).strip()}
    return Result(ok=True, output=Output.model_validate(output), metadata={"groundings": groundings})


if __name__ == "__main__":
    main(run, request_model=Request)
