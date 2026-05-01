from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .common import read_json, write_json

RequestT = TypeVar("RequestT", bound=BaseModel)


class Clip(BaseModel):
    video_id: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Frame(BaseModel):
    video_id: str | None = None
    timestamp_s: float | None = None
    frame_path: str | None = None
    relpath: str | None = None
    clip: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Region(BaseModel):
    label: str | None = None
    bbox: list[float] | None = None
    frame: dict[str, Any] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptSpan(BaseModel):
    start_s: float | None = None
    end_s: float | None = None
    text: str = ""
    speaker: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Runtime(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str | None = None
    run_dir: str | None = None
    workspace_root: str | None = None
    hf_cache: str | None = None
    ffmpeg_bin: str = "ffmpeg"
    device: str | None = None
    backend: str | None = None
    model: str | None = None
    model_name: str | None = None
    endpoint: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str
    task: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    runtime: Runtime = Field(default_factory=Runtime)
    context: dict[str, Any] = Field(default_factory=dict)


def read_payload(path: str | Path, request_model: type[RequestT]) -> tuple[ToolPayload, RequestT]:
    payload = ToolPayload.model_validate(read_json(path))
    request = request_model.model_validate(payload.request)
    return payload, request


def write_result(path: str | Path, result: BaseModel | dict[str, Any]) -> None:
    if hasattr(result, "model_dump"):
        write_json(path, result.model_dump(mode="json"))
    else:
        write_json(path, result)


def failure_envelope(tool_name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "output": {},
        "artifacts": [],
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        },
        "metadata": {},
    }


def main(run: Callable[[ToolPayload, RequestT], BaseModel], request_model: type[RequestT]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tool_name = request_model.__module__.split(".")[-1]
    try:
        payload, request = read_payload(args.input, request_model)
        tool_name = payload.tool
        result = run(payload, request)
        write_result(args.output, result)
    except BaseException as exc:  # pragma: no cover - defensive CLI wrapper
        write_json(args.output, failure_envelope(tool_name, exc))
