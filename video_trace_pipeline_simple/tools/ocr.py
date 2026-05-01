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
    media: dict[str, Any] = Field(default_factory=dict)
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    text: str = ""
    lines: list[dict[str, Any]] = Field(default_factory=list)
    reads: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    ok: bool
    tool: str = "ocr"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "ocr"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "ocr"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _scope_has_clips(request_payload: dict[str, Any]) -> bool:
    scope = dict(request_payload.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def _execute_paddleocr(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers import paddleocr_runner

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            request = dict(payload.get("request") or {})
            task = dict(payload.get("task") or {})
            runtime = dict(payload.get("runtime") or {})
            frame_out_dir = paddleocr_runner.scratch_dir(runtime, "ocr")
            request_items = paddleocr_runner._extract_request_items(request)
            engine = paddleocr_runner.create_paddleocr_engine(runtime)
            results = [
                paddleocr_runner._run_single_request(
                    item,
                    task=task,
                    runtime=runtime,
                    frame_out_dir=frame_out_dir,
                    engine=engine,
                )
                for item in request_items
            ]
            if len(results) <= 1:
                return results[0] if results else {"text": "", "lines": [], "query": None, "timestamp_s": 0.0, "source_frame_path": "", "backend": "paddleocr"}
            return {"results": results, "backend": "paddleocr"}
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def run(payload: ToolPayload, request: Request) -> Result:
    request_payload = request.model_dump(mode="json")
    media = dict(request.media or {})
    runner_request = {
        "tool_name": "ocr",
        "query": str(request.query or "").strip() or None,
        "frames": [dict(item or {}) for item in list(media.get("frames") or []) if isinstance(item, dict)],
        "regions": [dict(item or {}) for item in list(media.get("regions") or []) if isinstance(item, dict)],
        "clips": clips_from_request(request_payload, payload.task) if _scope_has_clips(request_payload) else [],
    }
    if not runner_request["frames"] and not runner_request["regions"] and media.get("segments"):
        runner_request["clips"] = [dict(item or {}) for item in list(media.get("segments") or []) if isinstance(item, dict)]
    if not runner_request["frames"] and not runner_request["regions"] and not runner_request["clips"]:
        raise RuntimeError("ocr requires request.media.frames, request.media.regions, or request.temporal_scope.clips")

    raw = _execute_paddleocr({"tool": "ocr", "task": payload.task, "request": runner_request, "runtime": _runtime(payload)})
    if isinstance(raw.get("results"), list):
        reads = [dict(item or {}) for item in list(raw.get("results") or []) if isinstance(item, dict)]
        lines: list[dict[str, Any]] = []
        texts: list[str] = []
        for item in reads:
            item_lines = [dict(line or {}) for line in list(item.get("lines") or []) if isinstance(line, dict)]
            lines.extend(item_lines)
            if str(item.get("text") or "").strip():
                texts.append(str(item.get("text")).strip())
        output = {"text": "\n\n".join(texts).strip(), "lines": lines, "reads": reads, "metadata": {"backend": raw.get("backend")}}
        return Result(ok=True, output=Output.model_validate(output))

    output = {
        "text": str(raw.get("text") or ""),
        "lines": [dict(item or {}) for item in list(raw.get("lines") or []) if isinstance(item, dict)],
        "reads": [dict(raw)],
        "metadata": {
            "backend": raw.get("backend"),
            "timestamp_s": raw.get("timestamp_s"),
            "source_frame_path": raw.get("source_frame_path"),
        },
    }
    return Result(ok=True, output=Output.model_validate(output))


if __name__ == "__main__":
    main(run, request_model=Request)
