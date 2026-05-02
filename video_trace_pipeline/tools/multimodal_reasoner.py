from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..backends.media import clips_from_request
from ..tool_io import ToolPayload, main


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    media: dict[str, Any] = Field(default_factory=dict)
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    answer: str = ""
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None


class Result(BaseModel):
    ok: bool
    tool: str = "multimodal_reasoner"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _runtime(payload: ToolPayload) -> dict[str, Any]:
    runtime = payload.runtime.model_dump(mode="json")
    runtime["tool"] = "generic_purpose"
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    runtime["extra"] = dict(runtime.get("extra") or {})
    run_dir = Path(str(runtime.get("run_dir") or ".")).expanduser().resolve()
    scratch_dir = run_dir / "scratch" / "multimodal_reasoner"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    return runtime


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    from video_trace_pipeline.tool_wrappers.qwen35vl_runner import execute_payload

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return execute_payload(payload)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _scope_has_clips(request_payload: dict[str, Any]) -> bool:
    scope = dict(request_payload.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def _media_text_contexts(media: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in list(media.get("texts") or []):
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            texts.append(text)
    for key, text_key in (
        ("lines", "text"),
        ("reads", "text"),
        ("captions", "caption"),
        ("captions", "visual"),
        ("transcript_segments", "text"),
        ("ocr_results", "text"),
    ):
        for item in list(media.get(key) or []):
            if isinstance(item, dict):
                text = str(item.get(text_key) or "").strip()
                if text:
                    texts.append(text)
    ordered = []
    seen = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _transcripts(media: dict[str, Any]) -> list[dict[str, Any]]:
    transcripts = [dict(item or {}) for item in list(media.get("transcripts") or []) if isinstance(item, dict)]
    if transcripts:
        return transcripts
    spans = [dict(item or {}) for item in list(media.get("transcript_segments") or []) if isinstance(item, dict)]
    if not spans:
        return []
    text = " ".join(str(item.get("text") or "").strip() for item in spans if str(item.get("text") or "").strip())
    clip = dict(spans[0].get("clip") or {}) if spans else {}
    return [{"clip": clip, "segments": spans, "text": text}]


def _runner_request(request: Request) -> dict[str, Any]:
    request_payload = request.model_dump(mode="json")
    media = dict(request.media or {})
    frames = [dict(item or {}) for item in list(media.get("frames") or []) if isinstance(item, dict)]
    text_contexts = _media_text_contexts(media)
    for region in [dict(item or {}) for item in list(media.get("regions") or []) if isinstance(item, dict)]:
        frame = dict(region.get("frame") or {})
        if frame:
            frames.append(frame)
        if region.get("label") or region.get("bbox"):
            text_contexts.append("Region %s bbox=%s" % (region.get("label") or "target", json.dumps(region.get("bbox"))))
    options = dict(request.options or {})
    return {
        "tool_name": "generic_purpose",
        "query": str(request.query or "").strip(),
        "clips": clips_from_request(request_payload, {}) if _scope_has_clips(request_payload) else [],
        "frames": frames,
        "transcripts": _transcripts(media),
        "text_contexts": text_contexts,
        "evidence_ids": [str(item).strip() for item in list(options.get("evidence_ids") or []) if str(item).strip()],
    }


def run(payload: ToolPayload, request: Request) -> Result:
    runner_request = _runner_request(request)
    request_payload = request.model_dump(mode="json")
    if _scope_has_clips(request_payload):
        runner_request["clips"] = clips_from_request(request_payload, payload.task)
    raw = _execute({"tool": "generic_purpose", "task": payload.task, "request": runner_request, "runtime": _runtime(payload)})
    output = {
        "answer": str(raw.get("answer") or ""),
        "reasoning": str(raw.get("analysis") or raw.get("answer") or ""),
        "evidence": [str(item).strip() for item in list(raw.get("supporting_points") or []) if str(item).strip()],
        "confidence": raw.get("confidence"),
    }
    return Result(ok=True, output=Output.model_validate(output))


if __name__ == "__main__":
    main(run, request_model=Request)
