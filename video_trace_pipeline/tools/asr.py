from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..backends.media import clips_from_request, extract_audio_clip
from ..tool_io import ToolPayload, main


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    temporal_scope: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)


class Result(BaseModel):
    ok: bool
    tool: str = "asr"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _scope_has_clips(request: dict[str, Any]) -> bool:
    scope = dict(request.get("temporal_scope") or {})
    return bool(list(scope.get("clips") or []) or list(scope.get("segments") or []) or list(scope.get("anchors") or []))


def run(payload: ToolPayload, request: Request) -> Result:
    from video_trace_pipeline.tools.local_asr import _transcribe_with_whisperx

    request_payload = request.model_dump(mode="json")
    if not _scope_has_clips(request_payload):
        raise RuntimeError("asr requires request.temporal_scope.clips")
    runtime = payload.runtime.model_dump(mode="json")
    extra = dict(runtime.get("extra") or {})
    model_name = str(extra.get("model_name") or runtime.get("model_name") or runtime.get("model") or "large-v3")
    language = extra.get("language")
    ffmpeg_bin = str(runtime.get("ffmpeg_bin") or "ffmpeg")
    transcript_segments: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(clips_from_request(request_payload, payload.task), start=1):
        start_s = float(clip.get("start_s") or 0.0)
        end_s = float(clip.get("end_s") or start_s)
        if end_s <= start_s:
            continue
        audio_path = None
        try:
            audio_path = extract_audio_clip(str(payload.task.get("video_path") or ""), ffmpeg_bin, start_s, end_s)
            result, runtime_warning = _transcribe_with_whisperx(
                audio_path,
                model_name=model_name,
                device_label=str(runtime.get("device") or "cpu"),
                language=language,
            )
            for segment_index, segment in enumerate(list(result.get("segments") or []), start=1):
                text = str(segment.get("text") or "").strip()
                if not text:
                    continue
                transcript_segments.append(
                    {
                        "id": "asr_%02d_%03d" % (clip_index, segment_index),
                        "start_s": round(start_s + float(segment.get("start") or 0.0), 3),
                        "end_s": round(start_s + float(segment.get("end") or segment.get("start") or 0.0), 3),
                        "text": text,
                        "speaker": segment.get("speaker"),
                        "clip": clip,
                        "metadata": {
                            "source": "whisperx_local",
                            "language": result.get("language") or language,
                            "runtime_warning": runtime_warning,
                        },
                    }
                )
        finally:
            if audio_path:
                shutil.rmtree(str(Path(audio_path).parent), ignore_errors=True)
    return Result(ok=True, output=Output.model_validate({"transcript_segments": transcript_segments}))


if __name__ == "__main__":
    main(run, request_model=Request)
