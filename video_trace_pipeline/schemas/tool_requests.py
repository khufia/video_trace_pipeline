from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

from ..common import hash_payload
from .artifacts import ClipRef, FrameRef, RegionRef, TranscriptRef


class ToolRequest(BaseModel):
    tool_name: str

    @validator("tool_name")
    def _validate_tool_name(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("tool_name must be non-empty")
        return value


class VisualTemporalGrounderRequest(ToolRequest):
    query: str
    top_k: int = 5


class FrameRetrieverRequest(ToolRequest):
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)
    time_hint: Optional[str] = None
    time_hints: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    num_frames: int = 5

    @model_validator(mode="after")
    def _require_clip_or_time_hint(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.clips and self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        if str(self.time_hint or "").strip() and not self.time_hints:
            self.time_hints = [str(self.time_hint).strip()]
        if self.time_hints and not str(self.time_hint or "").strip() and len(self.time_hints) == 1:
            self.time_hint = str(self.time_hints[0] or "").strip()
        if self.clips:
            return self
        if self.clip is not None:
            return self
        if self.time_hints:
            return self
        if str(self.time_hint or "").strip():
            return self
        raise ValueError("frame_retriever requires either clip or time_hint")


class AudioTemporalGrounderRequest(ToolRequest):
    query: str
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_clips(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.clips and self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        return self


class ASRRequest(ToolRequest):
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)
    speaker_attribution: bool = True

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.clips and self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        if self.clip is None and not self.clips:
            raise ValueError("asr requires at least one clip")
        return self


class DenseCaptionRequest(ToolRequest):
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)
    granularity: str = "segment"
    focus_query: str = ""

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.clips and self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        if self.clip is None and not self.clips:
            raise ValueError("dense_captioner requires at least one clip")
        return self


class OCRRequest(ToolRequest):
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)
    frame: Optional[FrameRef] = None
    frames: List[FrameRef] = Field(default_factory=list)
    region: Optional[RegionRef] = None
    regions: List[RegionRef] = Field(default_factory=list)
    query: Optional[str] = None

    @model_validator(mode="after")
    def _require_media(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.frame is not None and not self.frames:
            self.frames = [self.frame]
        if self.region is not None and not self.regions:
            self.regions = [self.region]
        if self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        if self.frame is None and len(self.frames) == 1:
            self.frame = self.frames[0]
        if self.region is None and len(self.regions) == 1:
            self.region = self.regions[0]
        # OCR operates on concrete frames/regions. When they are present,
        # keep the request specific and drop any fallback clip context.
        if self.regions or self.region is not None or self.frames or self.frame is not None:
            self.clip = None
            self.clips = []
        if self.regions or self.frames or self.clips or self.region is not None or self.frame is not None or self.clip is not None:
            return self
        raise ValueError("ocr requires at least one clip, frame, or region")


class SpatialGrounderRequest(ToolRequest):
    frame: Optional[FrameRef] = None
    frames: List[FrameRef] = Field(default_factory=list)
    query: str

    @model_validator(mode="after")
    def _require_frames(self):
        if self.frame is not None and not self.frames:
            self.frames = [self.frame]
        if self.frames and self.frame is None and len(self.frames) == 1:
            self.frame = self.frames[0]
        if self.frame is None and not self.frames:
            raise ValueError("spatial_grounder requires at least one frame")
        return self


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = list(value) if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_generic_transcript_item(value: Any) -> tuple[Any, List[str]]:
    if value is None or isinstance(value, TranscriptRef):
        return value, []
    if isinstance(value, str):
        cleaned = str(value or "").strip()
        return None, [cleaned] if cleaned else []
    if not isinstance(value, dict):
        return value, []

    transcript = dict(value)
    transcript_id = str(transcript.get("transcript_id") or "").strip()
    text = str(transcript.get("text") or "").strip()
    clip = transcript.get("clip")
    relpath = transcript.get("relpath")
    segments = list(transcript.get("segments") or [])
    metadata = dict(transcript.get("metadata") or {})
    backend = str(transcript.get("backend") or "").strip()
    if backend and "backend" not in metadata:
        metadata["backend"] = backend

    if transcript_id:
        normalized = dict(transcript)
        normalized["text"] = text
        normalized["segments"] = segments
        normalized["metadata"] = metadata
        return normalized, []

    if not any((text, clip is not None, relpath, segments)):
        return None, []

    return (
        {
            "transcript_id": "tx_%s"
            % hash_payload(
                {
                    "clip": clip,
                    "text": text,
                    "segments": segments,
                    "relpath": relpath,
                },
                12,
            ),
            "clip": clip,
            "relpath": relpath,
            "text": text,
            "segments": segments,
            "metadata": metadata,
        },
        [],
    )


class GenericPurposeRequest(ToolRequest):
    model_config = ConfigDict(extra="allow")

    query: str
    clip: Optional[ClipRef] = None
    clips: List[ClipRef] = Field(default_factory=list)
    frame: Optional[FrameRef] = None
    frames: List[FrameRef] = Field(default_factory=list)
    transcript: Optional[TranscriptRef] = None
    transcripts: List[TranscriptRef] = Field(default_factory=list)
    text_contexts: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_transcript_inputs(cls, value):
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        raw_transcripts: List[Any] = []
        if normalized.get("transcript") is not None:
            raw_transcripts.append(normalized.get("transcript"))
        if normalized.get("transcripts") is not None:
            if isinstance(normalized.get("transcripts"), list):
                raw_transcripts.extend(list(normalized.get("transcripts") or []))
            else:
                raw_transcripts.append(normalized.get("transcripts"))

        cleaned_transcripts: List[Any] = []
        transcript_signatures = set()
        text_contexts = _coerce_string_list(normalized.get("text_contexts"))
        seen_text_contexts = set(text_contexts)

        for item in raw_transcripts:
            normalized_item, extra_texts = _normalize_generic_transcript_item(item)
            for text_item in extra_texts:
                if text_item in seen_text_contexts:
                    continue
                seen_text_contexts.add(text_item)
                text_contexts.append(text_item)
            if normalized_item is None:
                continue
            signature_source: Dict[str, Any]
            if isinstance(normalized_item, TranscriptRef):
                signature_source = normalized_item.model_dump()
            elif isinstance(normalized_item, dict):
                signature_source = normalized_item
            else:
                signature_source = {"value": normalized_item}
            signature = hash_payload(signature_source, 16)
            if signature in transcript_signatures:
                continue
            transcript_signatures.add(signature)
            cleaned_transcripts.append(normalized_item)

        normalized["transcript"] = None
        normalized["transcripts"] = cleaned_transcripts
        normalized["text_contexts"] = text_contexts
        return normalized

    @model_validator(mode="after")
    def _normalize_media(self):
        if self.clip is not None and not self.clips:
            self.clips = [self.clip]
        if self.frame is not None and not self.frames:
            self.frames = [self.frame]
        if self.transcript is not None and not self.transcripts:
            self.transcripts = [self.transcript]
        if self.clip is None and len(self.clips) == 1:
            self.clip = self.clips[0]
        if self.frame is None and len(self.frames) == 1:
            self.frame = self.frames[0]
        if self.transcript is None and len(self.transcripts) == 1:
            self.transcript = self.transcripts[0]
        return self
