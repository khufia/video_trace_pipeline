from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

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
