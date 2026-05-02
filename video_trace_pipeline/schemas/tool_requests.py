from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

from .artifacts import ClipRef, FrameRef, RegionRef, TranscriptRef


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    clips: List[ClipRef] = Field(default_factory=list)
    time_hints: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    num_frames: int = 5
    sequence_mode: Literal["ranked", "anchor_window", "chronological"] = "ranked"
    neighbor_radius_s: float = 2.0
    include_anchor_neighbors: bool = True
    sort_order: Literal["ranked", "chronological"] = "ranked"

    @model_validator(mode="after")
    def _require_clips_or_time_hints(self):
        self.time_hints = [str(item).strip() for item in list(self.time_hints or []) if str(item).strip()]
        if self.clips or self.time_hints:
            return self
        raise ValueError("frame_retriever requires at least one clip or time_hints entry")


class AudioTemporalGrounderRequest(ToolRequest):
    query: str
    clips: List[ClipRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clips:
            return self
        raise ValueError("audio_temporal_grounder requires at least one clip")


class ASRRequest(ToolRequest):
    clips: List[ClipRef] = Field(default_factory=list)
    speaker_attribution: bool = True

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clips:
            return self
        raise ValueError("asr requires at least one clip")


class DenseCaptionRequest(ToolRequest):
    clips: List[ClipRef] = Field(default_factory=list)
    granularity: str = "segment"
    focus_query: str = ""

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clips:
            return self
        raise ValueError("dense_captioner requires at least one clip")


class OCRRequest(ToolRequest):
    clips: List[ClipRef] = Field(default_factory=list)
    frames: List[FrameRef] = Field(default_factory=list)
    regions: List[RegionRef] = Field(default_factory=list)
    query: Optional[str] = None

    @model_validator(mode="after")
    def _require_media(self):
        if self.regions or self.frames:
            self.clips = []
        if self.regions or self.frames or self.clips:
            return self
        raise ValueError("ocr requires at least one clip, frame, or region")


class SpatialGrounderRequest(ToolRequest):
    clips: List[ClipRef] = Field(default_factory=list)
    frames: List[FrameRef] = Field(default_factory=list)
    query: str

    @model_validator(mode="after")
    def _require_media(self):
        if self.frames:
            self.clips = []
            return self
        if self.clips:
            return self
        raise ValueError("spatial_grounder requires at least one frame or clip")


class GenericPurposeRequest(ToolRequest):
    query: str
    clips: List[ClipRef] = Field(default_factory=list)
    frames: List[FrameRef] = Field(default_factory=list)
    transcripts: List[TranscriptRef] = Field(default_factory=list)
    text_contexts: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_lists(self):
        self.text_contexts = [str(item).strip() for item in list(self.text_contexts or []) if str(item).strip()]
        self.evidence_ids = [str(item).strip() for item in list(self.evidence_ids or []) if str(item).strip()]
        return self

