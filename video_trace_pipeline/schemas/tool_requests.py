from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator, validator

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
    time_hint: Optional[str] = None
    query: Optional[str] = None
    num_frames: int = 5

    @model_validator(mode="after")
    def _require_clip_or_time_hint(self):
        if self.clip is not None:
            return self
        if str(self.time_hint or "").strip():
            return self
        raise ValueError("frame_retriever requires either clip or time_hint")


class AudioTemporalGrounderRequest(ToolRequest):
    query: str
    clip: Optional[ClipRef] = None


class ASRRequest(ToolRequest):
    clip: ClipRef
    speaker_attribution: bool = True


class DenseCaptionRequest(ToolRequest):
    clip: ClipRef
    granularity: str = "segment"
    focus_query: str = ""


class OCRRequest(ToolRequest):
    clip: Optional[ClipRef] = None
    frame: Optional[FrameRef] = None
    region: Optional[RegionRef] = None
    query: Optional[str] = None


class SpatialGrounderRequest(ToolRequest):
    frame: FrameRef
    query: str


class GenericPurposeRequest(ToolRequest):
    query: str
    clip: Optional[ClipRef] = None
    frame: Optional[FrameRef] = None
    transcript: Optional[TranscriptRef] = None
    evidence_ids: List[str] = Field(default_factory=list)
