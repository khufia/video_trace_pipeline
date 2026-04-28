from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator


class ArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    relpath: Optional[str] = None
    media_type: Optional[str] = None
    source_tool: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


class ClipRef(BaseModel):
    video_id: str
    start_s: float
    end_s: float
    artifact_id: Optional[str] = None
    relpath: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    @validator("end_s")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start_s", 0.0) or 0.0)
        if float(value) < start:
            raise ValueError("end_s must be >= start_s")
        return float(value)


class FrameRef(BaseModel):
    video_id: str
    timestamp_s: float
    artifact_id: Optional[str] = None
    relpath: Optional[str] = None
    clip: Optional[ClipRef] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


class RegionRef(BaseModel):
    frame: FrameRef
    bbox: List[float]
    label: Optional[str] = None
    artifact_id: Optional[str] = None
    relpath: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    @validator("bbox")
    def _validate_bbox(cls, value):  # noqa: N805
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements")
        return [float(item) for item in value]


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None


class TranscriptRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    clip: Optional[ClipRef] = None
    relpath: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)
    metadata: Dict[str, object] = Field(default_factory=dict)
