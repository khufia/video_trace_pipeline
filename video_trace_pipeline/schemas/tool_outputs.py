from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

from .artifacts import ClipRef, TranscriptRef


_GENERIC_CONFIDENCE_LABELS = {
    "very low": 0.1,
    "low": 0.25,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.85,
    "very high": 0.95,
}


def _coerce_optional_generic_confidence(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if 0.0 <= numeric <= 1.0 else None
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.lower()
    if normalized in {"none", "null", "unknown", "n/a", "na", "unavailable"}:
        return None
    if normalized in _GENERIC_CONFIDENCE_LABELS:
        return _GENERIC_CONFIDENCE_LABELS[normalized]
    if normalized.endswith("%"):
        try:
            numeric = float(normalized[:-1].strip()) / 100.0
        except ValueError:
            return None
        return numeric if 0.0 <= numeric <= 1.0 else None
    try:
        numeric = float(normalized)
    except ValueError:
        return None
    return numeric if 0.0 <= numeric <= 1.0 else None


class TimeRange(BaseModel):
    start_s: float
    end_s: float

    @validator("end_s")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start_s", 0.0) or 0.0)
        end = float(value or 0.0)
        if end < start:
            raise ValueError("end_s must be >= start_s")
        return end


class TemporalClipCandidate(BaseModel):
    video_id: str
    start_s: float
    end_s: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("end_s")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start_s", 0.0) or 0.0)
        end = float(value or 0.0)
        if end < start:
            raise ValueError("end_s must be >= start_s")
        return end

    def as_clip_ref(self) -> ClipRef:
        metadata = dict(self.metadata or {})
        if self.confidence is not None and "confidence" not in metadata:
            metadata["confidence"] = self.confidence
        return ClipRef(
            video_id=self.video_id,
            start_s=self.start_s,
            end_s=self.end_s,
            metadata=metadata,
        )


class VisualTemporalGrounderOutput(BaseModel):
    query: str
    clips: List[TemporalClipCandidate] = Field(default_factory=list)
    video_duration: Optional[float] = None
    retrieval_backend: Optional[str] = None
    query_absent: bool = False
    summary: str = ""
    prefilter: Dict[str, Any] = Field(default_factory=dict)


class AudioEventCandidate(BaseModel):
    event_label: str
    start_s: float
    end_s: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("end_s")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start_s", 0.0) or 0.0)
        end = float(value or 0.0)
        if end < start:
            raise ValueError("end_s must be >= start_s")
        return end


class AudioTemporalGrounderOutput(BaseModel):
    query: str
    clips: List[TemporalClipCandidate] = Field(default_factory=list)
    events: List[AudioEventCandidate] = Field(default_factory=list)
    retrieval_backend: Optional[str] = None
    query_absent: bool = False
    summary: str = ""


class RetrievedFrame(BaseModel):
    frame_path: str
    timestamp_s: float
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FrameRetrieverOutput(BaseModel):
    query: Optional[str] = None
    frames: List[RetrievedFrame] = Field(default_factory=list)
    mode: str = "clip_bounded"
    cache_metadata: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ASRSegmentOutput(BaseModel):
    start_s: float
    end_s: float
    text: str
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None

    @validator("end_s")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start_s", 0.0) or 0.0)
        end = float(value or 0.0)
        if end < start:
            raise ValueError("end_s must be >= start_s")
        return end


class ASROutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clips: List[ClipRef] = Field(default_factory=list)
    transcripts: List[TranscriptRef] = Field(default_factory=list)
    phrase_matches: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_clips_and_transcripts(self):
        if self.clips and self.transcripts:
            return self
        raise ValueError("asr output requires at least one clip and one transcript")


class DenseCaptionSpan(BaseModel):
    start: float
    end: float
    visual: str = ""
    audio: str = ""
    on_screen_text: str = ""
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)

    @validator("end")
    def _validate_range(cls, value, values):  # noqa: N805
        start = float(values.get("start", 0.0) or 0.0)
        end = float(value or 0.0)
        if end < start:
            raise ValueError("end must be >= start")
        return end


class DenseCaptionOutput(BaseModel):
    clips: List[ClipRef] = Field(default_factory=list)
    captioned_range: TimeRange
    captions: List[DenseCaptionSpan] = Field(default_factory=list)
    overall_summary: str = ""
    sampled_frames: List[Dict[str, Any]] = Field(default_factory=list)
    backend: Optional[str] = None

    @model_validator(mode="after")
    def _require_clips(self):
        if self.clips:
            return self
        raise ValueError("dense_captioner output requires at least one clip")


class OCRLineOutput(BaseModel):
    text: str
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None

    @validator("bbox")
    def _validate_bbox(cls, value):  # noqa: N805
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements")
        return [float(item) for item in value]


class OCROutput(BaseModel):
    text: str = ""
    lines: List[OCRLineOutput] = Field(default_factory=list)
    query: Optional[str] = None
    timestamp_s: Optional[float] = None
    source_frame_path: Optional[str] = None
    backend: Optional[str] = None


class SpatialDetectionOutput(BaseModel):
    label: str
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("bbox")
    def _validate_bbox(cls, value):  # noqa: N805
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements")
        return [float(item) for item in value]


class SpatialGrounderOutput(BaseModel):
    query: str
    timestamp_s: Optional[float] = None
    detections: List[SpatialDetectionOutput] = Field(default_factory=list)
    spatial_description: str = ""
    source_frame_path: Optional[str] = None
    backend: Optional[str] = None


class GenericPurposeOutput(BaseModel):
    answer: str = ""
    supporting_points: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    analysis: str = ""

    @validator("confidence", pre=True)
    def _normalize_confidence(cls, value):  # noqa: N805
        return _coerce_optional_generic_confidence(value)

