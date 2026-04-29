from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    frames: List[FrameRef] = Field(default_factory=list)
    query: str

    @model_validator(mode="after")
    def _require_frames(self):
        if self.frames:
            return self
        raise ValueError("spatial_grounder requires at least one frame")


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


class VerifierClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    claim_type: str = "option_mapping"
    expected_answer_option: Optional[str] = None

    @model_validator(mode="after")
    def _require_text_fields(self):
        self.claim_id = str(self.claim_id or "").strip()
        self.text = str(self.text or "").strip()
        self.claim_type = str(self.claim_type or "option_mapping").strip() or "option_mapping"
        if self.expected_answer_option is not None:
            self.expected_answer_option = str(self.expected_answer_option or "").strip() or None
        if not self.claim_id or not self.text:
            raise ValueError("verifier claims require claim_id and text")
        return self


class VerifierRequest(ToolRequest):
    query: str
    claims: List[VerifierClaimInput] = Field(default_factory=list)
    clips: List[ClipRef] = Field(default_factory=list)
    frames: List[FrameRef] = Field(default_factory=list)
    regions: List[RegionRef] = Field(default_factory=list)
    transcripts: List[TranscriptRef] = Field(default_factory=list)
    text_contexts: List[str] = Field(default_factory=list)
    ocr_results: List[Dict[str, Any]] = Field(default_factory=list)
    dense_captions: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)
    verification_policy: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_and_require_context(self):
        self.query = str(self.query or "").strip()
        self.text_contexts = [str(item).strip() for item in list(self.text_contexts or []) if str(item).strip()]
        self.evidence_ids = [str(item).strip() for item in list(self.evidence_ids or []) if str(item).strip()]
        if not self.query:
            raise ValueError("verifier requires a non-empty query")
        if not self.claims:
            raise ValueError("verifier requires at least one claim")
        has_context = any(
            [
                self.clips,
                self.frames,
                self.regions,
                self.transcripts,
                self.text_contexts,
                self.ocr_results,
                self.dense_captions,
                self.evidence_ids,
                self.observations,
                bool(self.retrieved_context),
            ]
        )
        if not has_context:
            raise ValueError("verifier requires at least one evidence/media/text context input")
        return self
