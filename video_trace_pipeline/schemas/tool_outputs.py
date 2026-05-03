from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

from .artifacts import ArtifactRef, ClipRef, TranscriptRef


_GENERIC_CONFIDENCE_LABELS = {
    "very low": 0.1,
    "low": 0.25,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.85,
    "very high": 0.95,
}

_TIME_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*s?\s*(?:-|\u2013|\u2014|to)\s*(?P<end>\d+(?:\.\d+)?)\s*s?\s*$",
    re.IGNORECASE,
)


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


def _coerce_time_token(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_time_interval(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        interval = dict(value)
        aliases = {
            "start_s": ("start", "start_time", "start_time_s", "begin", "begin_s"),
            "end_s": ("end", "end_time", "end_time_s", "finish", "finish_s"),
        }
        for canonical, keys in aliases.items():
            if canonical not in interval:
                for key in keys:
                    if key in interval:
                        interval[canonical] = interval[key]
                        break
        for key in ("start_s", "end_s"):
            coerced = _coerce_time_token(interval.get(key))
            if coerced is not None:
                interval[key] = coerced
        return interval
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = _TIME_RANGE_RE.match(text)
        if match:
            return {"start_s": float(match.group("start")), "end_s": float(match.group("end"))}
        timestamp = _coerce_time_token(text)
        if timestamp is not None:
            return {"start_s": timestamp, "end_s": timestamp}
        return {"text": text}
    return None


def _normalize_time_intervals(value: Any) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and not any(isinstance(part, (dict, list, tuple)) for part in item):
                start = _coerce_time_token(item[0])
                end = _coerce_time_token(item[1])
                if start is not None and end is not None:
                    intervals.append({"start_s": start, "end_s": end})
                    return
            for part in item:
                visit(part)
            return
        interval = _coerce_time_interval(item)
        if interval:
            intervals.append(interval)

    visit(value)
    return intervals


def _as_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_string_list(value: Any) -> List[str]:
    items: List[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
            return
        if isinstance(item, dict):
            for key in (
                "id",
                "observation_id",
                "evidence_id",
                "artifact_id",
                "claim_id",
                "value",
                "text",
                "gap",
                "details",
                "reason",
                "description",
            ):
                if item.get(key) is not None:
                    text = str(item.get(key) or "").strip()
                    if text:
                        items.append(text)
                    return
            return
        text = str(item or "").strip()
        if text:
            items.append(text)

    visit(value)
    return items


def _infer_artifact_kind(artifact_id: str, relpath: Optional[str] = None, media_type: Optional[str] = None) -> str:
    text = " ".join(str(item or "").lower() for item in (artifact_id, relpath, media_type))
    if "region" in text or "crop" in text:
        return "region"
    if "clip" in text or any(ext in text for ext in (".mp4", ".mov", ".mkv", ".webm")):
        return "clip"
    if "frame" in text or any(ext in text for ext in (".png", ".jpg", ".jpeg", ".webp")):
        return "frame"
    if "transcript" in text or "asr" in text:
        return "transcript"
    return "artifact"


def _normalize_artifact_refs(value: Any) -> List[Any]:
    refs: List[Any] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, ArtifactRef):
            refs.append(item)
            return
        if isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
            return
        if isinstance(item, dict):
            payload = dict(item)
            artifact_id = (
                payload.get("artifact_id")
                or payload.get("id")
                or payload.get("artifact")
                or payload.get("name")
            )
            relpath = payload.get("relpath") or payload.get("path") or payload.get("file_path")
            if not artifact_id and relpath:
                artifact_id = str(relpath).rstrip("/").split("/")[-1]
            if not artifact_id:
                text = str(payload.get("text") or payload.get("label") or "").strip()
                if text:
                    artifact_id = text
            if not artifact_id:
                return
            media_type = payload.get("media_type")
            metadata = payload.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                payload["metadata"] = {"raw_metadata": metadata}
            payload["artifact_id"] = str(artifact_id).strip()
            payload["kind"] = str(payload.get("kind") or _infer_artifact_kind(str(artifact_id), relpath, media_type))
            if relpath is not None:
                payload["relpath"] = str(relpath)
            if media_type is not None:
                payload["media_type"] = str(media_type)
            refs.append(payload)
            return
        text = str(item or "").strip()
        if not text:
            return
        refs.append(
            {
                "artifact_id": text,
                "kind": _infer_artifact_kind(text),
                "metadata": {"raw_ref": text},
            }
        )

    visit(value)
    return refs


def _normalize_dict_list(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in _as_sequence(value):
        if isinstance(item, dict):
            items.append(dict(item))
        elif item is not None:
            text = str(item or "").strip()
            if text:
                items.append({"text": text})
    return items


def _normalize_coverage(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)):
        checked = _normalize_string_list(value)
        return {"checked_inputs": checked, "missing_inputs": [], "sampling_summary": "Checked inputs listed by verifier."}
    text = str(value or "").strip()
    if text:
        return {"checked_inputs": [], "missing_inputs": [], "sampling_summary": text}
    return {}


def _normalize_claim_results(value: Any) -> List[Any]:
    if isinstance(value, dict):
        if any(key in value for key in ("claim_id", "verdict", "rationale", "answer_value")):
            return [dict(value)]
        normalized = []
        for claim_id, result in value.items():
            if isinstance(result, dict):
                payload = dict(result)
                payload.setdefault("claim_id", str(claim_id))
            else:
                payload = {
                    "claim_id": str(claim_id),
                    "verdict": "unknown",
                    "rationale": str(result or "").strip(),
                }
            normalized.append(payload)
        return normalized
    normalized = []
    for index, item in enumerate(_as_sequence(value), start=1):
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif item is not None:
            text = str(item or "").strip()
            if text:
                normalized.append({"claim_id": "claim_%d" % index, "verdict": "unknown", "rationale": text})
    return normalized


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


class VerifierCoverageOutput(BaseModel):
    checked_inputs: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list)
    sampling_summary: str = ""

    @validator("checked_inputs", "missing_inputs", pre=True)
    def _normalize_lists(cls, value):  # noqa: N805
        return _normalize_string_list(value)

    @validator("sampling_summary", pre=True)
    def _normalize_summary(cls, value):  # noqa: N805
        return str(value or "").strip()


class VerifierClaimResult(BaseModel):
    claim_id: str
    verdict: str = "unknown"
    confidence: Optional[float] = None
    answer_value: Any = None
    claimed_value: Any = None
    observed_value: Any = None
    match_status: str = ""
    target_presence: str = ""
    supporting_observation_ids: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    refuting_observation_ids: List[str] = Field(default_factory=list)
    refuting_evidence_ids: List[str] = Field(default_factory=list)
    time_intervals: List[Dict[str, Any]] = Field(default_factory=list)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    rationale: str = ""
    coverage: VerifierCoverageOutput = Field(default_factory=VerifierCoverageOutput)

    @validator("claim_id", "rationale", "match_status", "target_presence", pre=True)
    def _normalize_text_fields(cls, value):  # noqa: N805
        return str(value or "").strip()

    @validator("confidence", pre=True)
    def _normalize_confidence(cls, value):  # noqa: N805
        return _coerce_optional_generic_confidence(value)

    @validator(
        "supporting_observation_ids",
        "supporting_evidence_ids",
        "refuting_observation_ids",
        "refuting_evidence_ids",
        pre=True,
    )
    def _normalize_id_lists(cls, value):  # noqa: N805
        return _normalize_string_list(value)

    @validator("time_intervals", pre=True)
    def _normalize_time_intervals(cls, value):  # noqa: N805
        return _normalize_time_intervals(value)

    @validator("artifact_refs", pre=True)
    def _normalize_artifact_refs(cls, value):  # noqa: N805
        return _normalize_artifact_refs(value)

    @validator("coverage", pre=True)
    def _normalize_coverage(cls, value):  # noqa: N805
        return _normalize_coverage(value)

    @validator("verdict", pre=True)
    def _normalize_verdict_value(cls, value):  # noqa: N805
        if isinstance(value, dict):
            for key in ("verdict", "status", "answer", "value"):
                if value.get(key) is not None:
                    return value.get(key)
            return "unknown"
        if isinstance(value, (list, tuple)):
            return value[0] if value else "unknown"
        return value

    @validator("verdict")
    def _validate_verdict(cls, value):  # noqa: N805
        normalized = str(value or "").strip().lower()
        aliases = {
            "support": "supported",
            "supports": "supported",
            "true": "supported",
            "yes": "supported",
            "contradicted": "refuted",
            "false": "refuted",
            "no": "refuted",
            "partial": "partially_supported",
            "partially validated": "partially_supported",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"supported", "refuted", "unknown", "partially_supported"}:
            raise ValueError("verdict must be supported, refuted, unknown, or partially_supported")
        return normalized


class VerifierOutput(BaseModel):
    claim_results: List[VerifierClaimResult] = Field(default_factory=list)
    new_observations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_updates: List[Dict[str, Any]] = Field(default_factory=list)
    checklist_updates: List[Dict[str, Any]] = Field(default_factory=list)
    counter_updates: List[Dict[str, Any]] = Field(default_factory=list)
    referent_updates: List[Dict[str, Any]] = Field(default_factory=list)
    ocr_occurrence_updates: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_gaps: List[str] = Field(default_factory=list)

    @validator("claim_results", pre=True)
    def _normalize_claim_results(cls, value):  # noqa: N805
        return _normalize_claim_results(value)

    @validator(
        "new_observations",
        "evidence_updates",
        "checklist_updates",
        "counter_updates",
        "referent_updates",
        "ocr_occurrence_updates",
        pre=True,
    )
    def _normalize_dict_lists(cls, value):  # noqa: N805
        return _normalize_dict_list(value)

    @validator("unresolved_gaps", pre=True)
    def _normalize_unresolved_gaps(cls, value):  # noqa: N805
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def _require_claim_results(self):
        if not self.claim_results:
            raise ValueError("verifier output requires at least one claim_result")
        return self
