from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .artifacts import ArtifactRef


class ToolResult(BaseModel):
    tool_name: str
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    raw_output_text: Optional[str] = None
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    request_hash: Optional[str] = None
    cache_hit: bool = False
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AtomicObservation(BaseModel):
    observation_id: str
    subject: str
    subject_type: str
    predicate: str
    object_text: str = ""
    object_type: str = "text"
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    time_start_s: Optional[float] = None
    time_end_s: Optional[float] = None
    frame_ts_s: Optional[float] = None
    bbox: Optional[List[float]] = None
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None
    source_tool: str
    source_artifact_refs: List[str] = Field(default_factory=list)
    direct_or_derived: str = "direct"
    atomic_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("bbox")
    def _validate_bbox(cls, value):  # noqa: N805
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements")
        return [float(item) for item in value]


class EvidenceEntry(BaseModel):
    evidence_id: str
    tool_name: str
    evidence_text: str
    inference_hint: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "provisional"
    time_start_s: Optional[float] = None
    time_end_s: Optional[float] = None
    frame_ts_s: Optional[float] = None
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    observation_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("status")
    def _validate_status(cls, value):  # noqa: N805
        normalized = str(value or "provisional").strip().lower()
        if normalized not in {"validated", "provisional", "superseded"}:
            raise ValueError("status must be validated, provisional, or superseded")
        return normalized


class InferenceStep(BaseModel):
    step_id: int
    text: str
    supporting_observation_ids: List[str] = Field(default_factory=list)
    answer_relevance: str = "medium"
    time_start_s: Optional[float] = None
    time_end_s: Optional[float] = None
    frame_ts_s: Optional[float] = None

    @validator("step_id", pre=True)
    def _coerce_step_id(cls, value):
        if isinstance(value, bool):
            raise ValueError("step_id must not be a boolean")
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("step_id must be positive")
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("step_id must be an integer")
            value = int(value)
            if value <= 0:
                raise ValueError("step_id must be positive")
            return value
        text = str(value or "").strip()
        if not text:
            raise ValueError("step_id must be non-empty")
        if re.fullmatch(r"\d+", text):
            return int(text)
        matches = re.findall(r"\d+", text)
        if matches:
            parsed = int(matches[-1])
            if parsed <= 0:
                raise ValueError("step_id must be positive")
            return parsed
        raise ValueError("step_id must contain an integer value")

    @validator("text")
    def _validate_text(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("text must be non-empty")
        return value


class TracePackage(BaseModel):
    task_key: str
    mode: str
    evidence_entries: List[EvidenceEntry] = Field(default_factory=list)
    inference_steps: List[InferenceStep] = Field(default_factory=list)
    final_answer: str = ""
    benchmark_renderings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditFinding(BaseModel):
    severity: str
    category: str
    message: str
    evidence_ids: List[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    verdict: str = "FAIL"
    confidence: float = 0.0
    scores: Dict[str, int] = Field(default_factory=dict)
    findings: List[AuditFinding] = Field(default_factory=list)
    feedback: str = ""
    missing_information: List[str] = Field(default_factory=list)

    @validator("verdict")
    def _validate_verdict(cls, value):  # noqa: N805
        value = str(value or "FAIL").strip().upper()
        return value if value in {"PASS", "FAIL"} else "FAIL"

    @validator("scores", pre=True)
    def _normalize_scores(cls, value):  # noqa: N805
        if not isinstance(value, dict):
            return {}
        normalized = {}
        for key, raw_value in dict(value or {}).items():
            try:
                numeric = float(raw_value)
            except Exception:
                continue
            normalized[str(key)] = max(1, min(5, int(round(numeric))))
        return normalized
