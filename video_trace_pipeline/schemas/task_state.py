from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


TASK_CLAIM_STATUSES = {
    "unverified",
    "validated",
    "refuted",
    "unknown",
    "partially_validated",
}

TASK_EVIDENCE_STATUSES = {
    "candidate",
    "validated",
    "refuted",
    "irrelevant",
    "superseded",
    "stale",
    "unknown",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    ordered: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


class TaskClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    option: Optional[str] = None
    text: str
    claim_type: str = "option_mapping"
    required_modalities: List[str] = Field(default_factory=list)
    status: str = "unverified"
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    supporting_observation_ids: List[str] = Field(default_factory=list)
    refuting_evidence_ids: List[str] = Field(default_factory=list)
    refuting_observation_ids: List[str] = Field(default_factory=list)
    coverage_ids: List[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("claim_id", "text", "claim_type", "status", "notes")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value):
        normalized = _normalize_text(value).lower() or "unverified"
        if normalized not in TASK_CLAIM_STATUSES:
            raise ValueError("claim status must be one of: %s" % ", ".join(sorted(TASK_CLAIM_STATUSES)))
        return normalized

    @field_validator(
        "required_modalities",
        "supporting_evidence_ids",
        "supporting_observation_ids",
        "refuting_evidence_ids",
        "refuting_observation_ids",
        "coverage_ids",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class ReferentSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    referent_id: str
    description: str
    scope: str = ""
    status: str = "unresolved"
    linked_claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    observation_ids: List[str] = Field(default_factory=list)

    @field_validator("referent_id", "description", "scope", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("linked_claim_ids", "evidence_ids", "observation_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class CoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_id: str
    modality: str
    time_range: Dict[str, float] = Field(default_factory=dict)
    sampling: str = ""
    checked_by: str = ""
    status: str = "candidate"
    linked_claim_ids: List[str] = Field(default_factory=list)

    @field_validator("coverage_id", "modality", "sampling", "checked_by", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("linked_claim_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class CounterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counter_id: str
    target: str
    inclusion_rule: str = ""
    exclusion_rule: str = ""
    accepted_observation_ids: List[str] = Field(default_factory=list)
    rejected_observation_ids: List[str] = Field(default_factory=list)
    count: Optional[float] = None
    status: str = "open"

    @field_validator("counter_id", "target", "inclusion_rule", "exclusion_rule", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("accepted_observation_ids", "rejected_observation_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class OCRValueOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    kind: str = "text"
    raw_text: str
    normalized_value: Optional[float] = None
    nearby_label: str = ""
    source_artifact_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    status: str = "candidate"
    dedupe_key: Optional[str] = None

    @field_validator("occurrence_id", "kind", "raw_text", "nearby_label", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)


class TemporalEventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    description: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    source: str = ""
    status: str = "candidate"
    linked_claim_ids: List[str] = Field(default_factory=list)

    @field_validator("event_id", "description", "source", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("linked_claim_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class EvidenceStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    previous_status: str = ""
    new_status: str
    claim_id: Optional[str] = None
    reason: str = ""
    updated_by: str = ""
    round_index: Optional[int] = None

    @field_validator("evidence_id", "previous_status", "new_status", "claim_id", "reason", "updated_by")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("new_status")
    @classmethod
    def _validate_evidence_status(cls, value):
        normalized = _normalize_text(value).lower() or "candidate"
        if normalized not in TASK_EVIDENCE_STATUSES:
            raise ValueError("evidence status must be one of: %s" % ", ".join(sorted(TASK_EVIDENCE_STATUSES)))
        return normalized


class RetrievalMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_id: str
    query: str
    target: str = ""
    result_ids: List[str] = Field(default_factory=list)
    used_result_ids: List[str] = Field(default_factory=list)
    reason_unused: str = ""

    @field_validator("retrieval_id", "query", "target", "reason_unused")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("result_ids", "used_result_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class AnswerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option: str
    supporting_claim_ids: List[str] = Field(default_factory=list)
    refuting_claim_ids: List[str] = Field(default_factory=list)
    unknown_claim_ids: List[str] = Field(default_factory=list)
    status: str = "possible"

    @field_validator("option", "status")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("supporting_claim_ids", "refuting_claim_ids", "unknown_claim_ids", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "task_state_v1"
    task_key: str
    claim_results: List[TaskClaimResult] = Field(default_factory=list)
    referent_slots: List[ReferentSlot] = Field(default_factory=list)
    coverage_records: List[CoverageRecord] = Field(default_factory=list)
    counter_records: List[CounterRecord] = Field(default_factory=list)
    ocr_occurrences: List[OCRValueOccurrence] = Field(default_factory=list)
    temporal_events: List[TemporalEventRecord] = Field(default_factory=list)
    evidence_status_updates: List[EvidenceStatusUpdate] = Field(default_factory=list)
    retired_evidence: List[str] = Field(default_factory=list)
    retrieval_memory: List[RetrievalMemoryRecord] = Field(default_factory=list)
    answer_candidates: List[AnswerCandidate] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    ready_for_synthesis: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "task_key")
    @classmethod
    def _normalize_strings(cls, value):
        return _normalize_text(value)

    @field_validator("retired_evidence", "open_questions", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        return _normalize_list(value)
