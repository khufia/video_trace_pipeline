from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_step_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("step_id must not be a boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("step_id must be positive")
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("step_id must be an integer")
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("step_id must be positive")
        return parsed
    text = str(value or "").strip()
    if not text:
        raise ValueError("step_id must be non-empty")
    if re.fullmatch(r"\d+", text):
        parsed = int(text)
        if parsed <= 0:
            raise ValueError("step_id must be positive")
        return parsed
    matches = re.findall(r"\d+", text)
    if matches:
        parsed = int(matches[-1])
        if parsed <= 0:
            raise ValueError("step_id must be positive")
        return parsed
    raise ValueError("step_id must contain an integer value")


class InputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    field_path: str

    @field_validator("step_id", mode="before")
    @classmethod
    def _coerce_step_id(cls, value):
        return _normalize_step_id(value)

    @field_validator("field_path")
    @classmethod
    def _validate_field_path(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("field_path must be non-empty")
        return value


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    tool_name: str
    purpose: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    input_refs: Dict[str, List[InputRef]] = Field(default_factory=dict)
    expected_outputs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_id", mode="before")
    @classmethod
    def _coerce_step_id(cls, value):
        return _normalize_step_id(value)

    @field_validator("tool_name", "purpose")
    @classmethod
    def _validate_text(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, value):
        if not isinstance(value, dict):
            return value
        legacy_fields = [field for field in ("arguments", "depends_on", "use_summary") if field in value]
        if legacy_fields:
            raise ValueError("PlanStep uses removed field(s): %s" % ", ".join(sorted(legacy_fields)))
        refs = value.get("input_refs")
        if isinstance(refs, list):
            raise ValueError("input_refs must be a field-keyed object, not a list")
        return value

    @field_validator("input_refs", mode="before")
    @classmethod
    def _normalize_input_refs(cls, value):
        if value is None:
            return {}
        if isinstance(value, list):
            raise ValueError("input_refs must be a field-keyed object, not a list")
        if not isinstance(value, dict):
            raise ValueError("input_refs must be a field-keyed object")
        normalized: Dict[str, List[Any]] = {}
        for key, refs in value.items():
            field_name = str(key or "").strip()
            if not field_name:
                raise ValueError("input_refs field names must be non-empty")
            if refs is None:
                continue
            normalized[field_name] = refs if isinstance(refs, list) else [refs]
        return normalized


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    steps: List[PlanStep] = Field(default_factory=list)
    refinement_instructions: str = ""

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, value):
        if isinstance(value, dict) and "use_summary" in value:
            raise ValueError("ExecutionPlan uses removed field: use_summary")
        return value

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("strategy must be non-empty")
        return value


_RETRIEVAL_TARGETS = {
    "artifact_context",
    "asr_transcripts",
    "dense_captions",
    "evidence",
    "existing_evidence",
    "mixed",
    "observations",
    "preprocess",
    "prior_trace",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    ordered: List[str] = []
    seen = set()
    for item in values:
        text = _normalize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("time range values must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("time range values must be numeric") from exc


class PlannerRetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    target: str = "mixed"
    need: str
    query: str = ""
    modalities: List[str] = Field(default_factory=list)
    time_range: Dict[str, float] = Field(default_factory=dict)
    source_tools: List[str] = Field(default_factory=list)
    evidence_status: str = ""
    artifact_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    observation_ids: List[str] = Field(default_factory=list)
    limit: int = 20

    @field_validator("request_id", "need", "query", "evidence_status")
    @classmethod
    def _validate_text(cls, value):  # noqa: N805
        return _normalize_text(value)

    @field_validator("request_id", "need")
    @classmethod
    def _require_text(cls, value):  # noqa: N805
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @field_validator("target")
    @classmethod
    def _validate_target(cls, value):  # noqa: N805
        normalized = _normalize_text(value).lower()
        if normalized not in _RETRIEVAL_TARGETS:
            raise ValueError("target must be one of: %s" % ", ".join(sorted(_RETRIEVAL_TARGETS)))
        return normalized

    @field_validator("modalities", "source_tools", "artifact_ids", "evidence_ids", "observation_ids", mode="before")
    @classmethod
    def _validate_string_lists(cls, value):
        return _normalize_string_list(value)

    @field_validator("time_range", mode="before")
    @classmethod
    def _validate_time_range(cls, value):
        if value in (None, "", [], {}):
            return {}
        if not isinstance(value, dict):
            raise ValueError("time_range must be an object")
        normalized: Dict[str, float] = {}
        for source_key, target_key in (
            ("start_s", "start_s"),
            ("end_s", "end_s"),
            ("start", "start_s"),
            ("end", "end_s"),
        ):
            if source_key not in value:
                continue
            parsed = _normalize_optional_float(value.get(source_key))
            if parsed is not None:
                normalized[target_key] = parsed
        if (
            normalized.get("start_s") is not None
            and normalized.get("end_s") is not None
            and normalized["start_s"] > normalized["end_s"]
        ):
            raise ValueError("time_range.start_s must be <= time_range.end_s")
        return normalized

    @field_validator("limit", mode="before")
    @classmethod
    def _validate_limit(cls, value):
        if value in (None, ""):
            return 20
        if isinstance(value, bool):
            raise ValueError("limit must be an integer")
        parsed = int(value)
        return max(1, min(50, parsed))


class PlannerRetrievalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = "ready"
    rationale: str = ""
    requests: List[PlannerRetrievalQuery] = Field(default_factory=list)

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value):  # noqa: N805
        normalized = _normalize_text(value).lower()
        if normalized not in {"ready", "retrieve"}:
            raise ValueError("action must be ready or retrieve")
        return normalized

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value):  # noqa: N805
        return _normalize_text(value)

    @model_validator(mode="after")
    def _validate_requests_for_action(self):
        if self.action == "ready":
            self.requests = []
            return self
        if not self.requests:
            raise ValueError("retrieve action requires at least one request")
        return self
