from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


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


class InputRef(BaseModel):
    step_id: int
    field_path: str

    @validator("step_id", pre=True)
    def _coerce_step_id(cls, value):
        return _normalize_step_id(value)

    @validator("field_path")
    def _validate_field_path(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("field_path must be non-empty")
        return value


class ArgumentBinding(BaseModel):
    target_field: str
    source: InputRef

    @validator("target_field")
    def _validate_target_field(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("target_field must be non-empty")
        return value


class PlanStep(BaseModel):
    step_id: int
    tool_name: str
    purpose: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    input_refs: List[ArgumentBinding] = Field(default_factory=list)
    depends_on: List[int] = Field(default_factory=list)

    @validator("step_id", pre=True)
    def _coerce_step_id(cls, value):
        return _normalize_step_id(value)

    @validator("depends_on", pre=True)
    def _coerce_depends_on(cls, value):
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [_normalize_step_id(item) for item in items]

    @validator("tool_name", "purpose")
    def _validate_text(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value


class ExecutionPlan(BaseModel):
    strategy: str
    use_summary: bool = True
    steps: List[PlanStep] = Field(default_factory=list)
    refinement_instructions: str = ""

    @validator("strategy")
    def _validate_strategy(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("strategy must be non-empty")
        return value
