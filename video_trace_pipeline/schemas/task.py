from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from ..common import sanitize_for_persistence


class TaskSpec(BaseModel):
    benchmark: str
    sample_key: str
    question: str
    options: List[str] = Field(default_factory=list)
    video_path: str
    video_id: Optional[str] = None
    question_id: Optional[str] = None
    gold_answer: Optional[str] = None
    initial_trace_steps: Optional[List[str]] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    @validator("benchmark", "sample_key", "question", "video_path")
    def _validate_text(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    def persistable_dict(self) -> Dict[str, object]:
        payload = self.dict()
        payload.pop("video_path", None)
        return sanitize_for_persistence(payload)
