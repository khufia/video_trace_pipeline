from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

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


class PlannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["tool_call", "synthesize", "stop_unresolved"]
    rationale: str
    tool_name: Optional[str] = None
    tool_request: Dict[str, Any] = Field(default_factory=dict)
    expected_observation: str = ""
    synthesis_instructions: str = ""
    missing_information: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_common_action_envelopes(cls, value):
        if not isinstance(value, dict) or value.get("action_type"):
            return value

        for key in ("planner_action", "next_action", "action"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested

        if "synthesis_instructions" in value:
            instructions = str(value.get("synthesis_instructions") or "").strip()
            if instructions:
                return {
                    "action_type": "synthesize",
                    "rationale": str(value.get("rationale") or instructions).strip(),
                    "synthesis_instructions": instructions,
                }

        missing_information = value.get("missing_information")
        if missing_information:
            return {
                "action_type": "stop_unresolved",
                "rationale": str(value.get("rationale") or "The remaining answer-critical information is unresolved.").strip(),
                "missing_information": missing_information,
            }

        tool_request = value.get("tool_request") if isinstance(value.get("tool_request"), dict) else None
        tool_name = str(value.get("tool_name") or (tool_request or {}).get("tool_name") or "").strip()
        if not tool_name and tool_request is None:
            keys = {str(key) for key in value.keys()}
            if {"video_id", "start_s", "end_s"}.issubset(keys):
                tool_name = "frame_retriever"
            elif keys & {"ocr_sample_fps", "ocr_source"} and keys & {"clips", "frames", "regions"}:
                tool_name = "ocr"
            elif keys & {"sequence_mode", "neighbor_radius_s", "include_anchor_neighbors", "num_frames", "time_hints"}:
                tool_name = "frame_retriever"
            elif keys & {"speaker_attribution"} and "clips" in keys:
                tool_name = "asr"
            elif keys & {"granularity", "focus_query"} and "clips" in keys:
                tool_name = "dense_captioner"
            elif keys & {"text_contexts", "transcripts", "evidence_ids"}:
                tool_name = "generic_purpose"
            elif "query" in keys and "top_k" in keys:
                tool_name = "visual_temporal_grounder"

        if tool_name:
            request_fields = {
                key: item
                for key, item in value.items()
                if key
                not in {
                    "action_type",
                    "rationale",
                    "tool_name",
                    "tool_request",
                    "expected_observation",
                    "synthesis_instructions",
                    "missing_information",
                }
            }
            if tool_request is None and {"video_id", "start_s", "end_s"}.issubset(set(request_fields.keys())):
                clip = {
                    key: request_fields[key]
                    for key in ("video_id", "start_s", "end_s", "artifact_id", "relpath", "metadata")
                    if key in request_fields
                }
                normalized_request = {
                    "clips": [clip],
                    "query": str(value.get("query") or "clear answer-critical visual frames inside the supplied time window").strip(),
                    "sequence_mode": "chronological",
                    "sort_order": "chronological",
                }
            else:
                normalized_request = dict(tool_request or request_fields)
            normalized_request.setdefault("tool_name", tool_name)
            return {
                "action_type": "tool_call",
                "rationale": str(value.get("rationale") or "Run %s to collect the next answer-critical evidence." % tool_name).strip(),
                "tool_name": tool_name,
                "tool_request": normalized_request,
                "expected_observation": str(value.get("expected_observation") or "Answer-critical evidence returned by %s." % tool_name).strip(),
            }

        return value

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("rationale must be non-empty")
        return value

    @field_validator("tool_name", mode="before")
    @classmethod
    def _normalize_optional_tool_name(cls, value):
        text = str(value or "").strip()
        return text or None

    @field_validator("expected_observation", "synthesis_instructions", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value):
        return str(value or "").strip()

    @field_validator("missing_information", mode="before")
    @classmethod
    def _normalize_missing_information(cls, value):
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized = []
        seen = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    @model_validator(mode="after")
    def _validate_action_contract(self):
        if self.action_type == "tool_call":
            if not self.tool_name:
                raise ValueError("tool_call actions require tool_name")
            if self.tool_name == "verifier":
                raise ValueError("planner must not call the verifier tool")
            if not isinstance(self.tool_request, dict) or not self.tool_request:
                raise ValueError("tool_call actions require a non-empty tool_request")
        else:
            self.tool_name = None
            self.tool_request = {}
        if self.action_type == "synthesize" and not self.synthesis_instructions:
            self.synthesis_instructions = self.rationale
        return self
