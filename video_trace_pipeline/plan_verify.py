from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .config import CONTROL_TOOLS

TOOL_OUTPUTS = {
    "dense_captioner": ["output.captions", "output.artifacts"],
    "asr": ["output.transcript_segments"],
    "visual_temporal_grounder": ["output.segments", "output.summary"],
    "frame_retriever": ["output.frames"],
    "ocr": ["output.text", "output.lines", "output.reads"],
    "audio_temporal_grounder": ["output.segments", "output.summary"],
    "spatial_grounder": ["output.regions", "output.spatial_description"],
    "multimodal_reasoner": ["output.answer", "output.reasoning", "output.evidence", "output.confidence"],
}


def _runtime_output_path(path: str | None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    return "output.%s" % text


class RequestRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_step: str
    output: str

    @property
    def resolved_path(self) -> str:
        return _runtime_output_path(self.output)

    @field_validator("from_step", mode="before")
    @classmethod
    def normalize_from_step(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("from_step must be non-empty")
        return text

    @field_validator("output", mode="before")
    @classmethod
    def normalize_output(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("output must be non-empty")
        if text.startswith("output.") or text.startswith("result.output."):
            raise ValueError("output must be a field name like frames, not output.frames")
        return text


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    purpose: str
    request: dict[str, Any]
    request_refs: dict[str, list[RequestRef]]

    @field_validator("id", "tool", "purpose", mode="before")
    @classmethod
    def non_empty_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    steps: list[PlanStep]

    @field_validator("strategy", mode="before")
    @classmethod
    def strategy_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("strategy must be non-empty")
        return text


def _record_step_id(record: dict[str, Any]) -> str:
    step = dict(record.get("step") or {})
    return str(step.get("id") or "").strip()


def _record_tool(record: dict[str, Any]) -> str:
    step = dict(record.get("step") or {})
    return str(step.get("tool") or "").strip()


def normalize_plan_payload(plan_payload: dict[str, Any], previous_steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = dict(plan_payload.get("plan") or plan_payload) if isinstance(plan_payload, dict) else {}
    _ = previous_steps
    normalized = dict(payload)
    normalized_steps = []
    for raw_step in list(payload.get("steps") or []):
        if hasattr(raw_step, "model_dump"):
            raw_step = raw_step.model_dump(mode="json")
        step = dict(raw_step or {})
        if "id" in step:
            step["id"] = str(step["id"] or "").strip()
        if "tool" in step:
            step["tool"] = str(step["tool"] or "").strip()
        if "purpose" in step:
            step["purpose"] = str(step["purpose"] or "").strip()
        normalized_steps.append(step)
    if "strategy" in normalized:
        normalized["strategy"] = str(normalized["strategy"] or "").strip()
    if "steps" in normalized:
        normalized["steps"] = normalized_steps
    return normalized


def verify_plan(plan_payload: dict[str, Any], enabled_tools: dict[str, Any], previous_steps: list[dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    plan_payload = normalize_plan_payload(plan_payload, previous_steps)
    try:
        plan = Plan.model_validate(plan_payload)
    except Exception as exc:
        return ["plan schema validation failed: %s" % exc]

    seen_ids: set[str] = set()
    current_tools_by_id: dict[str, str] = {}
    previous_tools_by_id = {
        _record_step_id(record): _record_tool(record)
        for record in list(previous_steps or [])
        if _record_step_id(record) and _record_tool(record)
    }

    for step in plan.steps:
        if step.id in seen_ids:
            errors.append("duplicate step id: %s" % step.id)
        seen_ids.add(step.id)
        if step.tool in CONTROL_TOOLS:
            errors.append("control tool cannot be scheduled by planner: %s" % step.tool)
        if step.tool not in enabled_tools:
            errors.append("tool is not enabled or does not exist: %s" % step.tool)
        for target_field, refs in step.request_refs.items():
            if not target_field:
                errors.append("request_refs target field is empty in step %s" % step.id)
            for ref in refs:
                source_tool = current_tools_by_id.get(ref.from_step) or previous_tools_by_id.get(ref.from_step)
                if not source_tool:
                    errors.append("step %s references unknown or future step %s" % (step.id, ref.from_step))
                    continue
                path = ref.resolved_path
                allowed = TOOL_OUTPUTS.get(source_tool, [])
                if path not in allowed:
                    errors.append(
                        "step %s references %s from %s, but %s emits only %s"
                        % (step.id, path or "<empty>", ref.from_step, source_tool, ", ".join(allowed))
                    )
        current_tools_by_id[step.id] = step.tool
    return errors
