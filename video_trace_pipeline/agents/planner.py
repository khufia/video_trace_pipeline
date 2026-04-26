from __future__ import annotations

from typing import Any, Dict, List

from ..prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from ..schemas import ArgumentBinding, ExecutionPlan
from ..schemas.plans import _normalize_step_id


def _sanitize_input_refs(raw_step: Dict[str, Any]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for item in list(raw_step.get("input_refs") or []):
        if not isinstance(item, dict):
            continue
        try:
            binding = ArgumentBinding.model_validate(item)
        except Exception:
            continue
        sanitized.append(binding.model_dump())
    return sanitized


def _sanitize_depends_on(raw_step: Dict[str, Any]) -> List[int]:
    sanitized: List[int] = []
    for value in list(raw_step.get("depends_on") or []):
        try:
            sanitized.append(_normalize_step_id(value))
        except Exception:
            continue
    return sanitized


def _sanitize_execution_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    normalized_steps: List[Dict[str, Any]] = []
    for raw_step in list(normalized.get("steps") or []):
        if not isinstance(raw_step, dict):
            continue
        step_payload = dict(raw_step)
        if "input_refs" in step_payload:
            step_payload["input_refs"] = _sanitize_input_refs(step_payload)
        if "depends_on" in step_payload:
            step_payload["depends_on"] = _sanitize_depends_on(step_payload)
        normalized_steps.append(step_payload)
    normalized["steps"] = normalized_steps
    return normalized


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(
        self,
        task,
        mode,
        planner_segments,
        compact_rounds,
        retrieved_observations,
        audit_feedback,
        tool_catalog,
        preprocess_planning_memory=None,
    ):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            planner_segments=planner_segments,
            compact_rounds=compact_rounds,
            retrieved_observations=retrieved_observations,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
            preprocess_planning_memory=preprocess_planning_memory,
        )
        return {
            "endpoint_name": self.agent_config.endpoint or "default",
            "model_name": self.agent_config.model,
            "system_prompt": PLANNER_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }

    def complete_request(self, request):
        payload, raw = self.llm_client.complete_json(response_model=dict, **dict(request or {}))
        parsed = ExecutionPlan.model_validate(_sanitize_execution_plan_payload(payload))
        return raw, parsed

    def plan(
        self,
        task,
        mode,
        planner_segments,
        compact_rounds,
        retrieved_observations,
        audit_feedback,
        tool_catalog,
        preprocess_planning_memory=None,
    ):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                planner_segments=planner_segments,
                compact_rounds=compact_rounds,
                retrieved_observations=retrieved_observations,
                audit_feedback=audit_feedback,
                tool_catalog=tool_catalog,
                preprocess_planning_memory=preprocess_planning_memory,
            )
        )
