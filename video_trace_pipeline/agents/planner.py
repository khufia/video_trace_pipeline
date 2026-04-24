from __future__ import annotations

import re

from ..prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from ..schemas import ExecutionPlan


def _coerce_raw_step_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    matches = re.findall(r"-?\d+", text)
    if matches:
        return int(matches[-1])
    return None


def _repair_execution_plan_payload(payload):
    plan = dict(payload or {})
    repaired_steps = []
    existing_step_ids = set()

    for index, raw_step in enumerate(list(plan.get("steps") or []), start=1):
        step = dict(raw_step or {})
        raw_step_id = _coerce_raw_step_id(step.get("step_id"))
        if raw_step_id is None or raw_step_id <= 0 or raw_step_id in existing_step_ids:
            step_id = index
        else:
            step_id = raw_step_id
        existing_step_ids.add(step_id)
        step["step_id"] = step_id
        repaired_steps.append(step)

    def _repair_ref_id(value):
        parsed = _coerce_raw_step_id(value)
        if parsed is None:
            return None
        if parsed in existing_step_ids:
            return parsed
        shifted = parsed + 1
        if shifted in existing_step_ids:
            return shifted
        if parsed <= 0:
            return None
        return parsed

    for step in repaired_steps:
        repaired_refs = []
        for raw_binding in list(step.get("input_refs") or []):
            binding = dict(raw_binding or {})
            source = dict(binding.get("source") or {})
            repaired_step_id = _repair_ref_id(source.get("step_id"))
            if repaired_step_id is None:
                continue
            source["step_id"] = repaired_step_id
            binding["source"] = source
            repaired_refs.append(binding)
        step["input_refs"] = repaired_refs

        repaired_depends_on = []
        for raw_dep in list(step.get("depends_on") or []):
            repaired_dep = _repair_ref_id(raw_dep)
            if repaired_dep is None or repaired_dep == step["step_id"]:
                continue
            repaired_depends_on.append(repaired_dep)
        step["depends_on"] = repaired_depends_on

    plan["steps"] = repaired_steps
    return plan


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(self, task, mode, summary_text, compact_rounds, retrieved_observations, audit_feedback, tool_catalog):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            summary_text=summary_text,
            compact_rounds=compact_rounds,
            retrieved_observations=retrieved_observations,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
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
        parsed = ExecutionPlan.model_validate(_repair_execution_plan_payload(payload))
        return raw, parsed

    def plan(self, task, mode, summary_text, compact_rounds, retrieved_observations, audit_feedback, tool_catalog):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                summary_text=summary_text,
                compact_rounds=compact_rounds,
                retrieved_observations=retrieved_observations,
                audit_feedback=audit_feedback,
                tool_catalog=tool_catalog,
            )
        )
