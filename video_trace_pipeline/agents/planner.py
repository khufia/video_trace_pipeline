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
    raw_step_id_map = {}
    raw_step_id_counts = {}

    for index, raw_step in enumerate(list(plan.get("steps") or []), start=1):
        step = dict(raw_step or {})
        raw_step_id = _coerce_raw_step_id(step.get("step_id"))
        if raw_step_id is not None:
            raw_step_id_counts[raw_step_id] = raw_step_id_counts.get(raw_step_id, 0) + 1
        if raw_step_id is None or raw_step_id <= 0 or raw_step_id in existing_step_ids:
            step_id = index
        else:
            step_id = raw_step_id
        existing_step_ids.add(step_id)
        if raw_step_id is not None and raw_step_id not in raw_step_id_map:
            raw_step_id_map[raw_step_id] = step_id
        step["step_id"] = step_id
        repaired_steps.append(step)

    def _repair_ref_id(value, *, owner_step_id, ref_kind):
        parsed = _coerce_raw_step_id(value)
        if parsed is None:
            raise ValueError(
                "Planner returned invalid %s step_id %r for step %s; references must target steps in the same plan."
                % (ref_kind, value, owner_step_id)
            )
        if raw_step_id_counts.get(parsed, 0) > 1:
            raise ValueError(
                "Planner returned ambiguous %s step_id %s for step %s because multiple plan steps used that raw id."
                % (ref_kind, parsed, owner_step_id)
            )
        repaired = raw_step_id_map.get(parsed)
        if repaired is None:
            available_ids = sorted(raw_step_id_map)
            raise ValueError(
                "Planner returned invalid %s step_id %s for step %s; references must target steps in the same plan. "
                "Available raw step ids: %s. Do not use 0, retrieved_observations, previous rounds, or other pseudo-sources."
                % (ref_kind, parsed, owner_step_id, available_ids)
            )
        return repaired

    for step in repaired_steps:
        repaired_refs = []
        for raw_binding in list(step.get("input_refs") or []):
            binding = dict(raw_binding or {})
            source = dict(binding.get("source") or {})
            repaired_step_id = _repair_ref_id(
                source.get("step_id"),
                owner_step_id=step["step_id"],
                ref_kind="input_ref",
            )
            if repaired_step_id == step["step_id"]:
                raise ValueError(
                    "Planner returned self-referential input_ref for step %s; input_refs must point to earlier steps."
                    % step["step_id"]
                )
            source["step_id"] = repaired_step_id
            binding["source"] = source
            repaired_refs.append(binding)
        step["input_refs"] = repaired_refs

        repaired_depends_on = []
        for raw_dep in list(step.get("depends_on") or []):
            repaired_dep = _repair_ref_id(
                raw_dep,
                owner_step_id=step["step_id"],
                ref_kind="depends_on",
            )
            if repaired_dep == step["step_id"]:
                raise ValueError(
                    "Planner returned self-referential depends_on for step %s; dependencies must point to earlier steps."
                    % step["step_id"]
                )
            repaired_depends_on.append(repaired_dep)
        step["depends_on"] = repaired_depends_on

    plan["steps"] = repaired_steps
    return plan


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(
        self,
        task,
        mode,
        summary_text,
        compact_rounds,
        retrieved_observations,
        audit_feedback,
        tool_catalog,
        current_trace_cues=None,
        preprocess_planning_memory=None,
    ):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            summary_text=summary_text,
            compact_rounds=compact_rounds,
            retrieved_observations=retrieved_observations,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
            current_trace_cues=current_trace_cues,
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
        parsed = ExecutionPlan.model_validate(_repair_execution_plan_payload(payload))
        return raw, parsed

    def plan(
        self,
        task,
        mode,
        summary_text,
        compact_rounds,
        retrieved_observations,
        audit_feedback,
        tool_catalog,
        current_trace_cues=None,
        preprocess_planning_memory=None,
    ):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                summary_text=summary_text,
                compact_rounds=compact_rounds,
                retrieved_observations=retrieved_observations,
                audit_feedback=audit_feedback,
                tool_catalog=tool_catalog,
                current_trace_cues=current_trace_cues,
                preprocess_planning_memory=preprocess_planning_memory,
            )
        )
