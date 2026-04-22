from __future__ import annotations

from ..prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from ..schemas import ExecutionPlan


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
        parsed, raw = self.llm_client.complete_json(response_model=ExecutionPlan, **dict(request or {}))
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
