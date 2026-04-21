from __future__ import annotations

from ..prompts.templates import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from ..schemas import ExecutionPlan


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def plan(self, task, mode, summary_text, compact_rounds, retrieved_observations, audit_feedback, tool_names):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            summary_text=summary_text,
            compact_rounds=compact_rounds,
            retrieved_observations=retrieved_observations,
            audit_feedback=audit_feedback,
            tool_names=tool_names,
        )
        parsed, raw = self.llm_client.complete_json(
            endpoint_name=self.agent_config.endpoint or "default",
            model_name=self.agent_config.model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=ExecutionPlan,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        return raw, parsed
