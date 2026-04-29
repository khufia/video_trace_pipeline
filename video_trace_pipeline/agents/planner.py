from __future__ import annotations

from ..prompts.planner_prompt import (
    PLANNER_RETRIEVAL_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
    build_planner_retrieval_prompt,
)
from ..schemas import ExecutionPlan, PlannerRetrievalDecision


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(
        self,
        task,
        mode,
        planner_segments,
        retrieved_context,
        audit_feedback,
        tool_catalog,
        retrieval_catalog=None,
        task_state=None,
    ):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            planner_segments=planner_segments,
            retrieved_context=retrieved_context,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
            retrieval_catalog=retrieval_catalog,
            task_state=task_state,
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
        parsed = ExecutionPlan.model_validate(payload)
        return raw, parsed

    def build_retrieval_request(
        self,
        task,
        mode,
        retrieval_catalog,
        retrieved_context,
        audit_feedback,
        tool_catalog,
        iteration,
        max_iterations,
        task_state=None,
    ):
        prompt = build_planner_retrieval_prompt(
            task=task,
            mode=mode,
            retrieval_catalog=retrieval_catalog,
            retrieved_context=retrieved_context,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
            iteration=iteration,
            max_iterations=max_iterations,
            task_state=task_state,
        )
        return {
            "endpoint_name": self.agent_config.endpoint or "default",
            "model_name": self.agent_config.model,
            "system_prompt": PLANNER_RETRIEVAL_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }

    def complete_retrieval_request(self, request):
        payload, raw = self.llm_client.complete_json(response_model=dict, **dict(request or {}))
        parsed = PlannerRetrievalDecision.model_validate(payload)
        return raw, parsed

    def plan(
        self,
        task,
        mode,
        planner_segments,
        retrieved_context,
        audit_feedback,
        tool_catalog,
        task_state=None,
    ):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                planner_segments=planner_segments,
                retrieved_context=retrieved_context,
                audit_feedback=audit_feedback,
                tool_catalog=tool_catalog,
                task_state=task_state,
            )
        )
