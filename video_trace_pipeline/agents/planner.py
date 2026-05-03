from __future__ import annotations

from ..prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from ..schemas import PlannerAction


class PlannerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(
        self,
        task,
        mode,
        audit_feedback,
        tool_catalog,
        evidence_summary=None,
        preprocess_context=None,
        action_history=None,
        current_trace=None,
    ):
        prompt = build_planner_prompt(
            task=task,
            mode=mode,
            audit_feedback=audit_feedback,
            tool_catalog=tool_catalog,
            evidence_summary=evidence_summary,
            preprocess_context=preprocess_context,
            action_history=action_history,
            current_trace=current_trace,
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
        parsed = PlannerAction.model_validate(payload)
        return raw, parsed

    def plan(
        self,
        task,
        mode,
        audit_feedback,
        tool_catalog,
        evidence_summary=None,
        preprocess_context=None,
        action_history=None,
        current_trace=None,
    ):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                audit_feedback=audit_feedback,
                tool_catalog=tool_catalog,
                evidence_summary=evidence_summary,
                preprocess_context=preprocess_context,
                action_history=action_history,
                current_trace=current_trace,
            )
        )
