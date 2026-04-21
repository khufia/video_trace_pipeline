from __future__ import annotations

from ..prompts.templates import AUDITOR_SYSTEM_PROMPT, build_auditor_prompt
from ..schemas import AuditReport


class TraceAuditorAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def audit(self, task, trace_package, evidence_summary):
        prompt = build_auditor_prompt(task=task, trace_package=trace_package, evidence_summary=evidence_summary)
        parsed, raw = self.llm_client.complete_json(
            endpoint_name=self.agent_config.endpoint or "default",
            model_name=self.agent_config.model,
            system_prompt=AUDITOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=AuditReport,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        return raw, parsed
