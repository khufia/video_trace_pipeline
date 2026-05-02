from __future__ import annotations

from ..prompts.trace_auditor_prompt import AUDITOR_SYSTEM_PROMPT, build_auditor_prompt
from ..schemas import AuditReport


class TraceAuditorAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(self, task, trace_package, evidence_summary):
        prompt = build_auditor_prompt(
            task=task,
            trace_package=trace_package,
            evidence_summary=evidence_summary,
        )
        return {
            "endpoint_name": self.agent_config.endpoint or "default",
            "model_name": self.agent_config.model,
            "system_prompt": AUDITOR_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }

    def complete_request(self, request):
        parsed, raw = self.llm_client.complete_json(
            response_model=AuditReport,
            **dict(request or {})
        )
        return raw, parsed

    def audit(self, task, trace_package, evidence_summary):
        return self.complete_request(
            self.build_request(
                task=task,
                trace_package=trace_package,
                evidence_summary=evidence_summary,
            )
        )
