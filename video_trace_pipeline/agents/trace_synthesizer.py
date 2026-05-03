from __future__ import annotations

from ..prompts.trace_synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_prompt
from ..schemas import TracePackage


class TraceSynthesizerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def build_request(
        self,
        task,
        mode,
        round_evidence_entries,
        round_observations,
        current_trace,
        refinement_instructions,
        audit_feedback=None,
        preprocess_context=None,
    ):
        prompt = build_synthesizer_prompt(
            task=task,
            mode=mode,
            round_evidence_entries=round_evidence_entries,
            round_observations=round_observations,
            current_trace=current_trace,
            refinement_instructions=refinement_instructions,
            audit_feedback=audit_feedback,
            preprocess_context=preprocess_context,
        )
        return {
            "endpoint_name": self.agent_config.endpoint or "default",
            "model_name": self.agent_config.model,
            "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
        }

    def complete_request(self, request):
        parsed, raw = self.llm_client.complete_json(
            response_model=TracePackage,
            **dict(request or {})
        )
        return raw, parsed

    def synthesize(
        self,
        task,
        mode,
        round_evidence_entries,
        round_observations,
        current_trace,
        refinement_instructions,
        audit_feedback=None,
        preprocess_context=None,
    ):
        return self.complete_request(
            self.build_request(
                task=task,
                mode=mode,
                round_evidence_entries=round_evidence_entries,
                round_observations=round_observations,
                current_trace=current_trace,
                refinement_instructions=refinement_instructions,
                audit_feedback=audit_feedback,
                preprocess_context=preprocess_context,
            )
        )
