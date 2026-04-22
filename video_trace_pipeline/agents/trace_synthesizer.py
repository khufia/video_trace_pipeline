from __future__ import annotations

from ..prompts.trace_synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_prompt
from ..schemas import TracePackage


class TraceSynthesizerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def synthesize(self, task, mode, evidence_entries, observations, current_trace, refinement_instructions):
        prompt = build_synthesizer_prompt(
            task=task,
            mode=mode,
            evidence_entries=evidence_entries,
            observations=observations,
            current_trace=current_trace,
            refinement_instructions=refinement_instructions,
        )
        parsed, raw = self.llm_client.complete_json(
            endpoint_name=self.agent_config.endpoint or "default",
            model_name=self.agent_config.model,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=TracePackage,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        return raw, parsed
