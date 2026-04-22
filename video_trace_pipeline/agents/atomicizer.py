from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ..prompts.atomicizer_prompt import ATOMICIZER_SYSTEM_PROMPT


class AtomicFact(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object_text: str = ""
    object_type: str = "text"
    atomic_text: str


class AtomicFactResponse(BaseModel):
    facts: List[AtomicFact] = Field(default_factory=list)


class AtomicFactAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def atomicize(self, source_text: str, context_hint: str = "") -> List[dict]:
        text = str(source_text or "").strip()
        if not text:
            return []
        prompt = "\n".join(
            [
                "CONTEXT_HINT:",
                context_hint or "none",
                "",
                "SOURCE_TEXT:",
                text,
            ]
        )
        parsed, _ = self.llm_client.complete_json(
            endpoint_name=self.agent_config.endpoint or "default",
            model_name=self.agent_config.model,
            system_prompt=ATOMICIZER_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=AtomicFactResponse,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        return [item.model_dump() for item in parsed.facts]
