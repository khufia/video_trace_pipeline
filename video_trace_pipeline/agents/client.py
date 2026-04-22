from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..common import extract_json_object
from ..config import resolve_api_key
from ..schemas import MachineProfile, ModelsConfig

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at import time
    OpenAI = None


def _limit_kwargs(model_name: str, max_tokens: int) -> Dict[str, int]:
    name = str(model_name or "").strip().lower()
    if name.startswith("gpt-5") or name.startswith(("o1", "o2", "o3", "o4")):
        return {"max_completion_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens)}


def _temperature_kwargs(model_name: str, temperature: float) -> Dict[str, float]:
    return {"temperature": float(temperature)}


class OpenAIChatClient(object):
    def __init__(self, profile: MachineProfile, models_config: ModelsConfig):
        self.profile = profile
        self.models_config = models_config

    def _build_client(self, endpoint_name: str):
        endpoint = self.profile.agent_endpoints.get(endpoint_name)
        if endpoint is None:
            raise KeyError("Unknown endpoint: %s" % endpoint_name)
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        api_key = resolve_api_key(self.profile, endpoint_name)
        if not api_key:
            raise RuntimeError("No API key configured for endpoint %s" % endpoint_name)
        return OpenAI(base_url=endpoint.base_url, api_key=api_key)

    def _build_content(self, text: str, image_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        content = [{"type": "text", "text": text}]
        for image_path in image_paths or []:
            path = Path(image_path)
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            suffix = path.suffix.lower().lstrip(".") or "png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/%s;base64,%s" % (suffix, encoded)},
                }
            )
        return content

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete_text(
        self,
        endpoint_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        client = self._build_client(endpoint_name)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_content(user_prompt, image_paths=image_paths)},
            ],
            **_limit_kwargs(model_name, max_tokens),
            **_temperature_kwargs(model_name, temperature),
        )
        choice = response.choices[0]
        message = choice.message
        return str(getattr(message, "content", "") or "").strip()

    def complete_json(
        self,
        endpoint_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        image_paths: Optional[List[str]] = None,
    ):
        text = self.complete_text(
            endpoint_name=endpoint_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_paths=image_paths,
        )
        payload = extract_json_object(text)
        if payload is None:
            raise ValueError("Model did not return a JSON object: %s" % text[:1000])
        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(payload), text
        return response_model.parse_obj(payload), text
