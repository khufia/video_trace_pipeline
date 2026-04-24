from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from filelock import FileLock
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..common import ensure_dir, extract_json_object, fingerprint_file, hash_payload, write_json, write_text
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
    CACHE_VERSION = "v1"

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

    def _cache_root(self) -> Path:
        cache_root_value = str(self.profile.cache_root or "").strip()
        if cache_root_value:
            base_root = Path(cache_root_value).expanduser().resolve()
        else:
            base_root = Path(self.profile.workspace_root).expanduser().resolve() / "cache"
        return ensure_dir(base_root / "agent_responses")

    def _image_cache_payload(self, image_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        payload = []
        for image_path in list(image_paths or []):
            entry: Dict[str, Any] = {"path": str(Path(image_path).expanduser())}
            try:
                entry["fingerprint"] = fingerprint_file(image_path)
            except Exception:
                entry["fingerprint"] = None
            payload.append(entry)
        return payload

    def _cache_key(self, namespace: str, payload: Dict[str, Any]) -> str:
        return hash_payload({"version": self.CACHE_VERSION, "namespace": namespace, "payload": dict(payload or {})}, length=40)

    def _cache_paths(self, namespace: str, cache_key: str):
        cache_dir = ensure_dir(self._cache_root() / str(namespace or "text") / cache_key[:2] / cache_key)
        return (
            cache_dir,
            cache_dir / "response.txt",
            cache_dir / "request.json",
            FileLock(str(cache_dir / ".lock")),
        )

    def _request_payload(
        self,
        *,
        endpoint_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        response_model: Optional[Type] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "image_paths": self._image_cache_payload(image_paths=image_paths),
            "response_format": dict(response_format or {}) if response_format is not None else None,
        }
        if response_model is not None:
            payload["response_model"] = "%s.%s" % (
                getattr(response_model, "__module__", "response_model"),
                getattr(response_model, "__name__", str(response_model)),
            )
        return payload

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request_text(
        self,
        endpoint_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._build_client(endpoint_name)
        request_kwargs: Dict[str, Any] = {}
        if response_format is not None:
            request_kwargs["response_format"] = dict(response_format)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_content(user_prompt, image_paths=image_paths)},
            ],
            **request_kwargs,
            **_limit_kwargs(model_name, max_tokens),
            **_temperature_kwargs(model_name, temperature),
        )
        choice = response.choices[0]
        message = choice.message
        text = str(getattr(message, "content", "") or "").strip()
        if not text:
            raise ValueError("Model returned empty text.")
        return text

    def complete_text(
        self,
        endpoint_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        image_paths: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        request_payload = self._request_payload(
            endpoint_name=endpoint_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_paths=image_paths,
            response_format=response_format,
        )
        cache_key = self._cache_key("text", request_payload)
        _, response_path, request_path, lock = self._cache_paths("text", cache_key)
        with lock:
            if response_path.exists():
                return response_path.read_text(encoding="utf-8").strip()
            text = self._request_text(
                endpoint_name=endpoint_name,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                image_paths=image_paths,
                response_format=response_format,
            )
            write_json(request_path, request_payload)
            write_text(response_path, text)
            return text

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
        response_format = {"type": "json_object"}
        request_payload = self._request_payload(
            endpoint_name=endpoint_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_paths=image_paths,
            response_format=response_format,
            response_model=response_model,
        )
        cache_key = self._cache_key("json", request_payload)
        _, response_path, request_path, lock = self._cache_paths("json", cache_key)
        with lock:
            text = response_path.read_text(encoding="utf-8").strip() if response_path.exists() else ""
            payload = extract_json_object(text) if text else None
            if payload is None:
                text = self._request_text(
                    endpoint_name=endpoint_name,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    image_paths=image_paths,
                    response_format=response_format,
                )
                payload = extract_json_object(text)
                if payload is None:
                    raise ValueError("Model did not return a JSON object: %s" % text[:1000])
                write_json(request_path, request_payload)
                write_text(response_path, text)
        if response_model is dict:
            return payload, text
        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(payload), text
        return response_model.parse_obj(payload), text
