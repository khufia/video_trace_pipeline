from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..common import extract_json_objects
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

    def _local_qwen_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        local_system_prompt = "\n\n".join(
            item
            for item in [
                str(system_prompt or "").strip(),
                (
                    "LOCAL_QWEN_OUTPUT_POLICY:\n"
                    "- Return one complete final answer object or text response for the requested schema.\n"
                    "- Do not use ellipses, placeholders, omitted fields, or markdown fences.\n"
                    "- If JSON is requested, every required field must contain concrete content."
                ),
            ]
            if item
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": local_system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": str(user_prompt or "").strip()}],
            },
        ]

    def _local_qwen_text(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        agent_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        from ..tool_wrappers.local_multimodal import QwenStyleRunner
        from ..tool_wrappers.shared import resolve_generation_controls, resolve_model_path, resolved_device_label

        extra = dict(agent_extra or {})
        runtime = {
            "model_name": model_name,
            "workspace_root": self.profile.workspace_root,
            "hf_cache": self.profile.hf_cache,
            "device": extra.get("device") or self.profile.gpu_assignments.get("control_agent") or "cuda:0",
            "extra": {
                **extra,
                "temperature": extra.get("temperature", temperature),
                "do_sample": extra.get("do_sample", bool(float(temperature or 0.0) > 0.0)),
            },
        }
        model_path = resolve_model_path(str(model_name or ""), runtime)
        generation = resolve_generation_controls(runtime)
        runner = QwenStyleRunner(
            model_path=model_path,
            device_label=resolved_device_label(runtime),
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=str(extra.get("attn_implementation") or "").strip() or None,
            device_map=str(extra.get("device_map") or "").strip() or None,
            enable_thinking=None if extra.get("enable_thinking") is None else bool(extra.get("enable_thinking")),
        )
        try:
            max_new_tokens = int(extra.get("max_new_tokens") or max_tokens or 4096)
            text = runner.generate(
                self._local_qwen_messages(system_prompt, user_prompt),
                max_new_tokens=max_new_tokens,
            )
        finally:
            runner.close()
        text = str(text or "").strip()
        if not text:
            raise ValueError("Local Qwen model returned empty text.")
        return text

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

    def _image_cache_payload(self, image_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        payload = []
        for image_path in list(image_paths or []):
            entry: Dict[str, Any] = {"path": str(Path(image_path).expanduser())}
            try:
                entry["fingerprint"] = Path(image_path).expanduser().resolve().stat().st_mtime_ns
            except Exception:
                entry["fingerprint"] = None
            payload.append(entry)
        return payload

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
        backend: str = "openai",
        agent_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend": backend,
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "image_paths": self._image_cache_payload(image_paths=image_paths),
            "response_format": dict(response_format or {}) if response_format is not None else None,
            "agent_extra": dict(agent_extra or {}),
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
        backend: str = "openai",
        agent_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_backend = str(backend or "openai").strip().lower()
        if normalized_backend in {"local_qwen", "qwen_local", "qwen"}:
            if image_paths:
                raise ValueError("Local Qwen control-agent backend does not accept image_paths.")
            return self._local_qwen_text(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_extra=agent_extra,
            )
        if normalized_backend not in {"openai", "openai_chat"}:
            raise ValueError("Unsupported agent backend: %s" % backend)
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
        backend: str = "openai",
        agent_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._request_text(
            endpoint_name=endpoint_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_paths=image_paths,
            response_format=response_format,
            backend=backend,
            agent_extra=agent_extra,
        )

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
        backend: str = "openai",
        agent_extra: Optional[Dict[str, Any]] = None,
    ):
        response_format = {"type": "json_object"}

        def _schema_name() -> str:
            return getattr(response_model, "__name__", str(response_model))

        def _truncate_for_repair(value: object, limit: int = 4000) -> str:
            text_value = str(value or "").strip()
            if len(text_value) <= limit:
                return text_value
            return "%s\n...[truncated %d chars]" % (text_value[:limit], len(text_value) - limit)

        def _validate_candidates(payload_candidates):
            validation_error = None
            if response_model is dict:
                if payload_candidates:
                    return payload_candidates[0], None
                return None, None
            if getattr(response_model, "__name__", "") == "PlannerAction":
                payload_candidates = sorted(
                    list(payload_candidates or []),
                    key=lambda item: 0 if isinstance(item, dict) and item.get("action_type") else 1,
                )
            for candidate_payload in payload_candidates:
                try:
                    if hasattr(response_model, "model_validate"):
                        parsed_payload = response_model.model_validate(candidate_payload)
                    else:
                        parsed_payload = response_model.parse_obj(candidate_payload)
                except Exception as exc:
                    validation_error = exc
                    continue
                return parsed_payload, None
            return None, validation_error

        def _repair_invalid_payload(raw_text: str, validation_error: Exception):
            if response_model is dict:
                return None, None
            repair_system_prompt = "\n".join(
                [
                    str(system_prompt or "").strip(),
                    "",
                    "The previous JSON response failed schema validation.",
                    "Repair it into one valid JSON object for the requested schema.",
                    "Do not explain the repair.",
                ]
            ).strip()
            repair_user_prompt = "\n\n".join(
                [
                    str(user_prompt or "").strip(),
                    "SCHEMA_REPAIR_REQUEST:",
                    "The previous assistant response was JSON, but it did not validate as %s." % _schema_name(),
                    "Validation error:\n%s" % _truncate_for_repair(validation_error),
                    "Previous assistant response:\n%s" % _truncate_for_repair(raw_text),
                    "Return exactly one JSON object that validates as %s. Do not return a bare nested object or tool arguments by themselves." % _schema_name(),
                ]
            ).strip()
            repair_text = self._request_text(
                endpoint_name=endpoint_name,
                model_name=model_name,
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
                temperature=0.0,
                max_tokens=max(int(max_tokens), 4096),
                image_paths=image_paths,
                response_format=response_format,
                backend=backend,
                agent_extra=agent_extra,
            )
            repaired, repair_validation_error = _validate_candidates(extract_json_objects(repair_text))
            if repaired is not None:
                return repaired, repair_text
            if repair_validation_error is not None:
                raise repair_validation_error
            raise ValueError("Model repair JSON did not validate as %s: %s" % (_schema_name(), repair_text[:1000]))

        candidate_budgets = [int(max_tokens)]
        expanded_budget = max(int(max_tokens) + 4000, int(max_tokens) * 2)
        if expanded_budget > candidate_budgets[-1]:
            candidate_budgets.append(expanded_budget)

        text = ""
        payload = None
        last_validation_error = None
        for index, budget in enumerate(candidate_budgets):
            text = self._request_text(
                endpoint_name=endpoint_name,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=budget,
                image_paths=image_paths,
                response_format=response_format,
                backend=backend,
                agent_extra=agent_extra,
            )
            payload_candidates = extract_json_objects(text)
            payload = payload_candidates[0] if payload_candidates else None
            parsed, validation_error = _validate_candidates(payload_candidates)
            if parsed is not None:
                return parsed, text
            if validation_error is not None:
                last_validation_error = validation_error
                return _repair_invalid_payload(text, validation_error)
            if index + 1 >= len(candidate_budgets):
                break
            if not payload_candidates and not str(text or "").lstrip().startswith("{"):
                break
        if payload is None:
            raise ValueError("Model did not return a JSON object: %s" % text[:1000])
        if last_validation_error is not None:
            raise last_validation_error
        raise ValueError("Model JSON did not validate as %s: %s" % (_schema_name(), text[:1000]))
