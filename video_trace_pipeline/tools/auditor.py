from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from ..common import extract_json_object
from ..prompts.auditor_prompt import build_auditor_messages
from ..tool_io import ToolPayload, main


class Request(BaseModel):
    task: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    audit: dict[str, Any]


class Result(BaseModel):
    ok: bool
    tool: str = "auditor"
    output: Output
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _limit_kwargs(model_name: str, max_tokens: int) -> dict[str, int]:
    name = str(model_name or "").lower()
    if name.startswith("gpt-5") or name.startswith(("o1", "o2", "o3", "o4")):
        return {"max_completion_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens)}


def _call_openai_compatible(runtime: dict[str, Any], messages: dict[str, str]) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed") from exc
    api_key = runtime.get("api_key") or os.environ.get(str(runtime.get("api_key_env") or ""))
    if not api_key:
        raise RuntimeError("No API key configured for endpoint %s" % (runtime.get("endpoint") or "default"))
    client = OpenAI(base_url=runtime.get("base_url") or "https://api.openai.com/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=str(runtime.get("model") or runtime.get("model_name") or ""),
        messages=[{"role": "system", "content": messages["system"]}, {"role": "user", "content": messages["user"]}],
        response_format={"type": "json_object"},
        temperature=float(runtime.get("temperature") or 0.0),
        **_limit_kwargs(str(runtime.get("model") or ""), int(runtime.get("max_tokens") or 4096)),
    )
    return str(response.choices[0].message.content or "").strip()


def run(payload: ToolPayload, request: Request) -> Result:
    runtime = payload.runtime.model_dump(mode="json")
    task = request.task or payload.task
    context = dict(request.context or payload.context or {})
    raw = _call_openai_compatible(runtime, build_auditor_messages(task, context))
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Auditor model did not return a JSON object.")
    audit = dict(parsed.get("audit") or parsed)
    audit.setdefault("verdict", "FAIL")
    audit["verdict"] = str(audit.get("verdict") or "FAIL").upper()
    audit.setdefault("confidence", None)
    audit.setdefault("findings", [])
    audit.setdefault("missing_information", [])
    audit.setdefault("feedback", "")
    return Result(ok=True, output=Output(audit=audit), metadata={"raw_text": raw})


if __name__ == "__main__":
    main(run, request_model=Request)
