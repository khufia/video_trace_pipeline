from types import SimpleNamespace

from pydantic import BaseModel

from video_trace_pipeline.agents.client import OpenAIChatClient
from video_trace_pipeline.schemas import ApiEndpointConfig, MachineProfile, ModelsConfig


class _DummyResponseModel(BaseModel):
    strategy: str


class _FakeChatCompletions(object):
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.content[len(self.calls) - 1] if isinstance(self.content, list) else self.content
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def _make_client(fake_completions, *, workspace_root="/tmp/workspace", cache_root=None):
    profile = MachineProfile(
        workspace_root=str(workspace_root),
        cache_root=str(cache_root) if cache_root is not None else None,
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models_config = ModelsConfig(agents={}, tools={})
    client = OpenAIChatClient(profile=profile, models_config=models_config)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    client._build_client = lambda endpoint_name: fake_client
    return client


def test_complete_json_requests_json_object_response_format(tmp_path):
    fake_completions = _FakeChatCompletions('{"strategy": "focus on OCR"}')
    client = _make_client(
        fake_completions,
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
    )

    parsed, raw = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return JSON only.",
        user_prompt="Plan this task.",
        response_model=_DummyResponseModel,
    )

    assert parsed.strategy == "focus on OCR"
    assert raw == '{"strategy": "focus on OCR"}'
    assert fake_completions.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_does_not_persist_or_reuse_agent_response_cache(tmp_path):
    fake_completions = _FakeChatCompletions('{"strategy": "focus on OCR"}')
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        cache_root=str(tmp_path / "cache"),
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models_config = ModelsConfig(agents={}, tools={})
    client = OpenAIChatClient(profile=profile, models_config=models_config)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    client._build_client = lambda endpoint_name: fake_client

    first, first_raw = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return JSON only.",
        user_prompt="Plan this task.",
        response_model=_DummyResponseModel,
    )
    second, second_raw = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return JSON only.",
        user_prompt="Plan this task.",
        response_model=_DummyResponseModel,
    )

    assert first.strategy == "focus on OCR"
    assert second.strategy == "focus on OCR"
    assert first_raw == second_raw == '{"strategy": "focus on OCR"}'
    assert len(fake_completions.calls) == 2


def test_complete_json_retries_once_with_larger_budget_for_truncated_json(tmp_path):
    fake_completions = _FakeChatCompletions(
        ['{"strategy": "focus on OCR"', '{"strategy": "focus on OCR"}']
    )
    client = _make_client(
        fake_completions,
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
    )

    parsed, raw = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return JSON only.",
        user_prompt="Plan this task.",
        response_model=_DummyResponseModel,
        max_tokens=8000,
    )

    assert parsed.strategy == "focus on OCR"
    assert raw == '{"strategy": "focus on OCR"}'
    assert len(fake_completions.calls) == 2
    assert fake_completions.calls[0]["max_completion_tokens"] == 8000
    assert fake_completions.calls[1]["max_completion_tokens"] == 16000
