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
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def _make_client(fake_completions):
    profile = MachineProfile(
        workspace_root="/tmp/workspace",
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models_config = ModelsConfig(agents={}, tools={})
    client = OpenAIChatClient(profile=profile, models_config=models_config)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    client._build_client = lambda endpoint_name: fake_client
    return client


def test_complete_json_requests_json_object_response_format():
    fake_completions = _FakeChatCompletions('{"strategy": "focus on OCR"}')
    client = _make_client(fake_completions)

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
