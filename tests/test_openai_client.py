from types import SimpleNamespace

from pydantic import BaseModel

from video_trace_pipeline.agents.client import OpenAIChatClient
from video_trace_pipeline.tool_wrappers import local_multimodal
from video_trace_pipeline.tool_wrappers import shared as wrapper_shared
from video_trace_pipeline.schemas import ApiEndpointConfig, MachineProfile, ModelsConfig, PlannerAction, TracePackage


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


def test_complete_json_chooses_schema_valid_object_over_nested_time_interval(tmp_path):
    fake_completions = _FakeChatCompletions(
        """
        Here is the final trace JSON:
        {
          "task_key": "sample1",
          "mode": "generate",
          "evidence_entries": [],
          "inference_steps": [
            {
              "step_id": 1,
              "text": "The visual evidence supports option A.",
              "supporting_observation_ids": [],
              "answer_relevance": "high",
              "time_intervals": [{"start_s": 149.0, "end_s": 149.0}]
            }
          ],
          "final_answer": "A",
          "benchmark_renderings": {},
          "metadata": {}
        }
        """
    )
    client = _make_client(
        fake_completions,
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
    )

    parsed, _ = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return JSON only.",
        user_prompt="Synthesize this trace.",
        response_model=TracePackage,
    )

    assert parsed.task_key == "sample1"
    assert parsed.inference_steps[0].time_intervals[0].start_s == 149.0


def test_complete_json_chooses_planner_action_over_nested_clip_object(tmp_path):
    fake_completions = _FakeChatCompletions(
        """
        Candidate clip:
        {"video_id": "video_13", "start_s": 120.0, "end_s": 150.0, "artifact_id": "clip_120_00_150_00", "relpath": "artifacts/video_13/clips/clip_120_00_150_00.mp4"}

        Final action:
        {
          "action_type": "tool_call",
          "rationale": "Inspect the grounded clip for the answer-critical visual state.",
          "tool_name": "generic_purpose",
          "tool_request": {
            "tool_name": "generic_purpose",
            "query": "Determine the answer-critical visual state from the supplied clip.",
            "clips": [
              {"video_id": "video_13", "start_s": 120.0, "end_s": 150.0, "artifact_id": "clip_120_00_150_00", "relpath": "artifacts/video_13/clips/clip_120_00_150_00.mp4"}
            ]
          },
          "expected_observation": "Answer-critical visual state in the grounded clip."
        }
        """
    )
    client = _make_client(
        fake_completions,
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
    )

    parsed, _ = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return one PlannerAction JSON object.",
        user_prompt="Plan the next action.",
        response_model=PlannerAction,
    )

    assert parsed.action_type == "tool_call"
    assert parsed.tool_name == "generic_purpose"
    assert parsed.tool_request["clips"][0]["video_id"] == "video_13"


def test_planner_action_accepts_safe_bare_tool_request_shape():
    parsed = PlannerAction.model_validate(
        {
            "tool_name": "ocr",
            "clips": [{"video_id": "video_13", "start_s": 120.0, "end_s": 150.0}],
            "query": "read the chart values",
        }
    )

    assert parsed.action_type == "tool_call"
    assert parsed.tool_name == "ocr"
    assert parsed.tool_request["tool_name"] == "ocr"
    assert parsed.tool_request["clips"][0]["video_id"] == "video_13"


def test_complete_json_repairs_bare_ocr_runtime_fields_into_valid_planner_action(tmp_path):
    fake_completions = _FakeChatCompletions(
        [
            '{"ocr_sample_fps": 2.0, "ocr_source": "clip"}',
            """
            {
              "action_type": "synthesize",
              "rationale": "OCR evidence is already present; use it to answer from supported text.",
              "synthesis_instructions": "Use the collected OCR evidence and keep unsupported values unresolved."
            }
            """,
        ]
    )
    client = _make_client(
        fake_completions,
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
    )

    parsed, raw = client.complete_json(
        endpoint_name="default",
        model_name="gpt-5.4",
        system_prompt="Return one PlannerAction JSON object.",
        user_prompt="Plan the next action.",
        response_model=PlannerAction,
    )

    assert parsed.action_type == "synthesize"
    assert "OCR evidence is already present" in parsed.rationale
    assert "synthesize" in raw
    assert len(fake_completions.calls) == 2
    assert "SCHEMA_REPAIR_REQUEST" in fake_completions.calls[1]["messages"][1]["content"][0]["text"]


def test_complete_json_can_use_local_qwen_backend(tmp_path, monkeypatch):
    captured = {}

    class FakeQwenRunner(object):
        def __init__(self, **kwargs):
            captured["runner_kwargs"] = dict(kwargs)

        def generate(self, messages, max_new_tokens):
            captured["messages"] = messages
            captured["max_new_tokens"] = max_new_tokens
            return '{"strategy":"local qwen plan"}'

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(local_multimodal, "QwenStyleRunner", FakeQwenRunner)
    monkeypatch.setattr(wrapper_shared, "resolve_model_path", lambda *args, **kwargs: "/tmp/qwen")
    monkeypatch.setattr(wrapper_shared, "resolved_device_label", lambda runtime: runtime["device"])

    client = _make_client(
        _FakeChatCompletions('{"strategy": "should not call openai"}'),
        workspace_root=tmp_path / "workspace",
    )

    parsed, raw = client.complete_json(
        backend="local_qwen",
        endpoint_name="default",
        model_name="Qwen/Qwen3.5-9B",
        system_prompt="Return JSON only.",
        user_prompt="Plan this task.",
        response_model=_DummyResponseModel,
        max_tokens=1234,
        agent_extra={
            "device": "cuda:2",
            "device_map": "balanced_cuda:2,3",
            "attn_implementation": "flash_attention_2",
        },
    )

    assert parsed.strategy == "local qwen plan"
    assert raw == '{"strategy":"local qwen plan"}'
    assert captured["closed"] is True
    assert captured["runner_kwargs"]["device_label"] == "cuda:2"
    assert captured["runner_kwargs"]["device_map"] == "balanced_cuda:2,3"
    assert captured["runner_kwargs"]["attn_implementation"] == "flash_attention_2"
    assert captured["max_new_tokens"] == 1234
    assert captured["messages"][0]["role"] == "system"
