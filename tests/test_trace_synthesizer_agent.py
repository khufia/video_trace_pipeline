import json

from video_trace_pipeline.agents.trace_synthesizer import TraceSynthesizerAgent, _repair_trace_package_payload
from video_trace_pipeline.schemas import AgentConfig, TaskSpec


class FakeLLMClient(object):
    def __init__(self, payload):
        self.payload = dict(payload)
        self.calls = []

    def complete_text(self, **kwargs):
        self.calls.append(dict(kwargs))
        return json.dumps(self.payload)


def _task():
    return TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What is shown?",
        options=["A", "B"],
        video_path="video.mp4",
    )


def test_repair_trace_package_payload_backfills_tool_name_from_evidence_entries():
    payload = {
        "task_key": "sample1",
        "mode": "generate",
        "evidence_entries": [
            {
                "evidence_id": "ev_1",
                "evidence_text": "OCR reads 42 on the sign.",
                "observation_ids": ["obs_1"],
            }
        ],
        "inference_steps": [],
        "final_answer": "",
        "benchmark_renderings": {},
        "metadata": {},
    }

    repaired = _repair_trace_package_payload(
        payload,
        evidence_entries=[
            {
                "evidence_id": "ev_1",
                "tool_name": "ocr",
                "evidence_text": "OCR reads 42 on the sign.",
                "observation_ids": ["obs_1"],
            }
        ],
        observations=[],
        current_trace=None,
    )

    assert repaired["evidence_entries"][0]["tool_name"] == "ocr"


def test_trace_synthesizer_repairs_missing_tool_name_from_observations():
    client = FakeLLMClient(
        {
            "task_key": "sample1",
            "mode": "generate",
            "evidence_entries": [
                {
                    "evidence_id": "ev_1",
                    "evidence_text": "A person says hello.",
                    "observation_ids": ["obs_1"],
                }
            ],
            "inference_steps": [],
            "final_answer": "",
            "benchmark_renderings": {},
            "metadata": {},
        }
    )
    agent = TraceSynthesizerAgent(
        llm_client=client,
        agent_config=AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
    )

    raw, parsed = agent.synthesize(
        task=_task(),
        mode="generate",
        evidence_entries=[],
        observations=[
            {
                "observation_id": "obs_1",
                "source_tool": "asr",
            }
        ],
        current_trace=None,
        refinement_instructions="",
    )

    assert raw
    assert parsed.evidence_entries[0].tool_name == "asr"
    assert client.calls
    assert client.calls[0]["response_format"] == {"type": "json_object"}
