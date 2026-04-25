import json

from video_trace_pipeline.agents.trace_synthesizer import (
    TraceSynthesizerAgent,
    _normalize_evidence_status,
    _repair_trace_package_payload,
)
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


def test_normalize_evidence_status_maps_model_friendly_labels():
    assert _normalize_evidence_status("grounded") == "provisional"
    assert _normalize_evidence_status("supported") == "provisional"
    assert _normalize_evidence_status("obsolete") == "superseded"
    assert _normalize_evidence_status("validated") == "validated"
    assert _normalize_evidence_status("") == "provisional"


def test_repair_trace_package_payload_normalizes_grounded_status_from_cached_response():
    payload = {
        "task_key": "sample1",
        "mode": "generate",
        "evidence_entries": [
            {
                "evidence_id": "ev_syn_01",
                "tool_name": "asr",
                "evidence_text": "Example grounded evidence.",
                "status": "grounded",
                "observation_ids": ["obs_1"],
            },
            {
                "evidence_id": "ev_syn_02",
                "tool_name": "generic_purpose",
                "evidence_text": "Example supported evidence.",
                "status": "supported",
                "observation_ids": ["obs_2"],
            },
            {
                "evidence_id": "ev_syn_03",
                "tool_name": "ocr",
                "evidence_text": "Example obsolete evidence.",
                "status": "obsolete",
                "observation_ids": ["obs_3"],
            },
        ],
        "inference_steps": [],
        "final_answer": "",
        "benchmark_renderings": {},
        "metadata": {},
    }

    repaired = _repair_trace_package_payload(
        payload,
        evidence_entries=[],
        observations=[],
        current_trace=None,
    )

    statuses = [item.get("status") for item in repaired["evidence_entries"]]
    assert statuses == ["provisional", "provisional", "superseded"]
