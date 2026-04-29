import json

import pytest

from video_trace_pipeline.agents.trace_synthesizer import TraceSynthesizerAgent
from video_trace_pipeline.schemas import AgentConfig, TaskSpec, TracePackage


class FakeLLMClient(object):
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(dict(kwargs))
        response_model = kwargs["response_model"]
        if hasattr(response_model, "model_validate"):
            parsed = response_model.model_validate(self.payload)
        else:
            parsed = response_model.parse_obj(self.payload)
        return parsed, json.dumps(self.payload)


def _task():
    return TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What is shown?",
        options=["A", "B"],
        video_path="video.mp4",
    )


def test_trace_synthesizer_uses_tracepackage_response_model():
    payload = {
        "task_key": "sample1",
        "mode": "generate",
        "evidence_entries": [
            {
                "evidence_id": "ev_1",
                "tool_name": "ocr",
                "evidence_text": "OCR reads 42 on the sign.",
                "status": "validated",
                "observation_ids": ["obs_1"],
                "time_intervals": [{"start_s": 1.0, "end_s": 2.0}],
            }
        ],
        "inference_steps": [
            {
                "step_id": 1,
                "text": "The sign reads 42.",
                "supporting_observation_ids": ["obs_1"],
                "answer_relevance": "high",
                "time_intervals": [{"start_s": 1.0, "end_s": 2.0}],
            }
        ],
        "final_answer": "A",
        "benchmark_renderings": {},
        "metadata": {"round": 1},
    }
    client = FakeLLMClient(payload)
    agent = TraceSynthesizerAgent(
        llm_client=client,
        agent_config=AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
    )

    raw, parsed = agent.synthesize(
        task=_task(),
        mode="generate",
        round_evidence_entries=[],
        round_observations=[],
        current_trace=None,
        refinement_instructions="",
    )

    assert raw == json.dumps(payload)
    assert isinstance(parsed, TracePackage)
    assert parsed.evidence_entries[0].tool_name == "ocr"
    assert parsed.evidence_entries[0].status == "validated"
    assert parsed.evidence_entries[0].time_intervals[0].start_s == 1.0
    assert parsed.inference_steps[0].time_intervals[0].end_s == 2.0
    assert client.calls
    assert client.calls[0]["response_model"] is TracePackage


def test_trace_synthesizer_coerces_labeled_evidence_confidence():
    payload = {
        "task_key": "sample1",
        "mode": "generate",
        "evidence_entries": [
            {
                "evidence_id": "ev_1",
                "tool_name": "generic_purpose",
                "evidence_text": "The chart supports the value.",
                "confidence": "medium",
            },
            {
                "evidence_id": "ev_2",
                "tool_name": "generic_purpose",
                "evidence_text": "The label is clear.",
                "confidence": "high",
            },
        ],
        "inference_steps": [],
        "final_answer": "A",
        "benchmark_renderings": {},
        "metadata": {},
    }

    parsed = TracePackage.model_validate(payload)

    assert parsed.evidence_entries[0].confidence == 0.5
    assert parsed.evidence_entries[1].confidence == 0.85


def test_trace_synthesizer_rejects_noncanonical_trace_payloads():
    payload = {
        "task_key": "sample1",
        "mode": "generate",
        "evidence_entries": [
            {
                "evidence_id": "ev_1",
                "tool_name": "asr",
                "evidence_text": "A person says hello.",
                "status": "grounded",
                "observation_ids": ["obs_1"],
            }
        ],
        "inference_steps": [],
        "final_answer": "",
        "benchmark_renderings": {},
        "metadata": {},
    }
    client = FakeLLMClient(payload)
    agent = TraceSynthesizerAgent(
        llm_client=client,
        agent_config=AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
    )

    with pytest.raises(Exception, match="status must be one of"):
        agent.synthesize(
            task=_task(),
            mode="generate",
            round_evidence_entries=[],
            round_observations=[],
            current_trace=None,
            refinement_instructions="",
        )
