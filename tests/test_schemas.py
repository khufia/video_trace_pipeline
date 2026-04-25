import pytest

from video_trace_pipeline.common import traverse_path
from video_trace_pipeline.schemas import ASRRequest, AuditReport, ClipRef, ExecutionPlan, FrameRetrieverRequest, OCRRequest, TracePackage
from video_trace_pipeline.tools.base import ToolAdapter


class DummyFrameRetrieverAdapter(ToolAdapter):
    name = "frame_retriever"
    request_model = FrameRetrieverRequest


class DummyOCRAdapter(ToolAdapter):
    name = "ocr"
    request_model = OCRRequest


def test_frame_retriever_accepts_clip():
    clip = ClipRef(video_id="video1", start_s=0.0, end_s=5.0)
    request = FrameRetrieverRequest(tool_name="frame_retriever", clip=clip, query="scoreboard", num_frames=2)
    assert request.clip.start_s == 0.0
    assert request.num_frames == 2


def test_frame_retriever_accepts_time_hint_without_clip():
    request = FrameRetrieverRequest(
        tool_name="frame_retriever",
        query="scoreboard",
        time_hint="last 20% of the video",
        num_frames=2,
    )
    assert request.clip is None
    assert request.time_hint == "last 20% of the video"


def test_frame_retriever_requires_clip_or_time_hint():
    with pytest.raises(ValueError):
        FrameRetrieverRequest(tool_name="frame_retriever", query="scoreboard")


def test_tool_adapter_normalizes_frame_retriever_clip_list_and_top_k():
    adapter = DummyFrameRetrieverAdapter()

    request = adapter.parse_request(
        {
            "clip": [
                {"video_id": "video1", "start_s": 10.0, "end_s": 15.0},
                {"video_id": "video1", "start_s": 20.0, "end_s": 25.0},
            ],
            "top_k": 3,
        }
    )

    assert request.clip is None
    assert len(request.clips) == 2
    assert request.clips[0].start_s == 10.0
    assert request.clips[1].end_s == 25.0
    assert request.num_frames == 3


def test_asr_request_accepts_multiple_clips():
    request = ASRRequest(
        tool_name="asr",
        clips=[
            ClipRef(video_id="video1", start_s=0.0, end_s=5.0),
            ClipRef(video_id="video1", start_s=5.0, end_s=10.0),
        ],
    )

    assert request.clip is None
    assert len(request.clips) == 2


def test_ocr_request_prefers_specific_frame_inputs_over_clip_context():
    adapter = DummyOCRAdapter()

    request = adapter.parse_request(
        {
            "query": "read the chart values",
            "clip": {"video_id": "video1", "start_s": 0.0, "end_s": 5.0},
            "frames": [
                {
                    "video_id": "video1",
                    "timestamp_s": 2.5,
                    "clip": {"video_id": "video1", "start_s": 0.0, "end_s": 5.0},
                    "metadata": {"source_path": "frame_00.png"},
                }
            ],
        }
    )

    assert request.frame is not None
    assert request.frames[0].timestamp_s == 2.5
    assert request.clip is None
    assert request.clips == []


def test_clip_ref_validates_range():
    with pytest.raises(ValueError):
        ClipRef(video_id="video1", start_s=10.0, end_s=5.0)


def test_execution_plan_normalizes_string_step_ids():
    payload = {
        "strategy": "Use grounded steps.",
        "use_summary": True,
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "visual_temporal_grounder",
                "purpose": "Find the relevant segment.",
                "arguments": {"query": "person raises hand"},
                "input_refs": [],
                "depends_on": [],
            },
            {
                "step_id": "step_2",
                "tool_name": "frame_retriever",
                "purpose": "Pick a representative frame from the grounded clip.",
                "arguments": {"query": "raised hand"},
                "input_refs": [{"target_field": "clip", "source": {"step_id": "step_1", "field_path": "clips.0"}}],
                "depends_on": ["step_1"],
            },
        ],
        "refinement_instructions": "",
    }

    plan = ExecutionPlan.parse_obj(payload)

    assert [step.step_id for step in plan.steps] == [1, 2]
    assert plan.steps[1].input_refs[0].source.step_id == 1
    assert plan.steps[1].depends_on == [1]


def test_trace_package_normalizes_string_inference_step_ids():
    payload = {
        "task_key": "videomathqa_0",
        "mode": "generate",
        "evidence_entries": [],
        "inference_steps": [
            {
                "step_id": "step_1",
                "text": "The score shown on the board is 12.",
                "supporting_observation_ids": ["obs_1"],
                "answer_relevance": "high",
            }
        ],
        "final_answer": "12",
        "benchmark_renderings": {},
    }

    trace = TracePackage.parse_obj(payload)

    assert trace.inference_steps[0].step_id == 1


def test_traverse_path_matches_labeled_list_items():
    payload = {
        "regions": [
            {"label": "puzzle region", "bbox": [1, 2, 3, 4]},
            {"label": "label marker", "bbox": [5, 6, 7, 8]},
        ]
    }

    region = traverse_path(payload, "regions.puzzle")

    assert region["label"] == "puzzle region"


def test_audit_report_normalizes_scores_to_integer_band():
    report = AuditReport.parse_obj(
        {
            "verdict": "FAIL",
            "scores": {
                "logical_coherence": 0.18,
                "completeness": 5.7,
                "factual_correctness": "2.2",
            },
        }
    )

    assert report.scores == {
        "logical_coherence": 1,
        "completeness": 5,
        "factual_correctness": 2,
    }
