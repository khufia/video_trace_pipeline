import pytest

from video_trace_pipeline.common import traverse_path
from video_trace_pipeline.schemas import (
    ASRRequest,
    AuditReport,
    ClipRef,
    ExecutionPlan,
    EvidenceEntry,
    FrameRetrieverRequest,
    OCRRequest,
    TaskState,
    TracePackage,
    TranscriptRef,
    VerifierOutput,
    VerifierRequest,
)
from video_trace_pipeline.tools.base import ToolAdapter


class DummyFrameRetrieverAdapter(ToolAdapter):
    name = "frame_retriever"
    request_model = FrameRetrieverRequest


class DummyOCRAdapter(ToolAdapter):
    name = "ocr"
    request_model = OCRRequest


def test_frame_retriever_accepts_clip():
    clip = ClipRef(video_id="video1", start_s=0.0, end_s=5.0)
    request = FrameRetrieverRequest(tool_name="frame_retriever", clips=[clip], query="scoreboard", num_frames=2)
    assert request.clips[0].start_s == 0.0
    assert request.num_frames == 2


def test_frame_retriever_accepts_time_hint_without_clip():
    request = FrameRetrieverRequest(
        tool_name="frame_retriever",
        query="scoreboard",
        time_hints=["last 20% of the video"],
        num_frames=2,
    )
    assert request.clips == []
    assert request.time_hints == ["last 20% of the video"]


def test_frame_retriever_requires_clip_or_time_hint():
    with pytest.raises(ValueError):
        FrameRetrieverRequest(tool_name="frame_retriever", query="scoreboard")


def test_tool_adapter_rejects_noncanonical_frame_retriever_aliases():
    adapter = DummyFrameRetrieverAdapter()

    with pytest.raises(ValueError, match="Unexpected request field"):
        adapter.parse_request(
            {
                "clip": [
                    {"video_id": "video1", "start_s": 10.0, "end_s": 15.0},
                    {"video_id": "video1", "start_s": 20.0, "end_s": 25.0},
                ],
                "top_k": 3,
            }
        )


def test_asr_request_accepts_multiple_clips():
    request = ASRRequest(
        tool_name="asr",
        clips=[
            ClipRef(video_id="video1", start_s=0.0, end_s=5.0),
            ClipRef(video_id="video1", start_s=5.0, end_s=10.0),
        ],
    )

    assert len(request.clips) == 2


def test_transcript_ref_rejects_flattened_text_field():
    with pytest.raises(ValueError, match="Extra inputs"):
        TranscriptRef.model_validate(
            {
                "transcript_id": "tx_1",
                "clip": {"video_id": "video1", "start_s": 0.0, "end_s": 1.0},
                "text": "flattened transcript text",
                "segments": [{"start_s": 0.0, "end_s": 1.0, "text": "hello"}],
            }
        )


def test_ocr_request_prefers_specific_frame_inputs_over_clip_context():
    adapter = DummyOCRAdapter()

    request = adapter.parse_request(
        {
            "query": "read the chart values",
            "clips": [{"video_id": "video1", "start_s": 0.0, "end_s": 5.0}],
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

    assert request.frames[0].timestamp_s == 2.5
    assert request.clips == []


def test_clip_ref_validates_range():
    with pytest.raises(ValueError):
        ClipRef(video_id="video1", start_s=10.0, end_s=5.0)


def test_execution_plan_normalizes_string_step_ids():
    payload = {
        "strategy": "Use grounded steps.",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "visual_temporal_grounder",
                "purpose": "Find the relevant segment.",
                "inputs": {"query": "person raises hand"},
                "input_refs": {},
                "expected_outputs": {"clips": "candidate moments"},
            },
            {
                "step_id": "step_2",
                "tool_name": "frame_retriever",
                "purpose": "Pick a representative frame from the grounded clip.",
                "inputs": {"query": "raised hand"},
                "input_refs": {"clips": [{"step_id": "step_1", "field_path": "clips[0]"}]},
                "expected_outputs": {"frames": "representative frames"},
            },
        ],
        "refinement_instructions": "",
    }

    plan = ExecutionPlan.parse_obj(payload)

    assert [step.step_id for step in plan.steps] == [1, 2]
    assert plan.steps[1].input_refs["clips"][0].step_id == 1


def test_execution_plan_rejects_removed_fields():
    with pytest.raises(ValueError, match="use_summary"):
        ExecutionPlan.parse_obj({"strategy": "bad", "use_summary": True, "steps": []})

    with pytest.raises(ValueError, match="removed field"):
        ExecutionPlan.parse_obj(
            {
                "strategy": "bad",
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "visual_temporal_grounder",
                        "purpose": "bad",
                        "arguments": {"query": "old"},
                    }
                ],
            }
        )


def test_execution_plan_rejects_list_style_input_refs():
    with pytest.raises(ValueError, match="field-keyed object"):
        ExecutionPlan.parse_obj(
            {
                "strategy": "bad",
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "frame_retriever",
                        "purpose": "bad",
                        "inputs": {"query": "old"},
                        "input_refs": [{"target_field": "clips", "source": {"step_id": 1, "field_path": "clips"}}],
                    }
                ],
            }
        )


def test_evidence_statuses_use_canonical_values():
    assert EvidenceEntry(evidence_id="ev_1", tool_name="ocr", evidence_text="Candidate read.").status == "candidate"
    with pytest.raises(ValueError, match="status must be one of"):
        EvidenceEntry(evidence_id="ev_old", tool_name="ocr", evidence_text="Old.", status="provisional")


def test_verifier_request_and_output_are_strict_claim_contracts():
    request = VerifierRequest.model_validate(
        {
            "tool_name": "verifier",
            "query": "verify the visible price claim",
            "claims": [{"claim_id": "claim_price", "text": "The visible price is 42.", "claim_type": "ocr"}],
            "text_contexts": ["OCR read 42 from the localized price tag."],
        }
    )

    assert request.text_contexts == ["OCR read 42 from the localized price tag."]

    with pytest.raises(ValueError, match="at least one evidence"):
        VerifierRequest.model_validate(
            {
                "tool_name": "verifier",
                "query": "verify",
                "claims": [{"claim_id": "claim_1", "text": "A claim."}],
            }
        )

    output = VerifierOutput.model_validate(
        {
            "claim_results": [
                {
                    "claim_id": "claim_price",
                    "verdict": "true",
                    "confidence": "high",
                    "claimed_value": 42,
                    "observed_value": "42",
                    "match_status": "match",
                    "target_presence": "present",
                    "rationale": "The OCR text directly supports it.",
                }
            ]
        }
    )

    assert output.claim_results[0].verdict == "supported"
    assert output.claim_results[0].confidence == 0.85


def test_task_state_statuses_use_canonical_values():
    state = TaskState.model_validate(
        {
            "task_key": "sample1",
            "claim_results": [{"claim_id": "claim_1", "text": "A claim.", "status": "unknown"}],
            "evidence_status_updates": [{"evidence_id": "ev_1", "new_status": "candidate"}],
        }
    )

    assert state.claim_results[0].status == "unknown"
    assert state.evidence_status_updates[0].new_status == "candidate"


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
