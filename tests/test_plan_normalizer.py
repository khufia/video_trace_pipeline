import pytest

from video_trace_pipeline.orchestration.plan_normalizer import ExecutionPlanNormalizer
from video_trace_pipeline.schemas import (
    AudioTemporalGrounderOutput,
    AudioTemporalGrounderRequest,
    ASRRequest,
    ASROutput,
    ExecutionPlan,
    FrameRetrieverOutput,
    FrameRetrieverRequest,
    GenericPurposeOutput,
    GenericPurposeRequest,
    OCROutput,
    OCRRequest,
    PlanStep,
    SpatialGrounderOutput,
    SpatialGrounderRequest,
    TaskSpec,
    VisualTemporalGrounderOutput,
    VisualTemporalGrounderRequest,
)


class _Adapter(object):
    def __init__(self, request_model, output_model):
        self.request_model = request_model
        self.output_model = output_model


class _Registry(object):
    def __init__(self):
        self.adapters = {
            "visual_temporal_grounder": _Adapter(VisualTemporalGrounderRequest, VisualTemporalGrounderOutput),
            "audio_temporal_grounder": _Adapter(AudioTemporalGrounderRequest, AudioTemporalGrounderOutput),
            "frame_retriever": _Adapter(FrameRetrieverRequest, FrameRetrieverOutput),
            "spatial_grounder": _Adapter(SpatialGrounderRequest, SpatialGrounderOutput),
            "ocr": _Adapter(OCRRequest, OCROutput),
            "generic_purpose": _Adapter(GenericPurposeRequest, GenericPurposeOutput),
            "asr": _Adapter(ASRRequest, ASROutput),
        }

    def get_adapter(self, tool_name):
        return self.adapters[tool_name]


def _task():
    return TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="What is shown?",
        options=["A", "B"],
        video_path="video.mp4",
    )


def test_plan_normalizer_orders_steps_and_remaps_field_keyed_input_refs():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="  Read the chart carefully.  ",
        steps=[
            PlanStep(
                step_id=4,
                tool_name="generic_purpose",
                purpose=" Compute the answer. ",
                inputs={"query": "Answer from OCR."},
                input_refs={"text_contexts": [{"step_id": 3, "field_path": "text"}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose=" Grab readable frames. ",
                inputs={"num_frames": 2, "query": "Best chart frame."},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="ocr",
                purpose=" Read the chart. ",
                inputs={"query": "Extract chart text."},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose=" Find the right clip. ",
                inputs={"query": "Find the chart.", "top_k": 5},
            ),
        ],
        refinement_instructions="  Replace unsupported claims.  ",
    )

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.strategy == "Read the chart carefully."
    assert normalized.refinement_instructions == "Replace unsupported claims."
    assert [step.tool_name for step in normalized.steps] == [
        "visual_temporal_grounder",
        "frame_retriever",
        "ocr",
        "generic_purpose",
    ]
    assert normalized.steps[1].inputs == {"num_frames": 2, "query": "Best chart frame."}
    assert normalized.steps[1].input_refs["clips"][0].step_id == 1
    assert normalized.steps[3].input_refs["text_contexts"][0].step_id == 3


def test_plan_schema_rejects_removed_fields_before_normalization():
    with pytest.raises(ValueError, match="use_summary"):
        ExecutionPlan.parse_obj({"strategy": "bad", "use_summary": True, "steps": []})

    with pytest.raises(ValueError, match="removed field"):
        PlanStep.parse_obj(
            {
                "step_id": 1,
                "tool_name": "generic_purpose",
                "purpose": "bad",
                "arguments": {"query": "old"},
            }
        )

    with pytest.raises(ValueError, match="field-keyed object"):
        PlanStep.parse_obj(
            {
                "step_id": 1,
                "tool_name": "frame_retriever",
                "purpose": "bad",
                "inputs": {"query": "old"},
                "input_refs": [{"target_field": "clips", "source": {"step_id": 1, "field_path": "clips"}}],
            }
        )


def test_plan_normalizer_rejects_noncanonical_inputs_and_wiring():
    normalizer = ExecutionPlanNormalizer(_Registry())
    bad_alias_plan = ExecutionPlan(
        strategy="Reject aliases.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Use OCR text.",
                inputs={"query": "Answer.", "texts": "line 1"},
            ),
        ],
    )
    with pytest.raises(ValueError, match="unexpected input"):
        normalizer.normalize(_task(), bad_alias_plan)

    bad_ref_plan = ExecutionPlan(
        strategy="Reject non-structural media wiring.",
        steps=[
            PlanStep(step_id=1, tool_name="frame_retriever", purpose="Retrieve frames.", inputs={"query": "Best frame.", "num_frames": 2}),
            PlanStep(
                step_id=2,
                tool_name="ocr",
                purpose="Read the image.",
                inputs={"query": "Read text."},
                input_refs={"frames": [{"step_id": 1, "field_path": "summary"}]},
            ),
        ],
    )
    with pytest.raises(ValueError, match="must bind frames via a structural frames path"):
        normalizer.normalize(_task(), bad_ref_plan)

    bad_time_hint_plan = ExecutionPlan(
        strategy="Reject dynamic time hint wiring.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe speech.",
                inputs={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve frames around speech.",
                inputs={"query": "speech moment", "num_frames": 3},
                input_refs={
                    "clips": [{"step_id": 1, "field_path": "transcripts[].clip"}],
                    "time_hints": [{"step_id": 1, "field_path": "transcripts"}],
                },
            ),
        ],
    )
    with pytest.raises(ValueError, match="time_hints"):
        normalizer.normalize(_task(), bad_time_hint_plan)

    good_time_hint_plan = ExecutionPlan(
        strategy="Use ASR phrase timestamp wiring.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe and locate the quoted speech.",
                inputs={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve frames at the quoted speech timestamp.",
                inputs={
                    "query": "visible object on the table",
                    "num_frames": 3,
                    "sequence_mode": "anchor_window",
                    "sort_order": "chronological",
                },
                input_refs={
                    "clips": [{"step_id": 1, "field_path": "transcripts[].clip"}],
                    "time_hints": [{"step_id": 1, "field_path": "phrase_matches[].time_hint"}],
                },
            ),
        ],
    )
    normalized = normalizer.normalize(_task(), good_time_hint_plan)
    assert normalized.steps[1].input_refs["time_hints"][0].field_path == "phrase_matches[].time_hint"


def test_plan_normalizer_rejects_placeholder_time_hints_and_asr_visual_followup_without_anchor():
    normalizer = ExecutionPlanNormalizer(_Registry())
    placeholder_plan = ExecutionPlan(
        strategy="Reject placeholder hints.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="frame_retriever",
                purpose="Retrieve quote-neighbor frames.",
                inputs={
                    "query": "visible object on the table",
                    "clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 120.0}],
                    "time_hints": ["Use the timestamp of the ASR phrase match."],
                    "sequence_mode": "anchor_window",
                },
            )
        ],
    )
    with pytest.raises(ValueError, match="placeholder time_hints"):
        normalizer.normalize(_task(), placeholder_plan)

    missing_anchor_plan = ExecutionPlan(
        strategy="Reject ASR clip follow-up without phrase timestamp.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe quoted speech.",
                inputs={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 120.0}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve frames around the spoken phrase.",
                inputs={"query": "visible object on the table", "num_frames": 3},
                input_refs={"clips": [{"step_id": 1, "field_path": "transcripts[].clip"}]},
            ),
        ],
    )
    with pytest.raises(ValueError, match="without time_hints"):
        normalizer.normalize(_task(), missing_anchor_plan)


def test_plan_normalizer_rejects_asr_text_context_and_allows_transcripts():
    normalizer = ExecutionPlanNormalizer(_Registry())
    bad_plan = ExecutionPlan(
        strategy="Use ASR output structurally.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe speech.",
                inputs={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from transcript.",
                inputs={"query": "Use transcript."},
                input_refs={"text_contexts": [{"step_id": 1, "field_path": "text"}]},
            ),
        ],
    )
    with pytest.raises(ValueError, match="via transcripts -> transcripts"):
        normalizer.normalize(_task(), bad_plan)

    good_plan = ExecutionPlan(
        strategy="Use transcripts.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe speech.",
                inputs={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from transcript.",
                inputs={"query": "Use transcript."},
                input_refs={
                    "transcripts": [{"step_id": 1, "field_path": "transcripts"}],
                    "clips": [{"step_id": 1, "field_path": "clips"}],
                },
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), good_plan)
    assert sorted(normalized.steps[1].input_refs) == ["clips", "transcripts"]


def test_plan_normalizer_rejects_context_free_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject detached generic-purpose steps.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Inspect the chart and answer the question.",
                inputs={"query": "Determine the answer from the chart."},
            ),
        ],
    )

    with pytest.raises(ValueError, match="context-free generic_purpose"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_rejects_context_free_media_tools_before_execution():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject context-free frame retrieval.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="frame_retriever",
                purpose="Try to search the full video from a query alone.",
                inputs={"query": "the first dog in the shot", "num_frames": 3},
            )
        ],
    )

    with pytest.raises(ValueError, match="frame_retriever.*lacks required context"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_rejects_audio_first_for_visual_conditioned_audio_tasks():
    normalizer = ExecutionPlanNormalizer(_Registry())
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="How many different sounds are heard when the person is using the red sauce bottle?",
        options=["A. 1", "B. 2", "C. 3", "D. 4"],
        video_path="video.mp4",
    )
    bad_plan = ExecutionPlan(
        strategy="Start from audio even though the sound is conditioned on a visible action.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="audio_temporal_grounder",
                purpose="Find all sound events.",
                inputs={
                    "query": "different sounds heard while the red sauce bottle is used",
                    "clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 120.0}],
                },
            ),
            PlanStep(
                step_id=2,
                tool_name="visual_temporal_grounder",
                purpose="Find red sauce bottle use after audio.",
                inputs={"query": "person using the red sauce bottle", "top_k": 5},
            ),
        ],
    )

    with pytest.raises(ValueError, match="visual-conditioned audio task"):
        normalizer.normalize(task, bad_plan)

    good_plan = ExecutionPlan(
        strategy="Ground visual-use candidates first, then inspect audio only inside those clips.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find all visible red sauce bottle use candidates.",
                inputs={"query": "person visibly using the red sauce bottle", "top_k": 8},
            ),
            PlanStep(
                step_id=2,
                tool_name="audio_temporal_grounder",
                purpose="Find distinct sound events only inside the visible-use candidates.",
                inputs={"query": "distinct non-speech sounds heard during the visible red sauce bottle use"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(task, good_plan)

    assert [step.tool_name for step in normalized.steps] == [
        "visual_temporal_grounder",
        "audio_temporal_grounder",
    ]
    assert normalized.steps[1].input_refs["clips"][0].step_id == 1


def test_plan_normalizer_rejects_unretrieved_evidence_ids():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reuse evidence.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Answer from prior evidence.",
                inputs={"query": "Use existing evidence.", "evidence_ids": ["ev_missing"]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="were not retrieved"):
        normalizer.normalize(_task(), plan, retrieved_context={"evidence": [{"evidence_id": "ev_ready"}]})

    normalized = normalizer.normalize(
        _task(),
        ExecutionPlan(
            strategy="Reuse evidence.",
            steps=[
                PlanStep(
                    step_id=1,
                    tool_name="generic_purpose",
                    purpose="Answer from prior evidence.",
                    inputs={"query": "Use existing evidence.", "evidence_ids": ["ev_ready"]},
                ),
            ],
        ),
        retrieved_context={"observations": [{"observation_id": "obs_1", "evidence_id": "ev_ready"}]},
    )
    assert normalized.steps[0].inputs["evidence_ids"] == ["ev_ready"]


def test_plan_normalizer_rejects_nonexistent_and_wrong_target_output_fields():
    normalizer = ExecutionPlanNormalizer(_Registry())
    bad_generic_transcript = ExecutionPlan(
        strategy="Bad source field.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Describe frames.",
                inputs={"query": "Describe.", "text_contexts": ["context"]},
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Use impossible transcript field.",
                inputs={"query": "Answer."},
                input_refs={"transcripts": [{"step_id": 1, "field_path": "transcripts"}]},
            ),
        ],
    )
    with pytest.raises(ValueError, match="emits only"):
        normalizer.normalize(_task(), bad_generic_transcript)
