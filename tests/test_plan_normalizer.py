import pytest

from video_trace_pipeline.orchestration.plan_normalizer import ExecutionPlanNormalizer
from video_trace_pipeline.schemas import (
    ASRRequest,
    AudioTemporalGrounderRequest,
    DenseCaptionRequest,
    ExecutionPlan,
    FrameRetrieverRequest,
    GenericPurposeRequest,
    OCRRequest,
    PlanStep,
    SpatialGrounderRequest,
    TaskSpec,
    VisualTemporalGrounderRequest,
)


class _Adapter(object):
    def __init__(self, request_model):
        self.request_model = request_model


class _Registry(object):
    def __init__(self):
        self.adapters = {
            "visual_temporal_grounder": _Adapter(VisualTemporalGrounderRequest),
            "frame_retriever": _Adapter(FrameRetrieverRequest),
            "spatial_grounder": _Adapter(SpatialGrounderRequest),
            "ocr": _Adapter(OCRRequest),
            "generic_purpose": _Adapter(GenericPurposeRequest),
            "asr": _Adapter(ASRRequest),
            "dense_captioner": _Adapter(DenseCaptionRequest),
            "audio_temporal_grounder": _Adapter(AudioTemporalGrounderRequest),
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


def test_plan_normalizer_rejects_visual_grounding_frame_bridge_to_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Do broad visual count through frames.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find all candidate action intervals.",
                inputs={"query": "all object-use intervals", "top_k": 5},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve representative frames for each candidate interval.",
                inputs={"query": "representative frames showing the action", "num_frames": 5, "sort_order": "chronological"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="generic_purpose",
                purpose="Count the accepted action intervals.",
                inputs={"query": "Count the visible action occurrences from the supplied evidence."},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="Pass grounded clips directly to generic_purpose"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_rejects_dense_chronological_frame_bridge_to_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Do state reasoning through dense frames.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find all changing-state candidate clips.",
                inputs={"query": "object changing state over time", "top_k": 5},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve dense chronological coverage for count and state reasoning.",
                inputs={"query": "dense chronological visual coverage", "num_frames": 12, "sequence_mode": "chronological", "sort_order": "chronological"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="generic_purpose",
                purpose="Resolve the count and state from the dense sequence.",
                inputs={"query": "Count state transitions across the supplied sequence."},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="Pass grounded clips directly to generic_purpose"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_accepts_direct_grounded_clips_to_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Use grounded clips directly.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find all candidate object-state intervals.",
                inputs={"query": "all candidate object-state intervals", "top_k": 5},
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Determine which clips satisfy the state and count them.",
                inputs={"query": "Count only clips that visibly satisfy the requested state."},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), plan)

    assert [step.tool_name for step in normalized.steps] == ["visual_temporal_grounder", "generic_purpose"]
    assert normalized.steps[1].input_refs["clips"][0].field_path == "clips"


def test_plan_normalizer_accepts_direct_grounded_clips_to_spatial_grounder():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Localize directly from grounded clips.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find clips where the answer-critical object appears.",
                inputs={"query": "object visible in the required relation", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="spatial_grounder",
                purpose="Localize the object inside each grounded clip.",
                inputs={"query": "the answer-critical object"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), plan)

    assert [step.tool_name for step in normalized.steps] == ["visual_temporal_grounder", "spatial_grounder"]
    assert normalized.steps[1].input_refs["clips"][0].field_path == "clips"


def test_plan_normalizer_accepts_direct_grounded_clips_to_ocr():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Read visible text directly from grounded clips.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find clips where the text is visible.",
                inputs={"query": "visible sign text", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="ocr",
                purpose="Read visible text from the grounded clips.",
                inputs={"query": "read the visible text"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), plan)

    assert [step.tool_name for step in normalized.steps] == ["visual_temporal_grounder", "ocr"]
    assert normalized.steps[1].input_refs["clips"][0].field_path == "clips"


def test_plan_normalizer_rejects_visual_grounding_frame_bridge_to_spatial_without_frame_need():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Localize after unnecessary frame bridge.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find clips where the target object appears.",
                inputs={"query": "target object appears", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve representative frames for each candidate interval.",
                inputs={"query": "representative object frames", "num_frames": 3},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Localize the target object.",
                inputs={"query": "the target object"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="Pass grounded clips directly to spatial_grounder"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_rejects_spatial_grounder_to_ocr():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject spatial crops before OCR.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the sign interval.",
                inputs={"query": "visible sign with small text", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve a readable static high-resolution frame for the sign text.",
                inputs={"query": "readable still frame of the small sign text", "num_frames": 3},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Localize the sign text region in the readable frame.",
                inputs={"query": "the sign text region"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=4,
                tool_name="ocr",
                purpose="Read the exact sign text.",
                inputs={"query": "read the sign text exactly"},
                input_refs={"regions": [{"step_id": 3, "field_path": "regions"}]},
            ),
        ],
    )

    with pytest.raises(ValueError, match="OCR must use complete frames or grounded clips directly"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_accepts_explicit_frame_need_before_ocr_full_frames():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Use full frames because the sign must be readable.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the sign interval.",
                inputs={"query": "visible sign with small text", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve a readable static high-resolution frame for the sign text.",
                inputs={"query": "readable still frame of the small sign text", "num_frames": 3},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="ocr",
                purpose="Read the exact sign text from the complete frames.",
                inputs={"query": "read the sign text exactly from the full frame"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=4,
                tool_name="generic_purpose",
                purpose="Map OCR text to the answer option.",
                inputs={"query": "Choose the option supported by the OCR text."},
                input_refs={"text_contexts": [{"step_id": 3, "field_path": "text"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), plan)

    assert [step.tool_name for step in normalized.steps] == [
        "visual_temporal_grounder",
        "frame_retriever",
        "ocr",
        "generic_purpose",
    ]


def test_plan_normalizer_accepts_explicit_frame_need_before_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Inspect an exact local frame sequence.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the answer-critical action interval.",
                inputs={"query": "action at the requested moment", "top_k": 3},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve exact anchor-window neighboring frames around the timestamp.",
                inputs={"query": "anchor frame and neighbors", "num_frames": 5, "sequence_mode": "anchor_window", "sort_order": "chronological"},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="generic_purpose",
                purpose="Answer from the exact anchor-window frame sequence.",
                inputs={"query": "Determine the state at the timestamp from the neighboring frames."},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
        ],
    )

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[2].tool_name == "generic_purpose"
