import pytest

from video_trace_pipeline.orchestration.plan_normalizer import ExecutionPlanNormalizer
from video_trace_pipeline.schemas import (
    ASRRequest,
    ExecutionPlan,
    FrameRetrieverRequest,
    GenericPurposeRequest,
    OCRRequest,
    PlanStep,
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
            "ocr": _Adapter(OCRRequest),
            "generic_purpose": _Adapter(GenericPurposeRequest),
            "asr": _Adapter(ASRRequest),
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


def test_plan_normalizer_orders_steps_topologically_and_remaps_dependencies():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="  Read the chart carefully.  ",
        use_summary=False,
        steps=[
            PlanStep(
                step_id=4,
                tool_name="generic_purpose",
                purpose=" Compute the answer. ",
                arguments={"query": "Answer from OCR."},
                input_refs=[
                    {"target_field": "text_contexts", "source": {"step_id": 3, "field_path": "text"}},
                ],
                depends_on=[3],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose=" Grab readable frames. ",
                arguments={"num_frames": 2, "query": "Best chart frame."},
                input_refs=[
                    {"target_field": "clips", "source": {"step_id": 1, "field_path": "clips"}},
                ],
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool_name="ocr",
                purpose=" Read the chart. ",
                arguments={"query": "Extract chart text."},
                input_refs=[
                    {"target_field": "frames", "source": {"step_id": 2, "field_path": "frames"}},
                ],
                depends_on=[2],
            ),
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose=" Find the right clip. ",
                arguments={"query": "Find the chart.", "top_k": 5},
                input_refs=[],
                depends_on=[],
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
    assert normalized.steps[1].arguments == {"num_frames": 2, "query": "Best chart frame."}
    assert normalized.steps[1].input_refs[0].target_field == "clips"
    assert normalized.steps[1].depends_on == [1]
    assert normalized.steps[2].depends_on == [2]
    assert normalized.steps[3].depends_on == [3]
    assert normalized.steps[3].input_refs[0].source.step_id == 3


def test_plan_normalizer_rejects_noncanonical_argument_aliases():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject aliases.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Use OCR text.",
                arguments={"query": "Answer.", "texts": "line 1", "ocr_texts": ["line 2", "line 1"]},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    try:
        normalizer.normalize(_task(), plan)
    except ValueError as exc:
        assert "unexpected argument" in str(exc)
    else:
        raise AssertionError("Expected plan normalization to reject argument aliases.")


def test_plan_normalizer_rejects_noncanonical_plural_aliases():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject generic-purpose aliases.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Use singular aliases for plural fields.",
                arguments={"query": "Answer.", "text_context": "line 1", "evidence_id": "ev_1"},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    try:
        normalizer.normalize(_task(), plan)
    except ValueError as exc:
        assert "unexpected argument" in str(exc)
    else:
        raise AssertionError("Expected plan normalization to reject plural aliases.")


def test_plan_normalizer_rejects_noncanonical_input_ref_target_fields():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject non-canonical input_ref fields.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=7,
                tool_name="generic_purpose",
                purpose="Produce candidate time hints.",
                arguments={"query": "Find the likely chart moment."},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=1,
                tool_name="frame_retriever",
                purpose="Use candidate time windows.",
                arguments={"query": "Best frame."},
                input_refs=[
                    {"target_field": "time_hint", "source": {"step_id": 7, "field_path": "time_hints"}},
                ],
                depends_on=[7],
            ),
        ],
        refinement_instructions="",
    )

    with pytest.raises(ValueError, match="unexpected input_ref target_field"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_infers_dependencies_from_input_refs_and_reorders_steps():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Infer dependencies from input refs.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=10,
                tool_name="generic_purpose",
                purpose="Answer from frames.",
                arguments={"query": "Answer."},
                input_refs=[
                    {"target_field": "frames", "source": {"step_id": 20, "field_path": "frames"}},
                ],
                depends_on=[],
            ),
            PlanStep(
                step_id=20,
                tool_name="frame_retriever",
                purpose="Find frames.",
                arguments={"query": "Best frame.", "num_frames": 2},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    normalized = normalizer.normalize(_task(), plan)

    assert [step.tool_name for step in normalized.steps] == ["frame_retriever", "generic_purpose"]
    assert normalized.steps[1].depends_on == [1]
    assert normalized.steps[1].input_refs[0].source.step_id == 1


def test_plan_normalizer_rejects_context_free_generic_purpose_step():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject detached generic-purpose steps.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Inspect the chart and answer the question.",
                arguments={"query": "Determine the answer from the chart."},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    try:
        normalizer.normalize(_task(), plan)
    except ValueError as exc:
        assert "context-free generic_purpose request" in str(exc)
    else:
        raise AssertionError("Expected plan normalization to reject detached generic-purpose requests.")


def test_plan_normalizer_rejects_asr_text_binding_into_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Use ASR output structurally.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe the spoken dialogue.",
                arguments={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from the transcript.",
                arguments={"query": "Use the transcript to answer."},
                input_refs=[
                    {"target_field": "text_contexts", "source": {"step_id": 1, "field_path": "text"}},
                ],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    try:
        normalizer.normalize(_task(), plan)
    except ValueError as exc:
        assert "must bind ASR transcript content to generic_purpose via transcripts -> transcripts" in str(exc)
    else:
        raise AssertionError("Expected plan normalization to reject ASR text -> text_contexts binding.")


def test_plan_normalizer_allows_asr_clip_context_into_generic_purpose():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Use ASR-bounded clip and transcript context together.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe the spoken dialogue.",
                arguments={"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 30.0}]},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer using the bounded spoken moment with transcript support.",
                arguments={"query": "Use the bounded spoken moment to answer."},
                input_refs=[
                    {"target_field": "clips", "source": {"step_id": 1, "field_path": "clips"}},
                    {"target_field": "transcripts", "source": {"step_id": 1, "field_path": "transcripts"}},
                ],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[1].input_refs[0].target_field == "clips"
    assert normalized.steps[1].input_refs[1].target_field == "transcripts"


def test_plan_normalizer_rejects_input_ref_binding_into_evidence_ids():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject synthetic evidence-id wiring.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="ocr",
                purpose="Read chart text.",
                arguments={"query": "Read the chart.", "frames": [{"video_id": "sample1", "timestamp_s": 12.0}]},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from OCR.",
                arguments={"query": "Determine the answer."},
                input_refs=[
                    {"target_field": "evidence_ids", "source": {"step_id": 1, "field_path": "backend"}},
                ],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    with pytest.raises(ValueError, match="do not emit bindable evidence ids"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_rejects_non_structural_media_input_ref_bindings():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject non-structural media wiring.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="frame_retriever",
                purpose="Retrieve frames.",
                arguments={"query": "Best frame.", "num_frames": 2},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="ocr",
                purpose="Read the image.",
                arguments={"query": "Read text."},
                input_refs=[
                    {"target_field": "frames", "source": {"step_id": 1, "field_path": "summary"}},
                ],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    with pytest.raises(ValueError, match="must bind frames via a structural frames path"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_allows_structural_indexed_and_derived_media_bindings():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Allow structurally-derived media wiring.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find matching clips.",
                arguments={"query": "Locate the relevant moment.", "top_k": 2},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve a representative frame from the first grounded clip.",
                arguments={"query": "Representative frame.", "num_frames": 1},
                input_refs=[
                    {"target_field": "clips", "source": {"step_id": 1, "field_path": "clips[0]"}},
                ],
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool_name="asr",
                purpose="Transcribe the source clip for the retrieved frame.",
                arguments={"speaker_attribution": True},
                input_refs=[
                    {"target_field": "clips", "source": {"step_id": 2, "field_path": "frames[0].clip"}},
                ],
                depends_on=[2],
            ),
        ],
        refinement_instructions="",
    )

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[1].input_refs[0].source.field_path == "clips[0]"
    assert normalized.steps[2].input_refs[0].source.field_path == "frames[0].clip"


def test_plan_normalizer_rejects_non_textual_text_context_bindings():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject non-text text_contexts wiring.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="ocr",
                purpose="Read chart text.",
                arguments={"query": "Read the chart.", "frames": [{"video_id": "sample1", "timestamp_s": 12.0}]},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from OCR.",
                arguments={"query": "Determine the answer."},
                input_refs=[
                    {"target_field": "text_contexts", "source": {"step_id": 1, "field_path": "backend"}},
                ],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    with pytest.raises(ValueError, match="text_contexts only accepts textual outputs"):
        normalizer.normalize(_task(), plan)


def test_plan_normalizer_keeps_retrieval_query_verbatim():
    normalizer = ExecutionPlanNormalizer(_Registry())
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="Which scene is relevant to the question?",
        options=["Styled like Einstein", "Wearing a helmet"],
        video_path="video.mp4",
    )
    plan = ExecutionPlan(
        strategy="Avoid answer-option leakage.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Locate the relevant moment.",
                arguments={"query": "Locate the relevant man styled like Einstein in the scene."},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    normalized = normalizer.normalize(task, plan)

    assert normalized.steps[0].arguments["query"] == "Locate the relevant man styled like Einstein in the scene."


def test_plan_normalizer_keeps_asr_windows_verbatim_even_with_preprocess_memory():
    normalizer = ExecutionPlanNormalizer(_Registry())
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="Was the brother a premature baby?",
        options=["yes", "no"],
        video_path="video.mp4",
    )
    plan = ExecutionPlan(
        strategy="Ground the answer from audio and speech.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant crash-like sound.",
                arguments={"query": "repeated crashing or banging sounds from waves hitting a ship hull or bow"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="asr",
                purpose="Read the nearby dialogue.",
                arguments={"clips": [{"video_id": "sample1", "start_s": 59.0, "end_s": 116.5}]},
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    normalized = normalizer.normalize(
        task,
        plan,
        preprocess_planning_memory={
            "identity_memory": [{"label": "Mary", "kind": "speaker_id"}],
            "audio_event_memory": [{"label": "crashing or banging sounds"}],
        },
    )

    steps_by_tool = {step.tool_name: step for step in normalized.steps}

    assert steps_by_tool["visual_temporal_grounder"].arguments["query"] == (
        "repeated crashing or banging sounds from waves hitting a ship hull or bow"
    )
    assert steps_by_tool["asr"].arguments["clips"][0]["start_s"] == 59.0
    assert steps_by_tool["asr"].arguments["clips"][0]["end_s"] == 116.5


def test_plan_normalizer_rejects_transcript_payloads_inside_text_contexts():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Reject transcript payload aliases.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Answer from the transcript.",
                arguments={
                    "query": "Use transcript evidence.",
                    "text_contexts": [{"clip": {"video_id": "sample1", "start_s": 0.0, "end_s": 5.0}, "text": "hello"}],
                },
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    try:
        normalizer.normalize(_task(), plan)
    except ValueError as exc:
        assert "transcript payloads via text_contexts" in str(exc)
    else:
        raise AssertionError("Expected plan normalization to reject transcript payloads in text_contexts.")
