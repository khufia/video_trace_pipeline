from video_trace_pipeline.orchestration.plan_normalizer import ExecutionPlanNormalizer
from video_trace_pipeline.schemas import ExecutionPlan, FrameRetrieverRequest, GenericPurposeRequest, OCRRequest, PlanStep, TaskSpec, VisualTemporalGrounderRequest


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
                arguments={"k": 2, "prompt": "Best chart frame."},
                input_refs=[
                    {"target_field": "clip", "source": {"step_id": 1, "field_path": "clips"}},
                ],
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool_name="ocr",
                purpose=" Read the chart. ",
                arguments={"prompt": "Extract chart text."},
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


def test_plan_normalizer_maps_aliases_and_coerces_plural_fields_to_lists():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Normalize aliases.",
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

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[0].arguments == {"query": "Answer.", "text_contexts": ["line 1", "line 2"]}


def test_plan_normalizer_preserves_additional_plural_aliases():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Normalize generic-purpose aliases.",
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

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[0].arguments == {
        "evidence_ids": ["ev_1"],
        "query": "Answer.",
        "text_contexts": ["line 1"],
    }


def test_plan_normalizer_promotes_time_hint_bindings_to_time_hints():
    normalizer = ExecutionPlanNormalizer(_Registry())
    plan = ExecutionPlan(
        strategy="Normalize frame retriever bindings.",
        use_summary=True,
        steps=[
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

    normalized = normalizer.normalize(_task(), plan)

    assert normalized.steps[0].input_refs[0].target_field == "time_hints"
