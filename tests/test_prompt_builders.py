from video_trace_pipeline.prompts import (
    AUDITOR_SYSTEM_PROMPT,
    PLANNER_RETRIEVAL_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_auditor_prompt,
    build_planner_prompt,
    build_planner_retrieval_prompt,
    build_synthesizer_prompt,
    render_frame_sequence_context,
    render_tool_catalog,
)
from video_trace_pipeline.schemas import AgentConfig, FrameRetrieverOutput, FrameRetrieverRequest, ModelsConfig, TaskSpec, ToolConfig
from video_trace_pipeline.tools.registry import ToolRegistry


def _task():
    return TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="Which chart label has the highest value?",
        options=["A", "B", "C"],
        video_path="video.mp4",
    )


def test_build_planner_prompt_uses_rich_preprocess_and_retrieved_context():
    prompt = build_planner_prompt(
        task=_task(),
        mode="refine",
        planner_segments=[
            {
                "segment_id": "seg_001",
                "dense_caption": {
                    "overall_summary": "A person shows charts.",
                    "clips": [{"video_id": "video1", "start_s": 0.0, "end_s": 12.0}],
                    "captions": [{"visual": "A chart is visible.", "attributes": ["camera_state: static"]}],
                },
                "asr": {"transcript_spans": [{"text": "This slide compares the latest numbers."}]},
            }
        ],
        retrieved_context={
            "observations": [{"observation_id": "obs_1", "evidence_id": "ev_a", "atomic_text": "OCR reads Sales."}],
            "artifact_context": [
                {
                    "artifact_id": "frame_010.00",
                    "artifact_type": "frame",
                    "relpath": "artifacts/sample1/frames/frame_010.00.png",
                    "time": {"timestamp_s": 10.0},
                    "contains": ["A readable chart is visible."],
                },
                {
                    "artifact_id": "frame_011.00",
                    "artifact_type": "frame",
                    "relpath": "artifacts/sample1/frames/frame_011.00.png",
                    "time": {"timestamp_s": 11.0},
                    "contains": ["The same readable chart is fully visible."],
                }
            ],
            "audit_gaps": ["exact chart label"],
        },
        audit_feedback={"missing_information": ["exact chart label"], "feedback": "Need label."},
        tool_catalog={
            "frame_retriever": {"request_fields": ["clips", "query", "num_frames"]},
            "ocr": {"request_fields": ["frames", "query"]},
            "generic_purpose": {"request_fields": ["query", "text_contexts"]},
        },
        retrieval_catalog={"evidence_store": {"evidence_entry_count": 1}},
    )

    assert "RICH_PREPROCESS_SEGMENTS:" in prompt
    assert "PREPROCESS_TRANSCRIPTS_AVAILABLE:" in prompt
    assert "PREPROCESS_TRANSCRIPTS_USAGE_NOTE:" in prompt
    assert '"transcript_id": "preprocess_seg_001"' in prompt
    assert '"source": "preprocess"' in prompt
    assert "RETRIEVAL_CATALOG:" in prompt
    assert "RETRIEVED_CONTEXT:" in prompt
    assert "RETRIEVED_EVIDENCE_IDS_AVAILABLE:" in prompt
    assert "RETRIEVED_FRAME_REFS_AVAILABLE:" in prompt
    assert "RETRIEVED_FRAME_SEQUENCES_AVAILABLE:" in prompt
    assert '"ev_a"' in prompt
    assert '"artifact_id": "frame_010.00"' in prompt
    assert '"timestamp_s": 10.0' in prompt
    assert '"first_frame"' in prompt
    assert '"latest_frame"' in prompt
    assert '"chronological_frames"' in prompt
    assert '"artifact_id": "frame_011.00"' in prompt
    assert "candidate animated/progressive reveals" in prompt
    assert "artifact timestamps/relpaths beat prior trace prose" in prompt
    assert "PREPROCESS_PLANNING_MEMORY" not in prompt
    assert "PREVIOUS_ITERATIONS_SUMMARY" not in prompt
    assert "- never emit arguments, depends_on, use_summary" in prompt


def test_planner_system_prompt_documents_new_schema_and_icl_patterns():
    assert "steps: list of {step_id, tool_name, purpose, inputs, input_refs, expected_outputs}" in PLANNER_SYSTEM_PROMPT
    assert "field-keyed object" in PLANNER_SYSTEM_PROMPT
    assert "Do not emit removed fields" in PLANNER_SYSTEM_PROMPT
    assert "Pass ASR to generic_purpose through `transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "PREPROCESS_TRANSCRIPTS_AVAILABLE as structured `inputs.transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "Transcript already in preprocessing" in PLANNER_SYSTEM_PROMPT
    assert "run ASR only when transcript coverage is missing or insufficient" in PLANNER_SYSTEM_PROMPT
    assert "Do not bind current-plan outputs into `time_hints`" in PLANNER_SYSTEM_PROMPT
    assert "Example A, visible text region" in PLANNER_SYSTEM_PROMPT
    assert "Example C, sound trigger" in PLANNER_SYSTEM_PROMPT
    assert "Example K, object state anchored by speech" in PLANNER_SYSTEM_PROMPT
    assert "Example O, retrieved artifact frames in refine" in PLANNER_SYSTEM_PROMPT
    assert "Example P, progressive chart frame selection" in PLANNER_SYSTEM_PROMPT
    assert "Artifact timing and frame reuse" in PLANNER_SYSTEM_PROMPT
    assert "partially revealed chart" in PLANNER_SYSTEM_PROMPT
    assert "latest complete frame per display" in PLANNER_SYSTEM_PROMPT
    assert "RETRIEVED_FRAME_SEQUENCES_AVAILABLE groups adjacent retrieved frames" in PLANNER_SYSTEM_PROMPT
    assert "Choose frames by task semantics" in PLANNER_SYSTEM_PROMPT
    assert "first/earliest questions need first_frame plus neighbors" in PLANNER_SYSTEM_PROMPT
    assert "Wiring is not evidence" in PLANNER_SYSTEM_PROMPT
    assert "PREPROCESS_PLANNING_MEMORY" not in PLANNER_SYSTEM_PROMPT


def test_build_planner_retrieval_prompt_exposes_catalog_and_schema():
    prompt = build_planner_retrieval_prompt(
        task=_task(),
        mode="refine",
        retrieval_catalog={"preprocess": {"planner_segment_count": 3}, "evidence_store": {"evidence_entry_count": 2}},
        retrieved_context={"observations": [{"observation_id": "obs_1", "atomic_text": "OCR reads Sales."}]},
        audit_feedback={"missing_information": ["exact chart label"]},
        tool_catalog={"ocr": {"request_fields": ["frames", "query"]}},
        iteration=1,
        max_iterations=3,
    )

    assert "RETRIEVAL_CATALOG:" in prompt
    assert "CURRENT_RETRIEVED_CONTEXT:" in prompt
    assert "PlannerRetrievalDecision schema reminder" in prompt
    assert "action: \"ready\" or \"retrieve\"" in prompt
    assert "Example B, retrieve by audit gap and prior timestamp" in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert "Example E, conflicting prior trace timestamp" in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert "artifact-context times and ids" in PLANNER_RETRIEVAL_SYSTEM_PROMPT


def test_synthesizer_prompt_is_one_shot_with_icl_examples():
    prompt = build_synthesizer_prompt(
        task=_task(),
        mode="refine",
        round_evidence_entries=[{"evidence_id": "ev_1", "tool_name": "ocr"}],
        round_observations=[{"observation_id": "obs_1", "subject": "chart"}],
        current_trace={"final_answer": "A"},
        refinement_instructions="Replace the unsupported label claim.",
        audit_feedback={"missing_information": ["exact chart label"]},
    )

    assert "ROUND_EVIDENCE_ENTRIES:" in prompt
    assert "ROUND_ATOMIC_OBSERVATIONS:" in prompt
    assert "EVIDENCE_MEMORY" not in prompt
    assert "Example D, unresolved fine detail" in SYNTHESIZER_SYSTEM_PROMPT
    assert "Example J, ASR-to-visual anchor" in SYNTHESIZER_SYSTEM_PROMPT
    assert "choose the uniquely best-supported option" in SYNTHESIZER_SYSTEM_PROMPT


def test_auditor_prompt_has_complex_score_icl_without_evidence_memory():
    prompt = build_auditor_prompt(
        task=_task(),
        trace_package={"final_answer": "A", "inference_steps": []},
        evidence_summary={"observations": []},
    )

    assert "ordered, deduplicated, tool-agnostic list of atomic unresolved answer-critical needs" in prompt
    assert "diagnostics: object" in prompt
    assert "Example A, strong multimodal PASS" in AUDITOR_SYSTEM_PROMPT
    assert "Example J, truncated task" in AUDITOR_SYSTEM_PROMPT
    assert "EVIDENCE_MEMORY" not in prompt


def test_render_tool_catalog_includes_output_fields():
    rendered = render_tool_catalog(
        {
            "frame_retriever": {
                "description": "Retrieve bounded frames.",
                "model": "test-model",
                "request_fields": ["clips", "query", "num_frames"],
                "output_fields": ["query", "frames", "mode", "rationale"],
                "request_schema": ["clips: List[ClipRef]", "query: Optional[str]", "num_frames: int"],
                "output_schema": ["query: Optional[str]", "frames: List[RetrievedFrame]", "mode: str"],
                "request_nested": ["clips[] -> video_id: str, start_s: float, end_s: float"],
                "output_nested": ["frames[] -> frame_path: str, timestamp_s: float, relevance_score: Optional[float]"],
            }
        }
    )

    assert "args=clips, query, num_frames" in rendered
    assert "outputs=query, frames, mode, rationale" in rendered
    assert "output_nested: frames[] -> frame_path: str, timestamp_s: float, relevance_score: Optional[float]" in rendered


def test_render_frame_sequence_context_mentions_anchor_and_neighbors():
    rendered = render_frame_sequence_context(
        [
            {
                "timestamp_s": 6.0,
                "metadata": {
                    "requested_timestamp_s": 6.0,
                    "neighbor_radius_s": 2.0,
                    "sequence_mode": "anchor_window",
                    "sequence_role": "anchor",
                    "sequence_index": 1,
                },
            }
        ]
    )

    assert "chronological sequence centered on timestamp 6s" in rendered
    assert "neighboring frames before and after" in rendered


def test_tool_registry_catalog_exposes_new_asr_outputs():
    class _Adapter(object):
        request_model = FrameRetrieverRequest
        output_model = FrameRetrieverOutput

    registry = ToolRegistry.__new__(ToolRegistry)
    registry.models_config = ModelsConfig(
        agents={"planner": AgentConfig(backend="openai", model="gpt-5.4")},
        tools={"frame_retriever": ToolConfig(enabled=True, model="test-model", description="Retrieve frames.")},
    )
    registry.adapters = {"frame_retriever": _Adapter()}

    catalog = registry.tool_catalog()

    assert "clips" in catalog["frame_retriever"]["request_fields"]
    assert "frames" in catalog["frame_retriever"]["output_fields"]
    assert "sequence_mode: Literal[ranked, anchor_window, chronological]" in catalog["frame_retriever"]["request_schema"]
