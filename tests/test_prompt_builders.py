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
            "audit_gaps": ["exact chart label"],
        },
        audit_feedback={"missing_information": ["exact chart label"], "feedback": "Need label."},
        tool_catalog={
            "frame_retriever": {"request_fields": ["clips", "query", "num_frames"]},
            "ocr": {"request_fields": ["frames", "query"]},
            "generic_purpose": {"request_fields": ["query", "text_contexts"]},
            "verifier": {"request_fields": ["query", "claims", "frames", "ocr_results"]},
        },
        retrieval_catalog={"evidence_store": {"evidence_entry_count": 1}},
        task_state={
            "claim_results": [{"claim_id": "claim_1", "status": "unverified", "claim_type": "ocr"}],
            "open_questions": ["exact chart label"],
        },
    )

    assert "RICH_PREPROCESS_SEGMENTS:" in prompt
    assert "TASK_STATE:" in prompt
    assert "TASK_STATE_USAGE_NOTE:" in prompt
    assert "PREPROCESS_TRANSCRIPTS_AVAILABLE:" in prompt
    assert "PREPROCESS_TRANSCRIPTS_USAGE_NOTE:" in prompt
    assert '"transcript_id": "preprocess_seg_001"' in prompt
    assert '"source": "preprocess"' in prompt
    assert "RETRIEVAL_CATALOG:" in prompt
    assert "RETRIEVED_CONTEXT:" in prompt
    assert "RETRIEVED_EVIDENCE_IDS_AVAILABLE:" in prompt
    assert '"ev_a"' in prompt
    assert "artifact_context" not in prompt
    assert "RETRIEVED_FRAME_REFS_AVAILABLE:" not in prompt
    assert "RETRIEVED_FRAME_SEQUENCES_AVAILABLE:" not in prompt
    assert "PREPROCESS_PLANNING_MEMORY" not in prompt
    assert "PREVIOUS_ITERATIONS_SUMMARY" not in prompt
    assert "- never emit arguments, depends_on, use_summary" in prompt


def test_build_planner_prompt_adds_multi_referent_relation_hint():
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample_relation",
        question="What is the relation between the person holding the named object and the person holding the first object?",
        options=["A", "B"],
        video_path="video.mp4",
    )

    prompt = build_planner_prompt(
        task=task,
        mode="generate",
        planner_segments=[],
        retrieved_context={},
        audit_feedback=None,
        tool_catalog={"generic_purpose": {"request_fields": ["query", "frames", "transcripts"]}},
        retrieval_catalog={},
    )

    assert "QUESTION_STRUCTURE_HINTS:" in prompt
    assert "multi-referent relation question" in prompt
    assert "Resolve that ordinal over the question's full scope" in prompt


def test_planner_system_prompt_documents_new_schema_and_icl_patterns():
    assert "steps: list of {step_id, tool_name, purpose, inputs, input_refs, expected_outputs}" in PLANNER_SYSTEM_PROMPT
    assert "field-keyed object" in PLANNER_SYSTEM_PROMPT
    assert "Omit empty literal fields from `inputs`" in PLANNER_SYSTEM_PROMPT
    assert "Do not emit removed fields" in PLANNER_SYSTEM_PROMPT
    assert "Never invent helper fields such as `query_context`" in PLANNER_SYSTEM_PROMPT
    assert "Pass ASR to generic_purpose through `transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "Never bind `transcripts` from a clip/frame path" in PLANNER_SYSTEM_PROMPT
    assert "never earlier step inputs such as `inputs.transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "PREPROCESS_TRANSCRIPTS_AVAILABLE as structured `inputs.transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "Transcript already in preprocessing" in PLANNER_SYSTEM_PROMPT
    assert "run ASR only when transcript coverage is missing or insufficient" in PLANNER_SYSTEM_PROMPT
    assert "audio/count question is conditioned on a visible object, action, or state" in PLANNER_SYSTEM_PROMPT
    assert "the visible object/action is the anchor" in PLANNER_SYSTEM_PROMPT
    assert "non-speech audio option comparisons" in PLANNER_SYSTEM_PROMPT
    assert "Non-speech audio option comparison" in PLANNER_SYSTEM_PROMPT
    assert "Visual-conditioned audio/count" in PLANNER_SYSTEM_PROMPT
    assert "relationship or comparison questions" in PLANNER_SYSTEM_PROMPT
    assert "Multi-referent relation/comparison" in PLANNER_SYSTEM_PROMPT
    assert "Example Q, multi-referent relation" in PLANNER_SYSTEM_PROMPT
    assert "Task-state use:" in PLANNER_SYSTEM_PROMPT
    assert "Verifier use:" in PLANNER_SYSTEM_PROMPT
    assert "finish the chain with `verifier`" in PLANNER_SYSTEM_PROMPT
    assert "Text_contexts alone are not enough for visual-state verification" in PLANNER_SYSTEM_PROMPT
    assert "Avoid generic_purpose -> generic_purpose chains" in PLANNER_SYSTEM_PROMPT
    assert "extraction plus comparison in that single call" in PLANNER_SYSTEM_PROMPT
    assert "Multi-display chart/table comparison" in PLANNER_SYSTEM_PROMPT
    assert "blackboards, whiteboards, visible letters/words" in PLANNER_SYSTEM_PROMPT
    assert "arithmetic over visible/transcribed numbers" in PLANNER_SYSTEM_PROMPT
    assert "Localized visual state" in PLANNER_SYSTEM_PROMPT
    assert "do not preserve the prior anchor by default" in PLANNER_SYSTEM_PROMPT
    assert "Do not downgrade a repeated organization" in PLANNER_SYSTEM_PROMPT
    assert "Example M2, repeated place phrase" in PLANNER_SYSTEM_PROMPT
    assert "Do not bind current-plan outputs into `time_hints`" in PLANNER_SYSTEM_PROMPT
    assert "Example A, visible text region" in PLANNER_SYSTEM_PROMPT
    assert "Example C, sound trigger" in PLANNER_SYSTEM_PROMPT
    assert "Example C2, non-speech sound-effect option comparison" in PLANNER_SYSTEM_PROMPT
    assert "Example C3, visible-use anchored sound count" in PLANNER_SYSTEM_PROMPT
    assert "Example G2, arithmetic with a missing visual number" in PLANNER_SYSTEM_PROMPT
    assert "Example G3, exact blackboard or visible-letter task" in PLANNER_SYSTEM_PROMPT
    assert "Example K, object state anchored by speech" in PLANNER_SYSTEM_PROMPT
    assert "Example O, retrieved evidence in refine" in PLANNER_SYSTEM_PROMPT
    assert "Example P, progressive chart frame selection" in PLANNER_SYSTEM_PROMPT
    assert "partially revealed chart" in PLANNER_SYSTEM_PROMPT
    assert "do not pass answer-only generic evidence back as proof" in PLANNER_SYSTEM_PROMPT
    assert "Wiring is not evidence" in PLANNER_SYSTEM_PROMPT
    assert "artifact_context" not in PLANNER_SYSTEM_PROMPT
    assert "RETRIEVED_FRAME_REFS_AVAILABLE" not in PLANNER_SYSTEM_PROMPT
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
        task_state={"open_questions": ["exact chart label"]},
    )

    assert "TASK_STATE:" in prompt
    assert "RETRIEVAL_CATALOG:" in prompt
    assert "CURRENT_RETRIEVED_CONTEXT:" in prompt
    assert "PlannerRetrievalDecision schema reminder" in prompt
    assert "action: \"ready\" or \"retrieve\"" in prompt
    assert "Example B, retrieve by audit gap and prior timestamp" in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert "Example E, conflicting prior trace timestamp" in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert '"task_state"' in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert "artifact_context" not in PLANNER_RETRIEVAL_SYSTEM_PROMPT
    assert "artifact_ids" not in PLANNER_RETRIEVAL_SYSTEM_PROMPT


def test_synthesizer_prompt_is_one_shot_with_icl_examples():
    prompt = build_synthesizer_prompt(
        task=_task(),
        mode="refine",
        round_evidence_entries=[{"evidence_id": "ev_1", "tool_name": "ocr"}],
        round_observations=[{"observation_id": "obs_1", "subject": "chart"}],
        current_trace={"final_answer": "A"},
        refinement_instructions="Replace the unsupported label claim.",
        audit_feedback={"missing_information": ["exact chart label"]},
        task_state={"claim_results": [{"claim_id": "claim_1", "status": "unknown"}]},
    )

    assert "TASK_STATE:" in prompt
    assert "ROUND_EVIDENCE_ENTRIES:" in prompt
    assert "ROUND_ATOMIC_OBSERVATIONS:" in prompt
    assert "EVIDENCE_MEMORY" not in prompt
    assert "Example D, unresolved fine detail" in SYNTHESIZER_SYSTEM_PROMPT
    assert "Example J, ASR-to-visual anchor" in SYNTHESIZER_SYSTEM_PROMPT
    assert "relationship or comparison questions" in SYNTHESIZER_SYSTEM_PROMPT
    assert "Example K, relation slots" in SYNTHESIZER_SYSTEM_PROMPT
    assert "choose the uniquely best-supported option" in SYNTHESIZER_SYSTEM_PROMPT
    assert "Prefer the longest repeated matching name or phrase" in SYNTHESIZER_SYSTEM_PROMPT
    assert "Do not downgrade a repeated organization" in SYNTHESIZER_SYSTEM_PROMPT
    assert "verifier claim_results and TASK_STATE statuses" in SYNTHESIZER_SYSTEM_PROMPT


def test_auditor_prompt_has_complex_score_icl_without_evidence_memory():
    prompt = build_auditor_prompt(
        task=_task(),
        trace_package={"final_answer": "A", "inference_steps": []},
        evidence_summary={"observations": []},
        task_state={"claim_results": [{"claim_id": "claim_1", "status": "candidate"}]},
    )

    assert "TASK_STATE:" in prompt
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
