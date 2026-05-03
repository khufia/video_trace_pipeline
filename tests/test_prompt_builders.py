from video_trace_pipeline.prompts import (
    AUDITOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_auditor_prompt,
    build_planner_prompt,
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


def _removed_block(name: str) -> str:
    return name.replace("-", "_")


def test_build_planner_prompt_omits_removed_context_blocks():
    prompt = build_planner_prompt(
        task=_task(),
        mode="refine",
        audit_feedback={"missing_information": ["exact chart label"], "feedback": "Need label."},
        tool_catalog={
            "frame_retriever": {"request_fields": ["clips", "query", "num_frames"]},
            "ocr": {"request_fields": ["frames", "query"]},
            "generic_purpose": {"request_fields": ["query", "text_contexts"]},
        },
    )

    state_block = _removed_block("TASK" + "-STATE")
    catalog_block = _removed_block("RETRIEVAL" + "-CATALOG")
    context_block = _removed_block("RETRIEVED" + "-CONTEXT")
    assert f"{state_block}:" not in prompt
    assert f"{state_block}_USAGE_NOTE:" not in prompt
    assert f"{catalog_block}:" not in prompt
    assert f"{context_block}:" not in prompt
    assert "%s_EVIDENCE_IDS_AVAILABLE:" % context_block.replace("_CONTEXT", "") not in prompt
    assert "RETRIEVED_FRAME_REFS_AVAILABLE:" not in prompt
    assert "RETRIEVED_FRAME_SEQUENCES_AVAILABLE:" not in prompt
    assert "artifact_context" not in prompt
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
        audit_feedback=None,
        tool_catalog={"generic_purpose": {"request_fields": ["query", "frames", "transcripts"]}},
    )

    assert "QUESTION_STRUCTURE_HINTS:" in prompt
    assert "multi-referent relation question" in prompt
    assert "Resolve that ordinal over the question's full scope" in prompt


def test_build_planner_prompt_adds_complex_temporal_decomposition_hint():
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample_complex_temporal",
        question="Two shots before the second reviewed play, how many times does the spectator wave his flag?",
        options=["A", "B"],
        video_path="video.mp4",
    )

    prompt = build_planner_prompt(
        task=task,
        mode="generate",
        audit_feedback=None,
        tool_catalog={
            "visual_temporal_grounder": {"request_fields": ["query", "top_k"]},
            "generic_purpose": {"request_fields": ["query", "clips"]},
        },
    )

    assert "QUESTION_STRUCTURE_HINTS:" in prompt
    assert "multiple temporal/ordinal operators" in prompt
    assert "primitive/localizable sub-queries" in prompt


def test_build_planner_prompt_includes_previous_evidence_summary():
    prompt = build_planner_prompt(
        task=_task(),
        mode="refine",
        audit_feedback=None,
        tool_catalog={"generic_purpose": {"request_fields": ["query", "evidence_ids"]}},
        evidence_summary={
            "evidence_entry_count": 1,
            "observation_count": 1,
            "evidence_entries": [
                {
                    "evidence_id": "ev_1",
                    "tool_name": "frame_retriever",
                    "status": "candidate",
                    "evidence_text": "Frame at 129.125s was retrieved but no visual state was described.",
                }
            ],
            "recent_observations": [
                {
                    "observation_id": "obs_1",
                    "evidence_id": "ev_1",
                    "subject": "retrieved frame",
                    "predicate": "timestamp",
                    "value": "129.125s",
                }
            ],
        },
    )

    assert "PREVIOUS_EVIDENCE:" in prompt
    assert "ev_1" in prompt
    assert "129.125s" in prompt


def test_planner_system_prompt_documents_new_schema_and_icl_patterns():
    assert "steps: list of {step_id, tool_name, purpose, inputs, input_refs, expected_outputs}" in PLANNER_SYSTEM_PROMPT
    assert "field-keyed object" in PLANNER_SYSTEM_PROMPT
    assert "Omit empty literal fields from `inputs`" in PLANNER_SYSTEM_PROMPT
    assert "Do not emit removed fields" in PLANNER_SYSTEM_PROMPT
    assert "Never invent helper fields such as `query_context`" in PLANNER_SYSTEM_PROMPT
    assert "Pass ASR to generic_purpose through `transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "Never bind `transcripts` from a clip/frame path" in PLANNER_SYSTEM_PROMPT
    assert "never earlier step inputs such as `inputs.transcripts`" in PLANNER_SYSTEM_PROMPT
    assert "run ASR over grounded clips" in PLANNER_SYSTEM_PROMPT
    assert "audio/count question is conditioned on a visible object, action, or state" in PLANNER_SYSTEM_PROMPT
    assert "the visible object/action is the anchor" in PLANNER_SYSTEM_PROMPT
    assert "non-speech audio option comparisons" in PLANNER_SYSTEM_PROMPT
    assert "Non-speech audio option comparison" in PLANNER_SYSTEM_PROMPT
    assert "Visual-conditioned audio/count" in PLANNER_SYSTEM_PROMPT
    assert "relationship or comparison questions" in PLANNER_SYSTEM_PROMPT
    assert "Multi-referent relation/comparison" in PLANNER_SYSTEM_PROMPT
    assert "`visual_temporal_grounder` queries must be primitive/localizable" in PLANNER_SYSTEM_PROMPT
    assert "Do not ask `visual_temporal_grounder` to solve composite temporal logic" in PLANNER_SYSTEM_PROMPT
    assert "ground anchor candidates, ground target-event candidates" in PLANNER_SYSTEM_PROMPT
    assert "Complex visual-temporal ordinal" in PLANNER_SYSTEM_PROMPT
    assert "Example Q, multi-referent relation" in PLANNER_SYSTEM_PROMPT
    assert ("Task" + "-state use:") not in PLANNER_SYSTEM_PROMPT
    assert "Multimodal reasoning use:" in PLANNER_SYSTEM_PROMPT
    assert "Frame retrieval use:" in PLANNER_SYSTEM_PROMPT
    assert "`generic_purpose`, `ocr`, `spatial_grounder`, `asr`, `dense_captioner`, and `audio_temporal_grounder` can consume grounded clips directly" in PLANNER_SYSTEM_PROMPT
    assert "pass clips directly to these tools unless an actual frame artifact is required" in PLANNER_SYSTEM_PROMPT
    assert "Use `frame_retriever` after temporal grounding only when the next step explicitly needs frame artifacts" in PLANNER_SYSTEM_PROMPT
    assert "Do not insert `frame_retriever` before clip-capable tools" in PLANNER_SYSTEM_PROMPT
    assert "Do not use `spatial_grounder` as an OCR cropper" in PLANNER_SYSTEM_PROMPT
    assert "OCR must use complete frames or grounded clips directly" in PLANNER_SYSTEM_PROMPT
    assert "visual_temporal_grounder -> generic_purpose with clips" in PLANNER_SYSTEM_PROMPT
    assert "visual_temporal_grounder -> spatial_grounder with clips" in PLANNER_SYSTEM_PROMPT
    assert "Do not schedule a separate claim-checking tool" in PLANNER_SYSTEM_PROMPT
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
    assert "Bind current-plan outputs into `time_hints` only for explicit timestamp strings" in PLANNER_SYSTEM_PROMPT
    assert "Tool queries must be single-target and modality-specific" in PLANNER_SYSTEM_PROMPT
    assert "`frame_retriever` is temporal-independent" in PLANNER_SYSTEM_PROMPT
    assert "PREVIOUS_EVIDENCE contains text summaries" in PLANNER_SYSTEM_PROMPT
    assert "`sort_order: \"chronological\"` orders the selected returned frames" in PLANNER_SYSTEM_PROMPT
    assert "no point timestamp anchor" in PLANNER_SYSTEM_PROMPT
    assert "Example A, visible full-frame text" in PLANNER_SYSTEM_PROMPT
    assert "Example C, sound trigger" in PLANNER_SYSTEM_PROMPT
    assert "Example C2, non-speech sound-effect option comparison" in PLANNER_SYSTEM_PROMPT
    assert "Example C3, visible-use anchored sound count" in PLANNER_SYSTEM_PROMPT
    assert "Example D2, complex visual-temporal ordinal" in PLANNER_SYSTEM_PROMPT
    assert 'visual_temporal_grounder("the shot two shots before the second reviewed play' in PLANNER_SYSTEM_PROMPT
    assert "Example G2, arithmetic with a missing visual number" in PLANNER_SYSTEM_PROMPT
    assert "Example G3, exact blackboard or visible-letter task" in PLANNER_SYSTEM_PROMPT
    assert "Example K, object state anchored by speech" in PLANNER_SYSTEM_PROMPT
    assert "Example O, retrieved evidence in refine" not in PLANNER_SYSTEM_PROMPT
    assert "Example P, progressive chart frame selection" in PLANNER_SYSTEM_PROMPT
    assert "partially revealed chart" in PLANNER_SYSTEM_PROMPT
    assert "do not pass answer-only generic evidence back as proof" in PLANNER_SYSTEM_PROMPT
    assert "Wiring is not evidence" in PLANNER_SYSTEM_PROMPT
    assert "artifact_context" not in PLANNER_SYSTEM_PROMPT
    assert _removed_block("RETRIEVED" + "-CONTEXT") not in PLANNER_SYSTEM_PROMPT
    assert "RETRIEVED_FRAME_REFS_AVAILABLE" not in PLANNER_SYSTEM_PROMPT
    assert "PREPROCESS_PLANNING_MEMORY" not in PLANNER_SYSTEM_PROMPT


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

    assert f"{_removed_block('TASK' + '-STATE')}:" not in prompt
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
    assert _removed_block("TASK" + "-STATE") not in SYNTHESIZER_SYSTEM_PROMPT


def test_auditor_prompt_has_complex_score_icl_without_evidence_memory():
    prompt = build_auditor_prompt(
        task=_task(),
        trace_package={"final_answer": "A", "inference_steps": []},
        evidence_summary={"observations": []},
    )

    assert f"{_removed_block('TASK' + '-STATE')}:" not in prompt
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
