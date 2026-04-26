from video_trace_pipeline.prompts import (
    AUDITOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_auditor_prompt,
    build_planner_prompt,
    build_synthesizer_prompt,
    render_tool_catalog,
)
from video_trace_pipeline.schemas import (
    AgentConfig,
    FrameRetrieverOutput,
    FrameRetrieverRequest,
    ModelsConfig,
    TaskSpec,
    ToolConfig,
)
from video_trace_pipeline.tools.registry import ToolRegistry


def _task():
    return TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="Which chart label has the highest value?",
        options=["A", "B", "C"],
        video_path="video.mp4",
    )


def test_build_planner_prompt_formats_diagnosis_for_original_style_planning():
    prompt = build_planner_prompt(
        task=_task(),
        mode="refine",
        planner_segments=[
            {
                "start_s": 0.0,
                "end_s": 12.0,
                "dense_caption_spans": [
                    {
                        "start_s": 2.0,
                        "end_s": 10.0,
                        "visual": "A person shows several charts on screen.",
                        "on_screen_text": ["Sales", "Growth"],
                    }
                ],
                "transcript_spans": [
                    {
                        "start_s": 4.0,
                        "end_s": 6.0,
                        "text": "This slide compares the latest numbers.",
                    }
                ],
            }
        ],
        compact_rounds=[],
        retrieved_observations=[
            {
                "observation_id": "obs_1",
                "subject": "speaker",
                "predicate": "said",
                "object_text": "Safety is important.",
                "evidence_id": "ev_a",
            },
            {
                "observation_id": "obs_2",
                "subject": "speaker",
                "predicate": "said",
                "object_text": "No door.",
                "evidence_id": "ev_b",
            },
            {
                "observation_id": "obs_3",
                "subject": "speaker",
                "predicate": "said",
                "object_text": "Safety is important.",
                "evidence_id": "ev_a",
            },
        ],
        preprocess_planning_memory={
            "identity_memory": [{"label": "Phil", "kind": "speaker_id", "time_ranges": [{"start_s": 12.0, "end_s": 15.0}]}],
            "audio_event_memory": [{"label": "metallic bang", "time_ranges": [{"start_s": 14.0, "end_s": 15.0}]}],
        },
        audit_feedback={
            "feedback": "  Need the exact chart label and value.  ",
            "scores": {"logical_coherence": 2.0, "completeness": 1.0},
            "findings": [
                {
                    "severity": "LOW",
                    "category": "INFERENCE_ERROR",
                    "message": "The answer overstates the evidence.",
                    "evidence_ids": ["ev_b", "ev_a"],
                },
                {
                    "severity": "HIGH",
                    "category": "READING_GAP",
                    "message": "The trace never reads the exact chart label.",
                    "evidence_ids": ["ev_c"],
                },
            ],
            "missing_information": [" winning series value ", "exact chart label", "exact chart label"],
        },
        tool_catalog={
            "frame_retriever": {"request_fields": ["clips", "query", "num_frames"]},
            "ocr": {"request_fields": ["frames", "query"]},
            "generic_purpose": {"request_fields": ["query", "text_contexts"]},
        },
    )

    assert "DIAGNOSIS:" in prompt
    assert '"exact chart label"' in prompt
    assert '"winning series value"' in prompt
    assert prompt.index('"winning series value"') < prompt.index('"exact chart label"')
    assert prompt.index('"severity": "HIGH"') < prompt.index('"severity": "LOW"')
    assert prompt.index('"ev_a"') < prompt.index('"ev_b"')
    assert "PREPROCESS_PLANNING_MEMORY:" in prompt
    assert '"label": "Phil"' in prompt
    assert "PREPROCESS_SEGMENTS:" in prompt
    assert "VIDEO_CAPTION_SUMMARY:" not in prompt
    assert "exact-anchor continuity memory only" in prompt
    assert "CURRENT_TRACE_CUES:" not in prompt
    assert "- use_summary: boolean" not in prompt
    assert "they are not automatically complete support" in prompt
    assert "RETRIEVED_EVIDENCE_IDS_AVAILABLE:" in prompt
    assert '"ev_a"' in prompt
    assert '"ev_b"' in prompt
    assert prompt.count('"ev_a"') >= 2
    assert "copy one or more of these exact evidence_ids into the step arguments" in prompt


def test_planner_system_prompt_uses_original_style_repair_decomposition():
    assert "Your job is NOT to answer the question and NOT to rewrite the trace." in PLANNER_SYSTEM_PROMPT
    assert "Use `DIAGNOSIS.missing_information` as the canonical ordered gap list when it is present." in PLANNER_SYSTEM_PROMPT
    assert "There is no alias repair or post-processing." in PLANNER_SYSTEM_PROMPT
    assert "visual_temporal_grounder -> frame_retriever -> spatial_grounder" in PLANNER_SYSTEM_PROMPT
    assert "visual_temporal_grounder -> frame_retriever -> spatial_grounder -> ocr" in PLANNER_SYSTEM_PROMPT
    assert "visual_temporal_grounder -> frame_retriever -> generic_purpose" in PLANNER_SYSTEM_PROMPT
    assert "animated or evolving chart/table reading" in PLANNER_SYSTEM_PROMPT
    assert "asr -> frame_retriever -> spatial_grounder -> generic_purpose" in PLANNER_SYSTEM_PROMPT
    assert "Prefer it over OCR as the primary interpretation tool for charts/tables" in PLANNER_SYSTEM_PROMPT
    assert "compares frames across the bounded clip" in PLANNER_SYSTEM_PROMPT
    assert "pass `transcripts`, not flattened `text_contexts`" in PLANNER_SYSTEM_PROMPT
    assert "Do not bind current-plan outputs into `evidence_ids`" in PLANNER_SYSTEM_PROMPT
    assert "If you want `generic_purpose` to reason over previously retrieved observations" in PLANNER_SYSTEM_PROMPT
    assert "Use only literal reusable `evidence_ids`" in PLANNER_SYSTEM_PROMPT
    assert "`generic_purpose` cannot be the first step unless" in PLANNER_SYSTEM_PROMPT
    assert "If the question asks for a total across repeated occurrences" in PLANNER_SYSTEM_PROMPT
    assert "match the semantic target of the option" in PLANNER_SYSTEM_PROMPT
    assert "PREPROCESS_PLANNING_MEMORY" in PLANNER_SYSTEM_PROMPT
    assert "CURRENT_TRACE_CUES" not in PLANNER_SYSTEM_PROMPT
    assert "dialogue_claim_memory" not in PLANNER_SYSTEM_PROMPT
    assert "timeline_alignment_memory" not in PLANNER_SYSTEM_PROMPT
    assert "use_summary" not in PLANNER_SYSTEM_PROMPT


def test_build_auditor_prompt_mentions_atomic_missing_information_contract():
    prompt = build_auditor_prompt(
        task=_task(),
        trace_package={"final_answer": "A", "inference_steps": []},
        evidence_summary={"observations": []},
    )

    assert "ordered, deduplicated, tool-agnostic list of atomic unresolved answer-critical needs" in prompt
    assert "`missing_information` is the planner-facing canonical gap list." in AUDITOR_SYSTEM_PROMPT
    assert "object presence but not an answer-critical state" in AUDITOR_SYSTEM_PROMPT
    assert "Near-synonymous sound labels or repeated phases of the same action" in AUDITOR_SYSTEM_PROMPT
    assert 'free-form non-option answer such' in AUDITOR_SYSTEM_PROMPT
    assert 'Do not ask the planner to' in AUDITOR_SYSTEM_PROMPT
    assert '"declare ambiguity"' in AUDITOR_SYSTEM_PROMPT


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
    assert "request_schema: clips: List[ClipRef]; query: Optional[str]; num_frames: int" in rendered
    assert "output_nested: frames[] -> frame_path: str, timestamp_s: float, relevance_score: Optional[float]" in rendered


def test_tool_registry_catalog_exposes_request_and_output_fields():
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

    assert catalog["frame_retriever"]["request_fields"]
    assert "clips" in catalog["frame_retriever"]["request_fields"]
    assert catalog["frame_retriever"]["output_fields"]
    assert "frames" in catalog["frame_retriever"]["output_fields"]
    assert catalog["frame_retriever"]["request_schema"]
    assert "clips: List[ClipRef]" in catalog["frame_retriever"]["request_schema"]
    assert catalog["frame_retriever"]["output_schema"]
    assert "frames: List[RetrievedFrame]" in catalog["frame_retriever"]["output_schema"]
    assert any(line.startswith("clips[] -> ") for line in catalog["frame_retriever"]["request_nested"])
    assert any(line.startswith("frames[] -> ") for line in catalog["frame_retriever"]["output_nested"])


def test_synthesizer_system_prompt_is_closer_to_original_refiner_discipline():
    assert "SURGICAL EDITS, NOT REWRITES." in SYNTHESIZER_SYSTEM_PROMPT
    assert 'free-form text like "ambiguous/non-unique"' in SYNTHESIZER_SYSTEM_PROMPT


def test_synthesizer_prompt_is_round_local_and_audit_directed():
    prompt = build_synthesizer_prompt(
        task=_task(),
        mode="refine",
        round_evidence_entries=[{"evidence_id": "ev_1", "tool_name": "ocr"}],
        round_observations=[{"observation_id": "obs_1", "subject": "chart"}],
        current_trace={"final_answer": "A"},
        refinement_instructions="Replace the unsupported label claim.",
        audit_feedback={"missing_information": ["exact chart label"]},
    )

    assert "PRIOR_AUDIT_DIAGNOSIS:" in prompt
    assert "ROUND_EVIDENCE_ENTRIES:" in prompt
    assert "ROUND_ATOMIC_OBSERVATIONS:" in prompt
    assert "TRACE_WRITING_REQUIREMENTS:" not in prompt


def test_prompt_size_metrics_stay_bounded():
    planner_words = len(PLANNER_SYSTEM_PROMPT.split())
    synthesizer_words = len(SYNTHESIZER_SYSTEM_PROMPT.split())
    auditor_words = len(AUDITOR_SYSTEM_PROMPT.split())

    assert planner_words < 2600
    assert synthesizer_words < 900
    assert auditor_words < 1700
