from video_trace_pipeline.prompts import (
    AUDITOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_auditor_prompt,
    build_planner_prompt,
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
        summary_text="A person shows several charts on screen.",
        compact_rounds=[],
        retrieved_observations=[],
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
    assert "Use `missing_information` inside DIAGNOSIS as the canonical ordered repair-target list" in prompt
    assert "VIDEO_CAPTION_SUMMARY:" in prompt


def test_planner_system_prompt_uses_original_style_repair_decomposition():
    assert "Read DIAGNOSIS as a repair specification, not just a warning." in PLANNER_SYSTEM_PROMPT
    assert "Create the fewest tool calls that directly resolve the diagnosed evidence gaps." in PLANNER_SYSTEM_PROMPT
    assert "For identical prompt inputs" not in PLANNER_SYSTEM_PROMPT
    assert "Consult `AVAILABLE_TOOLS` for canonical argument names, top-level output fields, and the dynamically rendered request/output schemas." in PLANNER_SYSTEM_PROMPT
    assert "Use it when explicit text must be read or detected from a grounded frame or region." in PLANNER_SYSTEM_PROMPT
    assert "prefer `generic_purpose` for the frame analysis itself" in PLANNER_SYSTEM_PROMPT
    assert "common pattern is `visual_temporal_grounder -> frame_retriever -> generic_purpose`" in PLANNER_SYSTEM_PROMPT


def test_build_auditor_prompt_mentions_atomic_missing_information_contract():
    prompt = build_auditor_prompt(
        task=_task(),
        trace_package={"final_answer": "A", "inference_steps": []},
        evidence_summary={"observations": []},
    )

    assert "ordered, deduplicated, tool-agnostic list of atomic unresolved answer-critical needs" in prompt
    assert "`missing_information` is the planner-facing canonical gap list." in AUDITOR_SYSTEM_PROMPT


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
    assert "This pipeline splits evidence from reasoning:" in SYNTHESIZER_SYSTEM_PROMPT
    assert "The final `TracePackage` must stand on its own." in SYNTHESIZER_SYSTEM_PROMPT
