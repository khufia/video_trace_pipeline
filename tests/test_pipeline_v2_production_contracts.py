from __future__ import annotations

from types import SimpleNamespace

from video_trace_pipeline.orchestration.context_packer import build_evidence_cards, build_preprocess_context_pack
from video_trace_pipeline.orchestration.request_compiler import compile_planner_tool_request
from video_trace_pipeline.schemas import GenericPurposeOutput, PlannerAction
from video_trace_pipeline.tools.process_adapters import generic_output_quality_flags, generic_output_strict_json_usable


def _task():
    return SimpleNamespace(
        sample_key="sample_1",
        video_id="video_1",
        video_path="/tmp/video.mp4",
        question="After the narrator says Paris, what object is visible first?",
        options=["A. car", "B. plane"],
    )


def test_bare_clip_planner_action_becomes_chronological_frame_request():
    parsed = PlannerAction.model_validate(
        {
            "video_id": "video_13",
            "start_s": 120.0,
            "end_s": 150.0,
            "artifact_id": "clip_120_00_150_00",
            "relpath": "artifacts/video_13/clips/clip_120_00_150_00.mp4",
        }
    )

    assert parsed.action_type == "tool_call"
    assert parsed.tool_name == "frame_retriever"
    assert parsed.tool_request["clips"][0]["start_s"] == 120.0
    assert parsed.tool_request["sequence_mode"] == "chronological"


def test_frame_retriever_event_query_with_bounded_clip_gets_full_chronological_coverage():
    request = {
        "tool_name": "frame_retriever",
        "clips": [{"video_id": "video_12", "start_s": 40.0, "end_s": 70.0}],
        "query": "chronological frames where Ray receives a pass and shoots",
        "num_frames": 4,
    }

    compiled = compile_planner_tool_request("frame_retriever", request, task=_task())
    compiled_request = compiled["tool_request"]

    assert compiled["compiled_valid"]
    assert compiled["tool_name"] == "frame_retriever"
    assert compiled["action"] == "frame_retriever_bounded_event_query_to_chronological_coverage"
    assert compiled_request["clips"] == request["clips"]
    assert compiled_request["sequence_mode"] == "chronological"
    assert compiled_request["sort_order"] == "chronological"
    assert "receives" not in compiled_request["query"]


def test_frame_retriever_unbounded_event_query_reroutes_to_visual_temporal_grounder():
    request = {
        "tool_name": "frame_retriever",
        "query": "when Ray receives a pass and shoots",
        "num_frames": 4,
    }

    compiled = compile_planner_tool_request("frame_retriever", request, task=_task())

    assert compiled["compiled_valid"]
    assert compiled["tool_name"] == "visual_temporal_grounder"
    assert compiled["action"] == "reroute_unbounded_frame_event_query_to_visual_temporal_grounder"


def test_frame_retriever_time_range_hint_becomes_bounded_chronological_clip():
    request = {
        "tool_name": "frame_retriever",
        "time_hints": ["120.0s to 150.0s"],
        "query": "what happens during the interval",
    }

    compiled = compile_planner_tool_request("frame_retriever", request, task=_task())
    compiled_request = compiled["tool_request"]

    assert compiled["compiled_valid"]
    assert compiled_request["clips"] == [{"video_id": "video_1", "start_s": 120.0, "end_s": 150.0}]
    assert compiled_request["time_hints"] == []
    assert compiled_request["sequence_mode"] == "chronological"


def test_ocr_runtime_fields_are_stripped_before_schema_validation():
    request = {
        "tool_name": "ocr",
        "clips": [{"video_id": "video_13", "start_s": 120.0, "end_s": 150.0}],
        "ocr_sample_fps": 2.0,
        "ocr_source": "clip",
    }

    compiled = compile_planner_tool_request("ocr", request, task=_task())

    assert compiled["compiled_valid"]
    assert "ocr_sample_fps" not in compiled["tool_request"]
    assert "ocr_source" not in compiled["tool_request"]
    assert "dropped_unknown_field:ocr_sample_fps" in compiled["diagnostics"]


def test_generic_request_preserves_context_and_adds_typed_contract():
    request = {
        "tool_name": "generic_purpose",
        "query": "Count the visible planes and choose the option.",
        "frames": [{"video_id": "video_1", "timestamp_s": 10.0}],
        "text_contexts": ["evidence card text"],
        "evidence_ids": ["ev_01"],
    }

    compiled = compile_planner_tool_request("generic_purpose", request, task=_task())
    compiled_request = compiled["tool_request"]

    assert compiled["compiled_valid"]
    assert compiled_request["frames"] == request["frames"]
    assert compiled_request["text_contexts"] == request["text_contexts"]
    assert compiled_request["evidence_ids"] == request["evidence_ids"]
    assert "Return JSON only with keys answer, supporting_points, confidence, analysis" in compiled_request["query"]
    assert "evidence_table" not in compiled_request["query"]
    assert "indeterminate" in compiled_request["query"]


def test_preprocess_pack_combines_planner_segments_without_duplicate_dense_caption_channel():
    preprocess_context = {
        "source": "planner_segments.json",
        "cache_dir": "/cache",
        "manifest": {"video_id": "video_1"},
        "planner_segments": [
            {
                "start_s": 0.0,
                "end_s": 10.0,
                "transcript_segments": [{"start_s": 1.0, "end_s": 2.0, "text": "The narrator says Paris."}],
                "dense_caption_summary": "A plane is visible near a gate.",
            },
            {
                "start_s": 10.0,
                "end_s": 20.0,
                "transcript_segments": [{"start_s": 11.0, "end_s": 12.0, "text": "Other narration."}],
                "dense_caption_summary": "A car is visible.",
            },
        ],
    }

    packed = build_preprocess_context_pack(preprocess_context, _task(), target_token_budget=40, max_selected_segments=1)

    assert packed["kind"] == "preprocess_context_pack"
    assert packed["manifest"]["selection_policy"].startswith("task-term-ranked subset")
    assert len(packed["chunks"]) == 1
    assert "The narrator says Paris" in packed["chunks"][0]["asr_text"]
    assert "dense_captions" not in packed["chunks"][0]


def test_preprocess_pack_includes_all_segments_when_context_fits():
    preprocess_context = {
        "source": "planner_segments.json",
        "planner_segments": [
            {"start_s": 0.0, "end_s": 5.0, "transcript": "Paris appears.", "dense_caption_summary": "Plane."},
            {"start_s": 5.0, "end_s": 10.0, "transcript": "Second segment.", "dense_caption_summary": "Car."},
        ],
    }

    packed = build_preprocess_context_pack(preprocess_context, _task(), target_token_budget=10000, max_selected_segments=1)

    assert packed["manifest"]["full_preprocess_included_in_prompt"]
    assert len(packed["chunks"]) == 2


def test_preprocess_pack_reads_nested_planner_segment_cache_shape():
    preprocess_context = {
        "source": "planner_segments.json",
        "planner_segments": [
            {
                "start_s": 0.0,
                "end_s": 30.0,
                "asr": {
                    "transcript_spans": [
                        {"start_s": 0.1, "end_s": 2.0, "text": "Aldi milk costs $2.18."}
                    ]
                },
                "dense_caption": {
                    "overall_summary": "A price graphic shows milk and bread.",
                    "captions": [{"on_screen_text": ["FASTEST GROWING GROCERS IN 2022"]}],
                },
            }
        ],
    }

    packed = build_preprocess_context_pack(preprocess_context, _task(), target_token_budget=10000)

    assert "Aldi milk costs" in packed["chunks"][0]["asr_text"]
    assert "price graphic" in packed["chunks"][0]["dense_caption_text"]
    assert "FASTEST GROWING" in packed["chunks"][0]["ocr_text"]


def test_evidence_cards_include_text_not_only_ids():
    cards = build_evidence_cards(
        entries=[
            {
                "evidence_id": "ev_01",
                "tool_name": "asr",
                "status": "candidate",
                "evidence_text": "Transcript says Paris at 129s.",
                "observation_ids": ["obs_01"],
            }
        ],
        observations=[
            {
                "observation_id": "obs_01",
                "evidence_id": "ev_01",
                "atomic_text": "The transcript contains Paris.",
            }
        ],
    )

    assert cards[0]["evidence_id"] == "ev_01"
    assert "Transcript says Paris" in cards[0]["text"]
    assert "The transcript contains Paris" in cards[0]["text"]


def test_raw_qwen_output_is_marked_untrusted_for_planner_review():
    parsed = GenericPurposeOutput(
        answer="The user wants to identify the third plane. First I inspect the transcript. Wait, the transcript is ambiguous and I need more evidence.",
        supporting_points=[],
        confidence=None,
        analysis="Long raw reasoning text from Qwen.",
    )

    flags = generic_output_quality_flags(parsed, "The user wants to reason aloud.")

    assert "raw_reasoning_or_preamble_detected" in flags
    assert not generic_output_strict_json_usable(parsed, "The user wants to reason aloud.")
