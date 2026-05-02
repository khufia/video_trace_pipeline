import pytest

from video_trace_pipeline.schemas import ClipRef, FrameRef, FrameRetrieverRequest, GenericPurposeRequest, OCRRequest, SpatialGrounderRequest
from video_trace_pipeline.tools.base import ToolAdapter, _validate_request_payload
from video_trace_pipeline.tools.process_adapters import (
    GenericPurposeProcessAdapter,
    OCRProcessAdapter,
    SpatialGrounderProcessAdapter,
)


class _GenericPurposeAdapter(ToolAdapter):
    name = "generic_purpose"
    request_model = GenericPurposeRequest


class _OCRAdapter(ToolAdapter):
    name = "ocr"
    request_model = OCRRequest


def test_validate_request_payload_rejects_noncanonical_flat_clip_fields():
    with pytest.raises(ValueError, match="Unexpected request field"):
        _validate_request_payload(
            {
                "video_id": "video-1",
                "clip_start_s": 1.25,
                "clip_end_s": 4.5,
                "query": "look for the label",
            },
            FrameRetrieverRequest,
        )


def test_validate_request_payload_preserves_canonical_list_fields():
    payload = _validate_request_payload(
        {
            "query": "answer from OCR",
            "text_contexts": ["Alaska"],
            "evidence_ids": ["ocr-step-3"],
        },
        GenericPurposeRequest,
    )

    assert payload["text_contexts"] == ["Alaska"]
    assert payload["evidence_ids"] == ["ocr-step-3"]


@pytest.mark.parametrize(
    "removed_field, value",
    [
        ("sequence_mode", "anchor_window"),
        ("neighbor_radius_s", 2.0),
        ("include_anchor_neighbors", True),
        ("sort_order", "chronological"),
    ],
)
def test_frame_retriever_request_rejects_removed_fields(removed_field, value):
    with pytest.raises(ValueError, match="Unexpected request field"):
        _validate_request_payload(
            {
                "clips": [{"video_id": "video-1", "start_s": 0.0, "end_s": 10.0}],
                "time_hints": ["00:06"],
                removed_field: value,
            },
            FrameRetrieverRequest,
        )


def test_visual_adapters_append_frame_sequence_context(monkeypatch):
    frame = FrameRef(
        video_id="video-1",
        timestamp_s=6.0,
        metadata={"relevance_score": 0.9},
    )

    generic_seen = {}
    generic = GenericPurposeProcessAdapter(name="generic_purpose", model_name="qwen")
    monkeypatch.setattr(
        generic,
        "_run_json",
        lambda context, request_payload: generic_seen.update(request_payload)
        or ({"answer": "ok", "analysis": "ok", "supporting_points": [], "confidence": 0.7}, "{}"),
    )
    generic.execute(
        GenericPurposeRequest(tool_name="generic_purpose", query="What happens?", frames=[frame]),
        context=None,
    )
    assert "Frame timestamps: 6s" in generic_seen["text_contexts"][0]
    parsed_generic = generic.parse_request({"tool_name": "generic_purpose", "query": "What happens?", "frames": [frame.dict()]})
    assert "Frame timestamps: 6s" in parsed_generic.text_contexts[0]

    ocr_seen = {}
    ocr = OCRProcessAdapter(name="ocr", model_name="ocr")
    monkeypatch.setattr(
        ocr,
        "_run_json",
        lambda context, request_payload: ocr_seen.update(request_payload)
        or ({"text": "", "lines": [], "query": request_payload.get("query")}, "{}"),
    )
    ocr.execute(OCRRequest(tool_name="ocr", query="Read the label.", frames=[frame]), context=None)
    assert "Frame sequence context:" in ocr_seen["query"]
    parsed_ocr = ocr.parse_request({"tool_name": "ocr", "query": "Read the label.", "frames": [frame.dict()]})
    assert "Frame sequence context:" in parsed_ocr.query

    spatial_seen = {}
    spatial = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")
    monkeypatch.setattr(
        spatial,
        "_run_json",
        lambda context, request_payload: spatial_seen.update(request_payload)
        or (
            {
                "query": request_payload.get("query"),
                "detections": [],
                "spatial_description": "",
            },
            "{}",
        ),
    )
    spatial.execute(SpatialGrounderRequest(tool_name="spatial_grounder", query="Find the player.", frames=[frame]), context=None)
    assert "Frame sequence context:" in spatial_seen["query"]
    parsed_spatial = spatial.parse_request({"tool_name": "spatial_grounder", "query": "Find the player.", "frames": [frame.dict()]})
    assert "Frame sequence context:" in parsed_spatial.query


def test_spatial_grounder_request_and_adapter_accept_clips(monkeypatch):
    clip = ClipRef(video_id="video-1", start_s=10.0, end_s=15.0)
    frame = FrameRef(video_id="video-1", timestamp_s=12.0, clip=clip)

    clip_request = SpatialGrounderRequest(tool_name="spatial_grounder", query="Find the player.", clips=[clip])
    assert clip_request.clips == [clip]
    assert clip_request.frames == []

    frame_request = SpatialGrounderRequest(
        tool_name="spatial_grounder",
        query="Find the player.",
        clips=[clip],
        frames=[frame],
    )
    assert frame_request.frames == [frame]
    assert frame_request.clips == []

    spatial = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")
    parsed = spatial.parse_request(
        {
            "tool_name": "spatial_grounder",
            "query": "Find the player.",
            "clips": [clip.dict()],
        }
    )
    assert parsed.clips[0].start_s == 10.0

    seen = {}
    monkeypatch.setattr(
        spatial,
        "_run_json",
        lambda context, request_payload: seen.update(request_payload)
        or (
            {
                "query": request_payload.get("query"),
                "timestamp_s": 12.5,
                "detections": [{"label": "player", "bbox": [1, 2, 3, 4], "confidence": 0.9}],
                "spatial_description": "player is visible",
                "backend": "fake",
            },
            "{}",
        ),
    )

    result = spatial.execute(clip_request, context=None)

    assert seen["clips"][0]["start_s"] == 10.0
    assert result.data["frames"][0]["timestamp_s"] == 12.5
    assert result.data["frames"][0]["clip"]["start_s"] == 10.0
    assert result.data["regions"][0]["frame"]["timestamp_s"] == 12.5


def test_tool_adapter_rejects_noncanonical_argument_alias():
    with pytest.raises(ValueError, match="Unexpected request field"):
        _GenericPurposeAdapter().parse_request(
            {
                "prompt": "answer from OCR",
                "text_contexts": ["Alaska"],
            }
        )


def test_generic_purpose_request_requires_canonical_transcript_refs():
    with pytest.raises(Exception, match="transcript_id"):
        _GenericPurposeAdapter().parse_request(
            {
                "query": "answer from transcript",
                "transcripts": [
                    {
                        "clip": {"video_id": "video-1", "start_s": 0.0, "end_s": 4.0},
                        "text": "The label says Alaska.",
                        "segments": [],
                        "metadata": {"backend": "whisperx_local"},
                    }
                ],
            }
        )


def test_ocr_request_drops_clip_fields_when_frames_are_present():
    request = _OCRAdapter().parse_request(
        {
            "query": "read the percentages",
            "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 15.0}],
            "frames": [
                {
                    "video_id": "video-1",
                    "timestamp_s": 12.0,
                    "clip": {"video_id": "video-1", "start_s": 10.0, "end_s": 15.0},
                    "metadata": {"source_path": "frame_00.png"},
                },
                {
                    "video_id": "video-1",
                    "timestamp_s": 13.0,
                    "clip": {"video_id": "video-1", "start_s": 10.0, "end_s": 15.0},
                    "metadata": {"source_path": "frame_01.png"},
                },
            ],
        }
    )

    assert len(request.frames) == 2
    assert request.clips == []
