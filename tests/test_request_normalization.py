import pytest

from video_trace_pipeline.schemas import FrameRetrieverRequest, GenericPurposeRequest, OCRRequest
from video_trace_pipeline.tools.base import ToolAdapter, _validate_request_payload


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
