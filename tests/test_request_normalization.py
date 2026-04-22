from video_trace_pipeline.schemas import FrameRetrieverRequest, GenericPurposeRequest, OCRRequest
from video_trace_pipeline.tools.base import ToolAdapter, _normalize_request_payload


class _GenericPurposeAdapter(ToolAdapter):
    name = "generic_purpose"
    request_model = GenericPurposeRequest


class _OCRAdapter(ToolAdapter):
    name = "ocr"
    request_model = OCRRequest


def test_normalize_request_payload_builds_clip_from_flat_bounds():
    payload = _normalize_request_payload(
        {
            "video_id": "video-1",
            "clip_start_s": 1.25,
            "clip_end_s": 4.5,
            "query": "look for the label",
        },
        FrameRetrieverRequest,
    )

    assert payload["clip"]["video_id"] == "video-1"
    assert payload["clip"]["start_s"] == 1.25
    assert payload["clip"]["end_s"] == 4.5


def test_normalize_request_payload_coerces_scalar_list_fields():
    payload = _normalize_request_payload(
        {
            "query": "answer from OCR",
            "text_contexts": "Alaska\n\nAlaska",
            "evidence_ids": "ocr-step-3",
        },
        GenericPurposeRequest,
    )

    assert payload["text_contexts"] == ["Alaska\n\nAlaska"]
    assert payload["evidence_ids"] == ["ocr-step-3"]


def test_generic_purpose_request_moves_scalar_transcript_text_into_text_contexts():
    request = _GenericPurposeAdapter().parse_request(
        {
            "query": "answer from ASR fallback",
            "transcript": "ASR unavailable: weights-only failure",
        }
    )

    assert request.transcript is None
    assert request.transcripts == []
    assert request.text_contexts == ["ASR unavailable: weights-only failure"]


def test_generic_purpose_request_synthesizes_transcript_ref_for_asr_style_dict():
    request = _GenericPurposeAdapter().parse_request(
        {
            "query": "answer from transcript",
            "transcript": {
                "clip": {"video_id": "video-1", "start_s": 0.0, "end_s": 4.0},
                "text": "The label says Alaska.",
                "segments": [],
                "backend": "whisperx_local",
            },
        }
    )

    assert request.transcript is not None
    assert request.transcript.transcript_id.startswith("tx_")
    assert request.transcript.text == "The label says Alaska."
    assert request.transcript.metadata["backend"] == "whisperx_local"
    assert len(request.transcripts) == 1


def test_ocr_request_drops_clip_fields_when_frames_are_present():
    request = _OCRAdapter().parse_request(
        {
            "query": "read the percentages",
            "clip": {"video_id": "video-1", "start_s": 10.0, "end_s": 15.0},
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
    assert request.clip is None
    assert request.clips == []
