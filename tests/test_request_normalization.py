from video_trace_pipeline.schemas import FrameRetrieverRequest, GenericPurposeRequest
from video_trace_pipeline.tools.base import _normalize_request_payload


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
