from video_trace_pipeline.schemas import FrameRetrieverRequest
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
