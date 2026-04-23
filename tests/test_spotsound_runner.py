import json

from video_trace_pipeline.tool_wrappers import spotsound_runner


def test_intervals_from_response_supports_json_payload_and_clip_offset():
    raw = json.dumps(
        [
            {"start": 1.5, "end": 3.0},
            {"start_s": 4.0, "end_s": 5.5},
        ]
    )

    intervals = spotsound_runner._intervals_from_response(
        raw,
        clip_start_s=10.0,
        clip_end_s=20.0,
    )

    assert intervals == [(11.5, 13.0), (14.0, 15.5)]


def test_execute_payload_shapes_spotsound_output(monkeypatch):
    cleanup_calls = []

    monkeypatch.setattr(
        spotsound_runner,
        "_prepare_audio_input",
        lambda request, task, runtime: {
            "audio_path": "/tmp/spotsound.wav",
            "cleanup_path": "/tmp/spotsound.wav",
            "video_id": "video-13",
            "clip_start_s": 30.0,
            "clip_end_s": 45.0,
        },
    )
    monkeypatch.setattr(
        spotsound_runner,
        "_run_spotsound_inference",
        lambda *, query, audio_path, runtime: (
            "The query appears from 2.5 to 4.0 seconds.",
            {
                "repo_checkout": "/tmp/SpotSound",
                "base_model_path": "/models/audio-flamingo-3-hf",
                "adapter_path": "/models/spotsound",
                "device_label": "cuda:2",
            },
        ),
    )
    monkeypatch.setattr(spotsound_runner, "cleanup_temp_path", lambda path: cleanup_calls.append(path))

    payload = {
        "request": {"query": "spoken commercial phrase mentioning Bill's Ammunition"},
        "task": {"video_id": "video-13"},
        "runtime": {},
    }

    result = spotsound_runner.execute_payload(payload)

    assert result["query"] == "spoken commercial phrase mentioning Bill's Ammunition"
    assert result["retrieval_backend"] == "spotsound"
    assert result["query_absent"] is False
    assert result["clips"] == [
        {
            "video_id": "video-13",
            "start_s": 32.5,
            "end_s": 34.0,
            "confidence": None,
            "metadata": {
                "event_label": "spoken commercial phrase mentioning Bill's Ammunition",
                "tool_backend": "spotsound",
                "request_clip_start_s": 30.0,
                "request_clip_end_s": 45.0,
            },
        }
    ]
    assert result["events"] == [
        {
            "event_label": "spoken commercial phrase mentioning Bill's Ammunition",
            "start_s": 32.5,
            "end_s": 34.0,
            "confidence": None,
            "metadata": {
                "tool_backend": "spotsound",
                "request_clip_start_s": 30.0,
                "request_clip_end_s": 45.0,
            },
        }
    ]
    assert "32.500-34.000s" in result["summary"]
    assert cleanup_calls == ["/tmp/spotsound.wav"]
