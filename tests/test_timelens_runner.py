from video_trace_pipeline.tool_wrappers import timelens_runner


def test_timelens_runner_uses_full_video_and_configured_device_map(monkeypatch, tmp_path):
    video_path = tmp_path / "full_video.mp4"
    video_path.write_bytes(b"not-a-real-video")
    calls = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            calls["runner_kwargs"] = dict(kwargs)

        def generate(self, messages, *, max_new_tokens):
            calls["messages"] = messages
            calls["max_new_tokens"] = max_new_tokens
            return '{"found": true, "intervals": [{"start_s": 5, "end_s": 7, "confidence": 0.8}]}'

        def close(self):
            calls["closed"] = True

    def fail_iter_windows(*args, **kwargs):
        raise AssertionError("visual temporal grounder should not split the video into windows")

    monkeypatch.setattr(timelens_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(timelens_runner, "iter_windows", fail_iter_windows)
    monkeypatch.setattr(
        timelens_runner,
        "extracted_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not extract clips")),
        raising=False,
    )
    monkeypatch.setattr(timelens_runner, "resolve_model_path", lambda model_name, runtime: "/models/timelens")
    monkeypatch.setattr(timelens_runner, "resolved_device_label", lambda runtime: "cuda:0")
    monkeypatch.setattr(timelens_runner, "get_video_duration", lambda path: 360.0)

    result = timelens_runner.execute_payload(
        {
            "request": {"query": "the answer-critical visual event", "top_k": 3},
            "task": {"video_path": str(video_path), "video_id": "video-1"},
            "runtime": {
                "model_name": "TencentARC/TimeLens-8B",
                "device": "cuda:0",
                "extra": {
                    "fps": 2.0,
                    "max_new_tokens": 128,
                    "device_map": "balanced_cuda:0,1",
                },
            },
        }
    )

    assert calls["runner_kwargs"]["device_map"] == "balanced_cuda:0,1"
    assert calls["messages"][0]["content"][0]["video"] == str(video_path)
    assert "full video" in calls["messages"][0]["content"][1]["text"]
    assert "Return at most 3 intervals" in calls["messages"][0]["content"][1]["text"]
    assert "each interval must satisfy all query predicates" in calls["messages"][0]["content"][1]["text"]
    assert "aftermath or celebration" in calls["messages"][0]["content"][1]["text"]
    assert calls["max_new_tokens"] == 128
    assert calls["closed"] is True
    assert result["clips"] == [
        {
            "video_id": "video-1",
            "start_s": 5.0,
            "end_s": 7.0,
            "confidence": 0.8,
            "metadata": {
                "tool_backend": "timelens_transformers",
                "model_path": "/models/timelens",
            },
        }
    ]
    assert result["prefilter"]["enabled"] is False
    assert result["prefilter"]["reason"] == "full_video_mode"
