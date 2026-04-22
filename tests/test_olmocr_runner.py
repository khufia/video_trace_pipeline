from video_trace_pipeline.tool_wrappers import olmocr_runner


def test_olmocr_runner_requests_slow_processor(monkeypatch, tmp_path):
    captured = {}
    emitted = {}

    monkeypatch.setattr(olmocr_runner, "load_request", lambda: {"request": {"query": "read digits"}, "task": {}, "runtime": {}})
    monkeypatch.setattr(olmocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(olmocr_runner, "ensure_frame_for_request", lambda *args, **kwargs: (str(tmp_path / "frame.png"), 1.25))
    monkeypatch.setattr(olmocr_runner, "_prepare_olmocr_image", lambda frame_path, out_dir: frame_path)
    monkeypatch.setattr(olmocr_runner, "resolve_model_path", lambda *args, **kwargs: "/tmp/olmocr-model")
    monkeypatch.setattr(olmocr_runner, "resolved_device_label", lambda runtime: "cpu")
    monkeypatch.setattr(
        olmocr_runner,
        "run_qwen_style_messages",
        lambda **kwargs: captured.update(kwargs) or '{"text":"42","lines":[{"text":"42","bbox":null,"confidence":0.9}]}',
    )
    monkeypatch.setattr(olmocr_runner, "emit_json", lambda payload: emitted.update(payload))

    olmocr_runner.main()

    assert captured["processor_use_fast"] is False
    assert "processor_model_path" not in captured
    assert captured["generate_do_sample"] is False
    assert emitted["text"] == "42"
    assert emitted["lines"][0]["text"] == "42"


def test_olmocr_runner_returns_empty_result_when_no_text_is_found(monkeypatch, tmp_path):
    emitted = {}

    monkeypatch.setattr(olmocr_runner, "load_request", lambda: {"request": {"query": "read chart"}, "task": {}, "runtime": {}})
    monkeypatch.setattr(olmocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(olmocr_runner, "ensure_frame_for_request", lambda *args, **kwargs: (str(tmp_path / "frame.png"), 1.25))
    monkeypatch.setattr(olmocr_runner, "_prepare_olmocr_image", lambda frame_path, out_dir: frame_path)
    monkeypatch.setattr(olmocr_runner, "resolve_model_path", lambda *args, **kwargs: "/tmp/olmocr-model")
    monkeypatch.setattr(olmocr_runner, "resolved_device_label", lambda runtime: "cpu")
    monkeypatch.setattr(olmocr_runner, "run_qwen_style_messages", lambda **kwargs: "")
    monkeypatch.setattr(olmocr_runner, "emit_json", lambda payload: emitted.update(payload))

    olmocr_runner.main()

    assert emitted["text"] == ""
    assert emitted["lines"] == []
    assert emitted["backend"] == "olmocr_transformers"
