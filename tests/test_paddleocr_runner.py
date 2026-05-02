import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from video_trace_pipeline.tool_wrappers import paddleocr_runner


def test_paddleocr_runner_uses_paddle_engine_for_single_request(monkeypatch, tmp_path):
    emitted = {}
    seen = {}

    class _FakeEngine:
        pass

    monkeypatch.setattr(
        paddleocr_runner,
        "load_request",
        lambda: {"request": {"query": "read digits"}, "task": {}, "runtime": {"extra": {"use_textline_orientation": False}}},
    )
    monkeypatch.setattr(paddleocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "ensure_frame_for_request", lambda *args, **kwargs: (str(tmp_path / "frame.png"), 1.25))
    monkeypatch.setattr(paddleocr_runner, "_prepare_ocr_image", lambda frame_path, out_dir, *, max_longest_dim: frame_path)
    monkeypatch.setattr(paddleocr_runner, "create_paddleocr_engine", lambda runtime: _FakeEngine())
    monkeypatch.setattr(
        paddleocr_runner,
        "run_paddleocr_image",
        lambda engine, image_path, *, use_textline_orientation: seen.update(
            {"engine": engine, "image_path": image_path, "use_textline_orientation": use_textline_orientation}
        )
        or [{"text": "42", "bbox": None, "confidence": 0.9}],
    )
    monkeypatch.setattr(paddleocr_runner, "emit_json", lambda payload: emitted.update(payload))

    paddleocr_runner.main()

    assert isinstance(seen["engine"], _FakeEngine)
    assert seen["use_textline_orientation"] is False
    assert emitted["text"] == "42"
    assert emitted["lines"][0]["text"] == "42"
    assert emitted["backend"] == "paddleocr"


def test_paddleocr_runner_returns_empty_result_when_no_text_is_found(monkeypatch, tmp_path):
    emitted = {}

    monkeypatch.setattr(
        paddleocr_runner,
        "load_request",
        lambda: {"request": {"query": "read chart"}, "task": {}, "runtime": {"extra": {"use_textline_orientation": True}}},
    )
    monkeypatch.setattr(paddleocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "ensure_frame_for_request", lambda *args, **kwargs: (str(tmp_path / "frame.png"), 1.25))
    monkeypatch.setattr(paddleocr_runner, "_prepare_ocr_image", lambda frame_path, out_dir, *, max_longest_dim: frame_path)
    monkeypatch.setattr(paddleocr_runner, "create_paddleocr_engine", lambda runtime: object())
    monkeypatch.setattr(paddleocr_runner, "run_paddleocr_image", lambda engine, image_path, *, use_textline_orientation: [])
    monkeypatch.setattr(paddleocr_runner, "emit_json", lambda payload: emitted.update(payload))

    paddleocr_runner.main()

    assert emitted["text"] == ""
    assert emitted["lines"] == []
    assert emitted["backend"] == "paddleocr"


def test_create_paddleocr_engine_uses_cpu_directly_when_requested_gpu_is_unavailable(monkeypatch, tmp_path):
    attempts = []

    class _FakePaddleOCR:
        def __init__(self, **kwargs):
            attempts.append(dict(kwargs))

    monkeypatch.setattr(paddleocr_runner, "_configure_paddleocr_environment", lambda runtime: tmp_path)
    monkeypatch.setattr(
        paddleocr_runner,
        "_probe_paddleocr_device",
        lambda device_label: {
            "requested_device": device_label,
            "available": False,
            "reason": "gpu_runtime_unavailable",
        },
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=_FakePaddleOCR))

    engine = paddleocr_runner.create_paddleocr_engine({"extra": {"device": "gpu:2", "enable_mkldnn": False}})

    assert isinstance(engine, _FakePaddleOCR)
    assert len(attempts) == 1
    assert attempts[0]["device"] == "cpu"
    assert attempts[0]["enable_mkldnn"] is False


def test_create_paddleocr_engine_fails_when_gpu_is_required_but_unavailable(monkeypatch, tmp_path):
    captured = {}

    def _fake_fail_runtime(message, extra=None):
        captured["message"] = message
        captured["extra"] = dict(extra or {})
        raise RuntimeError(message)

    monkeypatch.setattr(paddleocr_runner, "_configure_paddleocr_environment", lambda runtime: tmp_path)
    monkeypatch.setattr(
        paddleocr_runner,
        "_probe_paddleocr_device",
        lambda device_label: {
            "requested_device": device_label,
            "available": False,
            "reason": "gpu_runtime_unavailable",
        },
    )
    monkeypatch.setattr(paddleocr_runner, "fail_runtime", _fake_fail_runtime)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=object))

    try:
        paddleocr_runner.create_paddleocr_engine({"extra": {"device": "gpu:2", "require_gpu": True}})
    except RuntimeError as exc:
        assert "require GPU execution" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected create_paddleocr_engine() to fail when GPU is required but unavailable.")

    assert captured["extra"]["hint"] == "Install a GPU-enabled `paddlepaddle-gpu` runtime in this environment."


def test_prepare_single_request_uses_region_frame_source(monkeypatch, tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color="white").save(frame_path)

    def _unexpected_ensure_frame(*args, **kwargs):
        raise AssertionError("ensure_frame_for_request should not run when region.frame already resolves.")

    monkeypatch.setattr(paddleocr_runner, "ensure_frame_for_request", _unexpected_ensure_frame)

    prepared_frame_path, source_frame_path, timestamp_s, query = paddleocr_runner._prepare_single_request(
        {
            "query": "read scoreboard",
            "regions": [
                {
                    "bbox": [207.0, 906.0, 795.0, 977.0],
                    "frame": {
                        "video_id": "video-1",
                        "timestamp_s": 127.0,
                        "metadata": {"source_path": str(frame_path)},
                    },
                }
            ],
        },
        task={},
        runtime={"workspace_root": str(tmp_path), "extra": {"max_longest_dim": 1600}},
        frame_out_dir=tmp_path,
    )

    assert source_frame_path == str(frame_path.resolve())
    assert timestamp_s == 127.0
    assert query == "read scoreboard"
    with Image.open(prepared_frame_path) as cropped:
        assert cropped.size == (196, 24)


def test_paddleocr_runner_samples_clip_at_default_fps_and_batches_predict(monkeypatch, tmp_path):
    emitted = {}
    sample_calls = []
    predict_calls = []
    sampled_paths = []

    class _FakeEngine:
        def predict(self, *, input, use_textline_orientation):
            predict_calls.append({"input": input, "use_textline_orientation": use_textline_orientation})
            assert isinstance(input, list)
            return [
                {
                    "rec_texts": [Path(path).stem],
                    "rec_scores": [0.9],
                    "dt_polys": [None],
                }
                for path in input
            ]

    def _fake_sample_frames(video_path, start_s, end_s, num_frames, out_dir, prefix):
        sample_calls.append(
            {
                "video_path": video_path,
                "start_s": start_s,
                "end_s": end_s,
                "num_frames": num_frames,
                "prefix": prefix,
            }
        )
        frames = []
        for index in range(num_frames):
            frame_path = tmp_path / ("clip_frame_%02d.jpg" % index)
            frame_path.touch()
            sampled_paths.append(str(frame_path.resolve()))
            frames.append({"frame_path": str(frame_path), "timestamp_s": 10.0 + (index * 0.5)})
        return frames

    def _unexpected_ensure_frame(*args, **kwargs):
        raise AssertionError("clip OCR should sample clip frames directly")

    monkeypatch.setattr(
        paddleocr_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "read rink text",
                "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 13.0}],
            },
            "task": {"video_path": "/tmp/video.mp4", "video_id": "video-1"},
            "runtime": {"extra": {}},
        },
    )
    monkeypatch.setattr(paddleocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "sample_frames", _fake_sample_frames)
    monkeypatch.setattr(paddleocr_runner, "ensure_frame_for_request", _unexpected_ensure_frame)
    monkeypatch.setattr(paddleocr_runner, "_prepare_ocr_image", lambda frame_path, out_dir, *, max_longest_dim: frame_path)
    monkeypatch.setattr(paddleocr_runner, "create_paddleocr_engine", lambda runtime: _FakeEngine())
    monkeypatch.setattr(paddleocr_runner, "emit_json", lambda payload: emitted.update(payload))

    paddleocr_runner.main()

    assert sample_calls == [
        {
            "video_path": "/tmp/video.mp4",
            "start_s": 10.0,
            "end_s": 13.0,
            "num_frames": 6,
            "prefix": "ocr_clip_01",
        }
    ]
    assert len(predict_calls) == 1
    assert predict_calls[0]["input"] == sampled_paths
    assert len(emitted["results"]) == 6
    assert emitted["results"][0]["clip"]["start_s"] == 10.0
    assert emitted["results"][0]["frame"]["clip"]["end_s"] == 13.0
    assert emitted["results"][0]["frame"]["metadata"]["ocr_sample_fps"] == 2.0
    assert emitted["results"][0]["timestamp_s"] == 10.0
    assert emitted["backend"] == "paddleocr"


def test_paddleocr_runner_honors_runtime_fps_for_clip_sampling(monkeypatch, tmp_path):
    emitted = {}
    sample_calls = []

    class _FakeEngine:
        def predict(self, *, input, use_textline_orientation):
            return [{"rec_texts": [Path(path).stem], "rec_scores": [0.9]} for path in input]

    def _fake_sample_frames(video_path, start_s, end_s, num_frames, out_dir, prefix):
        del video_path, start_s, end_s, out_dir, prefix
        sample_calls.append(num_frames)
        frames = []
        for index in range(num_frames):
            frame_path = tmp_path / ("fps_frame_%02d.jpg" % index)
            frame_path.touch()
            frames.append({"frame_path": str(frame_path), "timestamp_s": 20.0 + index})
        return frames

    monkeypatch.setattr(
        paddleocr_runner,
        "load_request",
        lambda: {
            "request": {"clips": [{"video_id": "video-1", "start_s": 20.0, "end_s": 23.0}]},
            "task": {"video_path": "/tmp/video.mp4", "video_id": "video-1"},
            "runtime": {"extra": {"fps": 1.0}},
        },
    )
    monkeypatch.setattr(paddleocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "sample_frames", _fake_sample_frames)
    monkeypatch.setattr(paddleocr_runner, "_prepare_ocr_image", lambda frame_path, out_dir, *, max_longest_dim: frame_path)
    monkeypatch.setattr(paddleocr_runner, "create_paddleocr_engine", lambda runtime: _FakeEngine())
    monkeypatch.setattr(paddleocr_runner, "emit_json", lambda payload: emitted.update(payload))

    paddleocr_runner.main()

    assert sample_calls == [3]
    assert len(emitted["results"]) == 3


def test_run_paddleocr_batch_falls_back_when_list_predict_fails():
    calls = []

    class _FakeEngine:
        def predict(self, *, input, use_textline_orientation):
            calls.append(input)
            if isinstance(input, list):
                raise RuntimeError("batch unsupported")
            return [{"rec_texts": [Path(input).stem], "rec_scores": [0.8]}]

    lines = paddleocr_runner.run_paddleocr_batch(
        _FakeEngine(),
        ["/tmp/one.png", "/tmp/two.png"],
        use_textline_orientation=False,
    )

    assert calls == [["/tmp/one.png", "/tmp/two.png"], "/tmp/one.png", "/tmp/two.png"]
    assert [item[0]["text"] for item in lines] == ["one", "two"]


def test_run_paddleocr_batch_uses_legacy_ocr_entrypoint_when_predict_is_missing():
    calls = []

    class _FakeEngine:
        def ocr(self, image_path, use_textline_orientation):
            calls.append({"image_path": image_path, "use_textline_orientation": use_textline_orientation})
            return [[None, [Path(image_path).stem, 0.7]]]

    lines = paddleocr_runner.run_paddleocr_batch(
        _FakeEngine(),
        ["/tmp/alpha.png", "/tmp/beta.png"],
        use_textline_orientation=True,
    )

    assert calls == [
        {"image_path": "/tmp/alpha.png", "use_textline_orientation": True},
        {"image_path": "/tmp/beta.png", "use_textline_orientation": True},
    ]
    assert [item[0]["text"] for item in lines] == ["alpha", "beta"]


def test_create_paddleocr_engine_sets_default_recognition_batch_size(monkeypatch, tmp_path):
    attempts = []

    class _FakePaddleOCR:
        def __init__(self, **kwargs):
            attempts.append(dict(kwargs))

    monkeypatch.setattr(paddleocr_runner, "_configure_paddleocr_environment", lambda runtime: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "_probe_paddleocr_device", lambda device_label: {"requested_device": device_label, "available": True})
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=_FakePaddleOCR))

    paddleocr_runner.create_paddleocr_engine({"extra": {"device": "cpu", "textline_orientation_batch_size": 4}})

    assert attempts[0]["text_recognition_batch_size"] == 16
    assert attempts[0]["textline_orientation_batch_size"] == 4
