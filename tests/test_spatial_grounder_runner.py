from contextlib import contextmanager

from PIL import Image

from video_trace_pipeline.schemas import ClipRef, FrameRef, SpatialGrounderRequest
from video_trace_pipeline.tools.process_adapters import SpatialGrounderProcessAdapter
from video_trace_pipeline.tool_wrappers import spatial_grounder_runner


def _patch_runner_setup(monkeypatch, calls, raw_text):
    class FakeRunner:
        def __init__(self, **kwargs):
            calls["runner_kwargs"] = dict(kwargs)

        def generate(self, messages, *, max_new_tokens):
            calls["messages"] = messages
            calls["max_new_tokens"] = max_new_tokens
            return raw_text

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(spatial_grounder_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(spatial_grounder_runner, "resolve_model_path", lambda model_name, runtime: "/models/qwen")
    monkeypatch.setattr(spatial_grounder_runner, "resolved_device_label", lambda runtime: "cuda:0")


def test_spatial_grounder_clip_input_uses_video_message(monkeypatch):
    calls = {}
    _patch_runner_setup(
        monkeypatch,
        calls,
        '{"timestamp_s": 12.25, "detections": [{"label": "player", "bbox": [1, 2, 3, 4], "confidence": 0.9}], "spatial_description": "player found"}',
    )

    @contextmanager
    def fake_extracted_clip(video_path, start_s, end_s, *, include_audio=False, **kwargs):
        calls["extracted_clip"] = {
            "video_path": video_path,
            "start_s": start_s,
            "end_s": end_s,
            "include_audio": include_audio,
        }
        yield "/tmp/spatial_clip.mp4"

    def fake_video_message(prompt, video_path, fps=2.0):
        calls["video_message"] = {"prompt": prompt, "video_path": video_path, "fps": fps}
        return [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt}]}]

    def fail_image_message(*args, **kwargs):
        raise AssertionError("clip inputs should not use image messages")

    def fail_ensure_frame(*args, **kwargs):
        raise AssertionError("clip inputs should not be converted to sampled frames")

    monkeypatch.setattr(spatial_grounder_runner, "extracted_clip", fake_extracted_clip)
    monkeypatch.setattr(spatial_grounder_runner, "make_qwen_video_message", fake_video_message)
    monkeypatch.setattr(spatial_grounder_runner, "make_qwen_image_messages", fail_image_message)
    monkeypatch.setattr(spatial_grounder_runner, "ensure_frame_for_request", fail_ensure_frame)

    result = spatial_grounder_runner.execute_payload(
        {
            "request": {
                "query": "the answer-critical player",
                "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 15.0}],
            },
            "task": {"video_path": "/videos/full.mp4", "video_id": "video-1"},
            "runtime": {
                "model_name": "Qwen/Qwen3.5-9B",
                "device": "cuda:0",
                "extra": {"fps": 1.5, "max_new_tokens": 96},
            },
        }
    )

    assert calls["extracted_clip"] == {
        "video_path": "/videos/full.mp4",
        "start_s": 10.0,
        "end_s": 15.0,
        "include_audio": False,
    }
    assert calls["video_message"]["video_path"] == "/tmp/spatial_clip.mp4"
    assert calls["video_message"]["fps"] == 1.5
    assert "original full video" in calls["video_message"]["prompt"]
    assert "Do not write prose outside the JSON object" in calls["video_message"]["prompt"]
    assert "bbox must be [x1, y1, x2, y2] in video-frame pixel coordinates" in calls["video_message"]["prompt"]
    assert "absent, occluded, or not in the requested relation" in calls["video_message"]["prompt"]
    assert calls["max_new_tokens"] == 96
    assert calls["closed"] is True
    assert result["timestamp_s"] == 12.25
    assert result["source_frame_path"] is None
    assert result["detections"] == [
        {"label": "player", "bbox": [1.0, 2.0, 3.0, 4.0], "confidence": 0.9, "metadata": {}}
    ]


def test_spatial_grounder_clip_local_timestamp_is_offset(monkeypatch):
    calls = {}
    _patch_runner_setup(
        monkeypatch,
        calls,
        '{"timestamp_s": 2.0, "detections": [], "spatial_description": "target visible"}',
    )

    @contextmanager
    def fake_extracted_clip(*args, **kwargs):
        yield "/tmp/spatial_clip.mp4"

    monkeypatch.setattr(spatial_grounder_runner, "extracted_clip", fake_extracted_clip)

    result = spatial_grounder_runner.execute_payload(
        {
            "request": {
                "query": "the player",
                "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 15.0}],
            },
            "task": {"video_path": "/videos/full.mp4", "video_id": "video-1"},
            "runtime": {"model_name": "Qwen/Qwen3.5-9B", "device": "cuda:0", "extra": {}},
        }
    )

    assert result["timestamp_s"] == 12.0


def test_spatial_grounder_clip_missing_timestamp_uses_midpoint(monkeypatch):
    calls = {}
    _patch_runner_setup(
        monkeypatch,
        calls,
        '{"detections": [], "spatial_description": "target visible"}',
    )

    @contextmanager
    def fake_extracted_clip(*args, **kwargs):
        yield "/tmp/spatial_clip.mp4"

    monkeypatch.setattr(spatial_grounder_runner, "extracted_clip", fake_extracted_clip)

    result = spatial_grounder_runner.execute_payload(
        {
            "request": {
                "query": "the player",
                "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 16.0}],
            },
            "task": {"video_path": "/videos/full.mp4", "video_id": "video-1"},
            "runtime": {"model_name": "Qwen/Qwen3.5-9B", "device": "cuda:0", "extra": {}},
        }
    )

    assert result["timestamp_s"] == 13.0


def test_spatial_grounder_frame_input_stays_image_mode(monkeypatch, tmp_path):
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (20, 10), color="white").save(frame_path)
    calls = {}
    _patch_runner_setup(
        monkeypatch,
        calls,
        '{"detections": [{"label": "player", "bbox": [1, 2, 30, 12], "confidence": 0.7}], "spatial_description": "player found"}',
    )

    def fake_image_message(prompt, image_paths):
        calls["image_message"] = {"prompt": prompt, "image_paths": list(image_paths)}
        return [{"role": "user", "content": [{"type": "image", "image": str(image_paths[0])}]}]

    def fail_video_message(*args, **kwargs):
        raise AssertionError("frame inputs should not use video messages")

    def fail_extracted_clip(*args, **kwargs):
        raise AssertionError("frame inputs should not extract video clips")

    monkeypatch.setattr(spatial_grounder_runner, "make_qwen_image_messages", fake_image_message)
    monkeypatch.setattr(spatial_grounder_runner, "make_qwen_video_message", fail_video_message)
    monkeypatch.setattr(spatial_grounder_runner, "extracted_clip", fail_extracted_clip)

    result = spatial_grounder_runner.execute_payload(
        {
            "request": {
                "query": "the player",
                "frames": [
                    {
                        "video_id": "video-1",
                        "timestamp_s": 8.0,
                        "metadata": {"source_path": str(frame_path)},
                    }
                ],
            },
            "task": {"video_path": "/videos/full.mp4", "video_id": "video-1"},
            "runtime": {"model_name": "Qwen/Qwen3.5-9B", "device": "cuda:0", "extra": {}},
        }
    )

    assert calls["image_message"]["image_paths"] == [str(frame_path.resolve())]
    assert "20x10 pixels" in calls["image_message"]["prompt"]
    assert "Do not write prose outside the JSON object" in calls["image_message"]["prompt"]
    assert "Every bbox must fit inside the actual image coordinate system" in calls["image_message"]["prompt"]
    assert result["timestamp_s"] == 8.0
    assert result["source_frame_path"] == str(frame_path.resolve())
    assert result["detections"][0]["bbox"] == [1.0, 2.0, 20.0, 10.0]


def test_spatial_grounder_adapter_clip_input_builds_virtual_frame_and_region(monkeypatch):
    adapter = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")
    calls = []

    def fake_run_json(context, request_payload):
        del context
        calls.append(request_payload)
        return (
            {
                "query": request_payload["query"],
                "timestamp_s": 12.25,
                "detections": [
                    {
                        "label": "player",
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "confidence": 0.9,
                    }
                ],
                "spatial_description": "player found",
                "source_frame_path": None,
                "backend": "qwen",
            },
            "{}",
        )

    monkeypatch.setattr(adapter, "_run_json", fake_run_json)
    clip = ClipRef(video_id="video-1", start_s=10.0, end_s=15.0)

    result = adapter.execute(
        SpatialGrounderRequest(tool_name="spatial_grounder", query="find player", clips=[clip]),
        context=None,
    )

    assert len(calls) == 1
    assert calls[0]["clips"][0]["start_s"] == 10.0
    assert result.data["frames"][0]["timestamp_s"] == 12.25
    assert result.data["frames"][0]["clip"]["start_s"] == 10.0
    assert result.data["regions"][0]["frame"]["clip"]["end_s"] == 15.0
    assert result.data["regions"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_spatial_grounder_adapter_clip_missing_timestamp_uses_midpoint(monkeypatch):
    adapter = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")

    monkeypatch.setattr(
        adapter,
        "_run_json",
        lambda context, request_payload: (
            {
                "query": request_payload["query"],
                "detections": [],
                "spatial_description": "candidate checked",
                "source_frame_path": None,
                "backend": "qwen",
            },
            "{}",
        ),
    )
    clip = ClipRef(video_id="video-1", start_s=10.0, end_s=16.0)

    result = adapter.execute(
        SpatialGrounderRequest(tool_name="spatial_grounder", query="find player", clips=[clip]),
        context=None,
    )

    assert result.data["frames"][0]["timestamp_s"] == 13.0
    assert result.data["frames"][0]["clip"]["start_s"] == 10.0


def test_spatial_grounder_adapter_multi_clip_fans_out_per_clip(monkeypatch):
    adapter = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")
    calls = []

    def fake_run_json(context, request_payload):
        del context
        calls.append(request_payload)
        clip = request_payload["clips"][0]
        timestamp_s = float(clip["start_s"]) + 1.0
        return (
            {
                "query": request_payload["query"],
                "timestamp_s": timestamp_s,
                "detections": [],
                "spatial_description": "checked %.1f" % float(clip["start_s"]),
                "source_frame_path": None,
                "backend": "qwen",
            },
            "{}",
        )

    monkeypatch.setattr(adapter, "_run_json", fake_run_json)
    clips = [
        ClipRef(video_id="video-1", start_s=10.0, end_s=15.0),
        ClipRef(video_id="video-1", start_s=20.0, end_s=25.0),
    ]

    result = adapter.execute(
        SpatialGrounderRequest(tool_name="spatial_grounder", query="find player", clips=clips),
        context=None,
    )

    assert [call["clips"][0]["start_s"] for call in calls] == [10.0, 20.0]
    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [11.0, 21.0]
    assert [frame["clip"]["start_s"] for frame in result.data["frames"]] == [10.0, 20.0]


def test_spatial_grounder_adapter_frame_input_still_uses_frame_ref(monkeypatch):
    adapter = SpatialGrounderProcessAdapter(name="spatial_grounder", model_name="qwen")
    frame = FrameRef(video_id="video-1", timestamp_s=8.0)

    monkeypatch.setattr(
        adapter,
        "_run_json",
        lambda context, request_payload: (
            {
                "query": request_payload["query"],
                "timestamp_s": 8.0,
                "detections": [],
                "spatial_description": "frame checked",
                "source_frame_path": None,
                "backend": "qwen",
            },
            "{}",
        ),
    )

    result = adapter.execute(
        SpatialGrounderRequest(tool_name="spatial_grounder", query="find player", frames=[frame]),
        context=None,
    )

    assert result.data["frames"] == [frame.dict()]
