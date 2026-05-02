import json
from pathlib import Path
from types import SimpleNamespace

from video_trace_pipeline.schemas import ArtifactRef
from video_trace_pipeline.tool_wrappers import paddleocr_runner
from video_trace_pipeline.tools.process_adapters import OCRProcessAdapter


def test_paddleocr_runner_reuses_one_engine_for_multi_frame_requests(monkeypatch, tmp_path):
    emitted = {}
    created_engines = []

    class _FakeEngine:
        def __init__(self):
            self.calls = []
            created_engines.append(self)

    def _fake_run_paddleocr_image(engine, image_path, *, use_textline_orientation):
        del use_textline_orientation
        engine.calls.append({"image_path": image_path})
        image_name = Path(str(image_path)).stem
        return [{"text": image_name, "bbox": None, "confidence": 0.9}]

    def _fake_ensure_frame_for_request(request, task, runtime, *, out_dir, prefix):
        del task, runtime, out_dir, prefix
        frame = dict((request.get("frames") or [{}])[0] or {})
        timestamp = float(frame.get("timestamp_s") or 0.0)
        return str(tmp_path / ("frame_%0.2f.png" % timestamp)), timestamp

    monkeypatch.setattr(
        paddleocr_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "read chart values",
                "frames": [
                    {"video_id": "video-1", "timestamp_s": 12.0},
                    {"video_id": "video-1", "timestamp_s": 13.0},
                ],
            },
            "task": {},
            "runtime": {"extra": {}},
        },
    )
    monkeypatch.setattr(paddleocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(paddleocr_runner, "ensure_frame_for_request", _fake_ensure_frame_for_request)
    monkeypatch.setattr(
        paddleocr_runner,
        "_prepare_ocr_image",
        lambda frame_path, out_dir, *, max_longest_dim: frame_path,
    )
    monkeypatch.setattr(paddleocr_runner, "create_paddleocr_engine", lambda runtime: _FakeEngine())
    monkeypatch.setattr(paddleocr_runner, "run_paddleocr_image", _fake_run_paddleocr_image)
    monkeypatch.setattr(paddleocr_runner, "emit_json", lambda payload: emitted.update(payload))

    paddleocr_runner.main()

    assert len(created_engines) == 1
    assert len(created_engines[0].calls) == 2
    assert [item["timestamp_s"] for item in emitted["results"]] == [12.0, 13.0]
    assert emitted["results"][0]["text"] == "frame_12.00"
    assert emitted["backend"] == "paddleocr"


def test_ocr_process_adapter_merges_batched_runner_results(monkeypatch):
    adapter = OCRProcessAdapter(name="ocr", model_name="PaddleOCR")
    seen = {}

    def _fake_run_json(context, request_payload):
        del context
        seen["request_payload"] = request_payload
        payload = {
            "results": [
                {
                    "text": "Walmart 95%",
                    "lines": [{"text": "Walmart 95%", "bbox": None, "confidence": 0.9}],
                    "query": "read chart values",
                    "timestamp_s": 12.0,
                    "source_frame_path": "/tmp/frame_12.00.png",
                    "backend": "paddleocr",
                },
                {
                    "text": "Target 91%",
                    "lines": [{"text": "Target 91%", "bbox": None, "confidence": 0.8}],
                    "query": "read chart values",
                    "timestamp_s": 13.0,
                    "source_frame_path": "/tmp/frame_13.00.png",
                    "backend": "paddleocr",
                },
            ],
            "backend": "paddleocr",
        }
        return payload, json.dumps(payload)

    class _FakeWorkspace:
        def store_file_artifact(self, source_frame_path, *, kind, source_tool, video_id):
            return ArtifactRef(
                artifact_id=Path(source_frame_path).stem,
                kind=kind,
                relpath="artifacts/%s/frames/%s" % (video_id, Path(source_frame_path).name),
                source_tool=source_tool,
            )

    monkeypatch.setattr(adapter, "_run_json", _fake_run_json)

    request = adapter.request_model.parse_obj(
        {
            "tool_name": "ocr",
            "query": "read chart values",
            "frames": [
                {"video_id": "video-1", "timestamp_s": 12.0},
                {"video_id": "video-1", "timestamp_s": 13.0},
            ],
        }
    )
    context = SimpleNamespace(
        workspace=_FakeWorkspace(),
        task=SimpleNamespace(video_id="video-1", sample_key="sample-1"),
    )

    result = adapter.execute(request, context)

    assert len(seen["request_payload"]["frames"]) == 2
    assert result.metadata["group_count"] == 2
    assert result.metadata["batch_execution"] == "single_subprocess"
    assert len(result.data["reads"]) == 2
    assert result.data["reads"][0]["frame"]["timestamp_s"] == 12.0
    assert "Walmart 95%" in result.data["text"]
    assert "Target 91%" in result.data["text"]
    assert len(result.artifact_refs) == 2


def test_ocr_process_adapter_merges_single_clip_sampled_frame_results(monkeypatch):
    adapter = OCRProcessAdapter(name="ocr", model_name="PaddleOCR")
    clip = {"video_id": "video-1", "start_s": 10.0, "end_s": 13.0}
    seen = {}

    def _fake_run_json(context, request_payload):
        del context
        seen["request_payload"] = request_payload
        payload = {
            "results": [
                {
                    "text": "HUGO'S",
                    "lines": [{"text": "HUGO'S", "bbox": None, "confidence": 0.9}],
                    "query": "read rink text",
                    "timestamp_s": 10.5,
                    "source_frame_path": "/tmp/clip_frame_01.jpg",
                    "backend": "paddleocr",
                    "clip": clip,
                    "frame": {
                        "video_id": "video-1",
                        "timestamp_s": 10.5,
                        "clip": clip,
                        "metadata": {"source_path": "/tmp/clip_frame_01.jpg", "ocr_source": "clip"},
                    },
                },
                {
                    "text": "ICE",
                    "lines": [{"text": "ICE", "bbox": None, "confidence": 0.8}],
                    "query": "read rink text",
                    "timestamp_s": 11.0,
                    "source_frame_path": "/tmp/clip_frame_02.jpg",
                    "backend": "paddleocr",
                    "clip": clip,
                    "frame": {
                        "video_id": "video-1",
                        "timestamp_s": 11.0,
                        "clip": clip,
                        "metadata": {"source_path": "/tmp/clip_frame_02.jpg", "ocr_source": "clip"},
                    },
                },
            ],
            "backend": "paddleocr",
        }
        return payload, json.dumps(payload)

    class _FakeWorkspace:
        def store_file_artifact(self, source_frame_path, *, kind, source_tool, video_id):
            return ArtifactRef(
                artifact_id=Path(source_frame_path).stem,
                kind=kind,
                relpath="artifacts/%s/frames/%s" % (video_id, Path(source_frame_path).name),
                source_tool=source_tool,
            )

    monkeypatch.setattr(adapter, "_run_json", _fake_run_json)

    request = adapter.request_model.parse_obj(
        {
            "tool_name": "ocr",
            "query": "read rink text",
            "clips": [clip],
        }
    )
    context = SimpleNamespace(
        workspace=_FakeWorkspace(),
        task=SimpleNamespace(video_id="video-1", sample_key="sample-1"),
    )

    result = adapter.execute(request, context)

    assert len(seen["request_payload"]["clips"]) == 1
    assert result.metadata["group_count"] == 2
    assert result.metadata["batch_execution"] == "single_subprocess"
    assert len(result.data["reads"]) == 2
    assert result.data["reads"][0]["clip"]["start_s"] == 10.0
    assert result.data["reads"][0]["frame"]["timestamp_s"] == 10.5
    assert result.data["reads"][0]["frame"]["clip"]["end_s"] == 13.0
    assert "HUGO'S" in result.data["text"]
    assert "ICE" in result.data["text"]
    assert len(result.artifact_refs) == 2
