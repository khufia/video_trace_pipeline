import json
from pathlib import Path
from types import SimpleNamespace

from video_trace_pipeline.schemas import ArtifactRef
from video_trace_pipeline.tool_wrappers import olmocr_runner
from video_trace_pipeline.tools.process_adapters import OCRProcessAdapter


def test_olmocr_runner_reuses_one_runner_for_multi_frame_requests(monkeypatch, tmp_path):
    emitted = {}
    created_runners = []

    class _FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.closed = False
            created_runners.append(self)

        def generate(self, messages, *, max_new_tokens):
            self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
            image_path = messages[0]["content"][0]["image"]
            image_name = Path(str(image_path)).stem
            return json.dumps(
                {
                    "text": image_name,
                    "lines": [{"text": image_name, "bbox": None, "confidence": 0.9}],
                }
            )

        def close(self):
            self.closed = True

    def _fake_ensure_frame_for_request(request, task, runtime, *, out_dir, prefix):
        del task, runtime, out_dir, prefix
        frame = dict(request.get("frame") or {})
        timestamp = float(frame.get("timestamp_s") or 0.0)
        return str(tmp_path / ("frame_%0.2f.png" % timestamp)), timestamp

    monkeypatch.setattr(
        olmocr_runner,
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
    monkeypatch.setattr(olmocr_runner, "scratch_dir", lambda runtime, tool_name: tmp_path)
    monkeypatch.setattr(olmocr_runner, "ensure_frame_for_request", _fake_ensure_frame_for_request)
    monkeypatch.setattr(olmocr_runner, "_prepare_olmocr_image", lambda frame_path, out_dir: frame_path)
    monkeypatch.setattr(olmocr_runner, "resolve_model_path", lambda *args, **kwargs: "/tmp/olmocr-model")
    monkeypatch.setattr(olmocr_runner, "resolved_device_label", lambda runtime: "cpu")
    monkeypatch.setattr(olmocr_runner, "cleanup_torch", lambda: None)
    monkeypatch.setattr(olmocr_runner, "QwenStyleRunner", _FakeRunner)
    monkeypatch.setattr(olmocr_runner, "emit_json", lambda payload: emitted.update(payload))

    olmocr_runner.main()

    assert len(created_runners) == 1
    assert len(created_runners[0].calls) == 2
    assert created_runners[0].closed is True
    assert [item["timestamp_s"] for item in emitted["results"]] == [12.0, 13.0]
    assert emitted["results"][0]["text"] == "frame_12.00"
    assert emitted["backend"] == "olmocr_transformers"


def test_ocr_process_adapter_merges_batched_runner_results(monkeypatch):
    adapter = OCRProcessAdapter(name="ocr", model_name="allenai/olmOCR-2-7B-1025-FP8")
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
                    "backend": "olmocr_transformers",
                },
                {
                    "text": "Target 91%",
                    "lines": [{"text": "Target 91%", "bbox": None, "confidence": 0.8}],
                    "query": "read chart values",
                    "timestamp_s": 13.0,
                    "source_frame_path": "/tmp/frame_13.00.png",
                    "backend": "olmocr_transformers",
                },
            ],
            "backend": "olmocr_transformers",
        }
        return payload, json.dumps(payload)

    class _FakeWorkspace:
        def store_file_artifact(self, source_frame_path, *, kind, source_tool):
            return ArtifactRef(
                artifact_id=Path(source_frame_path).stem,
                kind=kind,
                relpath=Path(source_frame_path).name,
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
    context = SimpleNamespace(workspace=_FakeWorkspace())

    result = adapter.execute(request, context)

    assert len(seen["request_payload"]["frames"]) == 2
    assert result.metadata["group_count"] == 2
    assert result.metadata["batch_execution"] == "single_subprocess"
    assert len(result.data["reads"]) == 2
    assert result.data["reads"][0]["frame"]["timestamp_s"] == 12.0
    assert "Walmart 95%" in result.data["text"]
    assert "Target 91%" in result.data["text"]
    assert len(result.artifact_refs) == 2
