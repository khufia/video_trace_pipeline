import transformers.utils.generic as hf_generic

from video_trace_pipeline.tool_wrappers import frame_retriever_runner
from video_trace_pipeline.tool_wrappers import reference_adapter
from video_trace_pipeline.schemas import ToolResult
from video_trace_pipeline.tools.process_adapters import FrameRetrieverProcessAdapter


class _FakeHarness:
    def __init__(self, *args, **kwargs):
        self.dataset_folder = "/tmp/reference"
        self.video_path = "/tmp/video.mp4"
        self.precompute_calls = 0

    def _list_dense_frame_paths(self, dataset_folder, video_path):
        return (
            [
                "/tmp/frame_159.00.png",
                "/tmp/frame_162.00.png",
                "/tmp/frame_206.00.png",
            ],
            "/tmp/reference/dense_frames",
        )

    def _ensure_dense_frames(self):
        raise AssertionError("dense frames should already be cached")

    def _timestamp_from_dense_frame_path(self, frame_path):
        return float(frame_path.split("_")[-1].replace(".png", ""))

    def _precompute_frame_embeddings_cache(self, candidates):
        self.precompute_calls += 1
        return True

    def _qwen_score_frames(self, query, bounded, top_k):
        return [
            {
                "frame_path": bounded[0]["frame_path"],
                "timestamp": bounded[0]["timestamp"],
                "relevance_score": 0.9,
            }
        ]

    def _release_frame_embedder(self):
        return None


def test_frame_retriever_runner_reports_cache_metadata(monkeypatch):
    emitted = {}
    fake_harness = _FakeHarness()

    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "best frame",
                "clip": {"start_s": 159.0, "end_s": 168.0},
                "num_frames": 1,
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(frame_retriever_runner, "ReferenceHarness", lambda *args, **kwargs: fake_harness)
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert fake_harness.precompute_calls == 1
    assert emitted["cache_metadata"]["dense_frame_cache_hit"] is True
    assert emitted["cache_metadata"]["embedding_cache_ready"] is True
    assert emitted["cache_metadata"]["bounded_frame_count"] == 2


def test_frame_retriever_process_adapter_merges_cache_metadata(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        clip = request.clip.dict()
        start_s = float(clip["start_s"])
        bounded_count = 10 if start_s < 100.0 else 8
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": [
                    {
                        "video_id": "video-1",
                        "timestamp_s": start_s,
                        "metadata": {"relevance_score": 0.8},
                    }
                ],
                "cache_metadata": {
                    "dense_frame_cache_hit": True,
                    "dense_frame_count": 362,
                    "bounded_frame_count": bounded_count,
                    "embedding_cache_ready": True,
                },
                "rationale": "clip-specific rationale",
            },
            summary="Retrieved 1 frame.",
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "best chart frame",
            "num_frames": 1,
            "clips": [
                {"video_id": "video-1", "start_s": 10.0, "end_s": 20.0},
                {"video_id": "video-1", "start_s": 110.0, "end_s": 120.0},
            ],
        }
    )

    result = adapter.execute(request, context=None)

    assert result.data["cache_metadata"]["dense_frame_cache_hit"] is True
    assert result.data["cache_metadata"]["dense_frame_count"] == 362
    assert result.data["cache_metadata"]["bounded_frame_count"] == 18
    assert result.data["cache_metadata"]["embedding_cache_ready"] is True
    assert len(result.data["cache_groups"]) == 2
    assert result.metadata["dense_frame_cache_hit"] is True


def test_reference_adapter_installs_check_model_inputs_compat(monkeypatch):
    monkeypatch.delattr(hf_generic, "check_model_inputs", raising=False)

    reference_adapter._install_transformers_generic_compat()

    assert hasattr(hf_generic, "check_model_inputs")

    @hf_generic.check_model_inputs
    def _decorated(value):
        return value

    assert _decorated("ok") == "ok"
