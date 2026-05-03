from contextlib import nullcontext
import sys
import types

import pytest

from video_trace_pipeline.tool_wrappers import frame_retriever_runner
from video_trace_pipeline.tool_wrappers import reference_adapter
from video_trace_pipeline.tool_wrappers import shared
from video_trace_pipeline.schemas import ToolResult
from video_trace_pipeline.tools.process_adapters import FrameRetrieverProcessAdapter


class _FakeHarness:
    def __init__(self, *args, **kwargs):
        self.dataset_folder = "/tmp/reference"
        self.video_path = "/tmp/video.mp4"
        self.precompute_calls = 0
        self.persist_cache = None

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

    def _frame_embedding_cache_ready(self):
        return False

    def _frame_embedder_runtime_metadata(self):
        return {"loaded_attn_implementation": "flash_attention_2", "retry_count": 0}

    def _qwen_score_frames(self, query, bounded, top_k, *, persist_cache=True):
        self.persist_cache = persist_cache
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
                "clips": [{"start_s": 159.0, "end_s": 168.0}],
                "num_frames": 1,
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert fake_harness.precompute_calls == 0
    assert fake_harness.persist_cache is False
    assert emitted["cache_metadata"]["dense_frame_cache_hit"] is True
    assert emitted["cache_metadata"]["embedding_cache_ready"] is False
    assert emitted["cache_metadata"]["persist_cache_on_bounded_request"] is False
    assert emitted["cache_metadata"]["bounded_frame_count"] == 2


def test_frame_retriever_runner_uses_time_hints_without_query(monkeypatch):
    emitted = {}
    fake_harness = _FakeHarness()

    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "clips": [{"start_s": 159.0, "end_s": 168.0}],
                "time_hints": ["start of the localized utterance"],
                "num_frames": 1,
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert emitted["cache_metadata"]["time_hints_applied"] is True
    assert emitted["frames"][0]["timestamp_s"] == 159.0
    assert emitted["rationale"] == "Frames were selected near the requested clip-local time hints."


def test_frame_retriever_parses_exact_timestamp_hints():
    assert frame_retriever_runner._time_hint_anchor_seconds("6s", 0.0, 10.0) == 6.0
    assert frame_retriever_runner._time_hint_anchor_seconds("frame at 6.00 seconds", 0.0, 10.0) == 6.0
    assert frame_retriever_runner._time_hint_anchor_seconds("00:06", 0.0, 10.0) == 6.0
    assert frame_retriever_runner._time_hint_anchor_seconds("1:23", 0.0, 100.0) == 83.0


def test_frame_retriever_exact_timestamp_returns_chronological_neighbors(monkeypatch):
    emitted = {}

    class _SequenceHarness(_FakeHarness):
        def _list_dense_frame_paths(self, dataset_folder, video_path):
            return (
                [
                    "/tmp/frame_4.00.png",
                    "/tmp/frame_5.00.png",
                    "/tmp/frame_6.00.png",
                    "/tmp/frame_7.00.png",
                    "/tmp/frame_8.00.png",
                ],
                "/tmp/reference/dense_frames",
            )

        def _qwen_score_frames(self, query, bounded, top_k, *, persist_cache=True):
            self.persist_cache = persist_cache
            return [
                {
                    "frame_path": item["frame_path"],
                    "timestamp": item["timestamp"],
                    "relevance_score": 1.0 - (abs(float(item["timestamp"]) - 6.0) * 0.05),
                }
                for item in bounded
            ]

    fake_harness = _SequenceHarness()
    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "what happens at timestamp 00:06",
                "clips": [{"start_s": 0.0, "end_s": 10.0}],
                "time_hints": ["00:06"],
                "num_frames": 3,
                "sequence_mode": "anchor_window",
                "neighbor_radius_s": 2.0,
                "sort_order": "chronological",
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert [frame["timestamp_s"] for frame in emitted["frames"]] == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert emitted["frames"][2]["metadata"]["sequence_role"] == "anchor"
    assert emitted["frames"][2]["metadata"]["requested_timestamp_s"] == 6.0
    assert emitted["frames"][0]["metadata"]["sequence_index"] == 0
    assert emitted["cache_metadata"]["anchor_window_applied"] is True


def test_frame_retriever_anchor_timestamp_narrows_full_video_pool(monkeypatch):
    emitted = {}

    class _LongHarness(_FakeHarness):
        def _list_dense_frame_paths(self, dataset_folder, video_path):
            return (
                [
                    "/tmp/frame_0.00.png",
                    "/tmp/frame_127.00.png",
                    "/tmp/frame_129.00.png",
                    "/tmp/frame_131.00.png",
                    "/tmp/frame_600.00.png",
                ],
                "/tmp/reference/dense_frames",
            )

        def _qwen_score_frames(self, query, bounded, top_k, *, persist_cache=True):
            self.persist_cache = persist_cache
            return [
                {
                    "frame_path": item["frame_path"],
                    "timestamp": item["timestamp"],
                    "relevance_score": 1.0,
                }
                for item in bounded
            ]

    fake_harness = _LongHarness()
    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "visible object on the table",
                "clips": [{"start_s": 0.0, "end_s": 1200.0}],
                "time_hints": ["129.000s"],
                "num_frames": 3,
                "sequence_mode": "anchor_window",
                "neighbor_radius_s": 2.0,
                "sort_order": "chronological",
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert emitted["cache_metadata"]["frame_pool_start_s"] == 127.0
    assert emitted["cache_metadata"]["frame_pool_end_s"] == 131.0
    assert emitted["cache_metadata"]["bounded_frame_count"] == 3
    assert [frame["timestamp_s"] for frame in emitted["frames"]] == [127.0, 129.0, 131.0]


def test_frame_retriever_chronological_without_anchor_returns_full_clip_sequence(monkeypatch):
    emitted = {}

    class _SequenceHarness(_FakeHarness):
        def _list_dense_frame_paths(self, dataset_folder, video_path):
            return (
                [
                    "/tmp/frame_0.00.png",
                    "/tmp/frame_1.00.png",
                    "/tmp/frame_2.00.png",
                    "/tmp/frame_3.00.png",
                    "/tmp/frame_4.00.png",
                ],
                "/tmp/reference/dense_frames",
            )

        def _qwen_score_frames(self, query, bounded, top_k, *, persist_cache=True):
            raise AssertionError("full chronological clip sequence should not score frames")

    fake_harness = _SequenceHarness()
    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "visible object on the table",
                "clips": [{"start_s": 1.0, "end_s": 3.0}],
                "num_frames": 1,
                "sequence_mode": "chronological",
                "sort_order": "chronological",
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert [frame["timestamp_s"] for frame in emitted["frames"]] == [1.0, 2.0, 3.0]
    assert [frame["metadata"]["sequence_index"] for frame in emitted["frames"]] == [0, 1, 2]
    assert emitted["frames"][0]["metadata"]["sequence_role"] == "interval_frame"
    assert emitted["cache_metadata"]["chronological_clip_sequence_applied"] is True
    assert emitted["cache_metadata"]["expanded_frame_pool_for_anchor_window"] is False


def test_anchor_window_expands_frame_pool_beyond_short_clip(monkeypatch):
    emitted = {}

    class _SequenceHarness(_FakeHarness):
        def _list_dense_frame_paths(self, dataset_folder, video_path):
            return (
                [
                    "/tmp/frame_9.00.png",
                    "/tmp/frame_10.00.png",
                    "/tmp/frame_11.00.png",
                    "/tmp/frame_12.00.png",
                    "/tmp/frame_13.00.png",
                    "/tmp/frame_14.00.png",
                    "/tmp/frame_15.00.png",
                ],
                "/tmp/reference/dense_frames",
            )

        def _qwen_score_frames(self, query, bounded, top_k, *, persist_cache=True):
            self.persist_cache = persist_cache
            return [
                {
                    "frame_path": item["frame_path"],
                    "timestamp": item["timestamp"],
                    "relevance_score": 1.0 - (abs(float(item["timestamp"]) - 12.0) * 0.05),
                }
                for item in bounded
            ]

    fake_harness = _SequenceHarness()
    monkeypatch.setattr(
        frame_retriever_runner,
        "load_request",
        lambda: {
            "request": {
                "query": "visible impact source",
                "clips": [{"start_s": 11.6, "end_s": 12.384}],
                "time_hints": ["11.60s-12.384s event interval; center near 11.99s"],
                "num_frames": 5,
                "sequence_mode": "anchor_window",
                "neighbor_radius_s": 2.5,
                "include_anchor_neighbors": True,
                "sort_order": "chronological",
            },
            "task": {},
            "runtime": {},
        },
    )
    monkeypatch.setattr(
        frame_retriever_runner,
        "_reference_harness_cls",
        lambda: (lambda *args, **kwargs: fake_harness),
    )
    monkeypatch.setattr(frame_retriever_runner, "emit_json", lambda payload: emitted.update(payload))
    monkeypatch.setattr(frame_retriever_runner, "resolved_device_label", lambda runtime: "cpu")

    frame_retriever_runner.main()

    assert [frame["timestamp_s"] for frame in emitted["frames"]] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert emitted["cache_metadata"]["frame_pool_start_s"] == 9.1
    assert emitted["cache_metadata"]["frame_pool_end_s"] == 14.884
    assert emitted["cache_metadata"]["expanded_frame_pool_for_anchor_window"] is True
    assert emitted["frames"][0]["metadata"]["frame_pool_start_s"] == 9.1


def test_frame_retriever_process_adapter_merges_cache_metadata(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        clip = request.clips[0].dict()
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
    assert result.data["cache_metadata"]["frame_count_policy"] == "per_clip"
    assert result.data["cache_metadata"]["frames_per_input"] == 1
    assert result.data["cache_metadata"]["returned_frame_count"] == 2
    assert len(result.data["cache_groups"]) == 2
    assert result.metadata["dense_frame_cache_hit"] is True


def test_frame_retriever_process_adapter_keeps_num_frames_per_clip(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        clip = request.clips[0].dict()
        start_s = float(clip["start_s"])
        if start_s < 100.0:
            frames = [
                {"video_id": "video-1", "timestamp_s": 12.0, "metadata": {"relevance_score": 0.95}},
                {"video_id": "video-1", "timestamp_s": 13.0, "metadata": {"relevance_score": 0.40}},
            ]
        else:
            frames = [
                {"video_id": "video-1", "timestamp_s": 112.0, "metadata": {"relevance_score": 0.80}},
                {"video_id": "video-1", "timestamp_s": 113.0, "metadata": {"relevance_score": 0.10}},
            ]
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": frames,
                "cache_metadata": {},
                "rationale": "clip-specific rationale",
            },
            summary="Retrieved %d frame(s)." % len(frames),
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "best chart frame",
            "num_frames": 2,
            "clips": [
                {"video_id": "video-1", "start_s": 10.0, "end_s": 20.0},
                {"video_id": "video-1", "start_s": 110.0, "end_s": 120.0},
            ],
        }
    )

    result = adapter.execute(request, context=None)

    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [12.0, 112.0, 13.0, 113.0]
    assert result.data["cache_metadata"]["frame_count_policy"] == "per_clip"
    assert result.data["cache_metadata"]["frames_per_input"] == 2
    assert result.data["cache_metadata"]["returned_frame_count"] == 4


def test_frame_retriever_process_adapter_chronological_multi_clip_keeps_each_clip(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        clip = request.clips[0].dict()
        start_s = float(clip["start_s"])
        if start_s < 100.0:
            frames = [
                {"video_id": "video-1", "timestamp_s": 14.0, "metadata": {"relevance_score": 0.20}},
                {"video_id": "video-1", "timestamp_s": 12.0, "metadata": {"relevance_score": 0.95}},
                {"video_id": "video-1", "timestamp_s": 13.0, "metadata": {"relevance_score": 0.40}},
            ]
        else:
            frames = [
                {"video_id": "video-1", "timestamp_s": 114.0, "metadata": {"relevance_score": 0.20}},
                {"video_id": "video-1", "timestamp_s": 112.0, "metadata": {"relevance_score": 0.80}},
                {"video_id": "video-1", "timestamp_s": 113.0, "metadata": {"relevance_score": 0.10}},
            ]
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": frames,
                "cache_metadata": {},
                "rationale": "clip-specific rationale",
            },
            summary="Retrieved %d frame(s)." % len(frames),
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "chart frame sequence",
            "num_frames": 2,
            "sort_order": "chronological",
            "clips": [
                {"video_id": "video-1", "start_s": 10.0, "end_s": 20.0},
                {"video_id": "video-1", "start_s": 110.0, "end_s": 120.0},
            ],
        }
    )

    result = adapter.execute(request, context=None)

    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [12.0, 13.0, 112.0, 113.0]
    assert result.data["cache_metadata"]["returned_frame_count"] == 4


def test_frame_retriever_process_adapter_filters_out_of_clip_time_hints(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")
    seen_requests = []

    def _fake_execute_single(request, context):
        del context
        seen_requests.append(request)
        clip = request.clips[0].dict()
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": [
                    {
                        "video_id": "video-1",
                        "timestamp_s": 196.0,
                        "metadata": {"relevance_score": 0.95},
                    }
                ],
                "cache_metadata": {},
                "rationale": "anchor clip",
            },
            summary="Retrieved 1 frame from %.1f." % float(clip["start_s"]),
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "boats by the shore",
            "num_frames": 1,
            "clips": [
                {"video_id": "video-1", "start_s": 180.0, "end_s": 210.0},
                {"video_id": "video-1", "start_s": 210.0, "end_s": 240.0},
            ],
            "time_hints": ["195.983s"],
            "sequence_mode": "anchor_window",
            "sort_order": "chronological",
        }
    )

    result = adapter.execute(request, context=None)

    assert len(seen_requests) == 1
    assert seen_requests[0].clips[0].start_s == 180.0
    assert seen_requests[0].time_hints == ["195.983s"]
    assert result.data["cache_metadata"]["skipped_out_of_window_time_hint_clip_count"] == 1
    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [196.0]


def test_frame_retriever_process_adapter_keeps_full_chronological_clip_sequences(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        clip = request.clips[0].dict()
        start_s = float(clip["start_s"])
        frames = []
        for index, timestamp_s in enumerate((start_s, start_s + 1.0, start_s + 2.0)):
            frames.append(
                {
                    "video_id": "video-1",
                    "timestamp_s": timestamp_s,
                    "metadata": {
                        "clip_start_s": start_s,
                        "sequence_mode": "chronological",
                        "sequence_role": "interval_frame",
                        "selection_reason": "chronological_clip_sequence",
                        "sequence_sort_order": "chronological",
                        "sequence_index": index,
                    },
                }
            )
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={"frames": frames, "cache_metadata": {}, "rationale": "full chronological sequence"},
            summary="Retrieved %d frame(s)." % len(frames),
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "interval frames",
            "num_frames": 1,
            "sequence_mode": "chronological",
            "sort_order": "chronological",
            "clips": [
                {"video_id": "video-1", "start_s": 10.0, "end_s": 12.0},
                {"video_id": "video-1", "start_s": 110.0, "end_s": 112.0},
            ],
        }
    )

    result = adapter.execute(request, context=None)

    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [10.0, 11.0, 12.0, 110.0, 111.0, 112.0]
    assert result.data["cache_metadata"]["returned_frame_count"] == 6


def test_frame_retriever_process_adapter_single_clip_uses_single_execution(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": [
                    {"video_id": "video-1", "timestamp_s": 12.0, "metadata": {"relevance_score": 0.95}},
                    {"video_id": "video-1", "timestamp_s": 13.0, "metadata": {"relevance_score": 0.40}},
                    {"video_id": "video-1", "timestamp_s": 14.0, "metadata": {"relevance_score": 0.20}},
                ],
                "cache_metadata": {"returned_frame_count": 3},
                "rationale": "single clip",
            },
            summary="Retrieved 3 frame(s).",
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "best chart frame",
            "num_frames": 2,
            "clips": [{"video_id": "video-1", "start_s": 10.0, "end_s": 20.0}],
        }
    )

    result = adapter.execute(request, context=None)

    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [12.0, 13.0, 14.0]
    assert result.data["cache_metadata"]["returned_frame_count"] == 3


def test_frame_retriever_process_adapter_preserves_chronological_sequence_merge(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")

    def _fake_execute_single(request, context):
        frames = []
        for timestamp_s, sequence_index in (
            (164.0, 2),
            (163.0, 1),
            (165.0, 3),
            (162.0, 0),
            (166.0, 4),
        ):
            frames.append(
                {
                    "video_id": "video-1",
                    "timestamp_s": timestamp_s,
                    "metadata": {
                        "clip_start_s": 150.0,
                        "requested_timestamp_s": 164.0,
                        "sequence_mode": "anchor_window",
                        "sequence_index": sequence_index,
                        "sequence_sort_order": "chronological",
                        "relevance_score": 1.0,
                    },
                }
            )
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={
                "frames": frames,
                "cache_metadata": {},
                "rationale": "chronological sequence",
            },
            summary="Retrieved 5 frame(s).",
        )

    monkeypatch.setattr(adapter, "_execute_single", _fake_execute_single)
    request = adapter.request_model.parse_obj(
        {
            "tool_name": "frame_retriever",
            "query": "map frames",
            "num_frames": 5,
            "clips": [{"video_id": "video-1", "start_s": 150.0, "end_s": 166.0}],
            "time_hints": ["164s", "165.98s utterance"],
            "sequence_mode": "anchor_window",
            "sort_order": "chronological",
        }
    )

    result = adapter.execute(request, context=None)

    assert [frame["timestamp_s"] for frame in result.data["frames"]] == [162.0, 163.0, 164.0, 165.0, 166.0]


def test_reference_adapter_installs_check_model_inputs_compat(monkeypatch):
    hf_generic = pytest.importorskip("transformers.utils.generic")
    monkeypatch.delattr(hf_generic, "check_model_inputs", raising=False)

    reference_adapter._install_transformers_generic_compat()

    assert hasattr(hf_generic, "check_model_inputs")

    @hf_generic.check_model_inputs
    def _decorated(value):
        return value

    assert _decorated("ok") == "ok"


def test_reference_adapter_loads_local_model_helper(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "qwen-embedder"
    scripts_dir = model_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "qwen3_vl_embedding.py").write_text(
        "class Qwen3VLEmbedder:\n"
        "    def __init__(self, model_name_or_path, **kwargs):\n"
        "        self.model_name_or_path = model_name_or_path\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reference_adapter, "get_video_duration", lambda _video_path: 12.0)

    harness = reference_adapter.ReferenceHarness(
        task={"video_path": str(tmp_path / "video.mp4"), "video_id": "video-1"},
        runtime={"workspace_root": str(tmp_path), "model_name": str(model_dir)},
        clip_duration_s=5.0,
        embedder_model=str(model_dir),
    )

    helper_cls = harness._load_model_helper_class(
        str(model_dir),
        "qwen3_vl_embedding.py",
        "Qwen3VLEmbedder",
        "_temporal_grounder_embedder_class",
    )

    assert helper_cls.__name__ == "Qwen3VLEmbedder"
    assert harness._load_model_helper_class(
        str(model_dir),
        "qwen3_vl_embedding.py",
        "Qwen3VLEmbedder",
        "_temporal_grounder_embedder_class",
    ) is helper_cls


def test_reference_adapter_frame_embedder_uses_balanced_cuda_device_map(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}
    module = types.ModuleType("fake_qwen3_vl_embedding_sharded")

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["model_kwargs"] = dict(kwargs)
            return cls()

        def eval(self):
            calls["model_eval"] = True
            return self

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls["processor_name_or_path"] = model_name_or_path
            calls["processor_kwargs"] = dict(kwargs)
            return cls()

    class _FakeEmbedder:
        pass

    _FakeEmbedder.__module__ = module.__name__
    module.Qwen3VLEmbedder = _FakeEmbedder
    module.Qwen3VLForEmbedding = _FakeModel
    module.Qwen3VLProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(reference_adapter, "get_video_duration", lambda _video_path: 12.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda _idx: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx: type("Props", (), {"total_memory": (idx + 1) * 1024})(),
    )

    harness = reference_adapter.ReferenceHarness(
        task={"video_path": str(tmp_path / "sample.mp4"), "video_id": "video-1"},
        runtime={
            "workspace_root": str(tmp_path),
            "model_name": str(tmp_path),
            "extra": {"device_map": "balanced_cuda:0,1"},
        },
        clip_duration_s=5.0,
        embedder_model=str(tmp_path),
    )
    monkeypatch.setattr(harness, "_load_model_helper_class", lambda *args, **kwargs: _FakeEmbedder)

    embedder = harness._get_or_load_frame_embedder()

    assert isinstance(embedder, _FakeEmbedder)
    assert calls["model_kwargs"]["device_map"] == "balanced"
    assert calls["model_kwargs"]["max_memory"] == {0: 1024, 1: 2048}
    assert calls["processor_kwargs"] == {"padding_side": "right"}
    assert harness._frame_embedder_device_index() == 0
    assert harness._frame_embedder_runtime_metadata()["device_map"] == "balanced_cuda:0,1"


def test_reference_adapter_frame_embedder_default_uses_helper_constructor(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}

    class _FakeEmbedder:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = dict(kwargs)

    monkeypatch.setattr(reference_adapter, "get_video_duration", lambda _video_path: 12.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    harness = reference_adapter.ReferenceHarness(
        task={"video_path": str(tmp_path / "sample.mp4"), "video_id": "video-1"},
        runtime={"workspace_root": str(tmp_path), "model_name": str(tmp_path)},
        clip_duration_s=5.0,
        embedder_model=str(tmp_path),
    )
    monkeypatch.setattr(harness, "_load_model_helper_class", lambda *args, **kwargs: _FakeEmbedder)

    embedder = harness._get_or_load_frame_embedder()

    assert isinstance(embedder, _FakeEmbedder)
    assert calls["model_name_or_path"] == str(tmp_path)
    assert "device_map" not in calls["kwargs"]
    assert harness._frame_embedder_runtime_metadata()["device_map"] is None


def test_reference_adapter_lists_dense_frames_in_timestamp_order(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_adapter, "get_video_duration", lambda _video_path: 12.0)
    harness = reference_adapter.ReferenceHarness(
        task={"video_path": str(tmp_path / "sample.mp4"), "video_id": "video-1"},
        runtime={"workspace_root": str(tmp_path), "model_name": str(tmp_path)},
        clip_duration_s=5.0,
        embedder_model=str(tmp_path),
    )
    dense_dir = tmp_path / "workspace" / "cache" / "tool_wrappers" / "reference" / "video-1" / "dense_frames" / "sample"
    dense_dir.mkdir(parents=True)
    for name in ("frame_10.00.png", "frame_2.00.png", "frame_1.50.png"):
        (dense_dir / name).write_bytes(b"png")

    frame_paths, dense_dir_path = harness._list_dense_frame_paths(harness.dataset_folder, harness.video_path)

    assert dense_dir_path.endswith("/sample")
    assert [path.rsplit("/", 1)[-1] for path in frame_paths] == [
        "frame_1.50.png",
        "frame_2.00.png",
        "frame_10.00.png",
    ]


def test_resolve_model_path_prefers_requested_snapshot_over_runtime_resolved(tmp_path, monkeypatch):
    timelens_dir = tmp_path / "timelens"
    timelens_dir.mkdir()
    qwen_snapshot = tmp_path / "qwen_snapshot"
    qwen_snapshot.mkdir()

    monkeypatch.setattr(
        shared,
        "resolve_model_snapshot",
        lambda model_name, hf_cache=None: qwen_snapshot if model_name == "Qwen/Qwen3-VL-Embedding-8B" else None,
    )

    runtime = {
        "model_name": "TencentARC/TimeLens-8B",
        "resolved_model_path": str(timelens_dir),
    }

    assert shared.resolve_model_path("TencentARC/TimeLens-8B", runtime) == str(timelens_dir.resolve())
    assert shared.resolve_model_path("Qwen/Qwen3-VL-Embedding-8B", runtime) == str(qwen_snapshot)


def test_reference_adapter_qwen_score_frames_retries_without_flash_attention(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(reference_adapter, "get_video_duration", lambda _video_path: 12.0)
    harness = reference_adapter.ReferenceHarness(
        task={"video_path": str(tmp_path / "sample.mp4"), "video_id": "video-1"},
        runtime={
            "workspace_root": str(tmp_path),
            "model_name": str(tmp_path),
            "extra": {
                "attn_implementation": "flash_attention_2",
                "dense_frame_embed_batch": 4,
            },
        },
        clip_duration_s=5.0,
        embedder_model=str(tmp_path),
    )

    class _FakeEmbedder:
        def __init__(self, attn_implementation):
            self.attn_implementation = attn_implementation

        def process(self, samples):
            if samples and samples[0].get("text") and self.attn_implementation == "flash_attention_2":
                raise ValueError("FlashAttention does not support inputs with dim=0.")
            rows = len(samples)
            return torch.ones(rows, 2, dtype=torch.float32)

    def _fake_get_or_load(self):
        if self._frame_embedder is None:
            attn = str(self._frame_embedder_attn_override or self._frame_embedder_requested_attn_implementation or "default")
            self._frame_embedder = _FakeEmbedder(attn)
            self._frame_embedder_loaded_attn_implementation = attn
            self._frame_embedder_diagnostics["loaded_attn_implementation"] = attn
        return self._frame_embedder

    def _fake_release(self):
        self._frame_embedder = None
        self._frame_embedder_loaded_attn_implementation = None

    monkeypatch.setattr(reference_adapter.ReferenceHarness, "_get_or_load_frame_embedder", _fake_get_or_load)
    monkeypatch.setattr(reference_adapter.ReferenceHarness, "_release_frame_embedder", _fake_release)
    monkeypatch.setattr(reference_adapter.ReferenceHarness, "_frame_embedder_inference_context", lambda self: nullcontext())

    scored = harness._qwen_score_frames(
        "find the chart frame",
        [{"frame_path": "/tmp/frame_159.00.png", "timestamp": 159.0}],
        1,
        persist_cache=False,
    )

    assert len(scored) == 1
    assert harness._frame_embedder_runtime_metadata()["loaded_attn_implementation"] == "sdpa"
    assert harness._frame_embedder_runtime_metadata()["retry_count"] >= 1


def test_frame_retriever_process_adapter_runs_persisted_payload_without_releasing_embedder(monkeypatch):
    adapter = FrameRetrieverProcessAdapter(name="frame_retriever", model_name="qwen-frame-reranker")
    fake_harness = object()
    seen = {}

    monkeypatch.setattr(adapter, "_persisted_frame_harness", lambda payload: fake_harness)
    monkeypatch.setattr(
        "video_trace_pipeline.tool_wrappers.frame_retriever_runner.execute_payload",
        lambda payload, *, harness=None, release_embedder=True: seen.update(
            {"payload": payload, "harness": harness, "release_embedder": release_embedder}
        )
        or {"frames": [], "cache_metadata": {}, "mode": "clip_bounded", "rationale": ""},
    )

    payload = {
        "request": {"tool_name": "frame_retriever", "query": "chart", "clips": [{"start_s": 1.0, "end_s": 2.0}]},
        "task": {"video_id": "video-1", "video_path": "/tmp/video.mp4"},
        "runtime": {"model_name": "Qwen/Qwen3-VL-Embedding-8B"},
    }

    result = adapter._run_persisted_json(payload)

    assert result["mode"] == "clip_bounded"
    assert seen["harness"] is fake_harness
    assert seen["release_embedder"] is False


def test_select_diverse_frames_prefers_structured_visual_plateau_center():
    ranked = [
        {"frame_path": "/tmp/frame_10.00.png", "timestamp": 10.0, "relevance_score": 0.90, "temporal_score": 0.95, "selection_reason": "structured_visual_temporal_rerank"},
        {"frame_path": "/tmp/frame_11.00.png", "timestamp": 11.0, "relevance_score": 0.91, "temporal_score": 0.96, "selection_reason": "structured_visual_temporal_rerank"},
        {"frame_path": "/tmp/frame_12.00.png", "timestamp": 12.0, "relevance_score": 0.92, "temporal_score": 0.97, "selection_reason": "structured_visual_temporal_rerank"},
        {"frame_path": "/tmp/frame_13.00.png", "timestamp": 13.0, "relevance_score": 0.90, "temporal_score": 0.95, "selection_reason": "structured_visual_temporal_rerank"},
        {"frame_path": "/tmp/frame_20.00.png", "timestamp": 20.0, "relevance_score": 0.60, "temporal_score": 0.60, "selection_reason": "structured_visual_temporal_rerank"},
    ]

    selected = frame_retriever_runner._select_diverse_frames(
        ranked,
        1,
        query="completed chart with all labels visible",
    )

    assert len(selected) == 1
    assert selected[0]["timestamp"] == 12.0
    assert selected[0]["selection_reason"] == "structured_visual_plateau_center"
