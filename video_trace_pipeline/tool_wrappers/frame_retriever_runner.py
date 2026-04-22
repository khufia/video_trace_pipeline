from __future__ import annotations

from typing import Any, Dict, List

from ..tools.local_asr import _clip_from_time_hint
from .protocol import emit_json, fail_runtime, load_request
from .reference_adapter import ReferenceHarness
from .shared import resolved_device_label


def _resolved_clip(request: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    clip = dict(request.get("clip") or {})
    if clip:
        return clip
    derived = _clip_from_time_hint(
        str(task.get("video_id") or task.get("sample_key") or "video"),
        str(task.get("video_path") or ""),
        request.get("time_hint"),
    )
    if derived is None:
        return {}
    if hasattr(derived, "model_dump"):
        return derived.model_dump()
    return derived.dict()


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    clip = _resolved_clip(request, task)
    query = str(request.get("query") or "").strip()
    if not clip:
        fail_runtime("frame_retriever requires a clip or resolvable time_hint")

    clip_start_s = float(clip.get("start_s") or 0.0)
    clip_end_s = float(clip.get("end_s") or clip_start_s)
    if clip_end_s < clip_start_s:
        clip_end_s = clip_start_s

    num_frames = max(1, int(request.get("num_frames") or 5))
    harness = ReferenceHarness(
        task=task,
        runtime=runtime,
        clip_duration_s=max(1.0, clip_end_s - clip_start_s),
        embedder_model=str(runtime.get("model_name") or ""),
        reranker_model=str((runtime.get("extra") or {}).get("reranker_model") or ""),
    )
    try:
        frame_items, _ = harness._list_dense_frame_paths(harness.dataset_folder, harness.video_path)
        dense_frame_cache_hit = bool(frame_items)
        if not frame_items:
            harness._ensure_dense_frames()
            frame_items, _ = harness._list_dense_frame_paths(harness.dataset_folder, harness.video_path)

        candidates = [
            {
                "frame_path": frame_path,
                "timestamp": harness._timestamp_from_dense_frame_path(frame_path),
            }
            for frame_path in frame_items
        ]
        embedding_cache_ready = bool(harness._precompute_frame_embeddings_cache(candidates))
        bounded = [
            item for item in candidates if clip_start_s <= float(item["timestamp"]) <= clip_end_s
        ] or candidates

        if query:
            ranked = harness._qwen_score_frames(query, bounded, num_frames)
        else:
            ranked = sorted(
                bounded,
                key=lambda item: abs(float(item["timestamp"]) - ((clip_start_s + clip_end_s) / 2.0)),
            )[:num_frames]
            ranked = [
                {
                    "frame_path": item["frame_path"],
                    "timestamp": item["timestamp"],
                    "relevance_score": 0.0,
                }
                for item in ranked
            ]

        frames: List[Dict[str, Any]] = []
        for item in ranked:
            frames.append(
                {
                    "frame_path": str(item["frame_path"]),
                    "timestamp_s": round(float(item["timestamp"]), 3),
                    "relevance_score": float(item.get("relevance_score") or 0.0),
                    "metadata": {
                        "device": resolved_device_label(runtime),
                        "clip_start_s": clip_start_s,
                        "clip_end_s": clip_end_s,
                    },
                }
            )
        emit_json(
            {
                "query": query or None,
                "frames": frames,
                "mode": "clip_bounded",
                "cache_metadata": {
                    "dense_frame_cache_hit": dense_frame_cache_hit,
                    "dense_frame_count": len(candidates),
                    "bounded_frame_count": len(bounded),
                    "embedding_cache_ready": embedding_cache_ready,
                },
                "rationale": (
                    "Frames were ranked within the requested clip using the configured Qwen visual embedder."
                    if query
                    else "No query was provided, so the frames nearest the clip midpoint were returned."
                ),
            }
        )
    finally:
        if getattr(harness, "_release_frame_embedder", None):
            try:
                harness._release_frame_embedder()
            except Exception:
                pass


if __name__ == "__main__":
    main()
