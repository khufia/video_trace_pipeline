from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..common import extract_json_object
from ..tools.media import get_video_duration
from .local_multimodal import QwenStyleRunner, make_qwen_video_message
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    extract_interval_candidates,
    extracted_clip,
    iter_windows,
    merge_intervals,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
    summarize_intervals,
)

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


_EMBEDDING_PREFILTER_ENABLED = False


def _reference_harness_cls():
    from .reference_adapter import ReferenceHarness

    return ReferenceHarness


def _window_candidates(raw_text: str, *, window_start_s: float) -> List[Dict[str, Any]]:
    payload = extract_json_object(raw_text) or {}
    intervals = payload.get("intervals") or payload.get("clips") or payload.get("segments") or []
    candidates: List[Dict[str, Any]] = []
    if isinstance(intervals, list):
        for item in intervals:
            if isinstance(item, dict):
                start = item.get("start_s", item.get("start"))
                end = item.get("end_s", item.get("end"))
                confidence = item.get("confidence")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = item[0], item[1]
                confidence = None
            else:
                continue
            try:
                candidates.append(
                    {
                        "start_s": float(start) + float(window_start_s),
                        "end_s": float(end) + float(window_start_s),
                        "confidence": None if confidence is None else float(confidence),
                    }
                )
            except Exception:
                continue
    if candidates:
        return candidates
    return [
        {"start_s": float(start), "end_s": float(end), "confidence": None}
        for start, end in extract_interval_candidates(raw_text, offset_s=float(window_start_s))
    ]


def _candidate_window_indices(
    scored_frames: List[Dict[str, Any]],
    windows: List[Tuple[float, float]],
    *,
    neighbor_radius: int,
    max_windows: int,
) -> List[int]:
    if not scored_frames or not windows:
        return []
    indices = set()
    for item in list(scored_frames or []):
        try:
            timestamp = float(item.get("timestamp") or 0.0)
        except Exception:
            continue
        anchor_index = None
        for index, (start_s, end_s) in enumerate(windows):
            if float(start_s) <= timestamp <= float(end_s):
                anchor_index = index
                break
        if anchor_index is None:
            if timestamp < float(windows[0][0]):
                anchor_index = 0
            else:
                anchor_index = len(windows) - 1
        for offset in range(-max(0, int(neighbor_radius)), max(0, int(neighbor_radius)) + 1):
            expanded = anchor_index + offset
            if 0 <= expanded < len(windows):
                indices.add(expanded)
    ordered = sorted(indices)
    if max_windows > 0:
        return ordered[:max_windows]
    return ordered


def _prefilter_windows(
    *,
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    query: str,
    duration_s: float,
    window_s: float,
    top_k: int,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    all_windows = list(iter_windows(duration_s, window_s))
    metadata: Dict[str, Any] = {
        "enabled": False,
        "total_windows": len(all_windows),
        "candidate_windows": len(all_windows),
    }
    extra = dict(runtime.get("extra") or {})
    metadata["configured"] = bool(extra.get("use_embedding_prefilter", True))
    if not _EMBEDDING_PREFILTER_ENABLED:
        metadata["reason"] = "prefilter_disabled_in_code"
        return all_windows, metadata
    if not all_windows or not query or not metadata["configured"]:
        return all_windows, metadata

    embedder_model = str(extra.get("prefilter_embedder_model") or "Qwen/Qwen3-VL-Embedding-8B").strip()
    neighbor_radius = max(0, int(extra.get("prefilter_neighbor_radius") or 1))
    max_windows = min(
        len(all_windows),
        max(
            int(top_k or 1),
            int(extra.get("prefilter_window_budget") or max(int(top_k or 1) * 4, 12)),
        ),
    )
    top_frames = max(
        int(top_k or 1) * 4,
        int(extra.get("prefilter_top_frames") or max_windows * 2),
    )

    harness = _reference_harness_cls()(
        task=task,
        runtime=runtime,
        clip_duration_s=max(1.0, float(window_s)),
        embedder_model=embedder_model,
    )
    try:
        frame_items, _ = harness._list_dense_frame_paths(harness.dataset_folder, harness.video_path)
        dense_frame_cache_hit = bool(frame_items)
        if not frame_items:
            harness._ensure_dense_frames()
            frame_items, _ = harness._list_dense_frame_paths(harness.dataset_folder, harness.video_path)
        candidates = [
            {
                "frame_path": str(frame_path),
                "timestamp": float(harness._timestamp_from_dense_frame_path(frame_path)),
            }
            for frame_path in frame_items
        ]
        if not candidates:
            metadata["reason"] = "no_dense_frames"
            return all_windows, metadata

        embedding_cache_ready = bool(harness._precompute_frame_embeddings_cache(candidates))
        scored_frames = harness._qwen_score_frames(query, candidates, min(len(candidates), int(top_frames)))
        candidate_indices = _candidate_window_indices(
            scored_frames,
            all_windows,
            neighbor_radius=neighbor_radius,
            max_windows=max_windows,
        )
        if not candidate_indices:
            metadata["reason"] = "no_prefilter_matches"
            metadata["dense_frame_cache_hit"] = dense_frame_cache_hit
            metadata["dense_frame_count"] = len(candidates)
            metadata["embedding_cache_ready"] = embedding_cache_ready
            return all_windows, metadata
        selected_windows = [all_windows[index] for index in candidate_indices]
        metadata.update(
            {
                "enabled": True,
                "dense_frame_cache_hit": dense_frame_cache_hit,
                "dense_frame_count": len(candidates),
                "embedding_cache_ready": embedding_cache_ready,
                "top_frame_count": len(scored_frames),
                "candidate_windows": len(selected_windows),
                "embedder_model": embedder_model,
            }
        )
        return selected_windows, metadata
    except Exception as exc:
        metadata["reason"] = "prefilter_error"
        metadata["error"] = str(exc)
        return all_windows, metadata
    finally:
        if getattr(harness, "_release_frame_embedder", None):
            try:
                harness._release_frame_embedder()
            except Exception:
                pass


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    video_path = str(task.get("video_path") or "").strip()
    query = str(request.get("query") or "").strip()
    if not video_path:
        fail_runtime("visual_temporal_grounder requires task.video_path")
    if not query:
        fail_runtime("visual_temporal_grounder requires a non-empty query")

    device_label = resolved_device_label(runtime)
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    window_s = float((runtime.get("extra") or {}).get("clip_duration_s") or 60.0)
    sample_fps = float((runtime.get("extra") or {}).get("fps") or 2.0)
    max_new_tokens = int((runtime.get("extra") or {}).get("max_new_tokens") or 256)
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    duration_s = float(get_video_duration(video_path) or 0.0)
    video_id = str(task.get("video_id") or task.get("sample_key") or "")
    top_k = max(1, int(request.get("top_k") or 5))
    candidate_windows, prefilter_metadata = _prefilter_windows(
        task=task,
        runtime=runtime,
        query=query,
        duration_s=duration_s,
        window_s=window_s,
        top_k=top_k,
    )

    runner = None
    owns_runner = False
    if runner_pool is not None:
        runner = runner_pool.acquire_qwen_style_runner(
            tool_name="visual_temporal_grounder",
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=attn_implementation,
        )
    if runner is None:
        runner = QwenStyleRunner(
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=attn_implementation,
        )
        owns_runner = True
    try:
        raw_candidates: List[Dict[str, Any]] = []
        for window_start_s, window_end_s in candidate_windows:
            prompt = (
                "You are a precise visual temporal grounding model.\n"
                "Find every interval in this clip where the queried visual event is present.\n"
                "If the event is absent, return an empty interval list.\n"
                "Return JSON only with keys: found, intervals, explanation.\n"
                "Each intervals item must be an object with start_s, end_s, confidence.\n"
                "Use seconds relative to this clip, not the full video.\n\n"
                f"Query: {query}"
            )
            with extracted_clip(video_path, window_start_s, window_end_s) as clip_path:
                raw_text = runner.generate(
                    make_qwen_video_message(prompt, clip_path, fps=sample_fps),
                    max_new_tokens=max_new_tokens,
                )
            raw_candidates.extend(_window_candidates(raw_text, window_start_s=window_start_s))
        merged = merge_intervals(
            [(item["start_s"], item["end_s"]) for item in raw_candidates],
            tolerance_s=0.75,
        )
        clips: List[Dict[str, Any]] = []
        for start_s, end_s in merged[:top_k]:
            confidence = max(
                (
                    float(item.get("confidence") or 0.5)
                    for item in raw_candidates
                    if float(item["start_s"]) <= float(end_s) and float(item["end_s"]) >= float(start_s)
                ),
                default=0.5,
            )
            clips.append(
                {
                    "video_id": video_id,
                    "start_s": round(float(start_s), 3),
                    "end_s": round(float(end_s), 3),
                    "confidence": round(float(confidence), 4),
                    "metadata": {
                        "tool_backend": "timelens_transformers",
                        "model_path": model_path,
                    },
                }
            )
        return {
            "query": query,
            "clips": clips,
            "video_duration": duration_s,
            "retrieval_backend": "timelens_transformers",
            "query_absent": not bool(clips),
            "summary": summarize_intervals([(item["start_s"], item["end_s"]) for item in clips]),
            "prefilter": prefilter_metadata,
        }
    finally:
        if owns_runner:
            runner.close()


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
