from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..tools.local_asr import _clip_from_time_hint
from .protocol import emit_json, fail_runtime, load_request
from .shared import resolved_device_label


def _normalize_query_text(value: str) -> str:
    cleaned = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in cleaned)
    return " ".join(cleaned.split())


def _parse_colon_timestamp_seconds(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    for match in re.finditer(r"(?<!\d)(\d{1,3}(?::\d{1,2}){1,2}(?:\.\d+)?)(?!\d)", text):
        parts = match.group(1).split(":")
        try:
            numbers = [float(part) for part in parts]
        except Exception:
            continue
        if len(numbers) == 2:
            minutes, seconds = numbers
            return (minutes * 60.0) + seconds
        if len(numbers) == 3:
            hours, minutes, seconds = numbers
            return (hours * 3600.0) + (minutes * 60.0) + seconds
    return None


def _parse_seconds_timestamp(value: str) -> Optional[float]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    colon_seconds = _parse_colon_timestamp_seconds(text)
    if colon_seconds is not None:
        return colon_seconds
    patterns = (
        r"(?:frame|timestamp|time|at|around|near|second|sec|s)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _has_explicit_timestamp_hint(time_hints: List[str]) -> bool:
    return any(_parse_seconds_timestamp(item) is not None for item in list(time_hints or []))


def _query_prefers_structured_visual_context(query: str) -> bool:
    normalized = _normalize_query_text(query)
    if not normalized:
        return False
    markers = (
        "chart",
        "graph",
        "table",
        "dashboard",
        "scoreboard",
        "diagram",
        "slide",
        "infographic",
        "timeline",
        "menu",
        "screen",
    )
    return any(marker in normalized for marker in markers)


def _time_hint_anchor_seconds(time_hint: str, clip_start_s: float, clip_end_s: float) -> Optional[float]:
    raw_hint = str(time_hint or "").strip()
    hint = _normalize_query_text(raw_hint)
    if not hint:
        return None
    clip_span_s = max(0.0, clip_end_s - clip_start_s)
    midpoint_s = clip_start_s + (clip_span_s / 2.0)

    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", hint)
    if percent_match:
        fraction = max(0.0, min(1.0, float(percent_match.group(1)) / 100.0))
        return clip_start_s + (clip_span_s * fraction)

    parsed_seconds = _parse_seconds_timestamp(raw_hint)
    if parsed_seconds is not None and clip_span_s >= 0.0:
        offset_s = float(parsed_seconds)
        if any(token in hint for token in ("after", "from start", "into", "after the start")):
            return min(clip_end_s, clip_start_s + offset_s)
        if any(token in hint for token in ("before end", "before the end", "from end")):
            return max(clip_start_s, clip_end_s - offset_s)
        if clip_start_s <= offset_s <= clip_end_s:
            return offset_s
        if any(token in hint for token in ("clip", "window", "segment", "local")) and offset_s <= clip_span_s:
            return min(clip_end_s, clip_start_s + offset_s)
        if clip_start_s <= 0.001 and offset_s <= clip_end_s:
            return offset_s

    if any(token in hint for token in ("start", "begin", "opening", "onset", "first")):
        return clip_start_s
    if any(token in hint for token in ("end", "finish", "closing", "last")):
        return clip_end_s
    if any(token in hint for token in ("middle", "midpoint", "mid", "center", "centre")):
        return midpoint_s
    return None


def _candidate_frames_for_time_hints(
    bounded: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    time_hints: List[str],
) -> tuple[List[Dict[str, Any]], List[float]]:
    anchors = [
        float(anchor)
        for anchor in (
            _time_hint_anchor_seconds(item, clip_start_s, clip_end_s)
            for item in list(time_hints or [])
        )
        if anchor is not None
    ]
    if not anchors:
        return bounded, []

    clip_span_s = max(0.0, clip_end_s - clip_start_s)
    anchor_window_s = max(1.0, min(2.5, clip_span_s * 0.2 if clip_span_s > 0.0 else 1.0))
    filtered = [
        item
        for item in list(bounded or [])
        if min(abs(float(item["timestamp"]) - anchor) for anchor in anchors) <= anchor_window_s
    ]
    return filtered or bounded, anchors


def _rank_without_query(
    bounded: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    time_hints: List[str],
) -> List[Dict[str, Any]]:
    candidates, anchors = _candidate_frames_for_time_hints(bounded, clip_start_s, clip_end_s, time_hints)
    if not anchors:
        anchors = [clip_start_s + ((clip_end_s - clip_start_s) / 2.0)]
    ranked = []
    for item in list(candidates or []):
        timestamp_s = float(item["timestamp"])
        anchor_distance_s = min(abs(timestamp_s - anchor) for anchor in anchors)
        ranked.append(
            {
                "frame_path": item["frame_path"],
                "timestamp": timestamp_s,
                "relevance_score": 0.0,
                "anchor_distance_s": round(anchor_distance_s, 3),
                "temporal_score": max(0.0, 1.0 - anchor_distance_s),
                "selection_reason": "time_hint" if time_hints else "clip_midpoint",
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item.get("anchor_distance_s") or 0.0),
            float(item["timestamp"]),
        )
    )
    return ranked


def _rerank_with_temporal_context(
    scored: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    query: str,
    time_hints: List[str],
) -> List[Dict[str, Any]]:
    if not scored:
        return []

    structured_visual = _query_prefers_structured_visual_context(query)
    clip_span_s = max(0.0, clip_end_s - clip_start_s)
    neighbor_window_s = max(1.0, min(2.0, clip_span_s * 0.15 if clip_span_s > 0.0 else 1.0))
    anchor_candidates, anchors = _candidate_frames_for_time_hints(scored, clip_start_s, clip_end_s, time_hints)
    anchor_lookup = {
        str(item["frame_path"]): item
        for item in list(anchor_candidates or [])
    }
    anchor_window_s = max(1.0, min(2.5, clip_span_s * 0.2 if clip_span_s > 0.0 else 1.0))

    by_time = sorted(list(scored or []), key=lambda item: float(item["timestamp"]))
    reranked: List[Dict[str, Any]] = []
    for item in by_time:
        timestamp_s = float(item["timestamp"])
        base_score = float(item.get("relevance_score") or 0.0)
        neighbor_scores = [
            float(other.get("relevance_score") or 0.0)
            for other in by_time
            if other is not item and abs(float(other["timestamp"]) - timestamp_s) <= neighbor_window_s
        ]
        stability_bonus = 0.0
        if neighbor_scores:
            stability_bonus = (0.18 if structured_visual else 0.08) * (
                sum(neighbor_scores) / float(len(neighbor_scores))
            )

        anchor_distance_s = None
        anchor_bonus = 0.0
        if anchors:
            anchor_distance_s = min(abs(timestamp_s - anchor) for anchor in anchors)
            anchor_bonus = 0.25 * max(0.0, 1.0 - (anchor_distance_s / anchor_window_s))
            if str(item["frame_path"]) not in anchor_lookup:
                anchor_bonus *= 0.5

        progressive_bonus = 0.0
        if structured_visual and clip_span_s > 0.0:
            progressive_bonus = 0.04 * ((timestamp_s - clip_start_s) / clip_span_s)

        reranked.append(
            {
                "frame_path": item["frame_path"],
                "timestamp": timestamp_s,
                "relevance_score": base_score,
                "anchor_distance_s": round(anchor_distance_s, 3) if anchor_distance_s is not None else None,
                "temporal_score": round(base_score + stability_bonus + anchor_bonus + progressive_bonus, 6),
                "selection_reason": (
                    "structured_visual_temporal_rerank"
                    if structured_visual
                    else "temporal_rerank"
                ),
            }
        )

    reranked.sort(
        key=lambda item: (
            -float(item.get("temporal_score") or 0.0),
            float(item.get("anchor_distance_s"))
            if item.get("anchor_distance_s") is not None
            else float("inf"),
            float(item["timestamp"]),
        )
    )
    return reranked


def _select_diverse_frames(
    ranked: List[Dict[str, Any]],
    num_frames: int,
    *,
    query: str,
) -> List[Dict[str, Any]]:
    if not ranked:
        return []
    structured_visual = _query_prefers_structured_visual_context(query)
    min_gap_s = 1.0 if structured_visual else 0.5
    selected: List[Dict[str, Any]] = []
    if structured_visual:
        best_score = max(float(item.get("temporal_score") or 0.0) for item in list(ranked or []))
        margin = max(0.02, best_score * 0.03)
        high_quality = [
            item
            for item in sorted(list(ranked or []), key=lambda candidate: float(candidate["timestamp"]))
            if (best_score - float(item.get("temporal_score") or 0.0)) <= margin
        ]
        if high_quality:
            groups: List[List[Dict[str, Any]]] = []
            current_group: List[Dict[str, Any]] = []
            for item in high_quality:
                if not current_group:
                    current_group = [item]
                    continue
                prev_ts = float(current_group[-1]["timestamp"])
                curr_ts = float(item["timestamp"])
                if abs(curr_ts - prev_ts) <= 1.25:
                    current_group.append(item)
                else:
                    groups.append(current_group)
                    current_group = [item]
            if current_group:
                groups.append(current_group)

            def _group_key(group: List[Dict[str, Any]]) -> tuple:
                scores = [float(item.get("temporal_score") or 0.0) for item in group]
                return (
                    -(sum(scores) / float(len(scores))),
                    -len(group),
                    -max(scores),
                    float(group[0]["timestamp"]),
                )

            best_group = sorted(groups, key=_group_key)[0]
            timestamps = [float(item["timestamp"]) for item in best_group]
            midpoint = (
                timestamps[len(timestamps) // 2]
                if len(timestamps) % 2 == 1
                else (timestamps[(len(timestamps) // 2) - 1] + timestamps[len(timestamps) // 2]) / 2.0
            )
            plateau_anchor = dict(
                sorted(
                    best_group,
                    key=lambda item: (
                        abs(float(item["timestamp"]) - midpoint),
                        -float(item.get("temporal_score") or 0.0),
                        float(item["timestamp"]),
                    ),
                )[0]
            )
            plateau_anchor["selection_reason"] = "structured_visual_plateau_center"
            selected.append(plateau_anchor)
            if len(selected) >= max(1, int(num_frames or 1)):
                return selected
    for item in list(ranked or []):
        timestamp_s = float(item["timestamp"])
        if any(abs(float(existing["timestamp"]) - timestamp_s) < min_gap_s for existing in selected):
            continue
        selected.append(item)
        if len(selected) >= max(1, int(num_frames or 1)):
            return selected
    for item in list(ranked or []):
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max(1, int(num_frames or 1)):
            break
    return selected


def _anchor_seconds_for_time_hints(
    time_hints: List[str],
    clip_start_s: float,
    clip_end_s: float,
) -> List[float]:
    anchors = []
    seen = set()
    for item in list(time_hints or []):
        anchor = _time_hint_anchor_seconds(item, clip_start_s, clip_end_s)
        if anchor is None:
            continue
        rounded = round(float(anchor), 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        anchors.append(rounded)
    return anchors


def _select_anchor_window_frames(
    candidates: List[Dict[str, Any]],
    anchors: List[float],
    *,
    clip_start_s: float,
    clip_end_s: float,
    num_frames: int,
    neighbor_radius_s: float,
    include_anchor_neighbors: bool,
    sort_order: str,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    by_time = sorted(list(candidates or []), key=lambda item: float(item["timestamp"]))
    if not anchors:
        anchors = [clip_start_s + ((clip_end_s - clip_start_s) / 2.0)]

    radius_s = max(0.0, float(neighbor_radius_s or 0.0))
    selected_by_path: Dict[str, Dict[str, Any]] = {}
    for anchor_s in anchors:
        window = [
            item
            for item in by_time
            if radius_s <= 0.0 or abs(float(item["timestamp"]) - float(anchor_s)) <= radius_s
        ]
        if not window:
            window = by_time

        closest = min(window, key=lambda item: (abs(float(item["timestamp"]) - float(anchor_s)), float(item["timestamp"])))
        must_keep = [closest]
        if include_anchor_neighbors:
            before = [item for item in by_time if float(item["timestamp"]) < float(anchor_s)]
            after = [item for item in by_time if float(item["timestamp"]) > float(anchor_s)]
            if before:
                must_keep.append(max(before, key=lambda item: float(item["timestamp"])))
            if after:
                must_keep.append(min(after, key=lambda item: float(item["timestamp"])))

        target_count = max(int(num_frames or 1), len({str(item["frame_path"]) for item in must_keep}))
        if include_anchor_neighbors:
            target_count = max(target_count, min(5, len(window)))
        fill = sorted(
            window,
            key=lambda item: (
                abs(float(item["timestamp"]) - float(anchor_s)),
                -float(item.get("relevance_score") or item.get("temporal_score") or 0.0),
                float(item["timestamp"]),
            ),
        )
        for item in must_keep + fill:
            frame_path = str(item["frame_path"])
            if frame_path in selected_by_path:
                continue
            timestamp_s = float(item["timestamp"])
            anchor_distance_s = abs(timestamp_s - float(anchor_s))
            role = "anchor" if frame_path == str(closest["frame_path"]) else ("before" if timestamp_s < anchor_s else "after")
            selected = dict(item)
            selected.update(
                {
                    "timestamp": timestamp_s,
                    "anchor_distance_s": round(anchor_distance_s, 3),
                    "temporal_score": item.get("temporal_score", item.get("relevance_score")),
                    "selection_reason": "anchor_window_sequence",
                    "requested_timestamp_s": round(float(anchor_s), 3),
                    "neighbor_radius_s": round(radius_s, 3),
                    "sequence_mode": "anchor_window",
                    "sequence_role": role,
                    "sequence_sort_order": "chronological" if sort_order == "chronological" else "ranked",
                }
            )
            selected_by_path[frame_path] = selected
            if len(selected_by_path) >= target_count:
                break

    selected_frames = list(selected_by_path.values())
    if sort_order == "chronological":
        selected_frames.sort(key=lambda item: float(item["timestamp"]))
    else:
        selected_frames.sort(
            key=lambda item: (
                float(item.get("anchor_distance_s") or 0.0),
                -float(item.get("relevance_score") or item.get("temporal_score") or 0.0),
                float(item["timestamp"]),
            )
        )
    for index, item in enumerate(selected_frames):
        item["sequence_index"] = index
    return selected_frames


def _resolved_clip(request: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    clip = dict((list(request.get("clips") or []) or [{}])[0] or {})
    if clip:
        return clip
    time_hints = [str(item).strip() for item in list(request.get("time_hints") or []) if str(item).strip()]
    time_hint = time_hints[0] if time_hints else None
    derived = _clip_from_time_hint(
        str(task.get("video_id") or task.get("sample_key") or "video"),
        str(task.get("video_path") or ""),
        time_hint,
    )
    if derived is None:
        return {}
    if hasattr(derived, "model_dump"):
        return derived.model_dump()
    return derived.dict()


def _reference_harness_cls():
    from .reference_adapter import ReferenceHarness

    return ReferenceHarness


def execute_payload(payload: Dict[str, Any], *, harness=None, release_embedder: bool = True) -> Dict[str, Any]:
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
    time_hints = [
        str(item).strip()
        for item in list(request.get("time_hints") or [])
        if str(item).strip()
    ]
    sequence_mode = str(request.get("sequence_mode") or "ranked").strip().lower()
    sort_order = str(request.get("sort_order") or "ranked").strip().lower()
    include_anchor_neighbors = bool(request.get("include_anchor_neighbors", True))
    try:
        neighbor_radius_s = float(request.get("neighbor_radius_s", 2.0) or 2.0)
    except Exception:
        neighbor_radius_s = 2.0
    owns_harness = harness is None
    if harness is None:
        harness = _reference_harness_cls()(
            task=task,
            runtime=runtime,
            clip_duration_s=max(1.0, clip_end_s - clip_start_s),
            embedder_model=str(runtime.get("model_name") or ""),
            reranker_model=str((runtime.get("extra") or {}).get("reranker_model") or ""),
        )
    try:
        started_at = time.perf_counter()
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
        embedding_cache_ready = bool(getattr(harness, "_frame_embedding_cache_ready", lambda: False)())
        anchors = _anchor_seconds_for_time_hints(time_hints, clip_start_s, clip_end_s)
        anchor_window_requested = (
            sequence_mode in {"anchor_window", "chronological"}
            or _has_explicit_timestamp_hint(time_hints)
        )
        expand_for_anchor_neighbors = bool(
            anchor_window_requested
            and (sequence_mode == "anchor_window" or _has_explicit_timestamp_hint(time_hints))
        )
        frame_pool_start_s = clip_start_s
        frame_pool_end_s = clip_end_s
        if expand_for_anchor_neighbors:
            expansion_s = max(0.0, float(neighbor_radius_s or 0.0))
            frame_pool_start_s = max(0.0, clip_start_s - expansion_s)
            frame_pool_end_s = clip_end_s + expansion_s
        bounded = [
            item for item in candidates if frame_pool_start_s <= float(item["timestamp"]) <= frame_pool_end_s
        ] or candidates
        effective_sort_order = (
            "chronological"
            if anchor_window_requested and sort_order == "ranked"
            else "chronological"
            if sort_order == "chronological" or sequence_mode == "chronological"
            else sort_order
        )

        if anchor_window_requested:
            if query:
                scored = harness._qwen_score_frames(query, bounded, len(bounded), persist_cache=False)
                candidate_pool = scored
            else:
                candidate_pool = [
                    {
                        "frame_path": item["frame_path"],
                        "timestamp": float(item["timestamp"]),
                        "relevance_score": 0.0,
                    }
                    for item in bounded
                ]
            ranked = _select_anchor_window_frames(
                candidate_pool,
                anchors,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                num_frames=num_frames,
                neighbor_radius_s=neighbor_radius_s,
                include_anchor_neighbors=include_anchor_neighbors,
                sort_order=effective_sort_order,
            )
        elif query:
            scored = harness._qwen_score_frames(query, bounded, len(bounded), persist_cache=False)
            ranked = _select_diverse_frames(
                _rerank_with_temporal_context(scored, clip_start_s, clip_end_s, query, time_hints),
                num_frames,
                query=query,
            )
        else:
            ranked = _select_diverse_frames(
                _rank_without_query(bounded, clip_start_s, clip_end_s, time_hints),
                num_frames,
                query=query,
            )

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
                        "frame_pool_start_s": round(float(frame_pool_start_s), 3),
                        "frame_pool_end_s": round(float(frame_pool_end_s), 3),
                        "temporal_score": item.get("temporal_score"),
                        "anchor_distance_s": item.get("anchor_distance_s"),
                        "selection_reason": item.get("selection_reason"),
                        **(
                            {
                                "requested_timestamp_s": item.get("requested_timestamp_s"),
                                "neighbor_radius_s": item.get("neighbor_radius_s"),
                                "sequence_mode": item.get("sequence_mode"),
                                "sequence_role": item.get("sequence_role"),
                                "sequence_index": item.get("sequence_index"),
                                "sequence_sort_order": item.get("sequence_sort_order"),
                            }
                            if item.get("sequence_mode")
                            else {}
                        ),
                    },
                }
            )

        if anchor_window_requested:
            rationale = (
                "Frames were returned as a chronological neighboring sequence around the requested timestamp "
                "or temporal anchor."
            )
        elif query and time_hints:
            rationale = (
                "Frames were ranked within the requested clip using the configured Qwen visual embedder, "
                "with time-hint-aware temporal reranking."
            )
        elif query and _query_prefers_structured_visual_context(query):
            rationale = (
                "Frames were ranked within the requested clip using the configured Qwen visual embedder, "
                "with temporal-context reranking for structured visuals."
            )
        elif query:
            rationale = "Frames were ranked within the requested clip using the configured Qwen visual embedder."
        elif time_hints:
            rationale = "Frames were selected near the requested clip-local time hints."
        else:
            rationale = "No query was provided, so the frames nearest the clip midpoint were returned."

        return {
            "query": query or None,
            "frames": frames,
            "mode": "clip_bounded",
            "cache_metadata": {
                "dense_frame_cache_hit": dense_frame_cache_hit,
                "dense_frame_count": len(candidates),
                "bounded_frame_count": len(bounded),
                "frame_pool_start_s": round(float(frame_pool_start_s), 3),
                "frame_pool_end_s": round(float(frame_pool_end_s), 3),
                "expanded_frame_pool_for_anchor_window": expand_for_anchor_neighbors,
                "embedding_cache_ready": embedding_cache_ready,
                "embedding_cache_scope": "full_video" if embedding_cache_ready else "none",
                "persist_cache_on_bounded_request": False,
                "time_hints_applied": bool(time_hints),
                "sequence_mode": "anchor_window" if anchor_window_requested else sequence_mode,
                "anchor_window_applied": bool(anchor_window_requested),
                "requested_anchor_timestamps_s": anchors,
                "neighbor_radius_s": neighbor_radius_s,
                "timings": {
                    "total_s": round(time.perf_counter() - started_at, 4),
                },
                "embedder": (
                    harness._frame_embedder_runtime_metadata()
                    if getattr(harness, "_frame_embedder_runtime_metadata", None)
                    else {}
                ),
            },
            "rationale": rationale,
        }
    finally:
        if owns_harness and release_embedder and getattr(harness, "_release_frame_embedder", None):
            try:
                harness._release_frame_embedder()
            except Exception:
                pass


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
