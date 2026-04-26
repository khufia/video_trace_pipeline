from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def normalize_time_intervals(intervals: Iterable[Tuple[Any, Any]]) -> List[Dict[str, float]]:
    normalized: List[Tuple[float, float]] = []
    for raw_start, raw_end in list(intervals or []):
        start_s = coerce_float(raw_start)
        end_s = coerce_float(raw_end)
        if start_s is None and end_s is None:
            continue
        if start_s is None:
            start_s = end_s
        if end_s is None:
            end_s = start_s
        assert start_s is not None
        assert end_s is not None
        start_s = round(start_s, 3)
        end_s = round(end_s, 3)
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        normalized.append((start_s, end_s))

    if not normalized:
        return []

    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: List[List[float]] = []
    for start_s, end_s in normalized:
        if not merged or start_s > merged[-1][1]:
            merged.append([start_s, end_s])
            continue
        merged[-1][1] = max(merged[-1][1], end_s)

    return [{"start_s": start_s, "end_s": end_s} for start_s, end_s in merged]


def temporal_payload_from_records(
    records: Iterable[Any],
    *,
    start_key: str = "time_start_s",
    end_key: str = "time_end_s",
    frame_key: str = "frame_ts_s",
) -> Dict[str, Any]:
    intervals: List[Tuple[Any, Any]] = []
    frame_timestamps: List[float] = []

    for item in list(records or []):
        start_s = coerce_float(_extract_value(item, start_key))
        end_s = coerce_float(_extract_value(item, end_key))
        frame_ts_s = coerce_float(_extract_value(item, frame_key))
        if start_s is not None or end_s is not None:
            intervals.append((start_s, end_s))
            continue
        if frame_ts_s is not None:
            frame_timestamps.append(round(frame_ts_s, 3))

    payload: Dict[str, Any] = {}
    normalized_intervals = normalize_time_intervals(intervals)
    if normalized_intervals:
        if len(normalized_intervals) == 1:
            payload["time_start_s"] = normalized_intervals[0]["start_s"]
            payload["time_end_s"] = normalized_intervals[0]["end_s"]
        else:
            payload["time_intervals"] = normalized_intervals

    unique_frames = sorted(set(frame_timestamps))
    if not normalized_intervals:
        if len(unique_frames) == 1:
            payload["frame_ts_s"] = unique_frames[0]
        elif unique_frames:
            payload["time_intervals"] = [
                {"start_s": timestamp_s, "end_s": timestamp_s} for timestamp_s in unique_frames
            ]

    return payload


def _format_seconds(value: Any) -> str:
    text = "%.3f" % float(value)
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _render_interval(start_s: Any, end_s: Any) -> str:
    start_s = coerce_float(start_s)
    end_s = coerce_float(end_s)
    if start_s is None and end_s is None:
        return ""
    if start_s is None:
        start_s = end_s
    if end_s is None:
        end_s = start_s
    assert start_s is not None
    assert end_s is not None
    if float(start_s) == float(end_s):
        return _format_seconds(start_s)
    return "%s to %s" % (_format_seconds(start_s), _format_seconds(end_s))


def render_temporal_anchor(item: Dict[str, Any] | None) -> str:
    payload = dict(item or {})
    rendered_intervals = [
        _render_interval(interval.get("start_s"), interval.get("end_s"))
        for interval in list(payload.get("time_intervals") or [])
        if isinstance(interval, dict)
    ]
    rendered_intervals = [interval for interval in rendered_intervals if interval]
    if rendered_intervals:
        return "; ".join(rendered_intervals)
    if payload.get("time_start_s") is not None or payload.get("time_end_s") is not None:
        return _render_interval(payload.get("time_start_s"), payload.get("time_end_s"))
    if payload.get("frame_ts_s") is not None:
        return _format_seconds(payload.get("frame_ts_s"))
    return ""


def clip_refs_from_intervals(
    intervals: Sequence[Dict[str, Any]],
    *,
    video_id: str,
    extra_fields: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    extras = dict(extra_fields or {})
    for interval in normalize_time_intervals(
        (interval.get("start_s"), interval.get("end_s"))
        for interval in list(intervals or [])
        if isinstance(interval, dict)
    ):
        clips.append(
            {
                "video_id": video_id,
                "start_s": interval["start_s"],
                "end_s": interval["end_s"],
                **extras,
            }
        )
    return clips
