from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for video frame sampling")


def get_video_duration(video_path: str) -> float:
    _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: %s" % video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return 0.0
    return max(0.0, frame_count / fps)


def sample_frame_times(start_s: float, end_s: float, num_frames: int) -> list[float]:
    count = max(1, int(num_frames or 1))
    start = max(0.0, float(start_s or 0.0))
    end = max(start, float(end_s if end_s is not None else start))
    if count == 1 or end <= start:
        return [(start + end) / 2.0]
    step = (end - start) / float(count - 1)
    return [start + (index * step) for index in range(count)]


def sample_frames(video_path: str, start_s: float, end_s: float, num_frames: int, out_dir: str | Path, prefix: str) -> list[dict[str, Any]]:
    _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: %s" % video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        capture.release()
        raise RuntimeError("Could not determine FPS for video: %s" % video_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sampled = []
    for index, timestamp in enumerate(sample_frame_times(start_s, end_s, num_frames), start=1):
        frame_index = max(0, int(round(timestamp * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        path = out_path / ("%s_%02d_%07.3f.jpg" % (prefix, index, float(timestamp)))
        cv2.imwrite(str(path), frame)
        sampled.append({"frame_path": str(path.resolve()), "timestamp_s": float(timestamp)})
    capture.release()
    return sampled


def parse_time_hint_seconds(value: str) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    colon = re.search(r"(?<!\d)(\d{1,3}(?::\d{1,2}){1,2}(?:\.\d+)?)(?!\d)", text)
    if colon:
        parts = [float(item) for item in colon.group(1).split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    if match:
        return float(match.group(1))
    return None


def _video_id(task: dict[str, Any]) -> str:
    return str(task.get("video_id") or task.get("sample_key") or "video")


def _duration(task: dict[str, Any]) -> float:
    try:
        return float(get_video_duration(str(task.get("video_path") or "")) or 0.0)
    except Exception:
        return 0.0


def _normalize_clip(value: dict[str, Any], task: dict[str, Any], duration: float) -> dict[str, Any]:
    start_s = max(0.0, float(value.get("start_s") or 0.0))
    end_value = value.get("end_s")
    end_s = float(end_value) if end_value is not None else duration
    if duration > 0.0:
        end_s = min(duration, end_s)
    if end_s < start_s:
        end_s = start_s
    clip = dict(value)
    clip["video_id"] = clip.get("video_id") or _video_id(task)
    clip["start_s"] = start_s
    clip["end_s"] = end_s
    clip.setdefault("metadata", {})
    return clip


def _scope_clips(scope: dict[str, Any], task: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    raw_clips = list(scope.get("clips") or [])
    if not raw_clips:
        raw_clips = list(scope.get("segments") or [])
    return [_normalize_clip(dict(item or {}), task, duration) for item in raw_clips if isinstance(item, dict)]


def _anchor_seconds(anchor: dict[str, Any]) -> float | None:
    if anchor.get("time_s") is not None:
        return float(anchor.get("time_s") or 0.0)
    if anchor.get("timecode") is not None:
        return parse_time_hint_seconds(str(anchor.get("timecode") or ""))
    return None


def clip_from_request(request: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    duration = _duration(task)
    scope = dict(request.get("temporal_scope") or {})
    clips = _scope_clips(scope, task, duration)
    if clips:
        base_clip = clips[0]
    else:
        base_clip = {"video_id": _video_id(task), "start_s": 0.0, "end_s": duration, "metadata": {}}
    anchors = [dict(item or {}) for item in list(scope.get("anchors") or []) if isinstance(item, dict)]
    if not anchors:
        return base_clip
    anchor = anchors[0]
    anchor_s = _anchor_seconds(anchor)
    if anchor_s is None:
        return base_clip
    if str(anchor.get("reference") or "video").lower() in {"clip", "scope"}:
        anchor_s = float(base_clip.get("start_s") or 0.0) + anchor_s
    radius_s = float(anchor.get("radius_s") if anchor.get("radius_s") is not None else scope.get("default_radius_s") or 2.0)
    base_start = float(base_clip.get("start_s") or 0.0)
    base_end = float(base_clip.get("end_s") if base_clip.get("end_s") is not None else anchor_s + radius_s)
    if not clips and base_end <= base_start:
        base_end = anchor_s + radius_s
    start_s = max(base_start, anchor_s - radius_s)
    end_s = min(base_end, anchor_s + radius_s)
    if end_s <= start_s:
        return base_clip
    metadata = dict(base_clip.get("metadata") or {})
    metadata["anchor"] = anchor
    return {"video_id": base_clip.get("video_id") or _video_id(task), "start_s": start_s, "end_s": end_s, "metadata": metadata}


def clips_from_request(request: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    duration = _duration(task)
    scope = dict(request.get("temporal_scope") or {})
    anchors = [dict(item or {}) for item in list(scope.get("anchors") or []) if isinstance(item, dict)]
    if anchors:
        return [clip_from_request(request, task)]
    clips = _scope_clips(scope, task, duration)
    if clips:
        return clips
    return [{"video_id": _video_id(task), "start_s": 0.0, "end_s": duration, "metadata": {}}]


def extract_audio_clip(video_path: str, ffmpeg_bin: str, start_s: float, end_s: float) -> str:
    start = max(0.0, float(start_s or 0.0))
    end = max(start, float(end_s or start))
    duration = end - start
    if duration <= 0.0:
        raise ValueError("audio clip duration must be positive")
    tmp_dir = tempfile.mkdtemp(prefix="vtps_audio_")
    audio_path = str(Path(tmp_dir) / "clip.wav")
    command = [
        str(ffmpeg_bin or "ffmpeg"),
        "-y",
        "-ss",
        "%.3f" % start,
        "-i",
        str(video_path),
        "-t",
        "%.3f" % duration,
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg audio extraction failed: %s" % (completed.stderr or completed.stdout or "").strip())
    return audio_path
