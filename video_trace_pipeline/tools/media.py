from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - optional in tests
    import cv2
except Exception:  # pragma: no cover - optional in tests
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


def normalize_clip_bounds(video_path: str, start_s: Optional[float], end_s: Optional[float]) -> Tuple[float, float]:
    duration = get_video_duration(video_path)
    start = max(0.0, float(start_s or 0.0))
    end = float(end_s) if end_s is not None else duration
    if duration > 0.0:
        end = min(duration, end)
    if end < start:
        end = start
    return start, end


def sample_frame_times(start_s: float, end_s: float, num_frames: int) -> List[float]:
    count = max(1, int(num_frames))
    if end_s <= start_s:
        return [float(start_s)]
    if count == 1:
        return [float((start_s + end_s) / 2.0)]
    step = (end_s - start_s) / float(count - 1)
    return [float(start_s + idx * step) for idx in range(count)]


def sample_frames(video_path: str, start_s: float, end_s: float, num_frames: int, out_dir: str, prefix: str) -> List[Dict[str, float]]:
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
        frame_path = out_path / ("%s_%02d_%07.3f.jpg" % (prefix, index, float(timestamp)))
        cv2.imwrite(str(frame_path), frame)
        sampled.append({"frame_path": str(frame_path), "timestamp_s": float(timestamp)})
    capture.release()
    return sampled


def midpoint_frame(video_path: str, start_s: float, end_s: float, out_dir: str, prefix: str) -> Optional[Dict[str, float]]:
    frames = sample_frames(video_path, start_s, end_s, 1, out_dir, prefix)
    return frames[0] if frames else None


def extract_audio_clip(video_path: str, ffmpeg_bin: str, start_s: float, end_s: float) -> str:
    start = max(0.0, float(start_s or 0.0))
    end = max(start, float(end_s or start))
    duration = max(0.0, end - start)
    if duration <= 0.0:
        raise ValueError("audio clip duration must be positive")
    tmp_dir = tempfile.mkdtemp(prefix="vtp_audio_")
    audio_path = str(Path(tmp_dir) / "clip.wav")
    cmd = [
        str(ffmpeg_bin),
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
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg audio extraction failed: %s" % (completed.stderr or completed.stdout or "").strip())
    return audio_path


def cleanup_temp_path(path: Optional[str]) -> None:
    if not path:
        return
    path_obj = Path(path)
    if path_obj.is_dir():
        shutil.rmtree(str(path_obj), ignore_errors=True)
        return
    try:
        parent = path_obj.parent
        path_obj.unlink()
        if parent.name.startswith("vtp_audio_"):
            shutil.rmtree(str(parent), ignore_errors=True)
    except Exception:
        pass
