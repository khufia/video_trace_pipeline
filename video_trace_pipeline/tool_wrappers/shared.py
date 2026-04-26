from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

from ..common import extract_json_object, make_run_id
from ..model_cache import download_model_snapshot, resolve_model_snapshot
from ..runtime_devices import resolve_device_label
from ..tools.media import get_video_duration, sample_frames


_INTERVAL_RE = re.compile(
    r"(?P<start>-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*(?:-|to|until|through)\s*"
    r"(?P<end>-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?",
    re.IGNORECASE,
)

_COMMON_BBOX_CANVASES: Tuple[Tuple[int, int], ...] = (
    (640, 360),
    (854, 480),
    (960, 540),
    (1024, 576),
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)


def repo_root_from_runtime(runtime: Dict[str, Any]) -> Path:
    workspace_root = Path(str(runtime.get("workspace_root") or ".")).expanduser().resolve()
    return workspace_root.parent if workspace_root.name == "workspace" else workspace_root


def workspace_root_from_runtime(runtime: Dict[str, Any]) -> Path:
    workspace_root = Path(str(runtime.get("workspace_root") or ".")).expanduser().resolve()
    if workspace_root.name == "workspace":
        return workspace_root
    return workspace_root / "workspace"


def tool_cache_root(runtime: Dict[str, Any], tool_name: str, video_id: str) -> Path:
    base = workspace_root_from_runtime(runtime) / "cache" / "tool_wrappers" / str(tool_name or "tool") / str(video_id or "video")
    base.mkdir(parents=True, exist_ok=True)
    return base


def scratch_dir(runtime: Dict[str, Any], tool_name: str) -> Path:
    raw = runtime.get("scratch_dir")
    if raw:
        path = Path(str(raw)).expanduser().resolve()
    else:
        path = workspace_root_from_runtime(runtime) / "tools" / "_scratch" / str(tool_name or "tool")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolved_device_label(runtime: Dict[str, Any]) -> str:
    return resolve_device_label(runtime.get("device"), default="cpu")


def device_index(device_label: str) -> Optional[int]:
    label = str(device_label or "").strip()
    if not label.startswith("cuda"):
        return None
    if ":" not in label:
        return 0
    try:
        return int(label.split(":", 1)[1])
    except Exception:
        return 0


def torch_dtype_for_device(device_label: str):
    import torch

    if str(device_label or "").startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def move_batch_to_device(batch: Any, device_label: str):
    import torch

    if hasattr(batch, "to"):
        try:
            return batch.to(device_label)
        except Exception:
            pass
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device_label)
            else:
                moved[key] = value
        return moved
    return batch


def resolve_generation_controls(runtime: Dict[str, Any]) -> Dict[str, Any]:
    extra = dict(runtime.get("extra") or {})
    do_sample = bool(extra.get("do_sample")) if "do_sample" in extra else False

    raw_temperature = extra.get("temperature")
    temperature = None
    if raw_temperature not in (None, ""):
        try:
            temperature = float(raw_temperature)
        except Exception:
            temperature = None

    if temperature is not None and temperature <= 0.0:
        do_sample = False
        temperature = None
    if not do_sample:
        temperature = None

    return {
        "do_sample": bool(do_sample),
        "temperature": temperature,
    }


def resolve_model_path(
    model_name: str,
    runtime: Dict[str, Any],
    allow_download: bool = False,
    *,
    prefer_runtime_resolved: bool = True,
) -> str:
    requested = str(model_name or "").strip()
    if not requested:
        raise RuntimeError("Model name is empty.")

    direct = Path(requested).expanduser()
    if direct.exists():
        return str(direct.resolve())

    resolved = str(runtime.get("resolved_model_path") or "").strip()
    runtime_model_name = str(runtime.get("model_name") or "").strip()
    if (
        prefer_runtime_resolved
        and resolved
        and Path(resolved).exists()
        and (
            not runtime_model_name
            or requested == runtime_model_name
        )
    ):
        return str(Path(resolved).resolve())

    snapshot = resolve_model_snapshot(requested, hf_cache=runtime.get("hf_cache"))
    if snapshot is not None:
        return str(snapshot)

    if allow_download and "/" in requested:
        downloaded = download_model_snapshot(requested, hf_cache=runtime.get("hf_cache"))
        return str(downloaded)

    if not allow_download or "/" not in requested:
        raise RuntimeError(
            "Model %r is not available in the local HF cache. "
            "Populate HF_HOME/HUGGINGFACE_HUB_CACHE first instead of downloading at runtime."
            % requested
        )
    raise RuntimeError("Runtime downloads are disabled for %r." % requested)


def resolve_aux_model_path(runtime: Dict[str, Any], field_name: str, allow_download: bool = True) -> Optional[str]:
    extra = dict(runtime.get("extra") or {})
    value = str(extra.get(field_name) or "").strip()
    if not value:
        return None
    return resolve_model_path(value, runtime, allow_download=allow_download, prefer_runtime_resolved=False)


def payload_clip(request: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    clip = dict((list(request.get("clips") or []) or [{}])[0] or {})
    if clip:
        return clip
    return {
        "video_id": str(task.get("video_id") or task.get("sample_key") or "video"),
        "start_s": 0.0,
        "end_s": float(get_video_duration(str(task.get("video_path") or "")) or 0.0),
    }


def absolute_frame_path(frame_payload: Dict[str, Any], runtime: Dict[str, Any]) -> Optional[str]:
    if not isinstance(frame_payload, dict):
        return None

    metadata = dict(frame_payload.get("metadata") or {})
    for candidate in (
        metadata.get("source_path"),
        frame_payload.get("frame_path"),
        frame_payload.get("source_frame_path"),
    ):
        raw = str(candidate or "").strip()
        if raw and Path(raw).exists():
            return str(Path(raw).resolve())

    relpath = str(frame_payload.get("relpath") or "").strip()
    if relpath:
        base = Path(str(runtime.get("workspace_root") or ".")).expanduser().resolve()
        candidate = (base / relpath).resolve()
        if candidate.exists():
            return str(candidate)
    return None


def ensure_frame_for_request(
    request: Dict[str, Any],
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str,
) -> Tuple[str, float]:
    frame_payload = dict((list(request.get("frames") or []) or [{}])[0] or {})
    frame_path = absolute_frame_path(frame_payload, runtime)
    if frame_path:
        timestamp_s = float(frame_payload.get("timestamp_s") or frame_payload.get("timestamp") or 0.0)
        return frame_path, timestamp_s

    clip = payload_clip(request, task)
    video_path = str(task.get("video_path") or "")
    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") or start_s)
    sampled = sample_frames(video_path, start_s, end_s, 1, str(out_dir), prefix=prefix)
    if not sampled:
        raise RuntimeError("Could not materialize a frame for the OCR/generic request.")
    item = sampled[0]
    return str(item["frame_path"]), float(item["timestamp_s"])


def sample_request_frames(
    request: Dict[str, Any],
    task: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str,
    num_frames: int,
) -> List[Dict[str, Any]]:
    clip = payload_clip(request, task)
    video_path = str(task.get("video_path") or "")
    return sample_frames(
        video_path,
        float(clip.get("start_s") or 0.0),
        float(clip.get("end_s") or 0.0),
        max(1, int(num_frames)),
        str(out_dir),
        prefix=prefix,
    )


def iter_windows(duration_s: float, window_s: float) -> Iterator[Tuple[float, float]]:
    duration_s = max(0.0, float(duration_s or 0.0))
    window_s = max(1.0, float(window_s or 1.0))
    start = 0.0
    while start < duration_s or (duration_s == 0.0 and start == 0.0):
        end = min(duration_s, start + window_s) if duration_s > 0.0 else window_s
        yield round(start, 3), round(max(start, end), 3)
        if end <= start:
            break
        start = end


def _ffmpeg_binary(preferred: Optional[str] = None) -> str:
    candidate = str(preferred or "").strip()
    if candidate:
        return candidate
    resolved = shutil.which("ffmpeg")
    if resolved:
        return str(resolved)
    with contextlib.suppress(Exception):
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    return "ffmpeg"


@contextmanager
def extracted_clip(
    video_path: str,
    start_s: float,
    end_s: float,
    *,
    ffmpeg_bin: Optional[str] = None,
    suffix: str = ".mp4",
    include_audio: bool = False,
):
    start_s = float(start_s or 0.0)
    end_s = float(end_s or start_s)
    if start_s <= 0.0 and end_s <= 0.0:
        yield str(Path(video_path).resolve())
        return

    ffmpeg = _ffmpeg_binary(ffmpeg_bin)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vtp_clip_"))
    clip_path = tmp_dir / ("clip_%s%s" % (make_run_id(), suffix))
    cmd = [
        str(ffmpeg),
        "-y",
        "-ss",
        "%.3f" % start_s,
        "-to",
        "%.3f" % end_s,
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
    ]
    if include_audio:
        cmd.extend(
            [
                "-map",
                "0:a?",
                "-c:a",
                "aac",
            ]
        )
    else:
        cmd.append("-an")
    cmd.append(str(clip_path))
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not clip_path.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise RuntimeError(
            "ffmpeg clip extraction failed: %s" % ((completed.stderr or completed.stdout or "").strip()[:4000])
        )
    try:
        yield str(clip_path.resolve())
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def extract_interval_candidates(text: str, *, offset_s: float = 0.0) -> List[Tuple[float, float]]:
    raw = str(text or "").strip()
    if not raw:
        return []

    payload = extract_json_object(raw)
    intervals: List[Tuple[float, float]] = []
    if isinstance(payload, dict):
        value = payload.get("intervals") or payload.get("clips") or payload.get("segments") or []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    start = item.get("start") if item.get("start") is not None else item.get("start_s")
                    end = item.get("end") if item.get("end") is not None else item.get("end_s")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    start, end = item[0], item[1]
                else:
                    continue
                try:
                    intervals.append((float(start) + offset_s, float(end) + offset_s))
                except Exception:
                    continue
    if intervals:
        return intervals

    for match in _INTERVAL_RE.finditer(raw):
        start = float(match.group("start"))
        end = float(match.group("end"))
        intervals.append((start + offset_s, end + offset_s))
    return intervals


def merge_intervals(
    intervals: Iterable[Tuple[float, float]],
    *,
    tolerance_s: float = 0.75,
) -> List[Tuple[float, float]]:
    ordered = sorted(
        (float(start), float(end)) for start, end in intervals if float(end) >= float(start)
    )
    if not ordered:
        return []
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end + float(tolerance_s):
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def normalize_xyxy_bbox(bbox: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        coords = [float(bbox[index]) for index in range(4)]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    left, right = sorted((coords[0], coords[2]))
    top, bottom = sorted((coords[1], coords[3]))
    return left, top, right, bottom


def _scaled_bbox_candidates(
    bbox: Tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> List[Tuple[float, float, float, float]]:
    if width <= 0 or height <= 0:
        return []
    if bbox[0] < 0.0 or bbox[1] < 0.0:
        return []
    max_ratio = max(
        float(bbox[2]) / float(width),
        float(bbox[3]) / float(height),
    )
    if max_ratio < 1.2:
        return []
    aspect_ratio = float(width) / float(height)
    candidates: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for canvas_width, canvas_height in _COMMON_BBOX_CANVASES:
        if canvas_width < width or canvas_height < height:
            continue
        scale_x = float(canvas_width) / float(width)
        scale_y = float(canvas_height) / float(height)
        if abs(scale_x - scale_y) > 0.05:
            continue
        if abs((float(canvas_width) / float(canvas_height)) - aspect_ratio) > 0.05:
            continue
        if bbox[2] > float(canvas_width) or bbox[3] > float(canvas_height):
            continue
        scale = (scale_x + scale_y) / 2.0
        if scale <= 1.0:
            continue
        candidates.append((scale, tuple(value / scale for value in bbox)))
    candidates.sort(key=lambda item: item[0])
    return [scaled_bbox for _, scaled_bbox in candidates]


def fit_bbox_to_image(
    bbox: Optional[Sequence[float]],
    *,
    image_size: Tuple[int, int],
    allow_scaled_canvas: bool = False,
) -> Optional[List[float]]:
    normalized = normalize_xyxy_bbox(bbox)
    if normalized is None:
        return None
    width = max(0, int(image_size[0]))
    height = max(0, int(image_size[1]))
    if width <= 0 or height <= 0:
        return [float(value) for value in normalized]

    candidates = [normalized]
    if allow_scaled_canvas:
        candidates.extend(_scaled_bbox_candidates(normalized, width=width, height=height))

    for candidate in candidates:
        x1, y1, x2, y2 = candidate
        if x1 >= 0.0 and y1 >= 0.0 and x2 <= float(width) and y2 <= float(height):
            return [float(x1), float(y1), float(x2), float(y2)]

    x1, y1, x2, y2 = candidates[0]
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]


def crop_region(frame_path: str, bbox: Optional[Sequence[float]], out_path: Path) -> str:
    if not bbox:
        return str(Path(frame_path).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        fitted_bbox = fit_bbox_to_image(
            bbox,
            image_size=image.size,
            allow_scaled_canvas=True,
        )
        if fitted_bbox is None:
            return str(Path(frame_path).resolve())
        x1_f, y1_f, x2_f, y2_f = fitted_bbox
        x1 = max(0, min(image.width, int(math.floor(x1_f))))
        y1 = max(0, min(image.height, int(math.floor(y1_f))))
        x2 = max(0, min(image.width, int(math.ceil(x2_f))))
        y2 = max(0, min(image.height, int(math.ceil(y2_f))))
        if x2 <= x1 or y2 <= y1:
            # An OCR region that is fully outside the frame should behave like "no text"
            # rather than crashing or scanning the entire image.
            Image.new("RGB", (1, 1), color="white").save(out_path)
            return str(out_path.resolve())
        cropped = image.crop((x1, y1, x2, y2))
        cropped.save(out_path)
        return str(out_path.resolve())


def cleanup_torch() -> None:
    try:
        import torch
    except Exception:
        return
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def summarize_intervals(intervals: Sequence[Tuple[float, float]]) -> str:
    if not intervals:
        return "The queried event is absent."
    parts = ["%.2fs-%.2fs" % (start, end) for start, end in intervals]
    return "The queried event appears at: %s." % ", ".join(parts)
