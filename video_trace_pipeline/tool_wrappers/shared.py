from __future__ import annotations

import gc
import json
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
from ..model_cache import resolve_model_snapshot
from ..runtime_devices import resolve_device_label
from ..tools.media import get_video_duration, sample_frames


_INTERVAL_RE = re.compile(
    r"(?P<start>-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*(?:-|to|until|through)\s*"
    r"(?P<end>-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?",
    re.IGNORECASE,
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


def resolve_model_path(model_name: str, runtime: Dict[str, Any], allow_download: bool = False) -> str:
    requested = str(model_name or "").strip()
    if not requested:
        raise RuntimeError("Model name is empty.")

    resolved = str(runtime.get("resolved_model_path") or "").strip()
    if resolved and Path(resolved).exists():
        return str(Path(resolved).resolve())

    direct = Path(requested).expanduser()
    if direct.exists():
        return str(direct.resolve())

    snapshot = resolve_model_snapshot(requested, hf_cache=runtime.get("hf_cache"))
    if snapshot is not None:
        return str(snapshot)

    if not allow_download or "/" not in requested:
        raise RuntimeError(
            "Model %r is not available in the local HF cache. "
            "Populate HF_HOME/HUGGINGFACE_HUB_CACHE first instead of downloading at runtime."
            % requested
        )

    raise RuntimeError(
        "Runtime downloads are disabled for %r. Populate HF_HOME/HUGGINGFACE_HUB_CACHE first."
        % requested
    )


def resolve_aux_model_path(runtime: Dict[str, Any], field_name: str, allow_download: bool = True) -> Optional[str]:
    extra = dict(runtime.get("extra") or {})
    value = str(extra.get(field_name) or "").strip()
    if not value:
        return None
    return resolve_model_path(value, runtime, allow_download=allow_download)


def payload_clip(request: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    clip = dict(request.get("clip") or {})
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
    frame_payload = dict(request.get("frame") or {})
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


@contextmanager
def extracted_clip(
    video_path: str,
    start_s: float,
    end_s: float,
    *,
    ffmpeg_bin: Optional[str] = None,
    suffix: str = ".mp4",
):
    start_s = float(start_s or 0.0)
    end_s = float(end_s or start_s)
    if start_s <= 0.0 and end_s <= 0.0:
        yield str(Path(video_path).resolve())
        return

    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
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
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        str(clip_path),
    ]
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


def crop_region(frame_path: str, bbox: Optional[Sequence[float]], out_path: Path) -> str:
    if not bbox:
        return str(Path(frame_path).resolve())
    image = Image.open(frame_path).convert("RGB")
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(image.width, x1))
    x2 = max(x1, min(image.width, x2))
    y1 = max(0, min(image.height, y1))
    y2 = max(y1, min(image.height, y2))
    cropped = image.crop((x1, y1, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
