from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..common import has_meaningful_text, is_low_signal_text
from ..tools.media import cleanup_temp_path, extract_audio_clip, get_video_duration
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    extract_interval_candidates,
    merge_intervals,
    repo_root_from_runtime,
    resolve_aux_model_path,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
)


logger = logging.getLogger(__name__)

_SPOTSOUND_REPO_URL = "https://github.com/LoieSun/SpotSound.git"
_SPOTSOUND_REPO_NAME = "SpotSound"
_DEFAULT_MAX_NEW_TOKENS = 500


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _cache_root_from_runtime(runtime: Dict[str, Any]) -> Path:
    cache_root = repo_root_from_runtime(runtime) / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _ensure_spotsound_repo(runtime: Dict[str, Any]) -> Path:
    repo_dir = (_cache_root_from_runtime(runtime) / _SPOTSOUND_REPO_NAME).resolve()
    if (repo_dir / "inference.py").is_file() and (repo_dir / "model" / "af3.py").is_file():
        return repo_dir
    if repo_dir.exists() and not (repo_dir / ".git").exists():
        raise RuntimeError("SpotSound checkout %s exists but is incomplete." % repo_dir)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", _SPOTSOUND_REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to clone SpotSound into %s: %s"
            % (repo_dir, (completed.stderr or completed.stdout or "").strip()[:4000])
        )
    return repo_dir


def _load_module_from_path(module_name: str, path: Path):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load module spec for %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_spotsound_classes(repo_dir: Path):
    processor_module = _load_module_from_path(
        "video_trace_pipeline._vendor_spotsound_processor_af3",
        repo_dir / "processor" / "af3.py",
    )
    if getattr(processor_module, "logger", None) is None:
        processor_module.logger = logging.getLogger("spotsound.processor")
    from transformers import AudioFlamingo3ForConditionalGeneration

    return (
        AudioFlamingo3ForConditionalGeneration,
        processor_module.AudioFlamingo3TemporalProcessor,
    )


def _resolve_ffmpeg_binary() -> str:
    resolved = shutil.which("ffmpeg")
    if resolved:
        return str(resolved)
    with contextlib.suppress(Exception):
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    return "ffmpeg"


def _request_clip(request: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    clips = list(request.get("clips") or [])
    if clips:
        return dict(clips[0] or {})
    duration_s = float(get_video_duration(str(task.get("video_path") or "")) or 0.0)
    return {
        "video_id": str(task.get("video_id") or task.get("sample_key") or "video"),
        "start_s": 0.0,
        "end_s": duration_s,
    }


def _prepare_audio_input(request: Dict[str, Any], task: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    video_path = str(task.get("video_path") or "").strip()
    if not video_path:
        raise RuntimeError("audio_temporal_grounder requires task.video_path")

    duration_s = float(get_video_duration(video_path) or 0.0)
    clip = _request_clip(request, task)
    start_s = float(clip.get("start_s") or 0.0)
    end_s = _coerce_float(clip.get("end_s"))
    if end_s is None:
        end_s = duration_s if duration_s > 0.0 else start_s
    if duration_s > 0.0:
        end_s = min(end_s, duration_s)
    if end_s <= start_s:
        if duration_s > start_s:
            end_s = duration_s
        else:
            end_s = start_s + 1.0

    audio_path = extract_audio_clip(video_path, _resolve_ffmpeg_binary(), start_s, end_s)
    return {
        "audio_path": audio_path,
        "cleanup_path": audio_path,
        "video_id": str(clip.get("video_id") or task.get("video_id") or task.get("sample_key") or "video"),
        "clip_start_s": round(start_s, 3),
        "clip_end_s": round(end_s, 3),
    }


def _build_query_prompt(query: str) -> str:
    return (
        "This is a sequence of audio stream. Your task is to identify the temporal window "
        "(start and end timestamps) when the given query appears. The query is: "
        f"{query}. Respond with timestamps in seconds. Answer: "
    )


def _normalize_response_text(value: Any) -> str:
    if isinstance(value, list):
        texts = [str(item or "").strip() for item in value if str(item or "").strip()]
        return "\n".join(texts).strip()
    return str(value or "").strip()


def _extract_intervals_from_payload(payload: Any, *, offset_s: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if isinstance(payload, dict):
        values = payload.get("intervals") or payload.get("clips") or payload.get("segments") or payload.get("events") or []
        return _extract_intervals_from_payload(values, offset_s=offset_s)
    if isinstance(payload, (list, tuple)):
        if len(payload) >= 2 and not isinstance(payload[0], (dict, list, tuple)):
            start = _coerce_float(payload[0])
            end = _coerce_float(payload[1])
            if start is not None and end is not None:
                intervals.append((start + offset_s, end + offset_s))
            return intervals
        for item in payload:
            if isinstance(item, dict):
                start = _coerce_float(item.get("start_s", item.get("start")))
                end = _coerce_float(item.get("end_s", item.get("end")))
                if start is not None and end is not None:
                    intervals.append((start + offset_s, end + offset_s))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start = _coerce_float(item[0])
                end = _coerce_float(item[1])
                if start is not None and end is not None:
                    intervals.append((start + offset_s, end + offset_s))
    return intervals


def _intervals_from_response(raw_text: str, *, clip_start_s: float, clip_end_s: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    with contextlib.suppress(Exception):
        payload = json.loads(str(raw_text or "").strip())
        intervals.extend(_extract_intervals_from_payload(payload, offset_s=clip_start_s))
    if not intervals:
        intervals.extend(extract_interval_candidates(raw_text, offset_s=clip_start_s))

    normalized: List[Tuple[float, float]] = []
    for start_s, end_s in merge_intervals(intervals, tolerance_s=0.5):
        start_value = max(float(clip_start_s), min(float(clip_end_s), float(start_s)))
        end_value = max(start_value, min(float(clip_end_s), float(end_s)))
        normalized.append((round(start_value, 3), round(end_value, 3)))
    return normalized


def _shorten_text(text: str, limit: int = 220) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)].rstrip() + "…"


def _summary_for_intervals(query: str, intervals: Iterable[Tuple[float, float]], raw_response: str) -> str:
    intervals = list(intervals)
    if intervals:
        bounds = ", ".join("%.3f-%.3fs" % (start_s, end_s) for start_s, end_s in intervals[:4])
        suffix = " (%d total)" % len(intervals) if len(intervals) > 4 else ""
        return "SpotSound localized %d candidate audio interval(s) for %r: %s%s." % (
            len(intervals),
            query,
            bounds,
            suffix,
        )
    if has_meaningful_text(raw_response):
        return "SpotSound returned no candidate intervals for %r. Response: %s" % (query, _shorten_text(raw_response))
    return "SpotSound returned no candidate intervals for %r." % query


def _device_map_for_label(device_label: str):
    label = str(device_label or "").strip() or "cpu"
    return {"": label}


def _model_device(model):
    with contextlib.suppress(Exception):
        return model.device
    with contextlib.suppress(Exception):
        return next(model.parameters()).device
    return "cpu"


def _run_spotsound_inference(
    *,
    query: str,
    audio_path: str,
    runtime: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    repo_dir = _ensure_spotsound_repo(runtime)
    model_cls, processor_cls = _load_spotsound_classes(repo_dir)

    base_model_path = resolve_aux_model_path(runtime, "base_model", allow_download=True)
    if not base_model_path:
        raise RuntimeError("SpotSound requires runtime.extra.base_model")
    adapter_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime, allow_download=True)
    device_label = resolved_device_label(runtime)
    generation = resolve_generation_controls(runtime)

    import torch
    from peft import PeftModel

    dtype = torch.float16 if str(device_label).startswith("cuda") else torch.float32
    processor = processor_cls.from_pretrained(base_model_path, local_files_only=True)
    model = model_cls.from_pretrained(
        base_model_path,
        device_map=_device_map_for_label(device_label),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()

    prompt = _build_query_prompt(query)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    model_device = _model_device(model)
    inputs = inputs.to(model_device).to(getattr(model, "dtype", dtype))

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": int((runtime.get("extra") or {}).get("max_new_tokens") or _DEFAULT_MAX_NEW_TOKENS),
        "do_sample": bool(generation.get("do_sample")),
    }
    if generation.get("temperature") is not None:
        generate_kwargs["temperature"] = float(generation["temperature"])
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)
    response = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return _normalize_response_text(response), {
        "repo_checkout": str(repo_dir),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "device_label": str(device_label),
    }


def execute_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("audio_temporal_grounder requires a non-empty query")

    audio_input = _prepare_audio_input(request, task, runtime)
    cleanup_path = str(audio_input.get("cleanup_path") or "")
    try:
        raw_response, runtime_info = _run_spotsound_inference(
            query=query,
            audio_path=str(audio_input["audio_path"]),
            runtime=runtime,
        )
    finally:
        cleanup_temp_path(cleanup_path)

    clip_start_s = float(audio_input["clip_start_s"])
    clip_end_s = float(audio_input["clip_end_s"])
    intervals = _intervals_from_response(
        raw_response,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )
    video_id = str(audio_input["video_id"])
    clips = [
        {
            "video_id": video_id,
            "start_s": start_s,
            "end_s": end_s,
            "confidence": None,
            "metadata": {
                "event_label": query,
                "tool_backend": "spotsound",
                "request_clip_start_s": clip_start_s,
                "request_clip_end_s": clip_end_s,
            },
        }
        for start_s, end_s in intervals
    ]
    events = [
        {
            "event_label": query,
            "start_s": start_s,
            "end_s": end_s,
            "confidence": None,
            "metadata": {
                "tool_backend": "spotsound",
                "request_clip_start_s": clip_start_s,
                "request_clip_end_s": clip_end_s,
            },
        }
        for start_s, end_s in intervals
    ]

    meaningful_response = "" if is_low_signal_text(raw_response) else raw_response
    return {
        "query": query,
        "clips": clips,
        "events": events,
        "retrieval_backend": "spotsound",
        "query_absent": not bool(intervals),
        "summary": _summary_for_intervals(query, intervals, meaningful_response),
        "model_response": meaningful_response or None,
        "runtime_info": runtime_info,
    }


def main() -> None:
    payload = load_request()
    try:
        emit_json(execute_payload(payload))
    except RuntimeError as exc:
        fail_runtime(str(exc))


if __name__ == "__main__":
    main()
