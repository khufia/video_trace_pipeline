from __future__ import annotations

import io
import shutil
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Callable

from .backends.media import extract_audio_clip
from .common import fingerprint_file, read_json, write_json
from .config import tool_runtime

SIMPLE_PREPROCESS_SCHEMA_VERSION = 6


def _video_duration(video_path: str) -> float:
    try:
        from .backends.media import get_video_duration

        return float(get_video_duration(video_path) or 0.0)
    except Exception:
        return 0.0


def _fingerprint_payload(video_path: str) -> dict[str, Any]:
    try:
        return {"video_path": str(Path(video_path).resolve()), "fingerprint": fingerprint_file(video_path)}
    except Exception:
        return {"video_path": str(video_path), "fingerprint": None}


def _cache_path(profile: dict[str, Any], task: dict[str, Any]) -> Path:
    workspace = Path(str(profile.get("workspace_root") or "workspace")).expanduser().resolve()
    return workspace / "preprocess_cache" / str(task.get("video_id") or task.get("sample_key") or "video") / "preprocess.json"


def _reuse_cache(path: Path, video_path: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if int((payload.get("metadata") or {}).get("schema_version") or 0) != SIMPLE_PREPROCESS_SCHEMA_VERSION:
        return None
    if (payload.get("metadata") or {}).get("video_fingerprint") == _fingerprint_payload(video_path):
        return payload
    return None


def _preprocess_settings(models: dict[str, Any]) -> dict[str, Any]:
    dense_cfg = dict(((models.get("tools") or {}).get("dense_captioner") or {}).get("extra") or {})
    settings = dict(dense_cfg.get("preprocess") or {})
    return {
        "clip_duration_s": max(1.0, float(settings.get("clip_duration_s") or settings.get("segment_duration_s") or 60.0)),
        "granularity": str(settings.get("granularity") or "segment"),
        "focus_query": str(settings.get("focus_query") or ""),
        "include_asr": bool(settings.get("include_asr", True)),
        "dense_caption": {
            key: settings[key]
            for key in (
                "sample_frames",
                "fps",
                "max_frames",
                "max_pixels",
                "video_max_pixels",
                "use_audio_in_video",
                "collect_sampled_frames",
                "max_new_tokens",
            )
            if key in settings
        },
    }


def _segment_ranges(duration_s: float, clip_duration_s: float) -> list[tuple[float, float]]:
    if duration_s <= 0.0:
        return [(0.0, 0.0)]
    ranges = []
    start_s = 0.0
    while start_s < duration_s:
        end_s = min(duration_s, start_s + clip_duration_s)
        ranges.append((round(start_s, 3), round(end_s, 3)))
        if end_s <= start_s:
            break
        start_s = end_s
    return ranges


def _execute_runner(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            return fn(*args, **kwargs)
    except SystemExit as exc:
        raise RuntimeError(stderr.getvalue().strip() or str(exc)) from exc


def _runtime(profile: dict[str, Any], models: dict[str, Any], tool_name: str, run_dir: str | Path, scratch_name: str) -> dict[str, Any]:
    runtime = tool_runtime(profile, models, tool_name, run_dir)
    runtime["model_name"] = runtime.get("model_name") or runtime.get("model")
    run_path = Path(str(run_dir)).expanduser().resolve()
    scratch_dir = run_path / "scratch" / scratch_name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime["scratch_dir"] = str(scratch_dir)
    runtime["extra"] = dict(runtime.get("extra") or {})
    return runtime


def _dense_caption_runtime(
    profile: dict[str, Any],
    models: dict[str, Any],
    run_dir: str | Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    runtime = _runtime(profile, models, "dense_captioner", run_dir, "preprocess_dense_captioner")
    extra = dict(runtime.get("extra") or {})
    extra.update(dict(settings.get("dense_caption") or {}))
    runtime["extra"] = extra
    return runtime


def _make_dense_runner(runtime: dict[str, Any]):
    from video_trace_pipeline.tool_wrappers.local_multimodal import TimeChatCaptionerRunner
    from video_trace_pipeline.tool_wrappers.shared import resolve_generation_controls, resolve_model_path, resolved_device_label

    generation = resolve_generation_controls(runtime)
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    extra = dict(runtime.get("extra") or {})
    return TimeChatCaptionerRunner(
        model_path=model_path,
        device_label=resolved_device_label(runtime),
        generate_do_sample=bool(generation.get("do_sample")),
        generate_temperature=generation.get("temperature"),
        use_audio_in_video=bool(extra.get("use_audio_in_video", True)),
        attn_implementation=str(extra.get("attn_implementation") or "").strip() or None,
    )


def _dense_caption_segments(
    task: dict[str, Any],
    profile: dict[str, Any],
    models: dict[str, Any],
    run_dir: str | Path,
    settings: dict[str, Any],
    duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    from video_trace_pipeline.tool_wrappers.timechat_dense_caption_runner import execute_payload

    video_id = str(task.get("video_id") or task.get("sample_key") or "video")
    runtime = _dense_caption_runtime(profile, models, run_dir, settings)
    ranges = _segment_ranges(duration_s, float(settings["clip_duration_s"]))
    runner = None
    segments: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        runner = _make_dense_runner(runtime)
        for index, (start_s, end_s) in enumerate(ranges, start=1):
            segment_id = "seg_%03d" % index
            clip = {"video_id": video_id, "start_s": start_s, "end_s": end_s}
            request = {
                "tool_name": "dense_captioner",
                "clips": [clip],
                "granularity": str(settings.get("granularity") or "segment"),
                "focus_query": str(settings.get("focus_query") or ""),
            }
            try:
                raw = _execute_runner(
                    execute_payload,
                    {"tool": "dense_captioner", "task": task, "request": request, "runtime": runtime},
                    runner=runner,
                )
            except Exception as exc:
                warnings.append("%s dense caption failed: %s" % (segment_id, exc))
                raw = {"captions": [], "overall_summary": ""}
            captions = [dict(item or {}) for item in list(raw.get("captions") or []) if isinstance(item, dict)]
            overall_summary = str(raw.get("overall_summary") or "").strip()
            if not overall_summary:
                visual_parts = [str(item.get("visual") or item.get("caption") or "").strip() for item in captions]
                overall_summary = " ".join(item for item in visual_parts if item).strip()
            sampled_frames = [dict(item or {}) for item in list(raw.get("sampled_frames") or []) if isinstance(item, dict)]
            for frame in sampled_frames:
                artifacts.append({"kind": "frame", **frame})
            segments.append(
                {
                    "id": segment_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "caption": overall_summary,
                    "dense_caption": {
                        "clips": list(raw.get("clips") or [clip]),
                        "captions": captions,
                        "overall_summary": overall_summary,
                        "captioned_range": raw.get("captioned_range") or {"start_s": start_s, "end_s": end_s},
                    },
                    "transcript": [],
                    "frames": sampled_frames,
                    "metadata": {"source": "dense_captioner", "sampled_frame_count": len(sampled_frames)},
                }
            )
    finally:
        if runner is not None:
            runner.close()
    return segments, artifacts, warnings


def _asr_segments(
    task: dict[str, Any],
    profile: dict[str, Any],
    models: dict[str, Any],
    run_dir: str | Path,
    duration_s: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    from video_trace_pipeline.tools.local_asr import _transcribe_with_whisperx

    warnings: list[str] = []
    runtime = _runtime(profile, models, "asr", run_dir, "preprocess_asr")
    extra = dict(runtime.get("extra") or {})
    model_name = str(extra.get("model_name") or runtime.get("model_name") or runtime.get("model") or "large-v3")
    language = extra.get("language")
    ffmpeg_bin = str(runtime.get("ffmpeg_bin") or "ffmpeg")
    audio_path = None
    try:
        audio_path = extract_audio_clip(str(task.get("video_path") or ""), ffmpeg_bin, 0.0, max(0.001, duration_s))
        raw, runtime_warning = _transcribe_with_whisperx(
            audio_path,
            model_name=model_name,
            device_label=str(runtime.get("device") or "cpu"),
            language=language,
        )
    except Exception as exc:
        return [], ["ASR preprocess failed: %s" % exc]
    finally:
        if audio_path:
            shutil.rmtree(str(Path(audio_path).parent), ignore_errors=True)
    if runtime_warning:
        warnings.append(str(runtime_warning))
    spans = []
    for index, item in enumerate(list(raw.get("segments") or []), start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start_s = float(item.get("start") or 0.0)
        end_s = float(item.get("end") or start_s)
        spans.append(
            {
                "id": "asr_%03d" % index,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "text": text,
                "speaker": item.get("speaker"),
                "metadata": {
                    "source": "whisperx_local",
                    "language": raw.get("language") or language,
                    "avg_logprob": item.get("avg_logprob"),
                },
            }
        )
    return spans, warnings


def _overlap_spans(spans: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    matched = []
    for span in spans:
        span_start = float(span.get("start_s") or 0.0)
        span_end = float(span.get("end_s") or span_start)
        if span_end < start_s or span_start > end_s:
            continue
        matched.append(dict(span))
    return matched


def preprocess(task: dict[str, Any], profile: dict[str, Any], models: dict[str, Any], run_dir: str | Path, use_cache: bool = True) -> dict[str, Any]:
    video_path = str(task.get("video_path") or "")
    cache_path = _cache_path(profile, task)
    if use_cache:
        cached = _reuse_cache(cache_path, video_path)
        if cached is not None:
            return cached

    duration_s = _video_duration(video_path)
    video_id = str(task.get("video_id") or task.get("sample_key") or "video")
    settings = _preprocess_settings(models)
    segments, artifacts, warnings = _dense_caption_segments(task, profile, models, run_dir, settings, duration_s)

    asr_spans: list[dict[str, Any]] = []
    if bool(settings.get("include_asr", True)):
        asr_spans, asr_warnings = _asr_segments(task, profile, models, run_dir, duration_s)
        warnings.extend(asr_warnings)
        for segment in segments:
            segment["transcript"] = _overlap_spans(asr_spans, float(segment.get("start_s") or 0.0), float(segment.get("end_s") or 0.0))

    result = {
        "ok": True,
        "video_id": video_id,
        "video_duration_s": duration_s,
        "segments": segments,
        "artifacts": artifacts,
        "asr_transcripts": [
            {
                "clip": {"video_id": video_id, "start_s": 0.0, "end_s": duration_s},
                "segments": asr_spans,
                "text": " ".join(str(item.get("text") or "").strip() for item in asr_spans if str(item.get("text") or "").strip()),
                "metadata": {"source": "whisperx_local"},
            }
        ]
        if asr_spans
        else [],
        "metadata": {
            "schema_version": SIMPLE_PREPROCESS_SCHEMA_VERSION,
            "source": "dense_caption_asr_preprocess",
            "settings": settings,
            "warnings": [item for item in warnings if item],
            "video_fingerprint": _fingerprint_payload(video_path),
            "cache_path": str(cache_path),
        },
    }
    if use_cache:
        write_json(cache_path, result)
    return result
