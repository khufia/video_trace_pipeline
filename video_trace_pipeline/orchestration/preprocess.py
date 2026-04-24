from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

from ..common import has_meaningful_text, hash_payload, is_low_signal_text, read_json, write_json, write_text
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation
from ..storage import WorkspaceManager


_DEFAULT_DENSE_CAPTION_PREPROCESS = {
    "clip_duration_s": 60.0,
    "sample_frames": 6,
    "fps": 1.0,
    "max_frames": 96,
    "use_audio_in_video": False,
    "include_asr": True,
    "summary_format": "dense_interleaved",
    "collect_sampled_frames": False,
    "max_new_tokens": 700,
}


def _coerce_float(value: Any, default: float, *, minimum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, parsed)


def _coerce_int(value: Any, default: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, parsed)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _coerce_string(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or str(default or "").strip()


def _dedupe_texts(values: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if not has_meaningful_text(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _format_seconds(value: Any) -> str:
    try:
        total_seconds = max(0, int(round(float(value or 0.0))))
    except Exception:
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)
    return "%02d:%02d" % (minutes, seconds)


def _format_interval(start_s: Any, end_s: Any) -> str:
    return "[%s-%s]" % (_format_seconds(start_s), _format_seconds(end_s))


def _clean_pipe_values(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = raw
    else:
        values = str(raw or "").split("|")
    return _dedupe_texts([str(value or "").strip() for value in values])


def _useful_attribute_values(raw_values: Any) -> List[str]:
    ignored_prefixes = (
        "camera_state:",
        "video_background:",
        "storyline:",
        "shooting_style:",
    )
    values = []
    for value in list(raw_values or []):
        text = str(value or "").strip()
        if not has_meaningful_text(text):
            continue
        lowered = text.casefold()
        if lowered.startswith(ignored_prefixes):
            continue
        values.append(text)
    return _dedupe_texts(values)


def _caption_line(caption: Dict[str, Any]) -> str:
    parts = []
    visual = str(caption.get("visual") or "").strip()
    if has_meaningful_text(visual):
        parts.append("Visual: %s" % visual)
    on_screen_text = _clean_pipe_values(caption.get("on_screen_text"))
    if on_screen_text:
        parts.append("Text: %s" % " | ".join(on_screen_text))
    actions = _dedupe_texts([str(value or "").strip() for value in list(caption.get("actions") or [])])
    if actions:
        parts.append("Actions: %s" % ", ".join(actions))
    objects = _dedupe_texts([str(value or "").strip() for value in list(caption.get("objects") or [])])
    if objects:
        parts.append("Objects: %s" % ", ".join(objects))
    attributes = _useful_attribute_values(caption.get("attributes"))
    if attributes:
        parts.append("Details: %s" % ", ".join(attributes))
    return " | ".join(parts).strip()


def _speech_line(segment: Dict[str, Any]) -> str:
    text = str(segment.get("text") or "").strip()
    if not has_meaningful_text(text):
        return ""
    speaker = str(segment.get("speaker_id") or "").strip()
    prefix = "Speech"
    if speaker and speaker.lower() != "unknown_speaker":
        prefix = "Speech (%s)" % speaker
    return "%s: %s" % (prefix, text)


def _assign_transcripts_to_segments(
    dense_segments: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prepared = []
    for segment in list(dense_segments or []):
        normalized = dict(segment or {})
        normalized["transcript_segments"] = []
        prepared.append(normalized)
    if not prepared:
        return prepared
    for raw_segment in list(transcript_segments or []):
        if not isinstance(raw_segment, dict):
            continue
        start_s = float(raw_segment.get("start_s", raw_segment.get("start", 0.0)) or 0.0)
        end_s = float(raw_segment.get("end_s", raw_segment.get("end", start_s)) or start_s)
        if end_s < start_s:
            end_s = start_s
        anchor = start_s if end_s <= start_s else (start_s + end_s) / 2.0
        assigned = False
        for index, window in enumerate(prepared):
            window_start = float(window.get("start", 0.0) or 0.0)
            window_end = float(window.get("end", window_start) or window_start)
            is_last = index == len(prepared) - 1
            if window_start <= anchor < window_end or (is_last and window_start <= anchor <= window_end):
                window["transcript_segments"].append(dict(raw_segment))
                assigned = True
                break
        if assigned:
            continue
        for window in prepared:
            window_start = float(window.get("start", 0.0) or 0.0)
            window_end = float(window.get("end", window_start) or window_start)
            if start_s < window_end and end_s > window_start:
                window["transcript_segments"].append(dict(raw_segment))
                assigned = True
                break
        if not assigned:
            prepared[-1]["transcript_segments"].append(dict(raw_segment))
    return prepared


def _dense_timeline_events(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    dense_caption = dict(segment.get("dense_caption") or {})
    for caption in list(dense_caption.get("captions") or []):
        if not isinstance(caption, dict):
            continue
        line = _caption_line(caption)
        if not has_meaningful_text(line):
            continue
        start_s = float(caption.get("start", segment.get("start", 0.0)) or 0.0)
        end_s = float(caption.get("end", caption.get("start", start_s)) or start_s)
        events.append({"start": start_s, "end": max(start_s, end_s), "kind": "visual", "text": line})
    for transcript in list(segment.get("transcript_segments") or []):
        if not isinstance(transcript, dict):
            continue
        line = _speech_line(transcript)
        if not has_meaningful_text(line):
            continue
        start_s = float(transcript.get("start_s", transcript.get("start", segment.get("start", 0.0))) or 0.0)
        end_s = float(transcript.get("end_s", transcript.get("end", start_s)) or start_s)
        events.append({"start": start_s, "end": max(start_s, end_s), "kind": "speech", "text": line})
    return sorted(events, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)), str(item.get("kind") or "")))


def _render_dense_interleaved_summary(segments: List[Dict[str, Any]]) -> str:
    lines = []
    seen = set()
    for segment in list(segments or []):
        for event in _dense_timeline_events(segment):
            text = str(event.get("text") or "").strip()
            if not has_meaningful_text(text):
                continue
            line = "%s %s" % (
                _format_interval(event.get("start"), event.get("end")),
                text,
            )
            signature = hash_payload({"interval": line.split(" ", 1)[0], "text": text.casefold()}, 16)
            if signature in seen:
                continue
            seen.add(signature)
            lines.append(line)
    rendered = "\n".join(lines).strip()
    return "" if is_low_signal_text(rendered) else rendered


def _normalize_bundle(base_dir: Path, bundle: Dict[str, Any]) -> Dict[str, Any] | None:
    normalized = dict(bundle or {})
    manifest = dict(normalized.get("manifest") or {})
    summary = str(normalized.get("summary") or "").strip()
    if has_meaningful_text(summary):
        normalized["summary"] = summary
        return normalized
    segments = list(normalized.get("segments") or [])
    if not segments:
        summary_status = str(manifest.get("summary_status") or "").strip()
        if summary_status.startswith("unavailable"):
            normalized["summary"] = ""
            normalized["manifest"] = manifest
            return normalized
        return None
    derived_summary = _render_dense_interleaved_summary(segments)
    if not has_meaningful_text(derived_summary):
        summary_status = str(manifest.get("summary_status") or "").strip()
        if summary_status.startswith("unavailable"):
            normalized["summary"] = ""
            normalized["manifest"] = manifest
            return normalized
        return None
    write_text(base_dir / "summary.txt", derived_summary)
    if manifest.get("summary_status") != "available":
        manifest["summary_status"] = "available"
        write_json(base_dir / "manifest.json", manifest)
    normalized["summary"] = derived_summary
    normalized["manifest"] = manifest
    return normalized


class DenseCaptionPreprocessor(object):
    def __init__(self, workspace: WorkspaceManager, tool_registry, models_config):
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.models_config = models_config

    def resolve_preprocess_settings(self, clip_duration_s: Optional[float] = None) -> Dict[str, Any]:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        preprocess_cfg = {}
        if dense_cfg is not None:
            preprocess_cfg = dict(dict(dense_cfg.extra or {}).get("preprocess") or {})
        settings = dict(_DEFAULT_DENSE_CAPTION_PREPROCESS)
        settings.update(preprocess_cfg)
        if clip_duration_s is not None:
            settings["clip_duration_s"] = clip_duration_s
        settings["clip_duration_s"] = _coerce_float(settings.get("clip_duration_s"), 60.0, minimum=1.0)
        settings["sample_frames"] = _coerce_int(settings.get("sample_frames"), 6, minimum=1)
        settings["fps"] = _coerce_float(settings.get("fps"), 1.0, minimum=0.1)
        settings["max_frames"] = _coerce_int(settings.get("max_frames"), 96, minimum=1)
        settings["use_audio_in_video"] = _coerce_bool(settings.get("use_audio_in_video"), False)
        settings["include_asr"] = _coerce_bool(settings.get("include_asr"), True)
        settings["summary_format"] = _coerce_string(settings.get("summary_format"), "dense_interleaved")
        settings["collect_sampled_frames"] = _coerce_bool(settings.get("collect_sampled_frames"), False)
        settings["max_new_tokens"] = _coerce_int(settings.get("max_new_tokens"), 700, minimum=1)
        return settings

    def get_or_build(self, task, clip_duration_s: Optional[float] = None) -> Dict[str, object]:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        implementation = tool_implementation("dense_captioner")
        model_name = dense_cfg.model if dense_cfg and dense_cfg.model else "dense_captioner"
        model_id = "%s__%s" % (implementation, model_name)
        prompt_version = dense_cfg.prompt_version if dense_cfg else "v1"
        preprocess_settings = self.resolve_preprocess_settings(clip_duration_s)
        effective_clip_duration_s = float(preprocess_settings["clip_duration_s"])
        preprocess_signature = hash_payload(preprocess_settings, 12)
        video_fingerprint = self.workspace.video_fingerprint(task.video_path)
        cache_dir = self.workspace.preprocess_dir(
            video_fingerprint_value=video_fingerprint,
            model_id=model_id,
            clip_duration_s=effective_clip_duration_s,
            prompt_version=prompt_version,
            settings_signature=preprocess_signature,
        )
        manifest_path = cache_dir / "manifest.json"
        segments_path = cache_dir / "segments.json"
        summary_path = cache_dir / "summary.txt"

        def _bundle_if_complete(base_dir: Path):
            candidate_manifest = base_dir / "manifest.json"
            candidate_segments = base_dir / "segments.json"
            candidate_summary = base_dir / "summary.txt"
            if candidate_manifest.exists() and candidate_segments.exists() and candidate_summary.exists():
                return {
                    "manifest": read_json(candidate_manifest),
                    "segments": read_json(candidate_segments),
                    "summary": candidate_summary.read_text(encoding="utf-8"),
                }
            return None

        lock = FileLock(str(cache_dir / ".lock"))
        with lock:
            bundle = _bundle_if_complete(cache_dir)
            if bundle is not None:
                bundle = _normalize_bundle(cache_dir, bundle)
            if bundle is not None:
                return {
                    "cache_hit": True,
                    "cache_dir": self.workspace.relative_path(cache_dir),
                    "manifest": bundle["manifest"],
                    "segments": bundle["segments"],
                    "summary": bundle["summary"],
                    "video_fingerprint": video_fingerprint,
                }

            class _PreprocessRun(object):
                def __init__(self, base_dir: Path):
                    self.tools_dir = base_dir

            preprocess_run = _PreprocessRun(cache_dir / "_tool_scratch")
            preprocess_context = ToolExecutionContext(
                workspace=self.workspace,
                run=preprocess_run,
                task=task,
                models_config=self.models_config,
                llm_client=self.tool_registry.llm_client,
                evidence_lookup=None,
                preprocess_bundle=None,
            )
            result = self.tool_registry.build_dense_caption_cache(
                task,
                effective_clip_duration_s,
                preprocess_context,
                preprocess_settings=preprocess_settings,
            )
            built_segments = list(result.get("segments") or [])
            asr_cfg = self.models_config.tools.get("asr")
            include_asr = bool(preprocess_settings.get("include_asr")) and bool(getattr(asr_cfg, "enabled", False))
            asr_result = None
            if include_asr and hasattr(self.tool_registry, "build_asr_preprocess_transcript"):
                asr_result = self.tool_registry.build_asr_preprocess_transcript(task, preprocess_context)
                built_segments = _assign_transcripts_to_segments(
                    built_segments,
                    list(dict(asr_result or {}).get("segments") or []),
                )
            built_summary = _render_dense_interleaved_summary(built_segments)
            summary_status = "available" if has_meaningful_text(built_summary) else "unavailable_low_signal"
            manifest = {
                "video_fingerprint": video_fingerprint,
                "clip_duration_s": effective_clip_duration_s,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "preprocess_settings": preprocess_settings,
                "preprocess_signature": preprocess_signature,
                "summary_format": str(preprocess_settings.get("summary_format") or "dense_interleaved"),
                "include_asr": include_asr,
                "transcript_segment_count": len(list(dict(asr_result or {}).get("segments") or [])),
                "segment_count": len(built_segments),
                "summary_status": summary_status,
            }
            write_json(manifest_path, manifest)
            write_json(segments_path, built_segments)
            write_text(summary_path, built_summary if summary_status == "available" else "")
            return {
                "cache_hit": False,
                "cache_dir": self.workspace.relative_path(cache_dir),
                "manifest": manifest,
                "segments": built_segments,
                "summary": built_summary if summary_status == "available" else "",
                "video_fingerprint": video_fingerprint,
            }
