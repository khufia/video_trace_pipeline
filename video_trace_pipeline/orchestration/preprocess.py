from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

from ..common import has_meaningful_text, hash_payload, read_json, write_json
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation
from ..storage import WorkspaceManager


_DEFAULT_DENSE_CAPTION_PREPROCESS = {
    "clip_duration_s": 60.0,
    "sample_frames": 6,
    "fps": 1.0,
    "max_frames": 96,
    "use_audio_in_video": True,
    "include_asr": True,
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


def _dedupe_texts(values: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in list(values or []):
        text = str(value or "").strip()
        alnum_len = len(re.sub(r"[^A-Za-z0-9]+", "", text))
        if not has_meaningful_text(text) and alnum_len < 3:
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


def _audio_chunks(raw_audio: Any) -> List[str]:
    chunks = []
    for value in re.split(r"[;\n]+", str(raw_audio or "")):
        text = str(value or "").strip()
        if not has_meaningful_text(text):
            continue
        lowered = text.casefold()
        if lowered in {"none", "unknown", "n/a"}:
            continue
        if lowered.startswith("speech:"):
            continue
        if lowered.startswith("acoustics:"):
            text = text.split(":", 1)[-1].strip()
            lowered = text.casefold()
        if not has_meaningful_text(text):
            continue
        chunks.append(text)
    return _dedupe_texts(chunks)


def _audio_line(caption: Dict[str, Any]) -> str:
    audio_chunks = _audio_chunks(caption.get("audio"))
    if not audio_chunks:
        return ""
    return "Audio: %s" % " | ".join(audio_chunks)


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


def _normalized_dense_caption_span(caption: Dict[str, Any], *, default_start: float, default_end: float) -> Dict[str, Any] | None:
    if not isinstance(caption, dict):
        return None
    start_s = float(caption.get("start", default_start) or default_start)
    end_s = float(caption.get("end", caption.get("start", start_s)) or start_s)
    if end_s < start_s:
        end_s = start_s
    normalized: Dict[str, Any] = {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
    }
    visual = " ".join(str(caption.get("visual") or "").split()).strip()
    if has_meaningful_text(visual):
        normalized["visual"] = visual
    audio_chunks = _audio_chunks(caption.get("audio"))
    if audio_chunks:
        normalized["audio"] = audio_chunks
    on_screen_text = _clean_pipe_values(caption.get("on_screen_text"))
    if on_screen_text:
        normalized["on_screen_text"] = on_screen_text
    actions = _dedupe_texts([str(value or "").strip() for value in list(caption.get("actions") or [])])
    if actions:
        normalized["actions"] = actions
    objects = _dedupe_texts([str(value or "").strip() for value in list(caption.get("objects") or [])])
    if objects:
        normalized["objects"] = objects
    attributes = _useful_attribute_values(caption.get("attributes"))
    if attributes:
        normalized["attributes"] = attributes
    return normalized if len(normalized) > 2 else None


def _normalized_transcript_span(transcript: Dict[str, Any], *, default_start: float, default_end: float) -> Dict[str, Any] | None:
    if not isinstance(transcript, dict):
        return None
    text = " ".join(str(transcript.get("text") or "").split()).strip()
    if not has_meaningful_text(text):
        return None
    start_s = float(transcript.get("start_s", transcript.get("start", default_start)) or default_start)
    end_s = float(transcript.get("end_s", transcript.get("end", start_s)) or start_s)
    if end_s < start_s:
        end_s = start_s
    normalized: Dict[str, Any] = {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "text": text,
    }
    speaker_id = str(transcript.get("speaker_id") or "").strip()
    if speaker_id and speaker_id.lower() != "unknown_speaker":
        normalized["speaker_id"] = speaker_id
    return normalized


def _planner_segments_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_segments = []
    for segment in list(segments or []):
        if not isinstance(segment, dict):
            continue
        start_s = float(segment.get("start", 0.0) or 0.0)
        end_s = float(segment.get("end", start_s) or start_s)
        if end_s < start_s:
            end_s = start_s
        normalized: Dict[str, Any] = {
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
        }
        dense_caption = dict(segment.get("dense_caption") or {})
        dense_caption_spans = [
            item
            for item in (
                _normalized_dense_caption_span(caption, default_start=start_s, default_end=end_s)
                for caption in list(dense_caption.get("captions") or [])
            )
            if item is not None
        ]
        if dense_caption_spans:
            normalized["dense_caption_spans"] = dense_caption_spans
        transcript_spans = [
            item
            for item in (
                _normalized_transcript_span(transcript, default_start=start_s, default_end=end_s)
                for transcript in list(segment.get("transcript_segments") or [])
            )
            if item is not None
        ]
        if transcript_spans:
            normalized["transcript_spans"] = transcript_spans
        normalized_segments.append(normalized)
    return normalized_segments


def _planner_segment_metrics(planner_segments: List[Dict[str, Any]]) -> Dict[str, int]:
    dense_caption_span_count = 0
    transcript_segment_count = 0
    for segment in list(planner_segments or []):
        if not isinstance(segment, dict):
            continue
        dense_caption_span_count += len(list(segment.get("dense_caption_spans") or []))
        transcript_segment_count += len(list(segment.get("transcript_spans") or []))
    return {
        "planner_segment_count": len(list(planner_segments or [])),
        "dense_caption_span_count": dense_caption_span_count,
        "transcript_segment_count": transcript_segment_count,
    }


def _normalize_memory_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _trim_memory_text(text: str, max_len: int = 180) -> str:
    rendered = " ".join(str(text or "").split()).strip()
    if len(rendered) <= max_len:
        return rendered
    return rendered[: max_len - 3].rstrip() + "..."

def _collect_identity_memory(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}

    def _add(label: str, *, kind: str, modality: str, start_s: float, end_s: float, snippet: str) -> None:
        cleaned_label = _trim_memory_text(label, max_len=90)
        cleaned_snippet = _trim_memory_text(snippet)
        label_key = _normalize_memory_key(cleaned_label)
        if len(re.sub(r"[^A-Za-z0-9]+", "", cleaned_label)) < 3 or not label_key or not has_meaningful_text(cleaned_snippet):
            return
        key = (kind, label_key)
        item = grouped.get(key)
        if item is None:
            item = {
                "label": cleaned_label,
                "kind": kind,
                "modalities": [],
                "time_ranges": [],
                "supporting_snippets": [],
                "_modality_keys": set(),
                "_time_keys": set(),
                "_snippet_keys": set(),
            }
            grouped[key] = item
        if modality and modality not in item["_modality_keys"]:
            item["_modality_keys"].add(modality)
            item["modalities"].append(modality)
        time_key = (round(float(start_s), 3), round(float(end_s), 3))
        if time_key not in item["_time_keys"]:
            item["_time_keys"].add(time_key)
            item["time_ranges"].append({"start_s": time_key[0], "end_s": time_key[1]})
        snippet_key = _normalize_memory_key(cleaned_snippet)
        if snippet_key and snippet_key not in item["_snippet_keys"]:
            item["_snippet_keys"].add(snippet_key)
            item["supporting_snippets"].append(cleaned_snippet)

    for segment in list(segments or []):
        dense_caption = dict(segment.get("dense_caption") or {})
        for caption in list(dense_caption.get("captions") or []):
            if not isinstance(caption, dict):
                continue
            start_s = float(caption.get("start", segment.get("start", 0.0)) or 0.0)
            end_s = float(caption.get("end", caption.get("start", start_s)) or start_s)
            caption_snippet = _caption_line(caption)
            for phrase in _clean_pipe_values(caption.get("on_screen_text")):
                _add(
                    phrase,
                    kind="on_screen_text",
                    modality="on_screen_text",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=caption_snippet or ('Text: %s' % phrase),
                )
        for transcript in list(segment.get("transcript_segments") or []):
            if not isinstance(transcript, dict):
                continue
            start_s = float(transcript.get("start_s", transcript.get("start", segment.get("start", 0.0))) or 0.0)
            end_s = float(transcript.get("end_s", transcript.get("end", start_s)) or start_s)
            speech_line = _speech_line(transcript)
            speaker_id = str(transcript.get("speaker_id") or "").strip()
            if speaker_id and speaker_id.lower() != "unknown_speaker":
                _add(
                    speaker_id,
                    kind="speaker_id",
                    modality="speech",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=speech_line or ('Speech (%s)' % speaker_id),
                )

    memory = []
    for item in grouped.values():
        snippets = list(item.get("supporting_snippets") or [])[:3]
        time_ranges = sorted(
            list(item.get("time_ranges") or []),
            key=lambda entry: (float(entry.get("start_s", 0.0)), float(entry.get("end_s", 0.0))),
        )[:4]
        memory.append(
            {
                "label": item.get("label"),
                "kind": item.get("kind"),
                "modalities": sorted(item.get("modalities") or []),
                "time_ranges": time_ranges,
                "supporting_snippets": snippets,
                "mention_count": len(time_ranges),
            }
        )
    memory.sort(
        key=lambda entry: (
            0 if entry.get("kind") == "speaker_id" else 1,
            -int(entry.get("mention_count") or 0),
            str(entry.get("label") or "").casefold(),
        )
    )
    return memory[:20]


def _collect_audio_event_memory(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for segment in list(segments or []):
        dense_caption = dict(segment.get("dense_caption") or {})
        for caption in list(dense_caption.get("captions") or []):
            if not isinstance(caption, dict):
                continue
            start_s = float(caption.get("start", segment.get("start", 0.0)) or 0.0)
            end_s = float(caption.get("end", caption.get("start", start_s)) or start_s)
            audio_line = _audio_line(caption)
            for chunk in _audio_chunks(caption.get("audio")):
                key = _normalize_memory_key(chunk)
                if not key:
                    continue
                item = grouped.get(key)
                if item is None:
                    item = {
                        "label": _trim_memory_text(chunk, max_len=100),
                        "time_ranges": [],
                        "supporting_snippets": [],
                        "_time_keys": set(),
                        "_snippet_keys": set(),
                    }
                    grouped[key] = item
                time_key = (round(float(start_s), 3), round(float(end_s), 3))
                if time_key not in item["_time_keys"]:
                    item["_time_keys"].add(time_key)
                    item["time_ranges"].append({"start_s": time_key[0], "end_s": time_key[1]})
                snippet = audio_line or ("Audio: %s" % chunk)
                snippet_key = _normalize_memory_key(snippet)
                if snippet_key and snippet_key not in item["_snippet_keys"]:
                    item["_snippet_keys"].add(snippet_key)
                    item["supporting_snippets"].append(_trim_memory_text(snippet))

    memory = []
    for item in grouped.values():
        time_ranges = sorted(
            list(item.get("time_ranges") or []),
            key=lambda entry: (float(entry.get("start_s", 0.0)), float(entry.get("end_s", 0.0))),
        )[:4]
        memory.append(
            {
                "label": item.get("label"),
                "time_ranges": time_ranges,
                "supporting_snippets": list(item.get("supporting_snippets") or [])[:3],
                "mention_count": len(time_ranges),
            }
        )
    memory.sort(
        key=lambda entry: (
            -int(entry.get("mention_count") or 0),
            str(entry.get("label") or "").casefold(),
        )
    )
    return memory[:20]


def _planner_context_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    identity_memory = _collect_identity_memory(segments)
    audio_event_memory = _collect_audio_event_memory(segments)
    return {
        "identity_memory": identity_memory,
        "audio_event_memory": audio_event_memory,
    }


def _normalize_manifest(
    manifest: Dict[str, Any],
    *,
    segments: List[Dict[str, Any]],
    planner_segments: List[Dict[str, Any]],
    planner_context: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(manifest or {})
    normalized.pop("summary_format", None)
    normalized.pop("summary_status", None)
    metrics = _planner_segment_metrics(planner_segments)
    normalized["segment_count"] = len(list(segments or []))
    normalized["planner_segment_count"] = int(metrics["planner_segment_count"])
    normalized["dense_caption_span_count"] = int(metrics["dense_caption_span_count"])
    normalized["transcript_segment_count"] = int(metrics["transcript_segment_count"])
    normalized["identity_memory_count"] = len(list(planner_context.get("identity_memory") or []))
    normalized["audio_event_memory_count"] = len(list(planner_context.get("audio_event_memory") or []))
    if "include_asr" not in normalized:
        normalized["include_asr"] = bool(metrics["transcript_segment_count"])
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
        settings.pop("summary_format", None)
        if clip_duration_s is not None:
            settings["clip_duration_s"] = clip_duration_s
        settings["clip_duration_s"] = _coerce_float(settings.get("clip_duration_s"), 60.0, minimum=1.0)
        settings["sample_frames"] = _coerce_int(settings.get("sample_frames"), 6, minimum=1)
        settings["fps"] = _coerce_float(settings.get("fps"), 1.0, minimum=0.1)
        settings["max_frames"] = _coerce_int(settings.get("max_frames"), 96, minimum=1)
        settings["use_audio_in_video"] = _coerce_bool(settings.get("use_audio_in_video"), True)
        settings["include_asr"] = _coerce_bool(settings.get("include_asr"), True)
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
        planner_segments_path = cache_dir / "planner_segments.json"
        planner_context_path = cache_dir / "planner_context.json"

        def _bundle_if_complete(base_dir: Path):
            candidate_manifest = base_dir / "manifest.json"
            candidate_segments = base_dir / "segments.json"
            candidate_planner_segments = base_dir / "planner_segments.json"
            candidate_planner_context = base_dir / "planner_context.json"
            if (
                not candidate_manifest.exists()
                or not candidate_segments.exists()
                or not candidate_planner_segments.exists()
                or not candidate_planner_context.exists()
            ):
                return None
            manifest = read_json(candidate_manifest)
            segments = read_json(candidate_segments)
            planner_segments = read_json(candidate_planner_segments)
            planner_context = read_json(candidate_planner_context)
            if (
                not isinstance(manifest, dict)
                or not isinstance(segments, list)
                or not isinstance(planner_segments, list)
                or not isinstance(planner_context, dict)
            ):
                return None
            normalized_manifest = _normalize_manifest(
                manifest,
                segments=segments,
                planner_segments=planner_segments,
                planner_context=planner_context,
            )
            if normalized_manifest != manifest:
                write_json(candidate_manifest, normalized_manifest)
            return {
                "manifest": normalized_manifest,
                "segments": segments,
                "planner_segments": planner_segments,
                "planner_context": planner_context,
            }

        lock = FileLock(str(cache_dir / ".lock"))
        with lock:
            bundle = _bundle_if_complete(cache_dir)
            if bundle is not None:
                return {
                    "cache_hit": True,
                    "cache_dir": self.workspace.relative_path(cache_dir),
                    "manifest": bundle["manifest"],
                    "segments": bundle["segments"],
                    "planner_segments": bundle["planner_segments"],
                    "planner_context": bundle["planner_context"],
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
            if include_asr and hasattr(self.tool_registry, "build_asr_preprocess_transcript"):
                asr_result = self.tool_registry.build_asr_preprocess_transcript(task, preprocess_context)
                built_segments = _assign_transcripts_to_segments(
                    built_segments,
                    list(dict(asr_result or {}).get("segments") or []),
                )
            planner_context = _planner_context_from_segments(built_segments)
            planner_segments = _planner_segments_from_segments(built_segments)
            manifest = _normalize_manifest(
                {
                    "video_fingerprint": video_fingerprint,
                    "clip_duration_s": effective_clip_duration_s,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    "preprocess_settings": preprocess_settings,
                    "preprocess_signature": preprocess_signature,
                    "include_asr": include_asr,
                },
                segments=built_segments,
                planner_segments=planner_segments,
                planner_context=planner_context,
            )
            write_json(manifest_path, manifest)
            write_json(segments_path, built_segments)
            write_json(planner_segments_path, planner_segments)
            write_json(planner_context_path, planner_context)
            return {
                "cache_hit": False,
                "cache_dir": self.workspace.relative_path(cache_dir),
                "manifest": manifest,
                "segments": built_segments,
                "planner_segments": planner_segments,
                "planner_context": planner_context,
                "video_fingerprint": video_fingerprint,
            }
