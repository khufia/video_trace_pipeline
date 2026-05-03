from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

from ..common import has_meaningful_text, hash_payload, read_json, write_json
from ..storage import WorkspaceManager
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation


_DEFAULT_DENSE_CAPTION_PREPROCESS = {
    "enabled": False,
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


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clean_string_list(value: Any) -> List[str]:
    raw_values = value if isinstance(value, list) else str(value or "").split("|")
    cleaned = []
    seen = set()
    for item in list(raw_values or []):
        text = _clean_text(item)
        if not has_meaningful_text(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "backend":
                continue
            pruned = _prune_empty(item)
            if pruned in (None, "", [], {}):
                continue
            cleaned[key] = pruned
        return cleaned
    if isinstance(value, list):
        return [item for item in (_prune_empty(item) for item in value) if item not in (None, "", [], {})]
    return value


def _clip_artifact_payload(workspace: WorkspaceManager, video_id: str, start_s: float, end_s: float) -> Dict[str, Any]:
    artifact = workspace.logical_clip_artifact(video_id, start_s, end_s, source_tool="preprocess")
    return {
        "video_id": video_id,
        "start_s": round(float(start_s), 3),
        "end_s": round(float(end_s), 3),
        "artifact_id": artifact.artifact_id,
        "relpath": artifact.relpath,
    }


def _normalize_clip_payload(
    clip: Dict[str, Any],
    *,
    workspace: WorkspaceManager,
    video_id: str,
    default_start: float,
    default_end: float,
) -> Dict[str, Any]:
    start_s = _float_or(clip.get("start_s", clip.get("start", default_start)), default_start)
    end_s = _float_or(clip.get("end_s", clip.get("end", default_end)), default_end)
    if end_s < start_s:
        end_s = start_s
    normalized = _clip_artifact_payload(workspace, str(clip.get("video_id") or video_id), start_s, end_s)
    metadata = _prune_empty(dict(clip.get("metadata") or {}))
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def _normalize_caption(caption: Dict[str, Any], *, default_start: float, default_end: float) -> Dict[str, Any] | None:
    if not isinstance(caption, dict):
        return None
    start_s = _float_or(caption.get("start_s", caption.get("start", default_start)), default_start)
    end_s = _float_or(caption.get("end_s", caption.get("end", start_s)), start_s)
    if end_s < start_s:
        end_s = start_s
    normalized: Dict[str, Any] = {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
    }
    for source_key, target_key in (("visual", "visual"), ("audio", "audio")):
        text = _clean_text(caption.get(source_key))
        if has_meaningful_text(text):
            normalized[target_key] = text
    on_screen_text = _clean_string_list(caption.get("on_screen_text"))
    if on_screen_text:
        normalized["on_screen_text"] = on_screen_text
    for list_key in ("actions", "objects", "attributes"):
        values = _clean_string_list(caption.get(list_key) or [])
        if values:
            normalized[list_key] = values
    for optional_key in ("confidence", "metadata"):
        value = _prune_empty(caption.get(optional_key))
        if value not in (None, "", [], {}):
            normalized[optional_key] = value
    return normalized if len(normalized) > 2 else None


def _normalize_dense_caption(
    dense_caption: Dict[str, Any],
    *,
    workspace: WorkspaceManager,
    video_id: str,
    default_start: float,
    default_end: float,
) -> Dict[str, Any]:
    dense_caption = dict(dense_caption or {})
    clips = [
        _normalize_clip_payload(
            dict(item or {}),
            workspace=workspace,
            video_id=video_id,
            default_start=default_start,
            default_end=default_end,
        )
        for item in list(dense_caption.get("clips") or [])
        if isinstance(item, dict)
    ]
    if not clips:
        clips = [_clip_artifact_payload(workspace, video_id, default_start, default_end)]

    captioned_range = dict(dense_caption.get("captioned_range") or {})
    if not captioned_range:
        captioned_range = {"start_s": default_start, "end_s": default_end}
    normalized: Dict[str, Any] = {
        "clips": clips,
        "captioned_range": {
            "start_s": round(_float_or(captioned_range.get("start_s", captioned_range.get("start", default_start)), default_start), 3),
            "end_s": round(_float_or(captioned_range.get("end_s", captioned_range.get("end", default_end)), default_end), 3),
        },
    }
    summary = _clean_text(dense_caption.get("overall_summary"))
    if summary:
        normalized["overall_summary"] = summary
    captions = [
        item
        for item in (
            _normalize_caption(caption, default_start=default_start, default_end=default_end)
            for caption in list(dense_caption.get("captions") or [])
        )
        if item is not None
    ]
    if captions:
        normalized["captions"] = captions
    sampled_frames = _prune_empty(dense_caption.get("sampled_frames"))
    if sampled_frames:
        normalized["sampled_frames"] = sampled_frames
    return _prune_empty(normalized)


def _normalize_transcript_segment(segment: Dict[str, Any], *, default_start: float, default_end: float) -> Dict[str, Any] | None:
    if not isinstance(segment, dict):
        return None
    text = _clean_text(segment.get("text"))
    if not has_meaningful_text(text):
        return None
    start_s = _float_or(segment.get("start_s", segment.get("start", default_start)), default_start)
    end_s = _float_or(segment.get("end_s", segment.get("end", start_s)), start_s)
    if end_s < start_s:
        end_s = start_s
    normalized: Dict[str, Any] = {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "text": text,
    }
    speaker_id = _clean_text(segment.get("speaker_id") or segment.get("speaker"))
    if speaker_id and speaker_id.lower() != "unknown_speaker":
        normalized["speaker_id"] = speaker_id
    confidence = segment.get("confidence")
    if confidence is not None:
        normalized["confidence"] = confidence
    return normalized


def _normalize_transcripts(asr_result: Dict[str, Any], *, workspace: WorkspaceManager, video_id: str) -> List[Dict[str, Any]]:
    raw_transcripts = list(dict(asr_result or {}).get("transcripts") or [])
    if not raw_transcripts:
        clip_payload = dict(dict(asr_result or {}).get("clip") or {})
        segments = list(dict(asr_result or {}).get("segments") or [])
        start_s = _float_or(clip_payload.get("start_s", clip_payload.get("start", 0.0)), 0.0)
        end_s = _float_or(clip_payload.get("end_s", clip_payload.get("end", start_s)), start_s)
        raw_transcripts = [
            {
                "clip": _normalize_clip_payload(
                    clip_payload,
                    workspace=workspace,
                    video_id=video_id,
                    default_start=start_s,
                    default_end=end_s,
                ),
                "segments": segments,
                "metadata": dict(asr_result or {}).get("metadata") or {},
            }
        ]

    transcripts: List[Dict[str, Any]] = []
    for index, transcript in enumerate(raw_transcripts, start=1):
        if not isinstance(transcript, dict):
            continue
        clip = dict(transcript.get("clip") or {})
        default_start = _float_or(clip.get("start_s", clip.get("start", 0.0)), 0.0)
        default_end = _float_or(clip.get("end_s", clip.get("end", default_start)), default_start)
        normalized_clip = _normalize_clip_payload(
            clip,
            workspace=workspace,
            video_id=video_id,
            default_start=default_start,
            default_end=default_end,
        )
        segments = [
            item
            for item in (
                _normalize_transcript_segment(segment, default_start=default_start, default_end=default_end)
                for segment in list(transcript.get("segments") or [])
            )
            if item is not None
        ]
        payload = {
            "transcript_id": transcript.get("transcript_id") or "tx_%s" % hash_payload({"clip": normalized_clip, "segments": segments}, 12),
            "clip": normalized_clip,
            "segments": segments,
            "metadata": _prune_empty(dict(transcript.get("metadata") or {})),
        }
        if not payload["segments"]:
            payload.pop("segments")
        transcripts.append(_prune_empty(payload))
    return transcripts


def _flatten_transcript_segments(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for transcript in list(transcripts or []):
        clip = dict(transcript.get("clip") or {})
        for segment in list(transcript.get("segments") or []):
            item = dict(segment)
            if clip and "clip" not in item:
                item["clip"] = clip
            flattened.append(item)
    return flattened


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
        start_s = _float_or(raw_segment.get("start_s", raw_segment.get("start", 0.0)), 0.0)
        end_s = _float_or(raw_segment.get("end_s", raw_segment.get("end", start_s)), start_s)
        if end_s < start_s:
            end_s = start_s
        anchor = start_s if end_s <= start_s else (start_s + end_s) / 2.0
        assigned = False
        for index, window in enumerate(prepared):
            window_start = _float_or(window.get("start", 0.0), 0.0)
            window_end = _float_or(window.get("end", window_start), window_start)
            is_last = index == len(prepared) - 1
            if window_start <= anchor < window_end or (is_last and window_start <= anchor <= window_end):
                window["transcript_segments"].append(dict(raw_segment))
                assigned = True
                break
        if assigned:
            continue
        for window in prepared:
            window_start = _float_or(window.get("start", 0.0), 0.0)
            window_end = _float_or(window.get("end", window_start), window_start)
            if start_s < window_end and end_s > window_start:
                window["transcript_segments"].append(dict(raw_segment))
                assigned = True
                break
        if not assigned:
            prepared[-1]["transcript_segments"].append(dict(raw_segment))
    return prepared


def _normalize_raw_segments(
    segments: List[Dict[str, Any]],
    *,
    workspace: WorkspaceManager,
    video_id: str,
) -> List[Dict[str, Any]]:
    raw_segments = []
    for index, segment in enumerate(list(segments or []), start=1):
        if not isinstance(segment, dict):
            continue
        start_s = _float_or(segment.get("start", segment.get("start_s", 0.0)), 0.0)
        end_s = _float_or(segment.get("end", segment.get("end_s", start_s)), start_s)
        if end_s < start_s:
            end_s = start_s
        dense_caption = _normalize_dense_caption(
            dict(segment.get("dense_caption") or {}),
            workspace=workspace,
            video_id=video_id,
            default_start=start_s,
            default_end=end_s,
        )
        transcript_segments = [
            item
            for item in (
                _normalize_transcript_segment(transcript, default_start=start_s, default_end=end_s)
                for transcript in list(segment.get("transcript_segments") or [])
            )
            if item is not None
        ]
        payload = {
            "segment_id": "seg_%03d" % index,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "dense_caption": dense_caption,
            "transcript_segments": transcript_segments,
        }
        raw_segments.append(_prune_empty(payload))
    return raw_segments


def _planner_segments_from_raw_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    planner_segments = []
    for index, segment in enumerate(list(raw_segments or []), start=1):
        start_s = _float_or(segment.get("start_s"), 0.0)
        end_s = _float_or(segment.get("end_s"), start_s)
        transcript_spans = [
            _prune_empty(
                {
                    "start_s": item.get("start_s"),
                    "end_s": item.get("end_s"),
                    "text": item.get("text"),
                    "speaker_id": item.get("speaker_id"),
                    "confidence": item.get("confidence"),
                }
            )
            for item in list(segment.get("transcript_segments") or [])
            if isinstance(item, dict)
        ]
        payload = {
            "segment_id": segment.get("segment_id") or "seg_%03d" % index,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "dense_caption": segment.get("dense_caption") or {},
            "asr": {"transcript_spans": transcript_spans},
        }
        planner_segments.append(_prune_empty(payload))
    return planner_segments


def _planner_segment_metrics(planner_segments: List[Dict[str, Any]]) -> Dict[str, int]:
    dense_caption_span_count = 0
    transcript_segment_count = 0
    for segment in list(planner_segments or []):
        dense_caption = dict(segment.get("dense_caption") or {})
        dense_caption_span_count += len(list(dense_caption.get("captions") or []))
        asr_payload = dict(segment.get("asr") or {})
        transcript_segment_count += len(list(asr_payload.get("transcript_spans") or []))
    return {
        "planner_segment_count": len(list(planner_segments or [])),
        "dense_caption_span_count": dense_caption_span_count,
        "transcript_segment_count": transcript_segment_count,
    }


def _normalize_manifest(
    manifest: Dict[str, Any],
    *,
    raw_segments: List[Dict[str, Any]],
    planner_segments: List[Dict[str, Any]],
    transcripts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = _prune_empty(dict(manifest or {}))
    metrics = _planner_segment_metrics(planner_segments)
    normalized["segment_count"] = len(list(raw_segments or []))
    normalized["planner_segment_count"] = int(metrics["planner_segment_count"])
    normalized["dense_caption_span_count"] = int(metrics["dense_caption_span_count"])
    normalized["transcript_segment_count"] = int(metrics["transcript_segment_count"])
    normalized["transcript_count"] = len(list(transcripts or []))
    normalized["include_asr"] = bool(metrics["transcript_segment_count"])
    return normalized


class DenseCaptionPreprocessor(object):
    def __init__(self, workspace: WorkspaceManager, tool_registry, models_config):
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.models_config = models_config

    def is_enabled(self) -> bool:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        preprocess_cfg = {}
        if dense_cfg is not None:
            preprocess_cfg = dict(dict(dense_cfg.extra or {}).get("preprocess") or {})
        return _coerce_bool(preprocess_cfg.get("enabled"), False)

    def resolve_preprocess_settings(self, clip_duration_s: Optional[float] = None) -> Dict[str, Any]:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        preprocess_cfg = {}
        if dense_cfg is not None:
            preprocess_cfg = dict(dict(dense_cfg.extra or {}).get("preprocess") or {})
        settings = dict(_DEFAULT_DENSE_CAPTION_PREPROCESS)
        settings.update(preprocess_cfg)
        settings.pop("summary_format", None)
        settings["enabled"] = _coerce_bool(settings.get("enabled"), False)
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

    def _bundle_if_complete(self, cache_dir: Path):
        manifest_path = cache_dir / "manifest.json"
        raw_segments_path = cache_dir / "raw_segments.json"
        planner_segments_path = cache_dir / "planner_segments.json"
        dense_segments_path = cache_dir / "dense_caption" / "segments.json"
        transcripts_path = cache_dir / "asr" / "transcripts.json"
        required = [manifest_path, raw_segments_path, planner_segments_path, dense_segments_path, transcripts_path]
        if any(not path.exists() for path in required):
            return None
        manifest = read_json(manifest_path)
        raw_segments = read_json(raw_segments_path)
        planner_segments = read_json(planner_segments_path)
        dense_segments = read_json(dense_segments_path)
        transcripts = read_json(transcripts_path)
        if (
            not isinstance(manifest, dict)
            or not isinstance(raw_segments, list)
            or not isinstance(planner_segments, list)
            or not isinstance(dense_segments, list)
            or not isinstance(transcripts, list)
        ):
            return None
        return {
            "manifest": manifest,
            "raw_segments": raw_segments,
            "dense_caption_segments": dense_segments,
            "asr_transcripts": transcripts,
            "planner_segments": planner_segments,
            "source": "planner_segments.json",
        }

    def _legacy_bundle_if_complete(self, cache_dir: Path, *, video_id: str):
        manifest_path = cache_dir / "manifest.json"
        preprocess_path = cache_dir / "preprocess.json"
        if not manifest_path.exists() or not preprocess_path.exists():
            return None
        manifest = read_json(manifest_path)
        preprocess = read_json(preprocess_path)
        if not isinstance(manifest, dict) or not isinstance(preprocess, dict):
            return None
        legacy_segments = preprocess.get("segments")
        if not isinstance(legacy_segments, list):
            return None

        raw_segments: List[Dict[str, Any]] = []
        for index, segment in enumerate(legacy_segments, start=1):
            if not isinstance(segment, dict):
                continue
            start_s = _float_or(segment.get("start_s", segment.get("start", 0.0)), 0.0)
            end_s = _float_or(segment.get("end_s", segment.get("end", start_s)), start_s)
            if end_s < start_s:
                end_s = start_s
            legacy_captions = segment.get("dense_captions")
            if not isinstance(legacy_captions, list):
                legacy_captions = []
            dense_caption = _normalize_dense_caption(
                {
                    "clips": [{"video_id": video_id, "start_s": start_s, "end_s": end_s}],
                    "captioned_range": {"start_s": start_s, "end_s": end_s},
                    "overall_summary": segment.get("dense_caption_summary") or segment.get("caption"),
                    "captions": legacy_captions,
                },
                workspace=self.workspace,
                video_id=video_id,
                default_start=start_s,
                default_end=end_s,
            )
            transcript_segments = [
                item
                for item in (
                    _normalize_transcript_segment(transcript, default_start=start_s, default_end=end_s)
                    for transcript in list(segment.get("transcript") or segment.get("transcript_segments") or [])
                )
                if item is not None
            ]
            raw_segments.append(
                _prune_empty(
                    {
                        "segment_id": segment.get("segment_id") or segment.get("id") or "seg_%03d" % index,
                        "start_s": round(start_s, 3),
                        "end_s": round(end_s, 3),
                        "dense_caption": dense_caption,
                        "transcript_segments": transcript_segments,
                    }
                )
            )

        planner_segments = _planner_segments_from_raw_segments(raw_segments)
        if not planner_segments:
            return None
        dense_caption_segments = [
            _prune_empty(
                {
                    "segment_id": segment.get("segment_id"),
                    "start_s": segment.get("start_s"),
                    "end_s": segment.get("end_s"),
                    "dense_caption": segment.get("dense_caption"),
                }
            )
            for segment in raw_segments
        ]
        transcripts = _normalize_transcripts(
            {"transcripts": preprocess.get("asr_transcripts") or []},
            workspace=self.workspace,
            video_id=video_id,
        )
        normalized_manifest = _normalize_manifest(
            {
                **manifest,
                "legacy_preprocess_cache": True,
            },
            raw_segments=raw_segments,
            planner_segments=planner_segments,
            transcripts=transcripts,
        )
        return {
            "manifest": normalized_manifest,
            "raw_segments": raw_segments,
            "dense_caption_segments": dense_caption_segments,
            "asr_transcripts": transcripts,
            "planner_segments": planner_segments,
            "source": "preprocess.json",
        }

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
        video_id = str(task.video_id or task.sample_key or "video")
        cache_dir = self.workspace.preprocess_dir(
            video_fingerprint_value=video_fingerprint,
            model_id=model_id,
            clip_duration_s=effective_clip_duration_s,
            prompt_version=prompt_version,
            settings_signature=preprocess_signature,
            video_id=video_id,
        )

        lock = FileLock(str(cache_dir / ".lock"))
        with lock:
            bundle = self._bundle_if_complete(cache_dir)
            if bundle is None:
                bundle = self._legacy_bundle_if_complete(cache_dir, video_id=video_id)
            if bundle is not None:
                return {
                    "cache_hit": True,
                    "cache_dir": self.workspace.relative_path(cache_dir),
                    "manifest": bundle["manifest"],
                    "segments": bundle["raw_segments"],
                    "raw_segments": bundle["raw_segments"],
                    "dense_caption_segments": bundle["dense_caption_segments"],
                    "asr_transcripts": bundle["asr_transcripts"],
                    "planner_segments": bundle["planner_segments"],
                    "planner_segments_source": bundle.get("source"),
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
            dense_segments = list(result.get("segments") or [])
            transcripts: List[Dict[str, Any]] = []
            asr_cfg = self.models_config.tools.get("asr")
            include_asr = bool(preprocess_settings.get("include_asr")) and bool(getattr(asr_cfg, "enabled", False))
            if include_asr and hasattr(self.tool_registry, "build_asr_preprocess_transcript"):
                asr_result = self.tool_registry.build_asr_preprocess_transcript(task, preprocess_context)
                transcripts = _normalize_transcripts(dict(asr_result or {}), workspace=self.workspace, video_id=video_id)
                dense_segments = _assign_transcripts_to_segments(dense_segments, _flatten_transcript_segments(transcripts))
            raw_segments = _normalize_raw_segments(dense_segments, workspace=self.workspace, video_id=video_id)
            planner_segments = _planner_segments_from_raw_segments(raw_segments)
            dense_caption_segments = [
                _prune_empty(
                    {
                        "segment_id": segment.get("segment_id"),
                        "start_s": segment.get("start_s"),
                        "end_s": segment.get("end_s"),
                        "dense_caption": segment.get("dense_caption"),
                    }
                )
                for segment in raw_segments
            ]
            manifest = _normalize_manifest(
                {
                    "video_id": video_id,
                    "video_fingerprint": video_fingerprint,
                    "clip_duration_s": effective_clip_duration_s,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    "preprocess_settings": preprocess_settings,
                    "preprocess_signature": preprocess_signature,
                },
                raw_segments=raw_segments,
                planner_segments=planner_segments,
                transcripts=transcripts,
            )
            write_json(cache_dir / "manifest.json", manifest)
            write_json(cache_dir / "raw_segments.json", raw_segments)
            write_json(cache_dir / "planner_segments.json", planner_segments)
            write_json(cache_dir / "dense_caption" / "segments.json", dense_caption_segments)
            write_json(cache_dir / "asr" / "transcripts.json", transcripts)
            return {
                "cache_hit": False,
                "cache_dir": self.workspace.relative_path(cache_dir),
                "manifest": manifest,
                "segments": raw_segments,
                "raw_segments": raw_segments,
                "dense_caption_segments": dense_caption_segments,
                "asr_transcripts": transcripts,
                "planner_segments": planner_segments,
                "video_fingerprint": video_fingerprint,
            }
