from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object, is_low_signal_text
from .local_multimodal import TimeChatCaptionerRunner, make_timechat_video_conversation
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    extracted_clip,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
    sample_request_frames,
    scratch_dir,
)

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TIMECODE_RE = re.compile(r"\d+(?::\d{2}){1,2}(?:\.\d+)?")
_QUOTED_TEXT_RE = re.compile(r"['\"]([^'\"]{2,200})['\"]")


def _coerce_second(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if is_low_signal_text(text) else text


def _list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_text(item) for item in value) if item]
    text = _text(value)
    return [text] if text else []


def _dedupe_texts(values: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        text = _text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _parse_timecode(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return None
    return None


def _timestamp_bounds(value: Any) -> tuple[float | None, float | None]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    matches = _TIMECODE_RE.findall(raw)
    if len(matches) >= 2:
        return _parse_timecode(matches[0]), _parse_timecode(matches[1])
    if len(matches) == 1:
        return _parse_timecode(matches[0]), None
    parts = re.split(r"\s*(?:-|to|until|through)\s*", raw, maxsplit=1)
    if len(parts) == 2:
        return _coerce_second(parts[0]), _coerce_second(parts[1])
    return _coerce_second(raw), None


def _quoted_texts(value: Any) -> List[str]:
    raw = str(value or "")
    return _dedupe_texts([match.group(1) for match in _QUOTED_TEXT_RE.finditer(raw)])


def _extract_json_list_items(text: str) -> List[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    with_json = None
    try:
        with_json = json.loads(raw)
    except Exception:
        with_json = None
    if isinstance(with_json, list):
        return [item for item in with_json if isinstance(item, dict)]
    list_start = raw.find("[")
    if list_start == -1:
        return []
    decoder = json.JSONDecoder()
    items: List[Dict[str, Any]] = []
    index = list_start + 1
    while index < len(raw):
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index >= len(raw):
            break
        if raw[index] == "]":
            break
        if raw[index] == ",":
            index += 1
            continue
        try:
            parsed, next_index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            items.append(parsed)
        index = next_index
    return items


def _summary_from_captions(captions: List[Dict[str, Any]]) -> str:
    segments = []
    for item in captions:
        if not isinstance(item, dict):
            continue
        parts = []
        for key in ("visual", "audio", "on_screen_text"):
            value = _text(item.get(key))
            if value:
                parts.append(value)
        for key in ("actions", "objects", "attributes"):
            values = _dedupe_texts([str(value).strip() for value in list(item.get(key) or [])])
            if values:
                parts.append("%s: %s" % (key, ", ".join(values)))
        combined = "; ".join(parts).strip()
        if combined:
            segments.append(combined)
    return " ".join(segments).strip()


def _normalize_times(raw: Dict[str, Any], *, start_s: float, end_s: float) -> tuple[float, float]:
    clip_duration = max(0.0, float(end_s) - float(start_s))
    raw_start = _coerce_second(raw.get("start"))
    raw_end = _coerce_second(raw.get("end"))
    if raw_start is None and raw_end is None:
        raw_start, raw_end = _timestamp_bounds(raw.get("timestamp") or raw.get("time"))
    if raw_start is None and raw_end is None:
        return float(start_s), float(end_s)
    if raw_start is None:
        raw_start = 0.0
    if raw_end is None:
        raw_end = clip_duration if clip_duration > 0.0 else raw_start
    if raw_end < raw_start:
        raw_end = raw_start

    relative_bounds = (
        clip_duration <= 0.0
        or (0.0 <= raw_start <= clip_duration + 1e-3 and 0.0 <= raw_end <= clip_duration + 1e-3)
    )
    absolute_bounds = (
        float(start_s) - 1e-3 <= raw_start <= float(end_s) + 1e-3
        and float(start_s) - 1e-3 <= raw_end <= float(end_s) + 1e-3
    )
    if relative_bounds and not absolute_bounds:
        start_value = float(start_s) + raw_start
        end_value = float(start_s) + raw_end
    else:
        start_value = raw_start
        end_value = raw_end

    start_value = max(float(start_s), min(float(end_s), float(start_value)))
    end_value = max(start_value, min(float(end_s), float(end_value)))
    return round(start_value, 3), round(end_value, 3)


def _normalize_span(raw: Dict[str, Any], *, start_s: float, end_s: float) -> Dict[str, Any]:
    span_start, span_end = _normalize_times(raw, start_s=start_s, end_s=end_s)
    audio_parts = _dedupe_texts(
        [raw.get("audio")]
        + [
            "speech: %s" % _text(raw.get("speech_content")) if _text(raw.get("speech_content")) else "",
            "acoustics: %s" % _text(raw.get("acoustics_content")) if _text(raw.get("acoustics_content")) else "",
        ]
    )
    on_screen_text = _dedupe_texts(
        _list(raw.get("on_screen_text"))
        + _list(raw.get("text_content"))
        + _list(raw.get("screen_text"))
        + _list(raw.get("text_overlay"))
        + _quoted_texts(raw.get("segment_detail_caption"))
    )
    attributes = _dedupe_texts(
        _list(raw.get("attributes"))
        + [
            "camera_state: %s" % _text(raw.get("camera_state")) if _text(raw.get("camera_state")) else "",
            "video_background: %s" % _text(raw.get("video_background")) if _text(raw.get("video_background")) else "",
            "storyline: %s" % _text(raw.get("storyline")) if _text(raw.get("storyline")) else "",
            "shooting_style: %s" % _text(raw.get("shooting_style")) if _text(raw.get("shooting_style")) else "",
        ]
    )
    return {
        "start": span_start,
        "end": span_end,
        "visual": _text(raw.get("visual") or raw.get("segment_detail_caption") or raw.get("caption") or raw.get("description")),
        "audio": "; ".join(audio_parts),
        "on_screen_text": " | ".join(on_screen_text),
        "actions": _dedupe_texts(_list(raw.get("actions")) + _list(raw.get("key_actions"))),
        "objects": _dedupe_texts(_list(raw.get("objects")) + _list(raw.get("entities")) + _list(raw.get("props"))),
        "attributes": attributes,
    }


def execute_payload(
    payload: Dict[str, Any],
    *,
    runner_pool: "PersistentModelPool | None" = None,
    runner: "TimeChatCaptionerRunner | None" = None,
) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    clip = dict((list(request.get("clips") or []) or [{}])[0] or {})
    if not clip and isinstance(request.get("clip"), dict):
        clip = dict(request.get("clip") or {})
    if not clip:
        fail_runtime("dense_captioner requires request.clips[0]")

    video_path = str(task.get("video_path") or "").strip()
    if not video_path:
        fail_runtime("dense_captioner requires task.video_path")

    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") or start_s)
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    out_dir = scratch_dir(runtime, "dense_captioner")
    collect_sampled_frames = bool((runtime.get("extra") or {}).get("collect_sampled_frames", True))
    sampled = []
    if collect_sampled_frames:
        sampled = sample_request_frames(
            request,
            task,
            out_dir=out_dir,
            prefix="dense_caption",
            num_frames=int((runtime.get("extra") or {}).get("sample_frames") or 10),
        )
        if not sampled:
            fail_runtime("dense_captioner could not sample any frames from the requested clip")

    focus_query = str(request.get("focus_query") or "").strip()
    granularity = str(request.get("granularity") or "segment").strip()
    generation = resolve_generation_controls(runtime)
    fps = float((runtime.get("extra") or {}).get("fps") or 2.0)
    max_frames = int((runtime.get("extra") or {}).get("max_frames") or 160)
    max_pixels = int((runtime.get("extra") or {}).get("max_pixels") or 297920)
    video_max_pixels = int((runtime.get("extra") or {}).get("video_max_pixels") or max_pixels)
    use_audio_in_video = bool((runtime.get("extra") or {}).get("use_audio_in_video", True))
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    modality_text = "audio-visual" if use_audio_in_video else "visual"
    prompt = (
        f"You are a dense {modality_text} captioning model for one bounded video clip.\n"
        "Return JSON only with keys: captions, overall_summary.\n"
        "Each captions item must contain: start, end, visual, audio, on_screen_text, actions, objects, attributes.\n"
        "Use timestamps as relative seconds inside the provided clip, starting at 0.0.\n"
        "Keep captions chronological and cover the important visible changes in the clip.\n"
        "Prioritize visible entities, actions, on-screen text, charts, quantities, labels, and event progression.\n"
        "When audio is available, use the audio field for distinctive non-speech sounds, music changes, crowd reactions, impacts, alarms, engines, or other audible events.\n"
        "Do not use the audio field for verbatim dialogue transcription unless a short spoken cue is itself the important event.\n"
        "Do not spend space on generic camera, background, or shooting-style narration unless it materially affects what happens.\n"
        "Use one atomic item per action/object/attribute entry.\n"
        "If a field is unavailable, use an empty string or an empty list.\n\n"
        f"Clip duration: {max(0.0, end_s - start_s):.3f}\n"
        f"Granularity: {granularity}\n"
        f"Focus query: {focus_query or '<none>'}"
    )
    if not use_audio_in_video:
        prompt += "\nAudio is disabled for this request. Leave audio as an empty string unless clear speech text appears on screen."
    owns_runner = False
    if runner is None and runner_pool is not None:
        runner = runner_pool.acquire_timechat_runner(
            tool_name="dense_captioner",
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            use_audio_in_video=use_audio_in_video,
            attn_implementation=attn_implementation,
        )
    if runner is None:
        runner = TimeChatCaptionerRunner(
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            use_audio_in_video=use_audio_in_video,
            attn_implementation=attn_implementation,
        )
        owns_runner = True
    try:
        with extracted_clip(
            video_path,
            start_s,
            end_s,
            include_audio=use_audio_in_video,
        ) as clip_path:
            raw_text = runner.generate(
                make_timechat_video_conversation(
                    prompt,
                    clip_path,
                    fps=fps,
                    max_frames=max_frames,
                    max_pixels=max_pixels,
                    video_max_pixels=video_max_pixels,
                ),
                max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 1024),
            )
    finally:
        if owns_runner:
            runner.close()
    parsed = extract_json_object(raw_text) or {}
    fallback_text = str(raw_text or "").strip()
    if is_low_signal_text(fallback_text):
        fallback_text = ""
    captions = parsed.get("captions")
    if not isinstance(captions, list):
        captions = parsed.get("segments")
    if not isinstance(captions, list):
        captions = _extract_json_list_items(raw_text)
    normalized_captions = [_normalize_span(item, start_s=start_s, end_s=end_s) for item in captions if isinstance(item, dict)]
    overall_summary = _text(parsed.get("overall_summary") or "")
    if not overall_summary and normalized_captions:
        overall_summary = _summary_from_captions(normalized_captions)
    if not overall_summary:
        overall_summary = _text(fallback_text)
    if not normalized_captions:
        normalized_captions = [
            {
                "start": 0.0,
                "end": max(0.0, end_s - start_s),
                "visual": overall_summary,
                "audio": "",
                "on_screen_text": "",
                "actions": [],
                "objects": [],
                "attributes": [],
            }
        ]

    return {
        "clips": [clip],
        "captioned_range": {"start_s": start_s, "end_s": end_s},
        "captions": normalized_captions,
        "overall_summary": overall_summary,
        "sampled_frames": sampled,
        "backend": "timechat_captioner_qwen25_omni",
    }


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
