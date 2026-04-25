from __future__ import annotations

import re
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
    "use_audio_in_video": True,
    "include_asr": True,
    "summary_format": "dense_interleaved",
    "collect_sampled_frames": False,
    "max_new_tokens": 700,
}


_PERSON_HINT_WORDS = {
    "actor",
    "anchor",
    "announcer",
    "boy",
    "bride",
    "candidate",
    "chef",
    "child",
    "coach",
    "customer",
    "dancer",
    "doctor",
    "driver",
    "employee",
    "father",
    "girl",
    "groom",
    "guest",
    "guy",
    "host",
    "lady",
    "man",
    "mother",
    "narrator",
    "person",
    "player",
    "police",
    "presenter",
    "reporter",
    "runner",
    "scientist",
    "seller",
    "shopper",
    "singer",
    "speaker",
    "student",
    "teacher",
    "tourist",
    "vendor",
    "waiter",
    "waitress",
    "woman",
}

_SOUND_EVENT_KEYWORDS = {
    "alarm",
    "applause",
    "bang",
    "beep",
    "bell",
    "buzzer",
    "cheer",
    "cheering",
    "choir",
    "clang",
    "clap",
    "click",
    "crack",
    "crash",
    "cry",
    "engine",
    "explosion",
    "footstep",
    "gunshot",
    "hiss",
    "horn",
    "knock",
    "laughter",
    "laughing",
    "metallic",
    "motor",
    "music",
    "noise",
    "ring",
    "ringing",
    "roar",
    "rumble",
    "scream",
    "shout",
    "siren",
    "slam",
    "song",
    "thud",
    "thunder",
    "voice-over",
    "whistle",
    "whoosh",
    "yell",
}

_GENERIC_NAME_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "but",
    "come",
    "day",
    "for",
    "from",
    "here",
    "how",
    "i",
    "if",
    "in",
    "it",
    "its",
    "look",
    "me",
    "my",
    "of",
    "oh",
    "on",
    "or",
    "our",
    "out",
    "please",
    "she",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "today",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
    "your",
}

_DIALOGUE_CLAIM_STOPWORDS = {
    "about",
    "after",
    "again",
    "almost",
    "also",
    "because",
    "been",
    "being",
    "came",
    "come",
    "could",
    "from",
    "have",
    "just",
    "like",
    "might",
    "really",
    "said",
    "says",
    "should",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "very",
    "want",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

_DIALOGUE_CONTRADICTION_MARKERS = (
    "did not",
    "didn't",
    "do not",
    "don't",
    "forgive me for lying",
    "i lied",
    "i'm lying",
    "i was lying",
    "is not",
    "isn't",
    "it was a lie",
    "lied to",
    "lying to",
    "never",
    "no,",
    "not true",
    "was not",
    "wasn't",
    "were not",
    "weren't",
)

_DIALOGUE_UNCERTAINTY_MARKERS = (
    "i guess",
    "i hope",
    "i think",
    "i wonder",
    "maybe",
    "perhaps",
    "probably",
)

_MULTI_TOKEN_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z]+(?:['’-][A-Za-z]+)?)(?:\s+(?:[A-Z][A-Za-z]+(?:['’-][A-Za-z]+)?)){1,3}\b"
)
_SINGLE_TOKEN_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:['’-][A-Za-z]+)?\b")


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
        if any(keyword in lowered for keyword in _SOUND_EVENT_KEYWORDS):
            chunks.append(text)
            continue
        if "'" in text or '"' in text:
            continue
        if len(text.split()) <= 6:
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


def _dense_timeline_events(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    dense_caption = dict(segment.get("dense_caption") or {})
    for caption in list(dense_caption.get("captions") or []):
        if not isinstance(caption, dict):
            continue
        line = _caption_line(caption)
        start_s = float(caption.get("start", segment.get("start", 0.0)) or 0.0)
        end_s = float(caption.get("end", caption.get("start", start_s)) or start_s)
        if has_meaningful_text(line):
            events.append({"start": start_s, "end": max(start_s, end_s), "kind": "visual", "text": line})
        audio_line = _audio_line(caption)
        if has_meaningful_text(audio_line):
            events.append({"start": start_s, "end": max(start_s, end_s), "kind": "audio", "text": audio_line})
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


def _normalize_memory_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _trim_memory_text(text: str, max_len: int = 180) -> str:
    rendered = " ".join(str(text or "").split()).strip()
    if len(rendered) <= max_len:
        return rendered
    return rendered[: max_len - 3].rstrip() + "..."


def _name_like_phrases(text: str) -> List[str]:
    raw = str(text or "")
    candidates = []
    candidates.extend(match.group(0) for match in _MULTI_TOKEN_NAME_RE.finditer(raw))
    for match in _SINGLE_TOKEN_NAME_RE.finditer(raw):
        token = str(match.group(0) or "").strip()
        lowered = token.casefold().strip(".,:;!?")
        if not token:
            continue
        if lowered in _GENERIC_NAME_STOPWORDS:
            continue
        if len(token) < 4 and "'" not in token:
            continue
        candidates.append(token)
    deduped = []
    seen = set()
    for candidate in candidates:
        cleaned = " ".join(str(candidate or "").split()).strip()
        key = _normalize_memory_key(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _looks_like_person_descriptor(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(
        re.search(r"\b%s\b" % re.escape(word), lowered)
        for word in _PERSON_HINT_WORDS
    )


def _overlaps_interval(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return max(float(start_a), float(start_b)) <= min(float(end_a), float(end_b))


def _claim_terms(text: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9']+", str(text or "").casefold()):
        cleaned = token.strip("'")
        if len(cleaned) < 4:
            continue
        if cleaned in _GENERIC_NAME_STOPWORDS or cleaned in _DIALOGUE_CLAIM_STOPWORDS:
            continue
        if cleaned.endswith("'s") and len(cleaned) > 4:
            cleaned = cleaned[:-2]
        elif cleaned.endswith("s") and len(cleaned) > 5:
            cleaned = cleaned[:-1]
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return terms


def _looks_like_dialogue_claim(text: str) -> bool:
    rendered = " ".join(str(text or "").split()).strip()
    if not has_meaningful_text(rendered):
        return False
    if rendered.endswith("?"):
        return False
    alnum_len = len(re.sub(r"[^A-Za-z0-9]+", "", rendered))
    if alnum_len < 10:
        return False
    return bool(_claim_terms(rendered))


def _dialogue_claim_stance(text: str) -> str:
    lowered = str(text or "").casefold()
    if any(marker in lowered for marker in _DIALOGUE_CONTRADICTION_MARKERS):
        return "conflicts"
    if any(marker in lowered for marker in _DIALOGUE_UNCERTAINTY_MARKERS):
        return "uncertain"
    return "supports"


def _collect_dialogue_claim_memory(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for segment in list(segments or []):
        segment_start = float(segment.get("start", 0.0) or 0.0)
        segment_end = float(segment.get("end", segment_start) or segment_start)
        for transcript in list(segment.get("transcript_segments") or []):
            if not isinstance(transcript, dict):
                continue
            text = " ".join(str(transcript.get("text") or "").split()).strip()
            if not _looks_like_dialogue_claim(text):
                continue
            terms = _claim_terms(text)
            if not terms:
                continue
            start_s = float(transcript.get("start_s", transcript.get("start", segment_start)) or segment_start)
            end_s = float(transcript.get("end_s", transcript.get("end", start_s)) or start_s)
            if end_s < start_s:
                end_s = start_s
            signature = " ".join(terms[:6])
            group = grouped.get(signature)
            if group is None:
                group = {
                    "claim_signature": signature,
                    "claim_terms": list(terms[:6]),
                    "entities": [],
                    "mentions": [],
                    "_entity_keys": set(),
                }
                grouped[signature] = group
            for entity in _name_like_phrases(text):
                entity_key = _normalize_memory_key(entity)
                if entity_key and entity_key not in group["_entity_keys"]:
                    group["_entity_keys"].add(entity_key)
                    group["entities"].append(entity)
            speaker_id = str(transcript.get("speaker_id") or "").strip()
            group["mentions"].append(
                {
                    "speaker_id": speaker_id or None,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "stance": _dialogue_claim_stance(text),
                    "text": _trim_memory_text(text, max_len=200),
                }
            )

    memory = []
    for group in grouped.values():
        mentions = sorted(
            list(group.get("mentions") or []),
            key=lambda item: (
                float(item.get("start_s", 0.0)),
                float(item.get("end_s", item.get("start_s", 0.0))),
                str(item.get("speaker_id") or ""),
            ),
        )
        supporting_mentions = [item for item in mentions if item.get("stance") == "supports"]
        conflicting_mentions = [item for item in mentions if item.get("stance") == "conflicts"]
        uncertain_mentions = [item for item in mentions if item.get("stance") == "uncertain"]
        contradiction_risk = "high" if supporting_mentions and conflicting_mentions else ("medium" if conflicting_mentions else "low")
        memory.append(
            {
                "claim_signature": group.get("claim_signature"),
                "claim_terms": list(group.get("claim_terms") or []),
                "entities": list(group.get("entities") or [])[:4],
                "supporting_mentions": supporting_mentions[:4],
                "conflicting_mentions": conflicting_mentions[:4],
                "uncertain_mentions": uncertain_mentions[:3],
                "mention_count": len(mentions),
                "contradiction_risk": contradiction_risk,
            }
        )
    memory.sort(
        key=lambda item: (
            0 if item.get("contradiction_risk") == "high" else (1 if item.get("contradiction_risk") == "medium" else 2),
            -int(item.get("mention_count") or 0),
            str(item.get("claim_signature") or ""),
        )
    )
    return memory[:24]


def _collect_timeline_alignment_memory(
    segments: List[Dict[str, Any]],
    identity_memory: List[Dict[str, Any]],
    audio_event_memory: List[Dict[str, Any]],
    dialogue_claim_memory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned = []
    for segment in list(segments or []):
        start_s = float(segment.get("start", 0.0) or 0.0)
        end_s = float(segment.get("end", start_s) or start_s)
        dense_caption = dict(segment.get("dense_caption") or {})
        visual_cues = []
        audio_cues = []
        speech_cues = []
        for caption in list(dense_caption.get("captions") or [])[:2]:
            if not isinstance(caption, dict):
                continue
            visual_line = _caption_line(caption)
            if has_meaningful_text(visual_line):
                visual_cues.append(_trim_memory_text(visual_line))
            audio_line = _audio_line(caption)
            if has_meaningful_text(audio_line):
                audio_cues.append(_trim_memory_text(audio_line))
        for transcript in list(segment.get("transcript_segments") or [])[:3]:
            if not isinstance(transcript, dict):
                continue
            speech_line = _speech_line(transcript)
            if has_meaningful_text(speech_line):
                speech_cues.append(_trim_memory_text(speech_line))

        aligned_names = []
        for item in list(identity_memory or []):
            if any(
                _overlaps_interval(start_s, end_s, entry.get("start_s", 0.0), entry.get("end_s", entry.get("start_s", 0.0)))
                for entry in list(item.get("time_ranges") or [])
            ):
                aligned_names.append(str(item.get("label") or "").strip())

        aligned_audio = []
        for item in list(audio_event_memory or []):
            if any(
                _overlaps_interval(start_s, end_s, entry.get("start_s", 0.0), entry.get("end_s", entry.get("start_s", 0.0)))
                for entry in list(item.get("time_ranges") or [])
            ):
                aligned_audio.append(str(item.get("label") or "").strip())

        aligned_claims = []
        for claim in list(dialogue_claim_memory or []):
            mentions = [
                mention
                for mention in list(claim.get("supporting_mentions") or [])
                + list(claim.get("conflicting_mentions") or [])
                + list(claim.get("uncertain_mentions") or [])
                if _overlaps_interval(
                    start_s,
                    end_s,
                    mention.get("start_s", 0.0),
                    mention.get("end_s", mention.get("start_s", 0.0)),
                )
            ]
            if not mentions:
                continue
            aligned_claims.append(
                {
                    "claim_signature": claim.get("claim_signature"),
                    "stances": sorted({str(item.get("stance") or "") for item in mentions if str(item.get("stance") or "").strip()}),
                    "snippets": [str(item.get("text") or "").strip() for item in mentions[:2] if str(item.get("text") or "").strip()],
                }
            )

        if not any((visual_cues, audio_cues, speech_cues, aligned_names, aligned_audio, aligned_claims)):
            continue
        aligned.append(
            {
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "named_entities": _dedupe_texts(aligned_names)[:4],
                "audio_events": _dedupe_texts(aligned_audio + audio_cues)[:4],
                "dialogue_claims": aligned_claims[:3],
                "visual_cues": _dedupe_texts(visual_cues)[:2],
                "speech_cues": _dedupe_texts(speech_cues)[:2],
            }
        )
    return aligned[:24]


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
                "aliases": [],
                "modalities": [],
                "time_ranges": [],
                "supporting_snippets": [],
                "_modality_keys": set(),
                "_time_keys": set(),
                "_snippet_keys": set(),
            }
            grouped[key] = item
        if cleaned_label not in item["aliases"]:
            item["aliases"].append(cleaned_label)
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
            for phrase in _name_like_phrases(caption.get("on_screen_text")):
                _add(
                    phrase,
                    kind="named_anchor",
                    modality="on_screen_text",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=caption_snippet or ('Text: %s' % phrase),
                )
            for phrase in _name_like_phrases(caption.get("visual")):
                _add(
                    phrase,
                    kind="named_anchor",
                    modality="visual",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=caption_snippet or ('Visual: %s' % phrase),
                )
            for obj in list(caption.get("objects") or []):
                obj_text = str(obj or "").strip()
                if not _looks_like_person_descriptor(obj_text):
                    continue
                _add(
                    obj_text,
                    kind="person_descriptor",
                    modality="visual",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=caption_snippet or obj_text,
                )
            visual_text = str(caption.get("visual") or "").strip()
            if _looks_like_person_descriptor(visual_text):
                _add(
                    visual_text,
                    kind="person_descriptor",
                    modality="visual",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=caption_snippet or visual_text,
                )
        for transcript in list(segment.get("transcript_segments") or []):
            if not isinstance(transcript, dict):
                continue
            start_s = float(transcript.get("start_s", transcript.get("start", segment.get("start", 0.0))) or 0.0)
            end_s = float(transcript.get("end_s", transcript.get("end", start_s)) or start_s)
            speech_line = _speech_line(transcript)
            for phrase in _name_like_phrases(transcript.get("text")):
                _add(
                    phrase,
                    kind="named_anchor",
                    modality="speech",
                    start_s=start_s,
                    end_s=end_s,
                    snippet=speech_line or str(transcript.get("text") or "").strip(),
                )

    memory = []
    for item in grouped.values():
        aliases = _dedupe_texts(item.get("aliases") or [])
        snippets = list(item.get("supporting_snippets") or [])[:3]
        time_ranges = sorted(
            list(item.get("time_ranges") or []),
            key=lambda entry: (float(entry.get("start_s", 0.0)), float(entry.get("end_s", 0.0))),
        )[:4]
        memory.append(
            {
                "label": aliases[0] if aliases else item.get("label"),
                "kind": item.get("kind"),
                "aliases": aliases[:4],
                "modalities": sorted(item.get("modalities") or []),
                "time_ranges": time_ranges,
                "supporting_snippets": snippets,
                "mention_count": len(time_ranges),
            }
        )
    memory.sort(
        key=lambda entry: (
            0 if entry.get("kind") == "named_anchor" else 1,
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
    dialogue_claim_memory = _collect_dialogue_claim_memory(segments)
    timeline_alignment_memory = _collect_timeline_alignment_memory(
        segments,
        identity_memory=identity_memory,
        audio_event_memory=audio_event_memory,
        dialogue_claim_memory=dialogue_claim_memory,
    )
    return {
        "identity_memory": identity_memory,
        "audio_event_memory": audio_event_memory,
        "dialogue_claim_memory": dialogue_claim_memory,
        "timeline_alignment_memory": timeline_alignment_memory,
    }


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
        settings["use_audio_in_video"] = _coerce_bool(settings.get("use_audio_in_video"), True)
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
                planner_context = _planner_context_from_segments(list(bundle.get("segments") or []))
                return {
                    "cache_hit": True,
                    "cache_dir": self.workspace.relative_path(cache_dir),
                    "manifest": bundle["manifest"],
                    "segments": bundle["segments"],
                    "summary": bundle["summary"],
                    "planner_context": planner_context,
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
            planner_context = _planner_context_from_segments(built_segments)
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
                "identity_memory_count": len(list(planner_context.get("identity_memory") or [])),
                "audio_event_memory_count": len(list(planner_context.get("audio_event_memory") or [])),
                "dialogue_claim_memory_count": len(list(planner_context.get("dialogue_claim_memory") or [])),
                "timeline_alignment_memory_count": len(list(planner_context.get("timeline_alignment_memory") or [])),
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
                "planner_context": planner_context,
                "video_fingerprint": video_fingerprint,
            }
