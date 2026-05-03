from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from ..common import sanitize_for_persistence


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(1, int(len(str(text)) / 4))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _first_text(payload: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _segment_time_bounds(segment: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    for start_key, end_key in (("start_s", "end_s"), ("start", "end"), ("clip_start_s", "clip_end_s")):
        if segment.get(start_key) is None or segment.get(end_key) is None:
            continue
        try:
            return float(segment.get(start_key)), float(segment.get(end_key))
        except Exception:
            continue
    clip = segment.get("clip") if isinstance(segment.get("clip"), dict) else {}
    if isinstance(clip, dict):
        try:
            if clip.get("start_s") is not None and clip.get("end_s") is not None:
                return float(clip.get("start_s")), float(clip.get("end_s"))
        except Exception:
            pass
    return None, None


def _segment_time_label(start_s: Optional[float], end_s: Optional[float]) -> str:
    if start_s is None and end_s is None:
        return ""
    if start_s is None:
        return "?.???-%.3fs" % float(end_s)
    if end_s is None:
        return "%.3f-?.???s" % float(start_s)
    return "%.3f-%.3fs" % (float(start_s), float(end_s))


def _extract_asr_text(segment: Dict[str, Any]) -> str:
    text = _first_text(
        segment,
        (
            "asr_text",
            "transcript_text",
            "transcript",
            "speech_text",
            "text",
        ),
    )
    if text:
        return text
    nested_asr = segment.get("asr") if isinstance(segment.get("asr"), dict) else {}
    if nested_asr:
        nested_text = _first_text(nested_asr, ("transcript_text", "transcript", "text", "overall_text"))
        if nested_text:
            return nested_text
        parts: List[str] = []
        for key in ("transcript_spans", "segments", "asr_segments"):
            value = nested_asr.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    part = _clean_text(item.get("text") or item.get("transcript"))
                else:
                    part = _clean_text(item)
                if part:
                    parts.append(part)
        if parts:
            return _clean_text(" ".join(parts))
    parts: List[str] = []
    for key in ("transcript_segments", "segments", "asr_segments"):
        value = segment.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                part = _clean_text(item.get("text") or item.get("transcript"))
            else:
                part = _clean_text(item)
            if part:
                parts.append(part)
    return _clean_text(" ".join(parts))


def _extract_dense_caption_text(segment: Dict[str, Any]) -> str:
    text = _first_text(
        segment,
        (
            "dense_caption_summary",
            "dense_captions_summary",
            "caption_summary",
            "visual_summary",
            "caption",
            "dense_caption",
        ),
    )
    if text:
        return text
    nested_dense = segment.get("dense_caption") if isinstance(segment.get("dense_caption"), dict) else {}
    if nested_dense:
        nested_summary = _first_text(
            nested_dense,
            (
                "overall_summary",
                "summary",
                "dense_caption_summary",
                "caption_summary",
                "visual_summary",
                "caption",
            ),
        )
        if nested_summary:
            return nested_summary
        parts: List[str] = []
        for item in list(nested_dense.get("captions") or []):
            if not isinstance(item, dict):
                part = _clean_text(item)
                if part:
                    parts.append(part)
                continue
            for key in ("visual", "caption", "text", "audio"):
                part = _clean_text(item.get(key))
                if part:
                    parts.append(part)
            attributes = item.get("attributes")
            if isinstance(attributes, list):
                parts.extend(_clean_text(attr) for attr in attributes if _clean_text(attr))
            on_screen_text = item.get("on_screen_text")
            if isinstance(on_screen_text, list):
                parts.extend(_clean_text(text_item) for text_item in on_screen_text if _clean_text(text_item))
        if parts:
            return _clean_text(" ".join(parts))
    parts: List[str] = []
    for key in ("dense_captions", "captions", "caption_segments"):
        value = segment.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                part = _clean_text(item.get("caption") or item.get("text") or item.get("summary"))
            else:
                part = _clean_text(item)
            if part:
                parts.append(part)
    return _clean_text(" ".join(parts))


def _extract_ocr_text(segment: Dict[str, Any]) -> str:
    text = _first_text(segment, ("ocr_text", "ocr_summary", "visible_text", "text_reading"))
    if text:
        return text
    nested_dense = segment.get("dense_caption") if isinstance(segment.get("dense_caption"), dict) else {}
    if nested_dense:
        parts: List[str] = []
        for item in list(nested_dense.get("captions") or []):
            if not isinstance(item, dict):
                continue
            on_screen_text = item.get("on_screen_text")
            if isinstance(on_screen_text, list):
                parts.extend(_clean_text(text_item) for text_item in on_screen_text if _clean_text(text_item))
            elif isinstance(on_screen_text, str):
                part = _clean_text(on_screen_text)
                if part:
                    parts.append(part)
        if parts:
            return _clean_text(" ".join(parts))
    parts: List[str] = []
    for key in ("ocr_lines", "reads", "ocr_results"):
        value = segment.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                part = _clean_text(item.get("text") or item.get("value"))
            else:
                part = _clean_text(item)
            if part:
                parts.append(part)
    return _clean_text(" ".join(parts))


def _task_terms(task: Any) -> set[str]:
    text = "%s %s" % (
        getattr(task, "question", "") or "",
        " ".join(str(item or "") for item in list(getattr(task, "options", []) or [])),
    )
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}", text)
        if len(token) >= 3
    }
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "what",
        "which",
        "when",
        "where",
        "while",
        "there",
        "about",
        "does",
        "most",
        "least",
        "after",
        "before",
        "option",
    }
    return {term for term in terms if term not in stop}


def _score_segment(segment_text: str, terms: set[str]) -> int:
    text = segment_text.casefold()
    if not text:
        return 0
    score = 0
    for term in terms:
        if term in text:
            score += 3
    if re.search(r"\b(first|last|before|after|then|next|while|during|start|end)\b", text):
        score += 1
    return score


def _row_to_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(row or {}).items()
        if key
        in {
            "segment_index",
            "time_start_s",
            "time_end_s",
            "time",
            "asr_text",
            "dense_caption_text",
            "ocr_text",
            "score",
        }
        and value not in (None, "")
    }


def build_preprocess_context_pack(
    preprocess_context: Optional[Dict[str, Any]],
    task: Any,
    *,
    target_token_budget: int = 24000,
    max_selected_segments: int = 24,
) -> Optional[Dict[str, Any]]:
    if not isinstance(preprocess_context, dict):
        return None
    planner_segments = preprocess_context.get("planner_segments")
    if not isinstance(planner_segments, list):
        return None

    terms = _task_terms(task)
    segment_rows: List[Dict[str, Any]] = []
    for index, segment in enumerate(planner_segments):
        if not isinstance(segment, dict):
            continue
        start_s, end_s = _segment_time_bounds(segment)
        asr_text = _extract_asr_text(segment)
        dense_caption_text = _extract_dense_caption_text(segment)
        ocr_text = _extract_ocr_text(segment)
        combined_text = _clean_text(" ".join(part for part in (asr_text, dense_caption_text, ocr_text) if part))
        score = _score_segment(combined_text, terms)
        row = {
            "segment_index": index,
            "time_start_s": start_s,
            "time_end_s": end_s,
            "time": _segment_time_label(start_s, end_s),
            "asr_text": asr_text,
            "dense_caption_text": dense_caption_text,
            "ocr_text": ocr_text,
            "combined_text": combined_text,
            "score": score,
        }
        segment_rows.append(row)

    all_chunks = [_row_to_chunk(row) for row in segment_rows]
    all_chunks = [chunk for chunk in all_chunks if chunk]
    full_normalized_token_estimate = estimate_tokens(all_chunks)
    full_token_estimate = estimate_tokens(planner_segments)
    selection_policy = "all planner_segments normalized into combined ASR+dense captions+OCR; no separate dense-caption duplication"
    full_preprocess_included = True
    if full_normalized_token_estimate <= int(target_token_budget):
        chunks = all_chunks
        token_total = full_normalized_token_estimate
    else:
        selected_indices = set()
        ranked = sorted(segment_rows, key=lambda item: (-int(item.get("score") or 0), item.get("segment_index") or 0))
        for row in ranked[: max(1, int(max_selected_segments))]:
            selected_indices.add(int(row.get("segment_index") or 0))
        if not selected_indices and segment_rows:
            selected_indices.add(0)

        chunks = []
        token_total = 0
        for row in segment_rows:
            if int(row.get("segment_index") or 0) not in selected_indices:
                continue
            chunk = _row_to_chunk(row)
            chunk_tokens = estimate_tokens(chunk)
            if chunks and token_total + chunk_tokens > target_token_budget:
                break
            token_total += chunk_tokens
            chunks.append(chunk)
        selection_policy = "task-term-ranked subset of combined ASR+dense captions+OCR from planner_segments; no separate dense-caption duplication"
        full_preprocess_included = False
    manifest = dict(preprocess_context.get("manifest") or {})
    pack = {
        "kind": "preprocess_context_pack",
        "source": preprocess_context.get("source") or "planner_segments.json",
        "cache_dir": preprocess_context.get("cache_dir"),
        "manifest": {
            **manifest,
            "segment_count": len(segment_rows),
            "selected_segment_count": len(chunks),
            "full_planner_segments_token_estimate": full_token_estimate,
            "full_normalized_preprocess_token_estimate": full_normalized_token_estimate,
            "packed_token_estimate": token_total,
            "target_token_budget": int(target_token_budget),
            "selection_policy": selection_policy,
            "full_preprocess_included_in_prompt": full_preprocess_included,
            "full_preprocess_available_in_run_artifacts": True,
        },
        "chunks": chunks,
    }
    return sanitize_for_persistence(pack)


def _entry_time_label(entry: Dict[str, Any]) -> str:
    if entry.get("frame_ts_s") is not None:
        try:
            return "%.3fs" % float(entry.get("frame_ts_s"))
        except Exception:
            pass
    if entry.get("time_start_s") is not None or entry.get("time_end_s") is not None:
        return _segment_time_label(entry.get("time_start_s"), entry.get("time_end_s"))
    intervals = entry.get("time_intervals")
    if isinstance(intervals, list) and intervals:
        labels = []
        for item in intervals[:4]:
            if isinstance(item, dict):
                labels.append(_segment_time_label(item.get("start_s"), item.get("end_s")))
        return ", ".join(label for label in labels if label)
    return ""


def build_evidence_cards(
    entries: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    *,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    observations_by_evidence_id: Dict[str, List[Dict[str, Any]]] = {}
    for observation in list(observations or []):
        if not isinstance(observation, dict):
            continue
        evidence_id = str(observation.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        observations_by_evidence_id.setdefault(evidence_id, []).append(dict(observation))

    cards: List[Dict[str, Any]] = []
    for entry in list(entries or [])[-max(1, int(limit)) :]:
        if not isinstance(entry, dict):
            continue
        evidence_id = str(entry.get("evidence_id") or "").strip()
        linked_observations = observations_by_evidence_id.get(evidence_id, [])
        observation_texts: List[str] = []
        observation_ids: List[str] = []
        for observation in linked_observations[:8]:
            observation_id = str(observation.get("observation_id") or "").strip()
            if observation_id:
                observation_ids.append(observation_id)
            text = _clean_text(observation.get("atomic_text") or observation.get("text") or observation.get("object_text"))
            if text and text not in observation_texts:
                observation_texts.append(text)
        evidence_text = _clean_text(entry.get("evidence_text"))
        text_parts = [part for part in [evidence_text] + observation_texts if part]
        card = {
            "evidence_id": evidence_id,
            "source_tool": entry.get("tool_name"),
            "status": entry.get("status") or "candidate",
            "time": _entry_time_label(entry),
            "text": "\n".join(text_parts)[:2400],
            "observation_ids": observation_ids,
            "artifact_refs": list(entry.get("artifact_refs") or [])[:6],
            "metadata": dict(entry.get("metadata") or {}),
        }
        if not card["text"]:
            card["text"] = "(no textual observation extracted; use only as media/artifact pointer)"
        cards.append(sanitize_for_persistence(card))
    return cards
