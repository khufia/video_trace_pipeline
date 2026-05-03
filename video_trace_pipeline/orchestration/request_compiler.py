from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from ..common import sanitize_for_persistence
from ..schemas.tool_requests import (
    ASRRequest,
    AudioTemporalGrounderRequest,
    DenseCaptionRequest,
    FrameRetrieverRequest,
    GenericPurposeRequest,
    OCRRequest,
    SpatialGrounderRequest,
    ToolRequest,
    VisualTemporalGrounderRequest,
)


REQUEST_MODELS: Dict[str, Type[ToolRequest]] = {
    "visual_temporal_grounder": VisualTemporalGrounderRequest,
    "frame_retriever": FrameRetrieverRequest,
    "audio_temporal_grounder": AudioTemporalGrounderRequest,
    "asr": ASRRequest,
    "dense_captioner": DenseCaptionRequest,
    "ocr": OCRRequest,
    "spatial_grounder": SpatialGrounderRequest,
    "generic_purpose": GenericPurposeRequest,
}


FRAME_EVENT_RE = re.compile(
    r"\b(before|after|when|while|during|start|starts|end|ends|transition|first|last|earliest|latest|then|next|"
    r"chronological|sequence|moment|event|appears?|disappears?|changes?|returns?|turns?|begins?|happens?|"
    r"receives?|shoots?|passes?|speaking|spoken|said|uses?|released?|shows?|visible at)\b",
    re.I,
)

TIME_RANGE_RE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*(?:-|to|through|until|–)\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
    re.I,
)

TIME_POINT_RE = re.compile(r"(?P<point>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", re.I)


def _model_fields(model_cls: Type[BaseModel]) -> set[str]:
    if hasattr(model_cls, "model_fields"):
        return set(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return set(getattr(model_cls, "__fields__").keys())
    return set()


def _model_validate(model_cls: Type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _task_dict(task: Any) -> Dict[str, Any]:
    if task is None:
        return {}
    if isinstance(task, dict):
        return dict(task)
    return {
        "sample_key": getattr(task, "sample_key", None),
        "video_id": getattr(task, "video_id", None),
        "video_path": getattr(task, "video_path", None),
        "question": getattr(task, "question", None),
        "options": list(getattr(task, "options", []) or []),
    }


def _video_id(task: Any, request: Dict[str, Any]) -> str:
    for clip in list(request.get("clips") or []):
        if isinstance(clip, dict) and str(clip.get("video_id") or "").strip():
            return str(clip.get("video_id")).strip()
    task_payload = _task_dict(task)
    return str(task_payload.get("video_id") or task_payload.get("sample_key") or "video").strip()


def _field_sanitize(tool_name: str, request: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    model_cls = REQUEST_MODELS.get(tool_name)
    if model_cls is None:
        return dict(request), []
    allowed = _model_fields(model_cls) | {"tool_name"}
    sanitized: Dict[str, Any] = {}
    dropped: List[str] = []
    for key, value in dict(request or {}).items():
        if key in allowed:
            sanitized[key] = value
        elif key in {"query_context", "context", "notes"} and tool_name == "generic_purpose":
            text_contexts = [str(item).strip() for item in list(sanitized.get("text_contexts") or []) if str(item).strip()]
            text_value = _normalize_text(value)
            if text_value:
                text_contexts.append(text_value)
                sanitized["text_contexts"] = text_contexts
            dropped.append(str(key))
        else:
            dropped.append(str(key))
    sanitized.setdefault("tool_name", tool_name)
    return sanitized, dropped


def _validate(tool_name: str, request: Dict[str, Any]) -> Tuple[bool, str]:
    model_cls = REQUEST_MODELS.get(tool_name)
    if model_cls is None:
        return False, "unknown tool"
    try:
        _model_validate(model_cls, request)
        return True, ""
    except (ValidationError, ValueError) as exc:
        return False, str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__


def _seconds_range_from_time_hints(time_hints: List[str]) -> Optional[Tuple[float, float]]:
    for hint in list(time_hints or []):
        match = TIME_RANGE_RE.search(str(hint or ""))
        if not match:
            continue
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        return start_s, end_s
    points: List[float] = []
    for hint in list(time_hints or []):
        for match in TIME_POINT_RE.finditer(str(hint or "")):
            points.append(float(match.group("point")))
    if len(points) >= 2:
        return min(points), max(points)
    return None


def _single_second_from_time_hints(time_hints: List[str]) -> Optional[float]:
    for hint in list(time_hints or []):
        match = TIME_POINT_RE.search(str(hint or ""))
        if match:
            return float(match.group("point"))
    return None


def _ensure_clip_from_time_hints(task: Any, request: Dict[str, Any], diagnostics: List[str]) -> Dict[str, Any]:
    compiled = dict(request)
    if compiled.get("clips"):
        return compiled
    time_hints = [str(item).strip() for item in list(compiled.get("time_hints") or []) if str(item).strip()]
    if not time_hints:
        return compiled
    video_id = _video_id(task, compiled)
    seconds_range = _seconds_range_from_time_hints(time_hints)
    if seconds_range is not None:
        start_s, end_s = seconds_range
        compiled["clips"] = [{"video_id": video_id, "start_s": round(start_s, 3), "end_s": round(end_s, 3)}]
        compiled["time_hints"] = []
        compiled["sequence_mode"] = "chronological"
        compiled["sort_order"] = "chronological"
        diagnostics.append("converted_time_hint_range_to_bounded_clip")
        return compiled
    point_s = _single_second_from_time_hints(time_hints)
    if point_s is not None:
        radius_s = max(0.5, float(compiled.get("neighbor_radius_s") or 2.0))
        compiled["clips"] = [
            {
                "video_id": video_id,
                "start_s": round(max(0.0, point_s - radius_s), 3),
                "end_s": round(point_s + radius_s, 3),
            }
        ]
        diagnostics.append("converted_single_time_hint_to_anchor_clip")
    return compiled


def _infer_task_type(task: Any, query: str) -> str:
    task_payload = _task_dict(task)
    text = _normalize_text(
        "%s %s %s"
        % (
            task_payload.get("question") or "",
            " ".join(str(item or "") for item in list(task_payload.get("options") or [])),
            query,
        )
    ).casefold()
    if "how many" in text or "count" in text or re.search(r"\bnumber of\b", text):
        return "counting"
    if any(term in text for term in ("order", "chronological", "sequence", "before", "after", "first", "last", "third")):
        return "temporal_order"
    if any(term in text for term in ("read", "text", "ocr", "chart", "percentage", "price", "sign", "label", "score")):
        return "exact_text_or_chart"
    if any(term in text for term in ("left", "right", "closest", "near", "far", "middle", "side")):
        return "spatial_identity"
    return "option_mapping"


def _typed_generic_query(task_type: str, original_query: str) -> str:
    base = _normalize_text(original_query)[:1200]
    schema = (
        "Return JSON only with keys answer, supporting_points, confidence, analysis. "
        "Keep answer to the final value/choice or indeterminate; put structured evidence rows in supporting_points; "
        "put one concise uncertainty/calculation note in analysis."
    )
    if task_type == "counting":
        contract = (
            "Use only the supplied media, transcripts, text, and evidence cards. "
            "%s In supporting_points, list each frame/time or clip, the directly visible/stated items counted, "
            "count_for_item, and why. If the count is not directly supported, set answer to indeterminate."
            % schema
        )
    elif task_type == "temporal_order":
        contract = (
            "Use only the supplied media, transcripts, text, and evidence cards. "
            "%s In supporting_points, list timestamp/range, event, source, and whether it is directly observed or inferred. "
            "Do not choose an option unless every answer-critical ordering claim is supported."
            % schema
        )
    elif task_type == "exact_text_or_chart":
        contract = (
            "Use only the supplied media, OCR/text, transcripts, and evidence cards. "
            "%s First extract visible text/value pairs in supporting_points, then compute or map to options in analysis. "
            "If text is unreadable or a value is missing, set answer to indeterminate."
            % schema
        )
    elif task_type == "spatial_identity":
        contract = (
            "Use only the supplied frames/clips and evidence cards. "
            "%s In supporting_points, list frame/time, candidate object, position cue, and visual support. "
            "If the object identity or position is ambiguous, set answer to indeterminate."
            % schema
        )
    else:
        contract = (
            "Use only the supplied media, transcripts, text, and evidence cards. "
            "%s List each answer-critical claim with its supporting source in supporting_points. "
            "If support is incomplete or conflicting, set answer to indeterminate."
            % schema
        )
    return "%s\n\nTask-specific query:\n%s" % (contract, base or "Resolve the answer-critical question from the supplied context.")


def compile_planner_tool_request(tool_name: str, request: Dict[str, Any], *, task: Any = None) -> Dict[str, Any]:
    original_tool_name = str(tool_name or (request or {}).get("tool_name") or "").strip()
    compiled_tool_name = original_tool_name
    original_request = dict(request or {})
    original_request.setdefault("tool_name", original_tool_name)
    diagnostics: List[str] = []
    action = "unchanged"

    compiled, dropped_fields = _field_sanitize(compiled_tool_name, original_request)
    for field_name in dropped_fields:
        diagnostics.append("dropped_unknown_field:%s" % field_name)

    if compiled_tool_name == "frame_retriever":
        compiled = _ensure_clip_from_time_hints(task, compiled, diagnostics)
        query = _normalize_text(compiled.get("query"))
        has_event_terms = bool(FRAME_EVENT_RE.search(query))
        has_clips = bool(compiled.get("clips"))
        has_time_hints = bool(compiled.get("time_hints"))
        if has_event_terms and has_clips:
            action = "frame_retriever_bounded_event_query_to_chronological_coverage"
            diagnostics.append("frame_retriever_temporal_query_removed")
            compiled["query"] = "clear answer-critical visual frames inside the supplied time window"
            compiled["sequence_mode"] = "chronological"
            compiled["sort_order"] = "chronological"
            compiled["time_hints"] = []
        elif has_event_terms and has_time_hints:
            action = "frame_retriever_temporal_anchor_query_to_anchor_window"
            diagnostics.append("frame_retriever_temporal_query_removed")
            compiled["query"] = "clear answer-critical visual frames near the supplied timestamp"
            compiled["sequence_mode"] = "anchor_window"
            compiled["sort_order"] = "chronological"
        elif has_event_terms:
            action = "reroute_unbounded_frame_event_query_to_visual_temporal_grounder"
            diagnostics.append("frame_retriever_unbounded_event_query_rerouted")
            compiled_tool_name = "visual_temporal_grounder"
            compiled = {
                "tool_name": "visual_temporal_grounder",
                "query": query or "answer-critical visual event",
                "top_k": 5,
            }
        elif not query:
            compiled["query"] = "clear visual evidence inside the supplied time window"
    elif compiled_tool_name == "generic_purpose":
        query = _normalize_text(compiled.get("query"))
        task_type = _infer_task_type(task, query)
        compiled["query"] = _typed_generic_query(task_type, query)
        diagnostics.append("typed_generic_contract:%s" % task_type)
        action = "typed_generic_contract"
        for key in ("clips", "frames", "transcripts", "text_contexts", "evidence_ids"):
            if key in original_request and key not in compiled:
                compiled[key] = original_request[key]
    elif compiled_tool_name == "ocr":
        if not _normalize_text(compiled.get("query")):
            compiled["query"] = "read exact visible text from the supplied media"

    compiled["tool_name"] = compiled_tool_name
    compiled, dropped_after = _field_sanitize(compiled_tool_name, compiled)
    for field_name in dropped_after:
        diagnostics.append("dropped_unknown_field_after_compile:%s" % field_name)

    original_valid, original_error = _validate(original_tool_name, original_request)
    compiled_valid, compiled_error = _validate(compiled_tool_name, compiled)
    return sanitize_for_persistence(
        {
            "original_tool_name": original_tool_name,
            "tool_name": compiled_tool_name,
            "action": action,
            "diagnostics": diagnostics,
            "original_request": original_request,
            "tool_request": compiled,
            "original_valid": original_valid,
            "original_error": original_error,
            "compiled_valid": compiled_valid,
            "compiled_error": compiled_error,
        }
    )
