from __future__ import annotations

import json
from typing import Any, Dict, List


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _coerce_seconds(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _format_seconds(value: Any) -> str:
    seconds = _coerce_seconds(value)
    if seconds is None:
        return ""
    text = "%.3f" % seconds
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _frame_metadata(frame: Any) -> Dict[str, Any]:
    if isinstance(frame, dict):
        return dict(frame.get("metadata") or {})
    return dict(getattr(frame, "metadata", {}) or {})


def _frame_timestamp(frame: Any) -> float | None:
    if isinstance(frame, dict):
        return _coerce_seconds(frame.get("timestamp_s", frame.get("timestamp")))
    return _coerce_seconds(getattr(frame, "timestamp_s", None))


def render_frame_sequence_context(frames: List[dict]) -> str:
    sequence_frames = []
    for frame in list(frames or []):
        metadata = _frame_metadata(frame)
        if not metadata:
            continue
        if not any(
            metadata.get(key) is not None
            for key in (
                "requested_timestamp_s",
                "neighbor_radius_s",
                "sequence_mode",
                "sequence_role",
                "sequence_index",
                "sequence_sort_order",
            )
        ):
            continue
        sequence_frames.append((frame, metadata))
    if not sequence_frames:
        return ""

    anchors = [
        _coerce_seconds(metadata.get("requested_timestamp_s"))
        for _, metadata in sequence_frames
        if _coerce_seconds(metadata.get("requested_timestamp_s")) is not None
    ]
    radii = [
        _coerce_seconds(metadata.get("neighbor_radius_s"))
        for _, metadata in sequence_frames
        if _coerce_seconds(metadata.get("neighbor_radius_s")) is not None
    ]
    timestamps = sorted(
        timestamp
        for timestamp in (_frame_timestamp(frame) for frame, _ in sequence_frames)
        if timestamp is not None
    )

    anchor_text = _format_seconds(anchors[0]) if anchors else "the requested timestamp"
    radius_text = _format_seconds(radii[0]) if radii else ""
    timestamp_text = ", ".join(_format_seconds(item) for item in timestamps[:12])

    parts = [
        "These frames are a chronological sequence centered on timestamp %s." % anchor_text,
        "Use neighboring frames before and after the anchor to understand action/state; do not treat one still frame as decisive.",
    ]
    if radius_text:
        parts.append("The requested neighbor radius is %s." % radius_text)
    if timestamp_text:
        parts.append("Frame timestamps in this sequence: %s." % timestamp_text)
    return " ".join(parts)


TOOL_PURPOSES = {
    "visual_temporal_grounder": "Find candidate visual time windows for a specific event, object state, chart appearance, or scene phase.",
    "audio_temporal_grounder": "Find candidate audio time windows for a sound event or spoken-content-related audio cue.",
    "frame_retriever": "Choose the most useful static frame(s) from a known clip; never use it as a full-video search.",
    "asr": "Transcribe speech in a clip, ideally with timestamps and speaker attribution.",
    "dense_captioner": "Summarize a bounded clip with dense visual/audio descriptions and on-screen text hints.",
    "ocr": "Read visible text or numbers from a frame or localized region.",
    "spatial_grounder": "Localize the answer-critical object, person, mark, or region inside already retrieved frame(s), especially when multiple same-type candidates appear.",
    "generic_purpose": "Perform targeted multimodal extraction or evidence-conditioned reasoning when no narrower tool fits.",
}


def render_tool_catalog(tool_catalog: Dict[str, Dict[str, object]]) -> str:
    lines = ["AVAILABLE_TOOLS:"]
    for name in sorted(tool_catalog):
        spec = tool_catalog[name] or {}
        description = str(spec.get("description") or TOOL_PURPOSES.get(name) or "").strip()
        model = str(spec.get("model") or "").strip()
        request_fields = [str(item).strip() for item in list(spec.get("request_fields") or []) if str(item).strip()]
        output_fields = [str(item).strip() for item in list(spec.get("output_fields") or []) if str(item).strip()]
        request_schema = [str(item).strip() for item in list(spec.get("request_schema") or []) if str(item).strip()]
        output_schema = [str(item).strip() for item in list(spec.get("output_schema") or []) if str(item).strip()]
        request_nested = [str(item).strip() for item in list(spec.get("request_nested") or []) if str(item).strip()]
        output_nested = [str(item).strip() for item in list(spec.get("output_nested") or []) if str(item).strip()]
        line = "- %s" % name
        details = []
        if description:
            details.append(description)
        if model:
            details.append("model=%s" % model)
        if request_fields:
            details.append("args=%s" % ", ".join(request_fields))
        if output_fields:
            details.append("outputs=%s" % ", ".join(output_fields))
        if details:
            line += ": " + " | ".join(details)
        lines.append(line)
        if request_schema:
            lines.append("  request_schema: %s" % "; ".join(request_schema))
        if output_schema:
            lines.append("  output_schema: %s" % "; ".join(output_schema))
        if request_nested:
            lines.append("  request_nested: %s" % " || ".join(request_nested))
        if output_nested:
            lines.append("  output_nested: %s" % " || ".join(output_nested))
    return "\n".join(lines)
