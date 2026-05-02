from __future__ import annotations

import json
from typing import Any, Dict, List


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def compact_json_rules() -> str:
    return (
        "Return one valid JSON object only. Do not include markdown fences, prose preambles, "
        "hidden chain-of-thought, or comments. Use empty arrays or null only when the schema needs them."
    )


def _clip_text(value: Any, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "."


def format_task(task: Dict[str, Any]) -> str:
    return pretty_json(
        {
            "benchmark": task.get("benchmark"),
            "sample_key": task.get("sample_key"),
            "video_id": task.get("video_id"),
            "question": task.get("question"),
            "options": task.get("options") or [],
            "initial_trace": task.get("initial_trace"),
            "initial_trace_steps": task.get("initial_trace_steps") or [],
        }
    )


def format_tool_outputs(previous_steps: List[Dict[str, Any]]) -> str:
    compact = []
    for record in list(previous_steps or [])[-20:]:
        step = dict(record.get("step") or {})
        result = dict(record.get("result") or {})
        compact.append(
            {
                "round": record.get("round"),
                "step": step,
                "ok": result.get("ok"),
                "output": result.get("output"),
                "error": result.get("error"),
            }
        )
    return pretty_json(compact)


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
    timestamps = sorted(
        timestamp
        for timestamp in (_frame_timestamp(frame) for frame in list(frames or []))
        if timestamp is not None
    )
    if not timestamps:
        return ""

    timestamp_text = ", ".join(_format_seconds(item) for item in timestamps[:12])
    return (
        "Frame timestamps: %s. Use frame timestamps when temporal order matters; "
        "do not infer chronology from retrieval order alone."
    ) % timestamp_text


TOOL_PURPOSES = {
    "visual_temporal_grounder": "Find candidate visual time windows for a specific event, object state, chart appearance, or scene phase.",
    "audio_temporal_grounder": "Find candidate audio time windows for a sound event or spoken-content-related audio cue.",
    "frame_retriever": "Materialize bounded frames only when an exact/readable/static frame, OCR-quality still, or true frame-by-frame inspection is explicitly required; never use it as a full-video search.",
    "asr": "Transcribe speech in a clip, ideally with timestamps and speaker attribution.",
    "dense_captioner": "Summarize a bounded clip with dense visual/audio descriptions and on-screen text hints.",
    "ocr": "Read visible text or numbers from grounded clips or complete frames.",
    "spatial_grounder": "Localize the answer-critical object, person, mark, or region inside grounded clip(s) or frame(s), especially when multiple same-type candidates appear.",
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
