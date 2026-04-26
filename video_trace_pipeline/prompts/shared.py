from __future__ import annotations

import json
from typing import Dict


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


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
