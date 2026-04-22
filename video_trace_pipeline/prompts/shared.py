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
    "spatial_grounder": "Locate an object, entity, chart region, or visual target in a frame.",
    "generic_purpose": "Perform targeted multimodal extraction or evidence-conditioned reasoning when no narrower tool fits.",
}


def render_tool_catalog(tool_catalog: Dict[str, Dict[str, object]]) -> str:
    lines = ["AVAILABLE_TOOLS:"]
    for name in sorted(tool_catalog):
        spec = tool_catalog[name] or {}
        description = str(spec.get("description") or TOOL_PURPOSES.get(name) or "").strip()
        model = str(spec.get("model") or "").strip()
        request_fields = [str(item).strip() for item in list(spec.get("request_fields") or []) if str(item).strip()]
        line = "- %s" % name
        details = []
        if description:
            details.append(description)
        if model:
            details.append("model=%s" % model)
        if request_fields:
            details.append("args=%s" % ", ".join(request_fields))
        if details:
            line += ": " + " | ".join(details)
        lines.append(line)
    return "\n".join(lines)
