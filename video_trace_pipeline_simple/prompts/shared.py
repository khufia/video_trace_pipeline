from __future__ import annotations

import json
from typing import Any


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


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


def format_task(task: dict[str, Any]) -> str:
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


def format_preprocess(preprocess: dict[str, Any]) -> str:
    segments = []
    for segment in list(preprocess.get("segments") or [])[:80]:
        segment = dict(segment or {})
        frames = [dict(item or {}) for item in list(segment.get("frames") or []) if isinstance(item, dict)]
        segments.append(
            {
                "id": segment.get("id"),
                "start_s": segment.get("start_s"),
                "end_s": segment.get("end_s"),
                "ocr_text": _clip_text(segment.get("ocr_text"), 700),
                "transcript": [
                    {
                        "start_s": item.get("start_s"),
                        "end_s": item.get("end_s"),
                        "text": _clip_text(item.get("text"), 300),
                    }
                    for item in list(segment.get("transcript") or [])[:8]
                    if isinstance(item, dict)
                ],
                "frame_times": [item.get("timestamp_s") for item in frames],
            }
        )
    compact = {
        "ok": preprocess.get("ok"),
        "video_id": preprocess.get("video_id"),
        "video_duration_s": preprocess.get("video_duration_s"),
        "source": (preprocess.get("metadata") or {}).get("source"),
        "cache_path": (preprocess.get("metadata") or {}).get("cache_path"),
        "settings": (preprocess.get("metadata") or {}).get("settings"),
        "segments": segments,
    }
    return pretty_json(compact)


def format_tool_outputs(previous_steps: list[dict[str, Any]]) -> str:
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
