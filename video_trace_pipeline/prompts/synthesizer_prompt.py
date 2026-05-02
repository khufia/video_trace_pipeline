from __future__ import annotations

from typing import Any

from .shared import compact_json_rules, format_task, format_tool_outputs, pretty_json


def build_synthesizer_messages(task: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    system = "\n".join(
        [
            "You are the Synthesizer in a simple video QA trace pipeline.",
            "Write a short evidence-grounded trace and final answer from supplied tool outputs.",
            "Do not invent observations that are absent from the provided text evidence.",
            compact_json_rules(),
        ]
    )
    user = "\n".join(
        [
            "Return JSON with shape:",
            '{"trace":{"answer":"","confidence":0.0,"reasoning":"","evidence":[],"trace_steps":[],"open_questions":[]}}',
            "",
            "TASK:",
            format_task(task),
            "",
            "PLAN:",
            pretty_json(context.get("plan")),
            "",
            "PREVIOUS_STEPS:",
            format_tool_outputs(context.get("previous_steps") or []),
            "",
            "LATEST_TRACE:",
            pretty_json(context.get("latest_trace")),
            "",
            "LATEST_AUDIT:",
            pretty_json(context.get("latest_audit")),
        ]
    )
    return {"system": system, "user": user.strip()}
