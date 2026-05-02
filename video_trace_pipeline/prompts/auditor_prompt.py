from __future__ import annotations

from typing import Any

from .shared import compact_json_rules, format_task, format_tool_outputs, pretty_json


def build_auditor_messages(task: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    system = "\n".join(
        [
            "You are the Auditor in a simple video QA trace pipeline.",
            "Diagnose whether the trace is supported by the supplied textual evidence.",
            "Return PASS only when the answer is justified by the available evidence.",
            compact_json_rules(),
        ]
    )
    user = "\n".join(
        [
            "Return JSON with shape:",
            '{"audit":{"verdict":"PASS","confidence":0.0,"findings":[],"missing_information":[],"feedback":""}}',
            "",
            "TASK:",
            format_task(task),
            "",
            "TRACE:",
            pretty_json(context.get("trace") or {}),
            "",
            "PREVIOUS_STEPS:",
            format_tool_outputs(context.get("previous_steps") or []),
        ]
    )
    return {"system": system, "user": user.strip()}
