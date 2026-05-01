from __future__ import annotations

from typing import Any

from ..config import HIDDEN_PLAN_TOOLS
from ..plan_verify import TOOL_OUTPUTS
from .shared import compact_json_rules, format_preprocess, format_task, format_tool_outputs, pretty_json


def _planner_bindable_outputs(available_tools: dict[str, Any]) -> dict[str, list[str]]:
    visible = set((available_tools or {}).keys())
    return {
        tool: [path.removeprefix("output.") for path in paths]
        for tool, paths in sorted(TOOL_OUTPUTS.items())
        if tool not in HIDDEN_PLAN_TOOLS and (not visible or tool in visible)
    }


def build_planner_messages(task: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    bindable_outputs = _planner_bindable_outputs(context.get("available_tools") or {})
    system = "\n".join(
        [
            "You are the Planner in a simple video QA trace pipeline.",
            "Choose only media/analysis tools for the next round. Do not answer the question.",
            "Control tools are forbidden in plans: planner, synthesizer, auditor.",
            "This pipeline executes steps exactly in the order returned.",
            "Every step must use keys: id, tool, purpose, request, request_refs.",
            "Every request must use query plus optional temporal_scope, media, and options.",
            "Use temporal_scope for video time windows and anchors. Use media for frames, regions, transcript_segments, captions, and texts.",
            "Use PREPROCESS segments and initial trace steps as hints, not as proof for answer-critical fine detail.",
            "visual_temporal_grounder is the broad visual search tool; give it a strong query and optional options.top_k, and let it scan the video rather than constraining it to weak preprocess OCR hits.",
            "frame_retriever is not a temporal search tool. Give it literal clips from preprocess/initial trace/temporal grounding or literal temporal_scope.anchors.",
            "Use request_refs only to bind outputs from earlier steps or previous rounds.",
            "Each request_refs value must be a list of objects shaped {\"from_step\":\"s1\",\"output\":\"frames\"}.",
            "Never use string shorthand such as {\"frames\":[\"s1\"]}; never use list-style refs without output.",
            "Use only the output names listed in BINDABLE_OUTPUTS.",
            compact_json_rules(),
        ]
    )
    user_parts = [
        "Return JSON with shape:",
        '{"strategy":"...","steps":[{"id":"s1","tool":"visual_temporal_grounder","purpose":"...","request":{"query":"find when the scoreboard is visible","temporal_scope":{},"options":{}},"request_refs":{}},{"id":"s2","tool":"frame_retriever","purpose":"...","request":{"query":"sample readable scoreboard frames","temporal_scope":{},"options":{"num_frames":6}},"request_refs":{"temporal_scope.clips":[{"from_step":"s1","output":"segments"}]}},{"id":"s3","tool":"ocr","purpose":"...","request":{"query":"read the visible text","media":{},"options":{}},"request_refs":{"media.frames":[{"from_step":"s2","output":"frames"}]}}]}',
        "",
        "CURRENT PLAN SCHEMA:",
        "- top-level: strategy, steps",
        "- each step: id, tool, purpose, request, request_refs",
        "- request.query is the tool's natural-language instruction",
        "- request.temporal_scope.clips is a list of absolute video-time ranges: {start_s, end_s}",
        "- request.temporal_scope.anchors is a list of anchor times: {time_s, radius_s, reference}; reference is \"video\" by default or \"clip\" for clip-relative time",
        "- request.media carries evidence objects: frames, regions, transcript_segments, captions, texts",
        "- request.options carries small tool-specific knobs, e.g. {\"num_frames\":6}",
        "- dependencies go in request_refs as target_field -> list of {from_step, output}",
        "- valid example: \"request_refs\":{\"media.frames\":[{\"from_step\":\"s2\",\"output\":\"frames\"}]}",
        "- valid temporal example: \"request_refs\":{\"temporal_scope.clips\":[{\"from_step\":\"s1\",\"output\":\"segments\"}]}",
        "- invalid examples: \"request_refs\":{\"frames\":[\"s1\"]}, \"request_refs\":{\"frames\":\"s1\"}, \"request_refs\":[...]",
        "",
        "BINDABLE_OUTPUTS:",
        pretty_json(bindable_outputs),
        "",
        "TASK:",
        format_task(task),
        "",
        "PREPROCESS:",
        format_preprocess(context.get("preprocess") or {}),
        "",
        "PREVIOUS_STEPS:",
        format_tool_outputs(context.get("previous_steps") or []),
        "",
        "LATEST_TRACE:",
        pretty_json(context.get("latest_trace")),
        "",
        "LATEST_AUDIT:",
        pretty_json(context.get("latest_audit")),
        "",
        "TASK_STATE:",
        pretty_json(context.get("task_state")),
        "",
        "AVAILABLE_TOOLS:",
        pretty_json(context.get("available_tools") or {}),
    ]
    if context.get("plan_errors"):
        user_parts.extend(["", "PLAN_ERRORS_TO_REPAIR:", pretty_json(context.get("plan_errors")), "REJECTED_PLAN:", pretty_json(context.get("rejected_plan"))])
    return {"system": system, "user": "\n".join(user_parts).strip()}
