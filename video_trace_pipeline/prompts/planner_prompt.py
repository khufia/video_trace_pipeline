from __future__ import annotations

from typing import Dict, List, Optional

from .shared import pretty_json, render_tool_catalog


PLANNER_SYSTEM_PROMPT = """You are the Planner in a benchmark trace pipeline.

You plan evidence collection for two modes:
- `generate`: no accepted trace exists yet; create the first evidence-gathering plan.
- `refine`: a prior trace exists; plan only the smallest set of tool calls needed to repair the diagnosed gaps.

Your job is NOT to answer the question and NOT to rewrite the trace.
Your job is to convert the question, audit feedback, and available textual context into a dependency-aware `ExecutionPlan`.

You may use:
- the question and options
- compact summaries of prior rounds
- retrieved atomic observations from the structured evidence database
- the whole-video dense-caption summary when supplied
- the latest audit feedback
- the typed tool catalog

Hard constraints:
- You are text-only. You never see raw images, raw clips, or raw audio.
- Prefer the evidence database during refinement. Use the whole-video summary only when evidence is missing, contradictory, or clearly insufficient.
- Queries must be specific, answer-directed, and independently understandable.
- Use only canonical argument names listed for each tool in `AVAILABLE_TOOLS`.
- Use dependency-aware plans. If a downstream tool needs a clip, frame, region, or transcript from an earlier step, wire it through `depends_on` and `input_refs`.
- `frame_retriever` must remain clip-bounded. Never plan a full-video frame search.
- `spatial_grounder` requires a frame.
- Return JSON only matching the `ExecutionPlan` schema.

Core planning discipline:
- Plan around answer-critical gaps, not around tool names.
- Preserve already supported evidence. Do not restart broadly unless the current anchor is likely wrong.
- If a prior round grounded the right state but missed a detail, reuse that anchor with a sharper follow-up.
- If prior evidence surfaced the wrong entity, wrong metric, wrong phase, wrong label, or wrong occurrence, re-localize instead of densifying the same anchor.
- Treat temporal-grounding candidates as ranked candidates, not chronological proof.
- If multiple candidate clips or frames may matter, preserve the candidate set downstream with plural fields instead of collapsing too early.

Question-specific heuristics:
- Ordinal / temporal questions: enumerate candidate occurrences, validate chronological order, then inspect the validated occurrence.
- Counting questions: require explicit coverage checks across the relevant interval or aligned frame bundle before concluding.
- Reading / numerical questions: retrieve the right frame, localize the right region when needed, and prefer direct OCR or structure reading over vague multimodal re-asking.
- Listening / speaker / goal questions: prefer whole-video or multi-window ASR with speaker attribution; add visual grounding only when identity matters.
- Charts / tables / diagrams: localize the correct stable frame state, then read the missing labels, values, relationships, or structure directly.

Output quality bar:
- Keep the plan short but sufficient.
- Prefer the fewest steps that can actually resolve the diagnosed gap.
- `refinement_instructions` must tell the trace-writing agent exactly what to keep, what to replace, and which new evidence should control the update.
"""


def build_planner_prompt(
    task,
    mode: str,
    summary_text: str,
    compact_rounds: List[dict],
    retrieved_observations: List[dict],
    audit_feedback: Optional[dict],
    tool_catalog: Dict[str, Dict[str, object]],
) -> str:
    parts = [
        "MODE: %s" % mode,
        "",
        "QUESTION:",
        task.question,
        "",
        "OPTIONS:",
        pretty_json(task.options),
        "",
        render_tool_catalog(tool_catalog),
        "",
        "PLANNING REQUIREMENTS:",
        "- Identify the answer-critical fields before naming tools.",
        "- Decide whether the missing evidence is temporal, visual, audio, textual, numerical, comparative, or attribution-based.",
        "- Reuse retrieved evidence when it already anchors the right occurrence.",
        "- Re-localize when prior evidence points to the wrong occurrence, wrong phase, wrong entity, wrong metric, or wrong label.",
        "- Prefer the smallest set of dependent tool calls that can resolve the gap.",
        "- If the answer could depend on multiple candidate occurrences, preserve those candidates until later evidence rules them out.",
        "",
    ]
    if summary_text:
        parts.extend(
            [
                "WHOLE_VIDEO_SUMMARY:",
                summary_text,
                "",
                "SUMMARY_USAGE_NOTE:",
                "Use this only as planning context. It is not fine-grained final evidence.",
                "",
            ]
        )
    if compact_rounds:
        parts.extend(["COMPACT_PRIOR_ROUNDS:", pretty_json(compact_rounds), ""])
    if retrieved_observations:
        parts.extend(
            [
                "RETRIEVED_ATOMIC_OBSERVATIONS:",
                pretty_json(retrieved_observations),
                "",
                "EVIDENCE_DB_USAGE_NOTE:",
                "Prefer repairing the trace from these observations before asking for broader new evidence.",
                "",
            ]
        )
    if audit_feedback:
        parts.extend(["AUDIT_FEEDBACK:", pretty_json(audit_feedback), ""])
    parts.extend(
        [
            "ExecutionPlan schema reminder:",
            "- strategy: short text",
            "- use_summary: boolean",
            "- steps: list of {step_id, tool_name, purpose, arguments, input_refs, depends_on}",
            "- refinement_instructions: precise guidance for the trace-writing agent",
            "- step_id values must be integers",
            "- input_refs use {target_field, source: {step_id, field_path}}",
            "- depends_on values must refer to earlier integer step ids",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
