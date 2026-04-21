from __future__ import annotations

import json
from typing import Iterable, List, Optional


PLANNER_SYSTEM_PROMPT = """You are the Planner in a video trace pipeline.

Your job is to decide the next tool calls using only text:
- the question and answer options
- compact summaries of prior rounds
- the evidence ledger and atomic observations
- the dense-caption whole-video summary when included
- the latest audit feedback

Rules:
- You never see raw images, raw clips, or raw audio.
- Use the whole-video summary by default in generation rounds.
- In refinement rounds, prefer the evidence ledger; only depend on the summary when evidence is missing, contradictory, or too weak.
- Make queries specific and question-directed, never vague.
- Use dependency-aware plans.
- `frame_retriever` requires a clip input and must never search the whole video.
- `spatial_grounder` requires a frame input.
- Ordinal and temporal questions: enumerate candidates, then plan a chronological sweep.
- Counting questions: plan explicit coverage checks before concluding.
- Reading and numerical questions: prefer frame retrieval, ROI localization, then OCR.
- Listening and speaker questions: prefer whole-video or multi-window ASR with speaker attribution.

Return JSON only matching the ExecutionPlan schema.
"""


SYNTHESIZER_SYSTEM_PROMPT = """You are the TraceSynthesizer in a video trace pipeline.

You generate or revise a canonical trace package with:
- evidence_entries: tool-facing, provenance-rich summaries
- inference_steps: answer-facing reasoning steps
- final_answer

Rules:
- Inference steps must be tool-free and understandable on their own.
- Evidence entries may mention tools, timestamps, and artifact provenance.
- Every inference step should be supported by the observations listed in supporting_observation_ids.
- Preserve still-valid evidence and reasoning when refining.
- Do not invent observations not present in the evidence packets.

Return JSON only matching the TracePackage schema.
"""


AUDITOR_SYSTEM_PROMPT = """You are the TraceAuditor in a video trace pipeline.

You only audit the textual trace package and evidence summaries you are given.
You do not execute tools.

Rules:
- Focus on unsupported claims, missing coverage, contradictions, and answer mismatches.
- Count repeated-event questions as incomplete unless coverage is explicit.
- Flag numerical answers that skip OCR / direct textual evidence when the question requires reading.
- Flag listening claims that lack transcript or speaker grounding.
- When the trace is sufficient, return PASS.

Return JSON only matching the AuditReport schema.
"""


ATOMICIZER_SYSTEM_PROMPT = """You are an evidence atomicizer.

Split the source text into atomic observations.

Rules:
- One attribute or fact per line.
- Split conjunctions.
- Keep timestamps and speaker identity when present.
- Do not infer beyond the source.

Return JSON: {"facts": [{"subject": "...", "subject_type": "...", "predicate": "...", "object_text": "...", "object_type": "...", "atomic_text": "..."}]}
"""


def render_tool_catalog(tool_names: Iterable[str]) -> str:
    lines = ["Available tools:"]
    for name in sorted(tool_names):
        lines.append("- %s" % name)
    return "\n".join(lines)


def build_planner_prompt(
    task,
    mode: str,
    summary_text: str,
    compact_rounds: List[dict],
    retrieved_observations: List[dict],
    audit_feedback: Optional[dict],
    tool_names: Iterable[str],
) -> str:
    parts = [
        "MODE: %s" % mode,
        "QUESTION:",
        task.question,
        "",
        "OPTIONS:",
        json.dumps(task.options, ensure_ascii=False, indent=2),
        "",
        render_tool_catalog(tool_names),
        "",
    ]
    if summary_text:
        parts.extend(["WHOLE_VIDEO_SUMMARY:", summary_text, ""])
    if compact_rounds:
        parts.extend(
            [
                "COMPACT_PRIOR_ROUNDS:",
                json.dumps(compact_rounds, ensure_ascii=False, indent=2),
                "",
            ]
        )
    if retrieved_observations:
        parts.extend(
            [
                "RETRIEVED_ATOMIC_OBSERVATIONS:",
                json.dumps(retrieved_observations, ensure_ascii=False, indent=2),
                "",
            ]
        )
    if audit_feedback:
        parts.extend(
            [
                "AUDIT_FEEDBACK:",
                json.dumps(audit_feedback, ensure_ascii=False, indent=2),
                "",
            ]
        )
    parts.extend(
        [
            "ExecutionPlan schema reminder:",
            "- strategy: string",
            "- use_summary: boolean",
            "- steps: list of {step_id, tool_name, purpose, arguments, input_refs, depends_on}",
            "- step_id must be an integer like 1, 2, 3",
            "- input_refs target fields with {target_field, source: {step_id, field_path}}",
            "- source.step_id and depends_on values must be integers, not strings like step_1",
            "- Use typed dependencies instead of placeholder strings.",
        ]
    )
    return "\n".join(parts).strip()


def build_synthesizer_prompt(
    task,
    mode: str,
    evidence_entries: List[dict],
    observations: List[dict],
    current_trace: Optional[dict],
    refinement_instructions: str,
) -> str:
    parts = [
        "TASK_KEY: %s" % task.sample_key,
        "MODE: %s" % mode,
        "QUESTION:",
        task.question,
        "",
        "OPTIONS:",
        json.dumps(task.options, ensure_ascii=False, indent=2),
        "",
        "EVIDENCE_ENTRIES:",
        json.dumps(evidence_entries, ensure_ascii=False, indent=2),
        "",
        "ATOMIC_OBSERVATIONS:",
        json.dumps(observations, ensure_ascii=False, indent=2),
        "",
    ]
    if current_trace:
        parts.extend(["CURRENT_TRACE_PACKAGE:", json.dumps(current_trace, ensure_ascii=False, indent=2), ""])
    if refinement_instructions:
        parts.extend(["REFINEMENT_INSTRUCTIONS:", refinement_instructions, ""])
    parts.extend(
        [
            "TracePackage schema reminder:",
            "- task_key: string",
            "- mode: string",
            "- evidence_entries: list",
            "- inference_steps: list[{step_id, text, supporting_observation_ids, answer_relevance}]",
            "- inference_steps[].step_id must be an integer like 1, 2, 3",
            "- final_answer: string",
            "- benchmark_renderings: object",
        ]
    )
    return "\n".join(parts).strip()


def build_auditor_prompt(task, trace_package: dict, evidence_summary: dict) -> str:
    return "\n".join(
        [
            "QUESTION:",
            task.question,
            "",
            "OPTIONS:",
            json.dumps(task.options, ensure_ascii=False, indent=2),
            "",
            "TRACE_PACKAGE:",
            json.dumps(trace_package, ensure_ascii=False, indent=2),
            "",
            "EVIDENCE_SUMMARY:",
            json.dumps(evidence_summary, ensure_ascii=False, indent=2),
            "",
            "AuditReport schema reminder:",
            "- verdict: PASS or FAIL",
            "- confidence: float",
            "- scores: map",
            "- findings: list[{severity, category, message, evidence_ids}]",
            "- feedback: string",
            "- missing_information: list[string]",
        ]
    ).strip()
