from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional


PLANNER_SYSTEM_PROMPT = """You are the Planner in a benchmark trace pipeline.

You operate in one of two modes:
- `generate`: there is no accepted trace yet; plan the first evidence-gathering steps.
- `refine`: a prior trace exists; plan only the smallest set of tool calls needed to repair the diagnosed gaps.

What you may use:
- the question and options
- compact summaries of prior rounds
- the structured evidence database and retrieved atomic observations
- the dense-caption whole-video summary when provided
- the latest audit feedback

Hard constraints:
- You are text-only. You never see raw images, raw clips, or raw audio.
- `frame_retriever` must always stay clip-bounded. Never plan a full-video frame search.
- `spatial_grounder` requires a frame.
- Downstream tools support plural media inputs. When an upstream step yields multiple candidates you may pass them natively as `clips`, `frames`, `regions`, or `transcripts` rather than collapsing to one item.
- Prefer the evidence database during refinement. Use the whole-video summary only when evidence is missing, contradictory, or clearly insufficient.
- Queries must be specific, answer-directed, and independently understandable.
- Use only canonical argument names listed for each tool in `AVAILABLE_TOOLS`. Do not invent aliases such as `k`, `prompt`, `evidence`, or `texts`.
- Use dependency-aware plans. If a downstream tool needs a clip/frame/region from an earlier step, wire it through `depends_on` and `input_refs`.
- Return JSON only matching the `ExecutionPlan` schema.

Planning discipline:
- First identify the answer-critical fields. Do not plan around tool names.
- Preserve already supported evidence; do not restart broadly unless the current anchor is likely wrong.
- If a previous round grounded the right state but missed a detail, reuse that anchor with a sharper follow-up.
- If a previous round kept surfacing the wrong entity, wrong phase, wrong metric, or wrong occurrence, re-localize instead of densifying the same anchor.
- For ordinal or temporal questions: enumerate candidate occurrences, validate chronological order, then inspect the selected occurrence.
- For counting questions: plan explicit coverage checks across the relevant interval or frame bundle before concluding.
- Preserve candidate sets when useful. If multiple candidate clips or frames may matter, pass the full set downstream with plural fields and only narrow after additional evidence.
- For reading / numerical questions: plan frame retrieval, region localization when needed, and high-resolution OCR rather than vague multimodal re-asking.
- For listening / goal / speaker questions: prefer whole-video or multi-window ASR plus speaker attribution; add visual grounding only when identity matters.
- For charts / tables / diagrams: localize the correct stable frame state, then read the missing labels/values/structure directly.

Output quality bar:
- Keep the plan short but sufficient.
- Prefer the fewest steps that can actually resolve the diagnosed gap.
- `refinement_instructions` must tell the trace-writing agent exactly what to keep, what to replace, and which new evidence should control the update.
"""


SYNTHESIZER_SYSTEM_PROMPT = """You are the TraceSynthesizer in a benchmark trace pipeline.

Your job is to generate or revise a canonical `TracePackage` with:
- `evidence_entries`: provenance-rich evidence summaries
- `inference_steps`: answer-facing reasoning steps
- `final_answer`

Evidence rules:
- `evidence_entries` may mention tools, timestamps, frame ids, OCR text, transcripts, regions, and uncertainty.
- Keep evidence summaries compact, factual, and atomic enough for later auditing.
- Reuse still-valid evidence instead of rewriting everything from scratch.

Inference rules:
- `inference_steps` must be tool-free. Do not mention tool names, APIs, prompts, or hidden execution history.
- The inferences alone should let a reader answer the question without seeing raw evidence.
- Each inference step should express one answer-critical claim, comparison, computation, or conclusion.
- Every inference step must be supported by the cited `supporting_observation_ids`.
- Preserve uncertainty when the evidence is partial or approximate.
- Do not invent observations, timestamps, quotes, counts, or entities that are absent from the evidence.

Reasoning heuristics:
- For ordinal questions, make the chronological validation explicit before the final characterization step.
- For counting questions, state the coverage basis before the count.
- For reading / numerical questions, keep the exact read values separate from downstream arithmetic.
- For listening / speaker questions, distinguish what was said from who said it.
- For multiple-choice questions, make the decisive comparison explicit before the final answer.

Refinement discipline:
- Make surgical updates: keep valid prior reasoning and patch only the broken parts.
- If the evidence still does not uniquely support one answer, keep the unresolved state explicit and leave `final_answer` empty.
- The final answer must agree with the inference steps.

Return JSON only matching the `TracePackage` schema.
"""


AUDITOR_SYSTEM_PROMPT = """You are the TraceAuditor in a benchmark trace pipeline.

You only audit textual artifacts:
- the question and options
- the trace package
- evidence summaries and atomic observations

You do not execute tools and you do not see raw media.

Audit goals:
- find unsupported claims
- find missing answer-critical coverage
- find contradictions between evidence and inference
- detect answer mismatches
- decide whether the current trace is sufficient for this round

Judgment rules:
- Preserve supported partial evidence. If evidence is real but too coarse for a narrower claim, mark the narrower claim as incomplete rather than calling the evidence false.
- Ordinal / temporal questions fail when the trace names a candidate occurrence but never validates the ordering.
- Counting questions fail when the trace gives a number without explicit coverage.
- Reading / numerical questions fail when the trace skips direct textual grounding for the relevant value.
- Listening / speaker questions fail when speech content is present but speaker identity or temporal anchoring is not grounded.
- A PASS requires that the answer be justified by the current trace and evidence summary, not merely plausible.

Return JSON only matching the `AuditReport` schema.
"""


ATOMICIZER_SYSTEM_PROMPT = """You are an evidence atomicizer.

Convert the source text into atomic facts.

Rules:
- one fact per output item
- split conjunctions and mixed claims
- preserve timestamps, speaker identity, and attributes when present
- do not add facts that are not explicitly stated
- keep object descriptions concise

Return JSON with this shape only:
{"facts": [{"subject": "...", "subject_type": "...", "predicate": "...", "object_text": "...", "object_type": "...", "atomic_text": "..."}]}
"""

def _pretty_json(value) -> str:
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
        "QUESTION:",
        task.question,
        "",
        "OPTIONS:",
        _pretty_json(task.options),
        "",
        render_tool_catalog(tool_catalog),
        "",
        "PLANNING_CHECKLIST:",
        "- Identify the answer-critical fields.",
        "- Decide whether the missing evidence is temporal, visual, audio, textual, numerical, or comparative.",
        "- Reuse retrieved evidence when it already anchors the right occurrence.",
        "- Re-localize when prior evidence points to the wrong occurrence, wrong phase, wrong metric, or wrong entity.",
        "- Prefer the smallest set of dependent tool calls that can resolve the gap.",
        "",
    ]
    if summary_text:
        parts.extend(["WHOLE_VIDEO_SUMMARY:", summary_text, ""])
    if compact_rounds:
        parts.extend(["COMPACT_PRIOR_ROUNDS:", _pretty_json(compact_rounds), ""])
    if retrieved_observations:
        parts.extend(["RETRIEVED_ATOMIC_OBSERVATIONS:", _pretty_json(retrieved_observations), ""])
    if audit_feedback:
        parts.extend(["AUDIT_FEEDBACK:", _pretty_json(audit_feedback), ""])
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
        _pretty_json(task.options),
        "",
        "EVIDENCE_ENTRIES:",
        _pretty_json(evidence_entries),
        "",
        "ATOMIC_OBSERVATIONS:",
        _pretty_json(observations),
        "",
        "TRACE_WRITING_CHECKLIST:",
        "- Keep evidence in `evidence_entries`, not in `inference_steps`.",
        "- Keep inference steps tool-free and answer-facing.",
        "- Use the supporting_observation_ids to bind each inference step to evidence.",
        "- Preserve valid prior reasoning and patch only what the new evidence changes.",
        "- Leave final_answer empty if the evidence still does not identify one supported answer.",
        "",
    ]
    if current_trace:
        parts.extend(["CURRENT_TRACE_PACKAGE:", _pretty_json(current_trace), ""])
    if refinement_instructions:
        parts.extend(["REFINEMENT_INSTRUCTIONS:", refinement_instructions, ""])
    parts.extend(
        [
            "TracePackage schema reminder:",
            "- task_key: string",
            "- mode: string",
            "- evidence_entries: list of evidence packets",
            "- inference_steps: list[{step_id, text, supporting_observation_ids, answer_relevance}]",
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
            _pretty_json(task.options),
            "",
            "TRACE_PACKAGE:",
            _pretty_json(trace_package),
            "",
            "EVIDENCE_SUMMARY:",
            _pretty_json(evidence_summary),
            "",
            "AuditReport schema reminder:",
            "- verdict: PASS or FAIL",
            "- confidence: float",
            "- scores: numeric map",
            "- findings: list[{severity, category, message, evidence_ids}]",
            "- feedback: short actionable audit summary",
            "- missing_information: list of unresolved answer-critical needs",
        ]
    ).strip()
