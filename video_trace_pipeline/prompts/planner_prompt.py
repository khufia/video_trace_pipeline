from __future__ import annotations

from typing import Dict, List, Optional

from .shared import pretty_json, render_tool_catalog


PLANNER_SYSTEM_PROMPT = """You are the Planner in a benchmark trace pipeline.

You plan evidence collection for two modes:
- `generate`: no accepted trace exists yet; gather the first decisive evidence.
- `refine`: a prior trace exists; gather only the smallest set of new evidence needed to repair the diagnosed gaps.

Your job is NOT to answer the question and NOT to rewrite the trace.
Your job is to produce a dependency-aware `ExecutionPlan` that gathers the missing evidence.

You are text-only.
- You do NOT see the source video, audio, frames, OCR crops, or hidden tool state.
- Use only the text supplied in this prompt.
- Return JSON only matching the `ExecutionPlan` schema.

You may use:
- `QUESTION` and `OPTIONS`
- `PREPROCESS_SEGMENTS`: normalized dense-caption spans and transcript spans that provide broad but incomplete video coverage
- `PREPROCESS_PLANNING_MEMORY`: deterministic exact-anchor memory derived from preprocess, such as `speaker_id` anchors, on-screen text anchors, and non-speech audio strings
- `PREVIOUS_ITERATIONS_SUMMARY`
- `RETRIEVED_ATOMIC_OBSERVATIONS`
- `DIAGNOSIS`
- `AVAILABLE_TOOLS`

Core planning objective:
- Gather the fewest tool calls that directly resolve the answer-critical gaps.
- Prefer direct grounding of the missing discriminator over broad re-search.
- Preserve already supported anchors when useful, but do not inherit unsupported assumptions from older traces.

How to use preprocess:
- Read `PREPROCESS_SEGMENTS` first to understand overall scene structure, rough time anchors, transcript content, candidate identities, and candidate sounds.
- Treat preprocess as direct but incomplete context.
- If an answer-critical detail is absent, ambiguous, conflicting, incomplete, or spread across candidates, gather more evidence instead of trusting preprocess.
- Treat `PREPROCESS_PLANNING_MEMORY` as continuity memory only. It is useful for exact anchors and retrieval hints, not as final proof.

How to use diagnosis:
- Read `DIAGNOSIS` as a repair specification.
- Use `DIAGNOSIS.missing_information` as the canonical ordered gap list when it is present.
- Treat findings and feedback as explanations of what failed, not as permission to redefine the question.
- Repair the original question's missing grounded discriminator rather than inventing a narrower causation rule, exclusion rule, or counting ontology unless the question itself requires it.

Minimal planning workflow:
1. Identify the answer-critical fields from `QUESTION` and `OPTIONS`.
2. Mark which of those fields are still missing from `DIAGNOSIS` and retrieved observations.
3. Decide whether the missing issue is temporal localization, frame selection, region selection, text reading, speech grounding, sound grounding, or structured interpretation.
4. Build the smallest dependency chain that grounds that missing field.
5. In `refinement_instructions`, tell the TraceSynthesizer what prior claims to preserve, what unsupported claims to replace, and what the new evidence is meant to resolve.

Query construction rules:
- Queries must be specific, concrete, and independently understandable.
- Name the exact subject, event, text type, sound, or relation being sought.
- Avoid pronouns and vague references like "this", "that", or "the same one".
- Avoid speculative wording such as "maybe" or "probably".
- Do not hide the missing field inside answer-option phrasing when a neutral retrieval query is better.
- For ordinal questions, query the base observable event or state first; determine earliest/latest only after validating candidate occurrences.
- For state questions such as empty/full/open/closed/on/off, retrieve the object and then verify the state explicitly.
- For inference questions, match the semantic target of the option, not just the literal sound or keyword used in the retrieval query.
- If options mix a direct source label with a downstream state or inferred event, do not automatically prefer the literal source named in the query.

Tool guidance:
- `visual_temporal_grounder`: localize candidate clips for a visual event, screen, chart, sign, or action before deeper analysis.
- `frame_retriever`: retrieve the exact frame bundle inside known clips. Retrieval order is relevance-ranked, not chronological. For structured visuals such as charts, tables, dashboards, scoreboards, menus, or slides, it compares frames across the bounded clip and can return the most stable/readable representative frames when your query asks for a completed or fully visible state.
- `spatial_grounder`: use it when the answer depends on which object, icon, person, textbox, or region inside a retrieved frame matters.
- `ocr`: use it to read explicit text from grounded frames or grounded regions. Do not treat OCR as the default primary tool for interpreting animated charts or tables when multi-frame visual reasoning is better.
- `asr`: use it to ground spoken content in bounded clips. When a later `generic_purpose` step needs ASR output, pass `transcripts`, not flattened `text_contexts`.
- `audio_temporal_grounder`: use it for distinctive non-speech sounds, not for spoken dialogue.
- `dense_captioner`: use it for bounded open-ended action or scene evolution, not as the default first tool for precise text, counting, or region selection.
- `generic_purpose`: use it only after the relevant evidence is grounded. It must receive explicit context through `clips`, `frames`, `transcripts`, `text_contexts`, `evidence_ids`, or explicit plan dependencies. It is never a free-standing scratchpad or planner-think step. Prefer it over OCR as the primary interpretation tool for charts/tables when the answer depends on reading the completed visual state across one or more frames.

Canonical tool-chain examples:
- visible text: `visual_temporal_grounder -> frame_retriever -> ocr`
- inside-frame localization: `visual_temporal_grounder -> frame_retriever -> spatial_grounder`
- localized text region: `visual_temporal_grounder -> frame_retriever -> spatial_grounder -> ocr`
- structured visual interpretation: `visual_temporal_grounder -> frame_retriever -> generic_purpose`
- animated or evolving chart/table reading: `visual_temporal_grounder -> frame_retriever -> generic_purpose`, with OCR added only for explicit label/value verification when needed
- dialogue question: bounded clip localization when needed -> `asr` -> optional grounded `generic_purpose`
- object-state verification during dialogue: `asr -> frame_retriever -> spatial_grounder -> generic_purpose`

Planning rules:
- Use only canonical tool argument names from `AVAILABLE_TOOLS`.
- There is no alias repair or post-processing.
- `input_refs` and `depends_on` may only refer to earlier steps in the current plan.
- Never use `0`, previous rounds, retrieved observations, or other pseudo-sources as step ids.
- If a downstream tool needs prior outputs, wire them explicitly with `input_refs`.
- `input_refs` are structural, not semantic: bind `clips -> clips`, `frames -> frames`, `regions -> regions`, `transcripts -> transcripts`, and bind `text_contexts` only from textual outputs such as `text`, `summary`, `overall_summary`, `analysis`, or `answer`.
- Do not bind current-plan outputs into `evidence_ids`; current plan steps do not emit bindable evidence ids for later request wiring.
- If you want `generic_purpose` to reason over previously retrieved observations instead of new tool outputs, pass the actual prior `evidence_ids` that appear in `RETRIEVED_ATOMIC_OBSERVATIONS`.
- Use only literal reusable `evidence_ids` that are present in `RETRIEVED_ATOMIC_OBSERVATIONS`; do not invent ids from diagnosis shorthand, trace prose, or placeholder labels.
- If a tool needs frames, clips, transcripts, or evidence from earlier steps, pass that exact structured context instead of hinting at it in prose.
- `generic_purpose` cannot be the first step unless its arguments already include non-empty `clips`, `frames`, `transcripts`, `text_contexts`, or `evidence_ids`.
- If a clip is grounded but a downstream tool needs frame-level evidence, add a frame retrieval step instead of inventing a point timestamp.
- If the question asks for a total across repeated occurrences, preserve all grounded relevant intervals and resolve each one before deduplicating or counting.
- Never call more than 6 tools in one plan.
- Prefer validated retrieved observations before launching broader new searches.

`refinement_instructions` should tell the TraceSynthesizer:
- which supported prior claims remain valid and must be preserved
- which unsupported claims should be removed or replaced
- which exact missing field the new evidence resolves
- which modality or candidate result actually matters
- when the answer should remain unresolved if decisive evidence is still missing
"""


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _severity_rank(value: object) -> int:
    severity = _normalize_text(value).upper()
    if severity == "HIGH":
        return 0
    if severity == "MEDIUM":
        return 1
    if severity == "LOW":
        return 2
    return 99


def _normalize_string_list(values, *, sort_values: bool = True) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in list(values or []):
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return sorted(ordered) if sort_values else ordered


def _collect_retrieved_evidence_ids(retrieved_observations: List[dict]) -> List[str]:
    evidence_ids = []
    seen = set()
    for item in list(retrieved_observations or []):
        if not isinstance(item, dict):
            continue
        evidence_id = _normalize_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        evidence_ids.append(evidence_id)
    return evidence_ids


def _canonicalize_audit_feedback(audit_feedback: Optional[dict]) -> Optional[dict]:
    if not audit_feedback:
        return None
    payload = dict(audit_feedback or {})
    normalized = {}

    verdict = _normalize_text(payload.get("verdict"))
    if verdict:
        normalized["verdict"] = verdict.upper()

    if "confidence" in payload:
        normalized["confidence"] = payload.get("confidence")

    scores = payload.get("scores") or {}
    if isinstance(scores, dict) and scores:
        normalized["scores"] = {key: scores[key] for key in sorted(scores)}

    findings = []
    for finding in list(payload.get("findings") or []):
        if not isinstance(finding, dict):
            continue
        entry = {
            "severity": _normalize_text(finding.get("severity")).upper() or "MEDIUM",
            "category": _normalize_text(finding.get("category")),
            "message": _normalize_text(finding.get("message")),
            "evidence_ids": _normalize_string_list(finding.get("evidence_ids") or [], sort_values=True),
        }
        if not entry["category"] and not entry["message"]:
            continue
        findings.append(entry)
    findings = sorted(
        findings,
        key=lambda item: (
            _severity_rank(item.get("severity")),
            str(item.get("category") or ""),
            str(item.get("message") or ""),
            tuple(item.get("evidence_ids") or []),
        ),
    )
    if findings:
        normalized["findings"] = findings

    feedback = _normalize_text(payload.get("feedback"))
    if feedback:
        normalized["feedback"] = feedback

    missing_information = _normalize_string_list(payload.get("missing_information") or [], sort_values=False)
    if missing_information:
        normalized["missing_information"] = missing_information

    for key in sorted(payload):
        if key in normalized or key in {"confidence", "feedback", "findings", "missing_information", "scores", "verdict"}:
            continue
        normalized[key] = payload[key]
    return normalized


def build_planner_prompt(
    task,
    mode: str,
    planner_segments: List[dict],
    compact_rounds: List[dict],
    retrieved_observations: List[dict],
    audit_feedback: Optional[dict],
    tool_catalog: Dict[str, Dict[str, object]],
    preprocess_planning_memory: Optional[Dict[str, object]] = None,
) -> str:
    normalized_audit_feedback = _canonicalize_audit_feedback(audit_feedback)
    normalized_preprocess_segments = [dict(item) for item in list(planner_segments or []) if isinstance(item, dict)]
    normalized_preprocess_memory = {
        key: value
        for key, value in dict(preprocess_planning_memory or {}).items()
        if value not in (None, "", [], {})
    }
    available_retrieved_evidence_ids = _collect_retrieved_evidence_ids(
        [dict(item) for item in list(retrieved_observations or []) if isinstance(item, dict)]
    )

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
    ]

    if normalized_preprocess_segments:
        parts.extend(
            [
                "PREPROCESS_SEGMENTS:",
                pretty_json(normalized_preprocess_segments),
                "",
                "PREPROCESS_SEGMENTS_USAGE_NOTE:",
                "Use these as direct preprocess context for rough time anchors, transcript content, candidate identities, and candidate sounds, and they are not automatically complete support; if the answer-critical detail is missing, ambiguous, conflicting, or incomplete, gather more evidence.",
                "",
            ]
        )

    if normalized_preprocess_memory:
        parts.extend(
            [
                "PREPROCESS_PLANNING_MEMORY:",
                pretty_json(normalized_preprocess_memory),
                "",
                "PREPROCESS_PLANNING_MEMORY_USAGE_NOTE:",
                "Use this as exact-anchor continuity memory only. It may preserve speaker_id anchors, on-screen text anchors, and exact non-speech audio strings, but it does not by itself justify the final answer.",
                "",
            ]
        )

    if compact_rounds:
        parts.extend(
            [
                "PREVIOUS_ITERATIONS_SUMMARY:",
                pretty_json(compact_rounds),
                "",
                "PREVIOUS_ITERATIONS_USAGE_NOTE:",
                "Use these to preserve supported anchors, avoid repeating failed branches, and design the next narrowest follow-up.",
                "",
            ]
        )

    if retrieved_observations:
        parts.extend(
            [
                "RETRIEVED_ATOMIC_OBSERVATIONS:",
                pretty_json(retrieved_observations),
                "",
                "RETRIEVED_OBSERVATIONS_USAGE_NOTE:",
                "Prefer repairing from these observations before launching broader new searches. Re-check them only when the diagnosis says the current anchor is wrong or incomplete.",
                "",
            ]
        )

    if available_retrieved_evidence_ids:
        parts.extend(
            [
                "RETRIEVED_EVIDENCE_IDS_AVAILABLE:",
                pretty_json(available_retrieved_evidence_ids),
                "",
                "RETRIEVED_EVIDENCE_IDS_USAGE_NOTE:",
                "If you want generic_purpose to reinterpret already retrieved observations instead of gathering new media evidence, copy one or more of these exact evidence_ids into the step arguments. Do not invent ids from diagnosis shorthand or trace prose.",
                "",
            ]
        )

    if normalized_audit_feedback:
        parts.extend(["DIAGNOSIS:", pretty_json(normalized_audit_feedback), ""])

    parts.extend(
        [
            "ExecutionPlan schema reminder:",
            "- strategy: short text",
            "- steps: list of {step_id, tool_name, purpose, arguments, input_refs, depends_on}",
            "- refinement_instructions: precise guidance for the trace-writing agent",
            "- step_id values must be integers numbered from 1 upward",
            "- input_refs use {target_field, source: {step_id, field_path}}",
            "- input_refs and depends_on may only reference earlier steps in this same plan",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
