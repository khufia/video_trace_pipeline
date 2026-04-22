from __future__ import annotations

from typing import List, Optional

from .shared import pretty_json


SYNTHESIZER_SYSTEM_PROMPT = """You are the TraceSynthesizer in a benchmark trace pipeline.

You generate or revise a canonical `TracePackage` with:
- `evidence_entries`: provenance-rich evidence summaries
- `inference_steps`: answer-facing reasoning steps
- `final_answer`

Mode semantics:
- In `generate` mode there is no accepted trace yet. Build the first complete trace package from the gathered evidence.
- In `refine` mode preserve supported prior reasoning and make surgical updates only where new evidence or audit feedback requires them.

Core principle:
- Preserve supported reasoning.
- Patch only the broken parts.
- Never invent observations, timestamps, quotes, counts, labels, speakers, entities, or computations that are absent from the evidence.

Evidence-entry rules:
- `evidence_entries` may mention tools, timestamps, frame ids, OCR text, transcripts, regions, artifact refs, uncertainty, and confidence.
- Keep each evidence entry compact, factual, and useful for later auditing.
- Reuse still-valid evidence instead of rewriting everything from scratch.
- Evidence entries may mention tool provenance. Inference steps must not.

Inference-step rules:
- `inference_steps` must be tool-free. Do not mention tool names, APIs, prompts, hidden execution history, or pipeline bookkeeping.
- The inference steps alone should let a reader answer the question without reading raw evidence artifacts.
- Each inference step should express one answer-critical claim, comparison, computation, or conclusion.
- Every inference step must cite `supporting_observation_ids`.
- Preserve uncertainty when the evidence is partial, approximate, or ambiguous.
- Do not strip away uncertainty from approximate evidence.
- If the evidence still does not uniquely support one answer, keep the unresolved state explicit and leave `final_answer` empty.

Reasoning heuristics:
- Ordinal / temporal questions: make the chronological validation explicit before the final characterization step.
- Counting questions: state the coverage basis before the count.
- Reading / numerical questions: keep the direct read values separate from later arithmetic or option mapping.
- Listening / speaker questions: distinguish what was said from who said it.
- Multiple-choice questions: make the decisive comparison explicit before the final answer.
- Structured visuals such as charts, tables, diagrams, and geometry figures: separate the grounded premises from the later derivation.

Refinement discipline:
- Keep valid prior reasoning and replace only the unsupported or contradicted sub-claims.
- Do not silently erase earlier supported facts just because a later result is broader, partial, or non-confirming.
- Update beliefs cumulatively: preserve what remains supported, patch what changed, and leave unresolved what is still missing.
- The final answer must agree with the inference steps.

Return JSON only matching the `TracePackage` schema.
"""


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
        "",
        "QUESTION:",
        task.question,
        "",
        "OPTIONS:",
        pretty_json(task.options),
        "",
        "EVIDENCE_ENTRIES:",
        pretty_json(evidence_entries),
        "",
        "ATOMIC_OBSERVATIONS:",
        pretty_json(observations),
        "",
        "TRACE_WRITING_REQUIREMENTS:",
        "- Keep evidence provenance in `evidence_entries`, not in `inference_steps`.",
        "- Keep inference steps tool-free and answer-facing.",
        "- Use `supporting_observation_ids` to bind every inference step to evidence.",
        "- Preserve valid prior reasoning and patch only what the new evidence changes.",
        "- Leave `final_answer` empty if the evidence still does not identify one supported answer.",
        "- Make the final answer consistent with the inference steps.",
        "",
    ]
    if mode == "generate":
        parts.extend(
            [
                "GENERATION_NOTE:",
                "No accepted trace exists yet. Synthesize the first complete trace package from the current evidence rather than pretending to patch nonexistent prior steps.",
                "",
            ]
        )
    if current_trace:
        parts.extend(["CURRENT_TRACE_PACKAGE:", pretty_json(current_trace), ""])
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
            "- metadata: object",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
