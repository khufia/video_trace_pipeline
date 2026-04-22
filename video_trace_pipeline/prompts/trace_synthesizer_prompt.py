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
- In `refine` mode preserve supported prior reasoning and make surgical updates only where new evidence or refinement instructions require them.

━━━━━━━━ Core Principle ━━━━━━━━
SURGICAL EDITS, NOT REWRITES.
Preserve everything in the current trace package that is still supported.
Only modify the specific claims, comparisons, timestamps, or answer choices
that are unsupported, contradicted, or incomplete under the new evidence.

Never invent observations, timestamps, quotes, counts, labels, speakers,
entities, computations, or conclusions that are absent from the evidence.

━━━━━━━━ Evidence-Carrying Package Requirement ━━━━━━━━
This pipeline splits evidence from reasoning:
- `evidence_entries` carry provenance-rich grounded evidence
- `inference_steps` carry reader-facing reasoning

Use that split carefully:
- Put tool provenance, timestamps, frame ids, OCR text, transcript snippets,
  regions, uncertainty, and confidence in `evidence_entries`.
- Keep `inference_steps` tool-free. They should not mention tool names, APIs,
  prompts, hidden execution history, or pipeline bookkeeping.
- Every answer-critical inference step must cite `supporting_observation_ids`.
- The combination of `evidence_entries` plus `inference_steps` should be enough
  for an auditor to see both what was grounded and how the answer was derived.

Evidence-entry rules:
- Each evidence entry should be compact, factual, and directly useful for later auditing.
- Reuse still-valid evidence instead of rewriting everything from scratch.
- Prefer concrete anchors when available:
  - tool name
  - timestamp or time range
  - frame identifier
  - bbox or region
  - quoted OCR or ASR text
  - reported numeric value, label, or confidence
- Preserve uncertainty from the source evidence. If a tool output says
  "approximately", "about", or "estimated", do not turn it into an exact fact.
- If multiple evidence entries describe the same stable state, keep the
  mutually consistent primitives rather than duplicating near-identical entries.

━━━━━━━━ Reader-Facing Inference Requirement ━━━━━━━━
The `inference_steps` must read like a compact argument for a reader, not like
an execution log.

Prefer this shape whenever the question allows it:
1. answer-critical anchor or decisive grounded premise
2. comparison, arithmetic, ordering, or disambiguation
3. final conclusion

Style constraints for `inference_steps`:
- Prefer one main claim per step.
- Keep each step short and answer-facing.
- Include only answer-critical evidence consequences, comparisons, or arithmetic.
- Omit filler, repeated setup, broad search history, and planner/executor narration.
- Start with the decisive grounded premise, not with generic setup.
- Group multi-part questions by subgoal rather than by tool chronology.
- Mention negative evidence only when it directly rules out an option, resolves
  an ambiguity, or explains why the answer must remain unresolved.

━━━━━━━━ Self-Contained Package Requirement ━━━━━━━━
The final `TracePackage` must stand on its own.

Therefore:
- `inference_steps` must NOT refer to "the original trace", "the previous
  answer", "the repair", "the planner", "the auditor", "the refiner", tool
  step numbers, or iteration history.
- `evidence_entries` may mention provenance, but should still avoid pipeline
  bookkeeping language such as "planner step 3" or "repair attempt".
- Restate actual grounded evidence, not pipeline history.

Good inference-step style:
- "The earliest validated clip shows the sign before the bus arrives."
- "The read values are 24 and 18, so the difference is 6."
- "Only option B matches the grounded comparison."

Bad inference-step style:
- "The previous trace claimed ..."
- "Tool output step 2 says ..."
- "The repair confirms ..."

━━━━━━━━ Refinement Discipline ━━━━━━━━
When refining:
- Keep valid prior reasoning and replace only the unsupported or contradicted sub-claims.
- Do not silently erase earlier supported facts just because a later result is
  broader, partial, or non-confirming.
- Update beliefs cumulatively: preserve what remains supported, patch what
  changed, and leave unresolved what is still missing.
- If a follow-up resolves only one missing sub-detail, keep the already
  grounded parts of the earlier reasoning and patch only the unresolved portion.
- Only replace an earlier supported claim when later evidence directly
  contradicts it, grounds a more precise correction, or explains that the
  earlier claim came from the wrong frame, span, entity, or phase.

━━━━━━━━ Reasoning Discipline ━━━━━━━━
- Ordinal or temporal questions: make the chronological validation explicit
  before the final characterization step.
- Counting questions: state the coverage basis before the count.
- Reading or numerical questions: keep direct read values separate from later
  arithmetic or option mapping.
- Listening or speaker questions: distinguish what was said from who said it.
- Multiple-choice questions: make the decisive comparison explicit before the final answer.
- Structured visuals such as charts, tables, dashboards, diagrams, or geometry
  figures: separate grounded premises from later derivation.
- Do not upgrade candidate evidence into verified global chronology unless the
  evidence actually establishes order.
- Do not map a computed value to a different option unless the inference steps
  explicitly justify that mapping.

━━━━━━━━ Uncertainty And Unresolved Cases ━━━━━━━━
- Preserve uncertainty when the evidence is partial, approximate, or ambiguous.
- If multiple answer choices remain compatible, or an answer-critical premise
  is still unresolved, leave `final_answer` empty.
- Do not force a "closest option" answer unless the evidence actually justifies that approximation.
- If different candidate occurrences support different answers and the conflict
  is not resolved, keep the answer unresolved.
- If evidence coverage is partial, reflect that limitation in the package rather
  than writing as if the whole interval or whole video was exhaustively checked.

━━━━━━━━ Consistency Requirements ━━━━━━━━
- The final answer must agree with the inference steps.
- The inference steps must be supported by cited observations.
- The evidence entries must not smuggle in unsupported conclusions that the
  inference steps never justify.
- Prefer the strongest supported granularity. If one evidence entry establishes
  a specific fact and another is broader but less specific, preserve the
  specific fact unless the broader one directly contradicts it.

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
        "- Keep provenance, timestamps, frame ids, OCR text, transcript snippets, regions, and confidence in `evidence_entries`.",
        "- Keep `inference_steps` tool-free, self-contained, and answer-facing.",
        "- Use `supporting_observation_ids` to bind every inference step to evidence.",
        "- Preserve valid prior reasoning and patch only what the new evidence changes.",
        "- Leave `final_answer` empty if the evidence still does not identify one supported answer.",
        "- Make the final answer consistent with the inference steps.",
        "- Do not mention planner, auditor, repair history, or tool step numbers in `inference_steps`.",
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
        parts.extend(
            [
                "CURRENT_TRACE_PACKAGE:",
                pretty_json(current_trace),
                "",
                "CURRENT_TRACE_USAGE_NOTE:",
                "Preserve any evidence entries and inference steps that remain supported. Replace only the unsupported, contradicted, or incomplete parts.",
                "",
            ]
        )
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
