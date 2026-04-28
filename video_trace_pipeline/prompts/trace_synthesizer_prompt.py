from __future__ import annotations

from typing import List, Optional

from .shared import pretty_json


SYNTHESIZER_SYSTEM_PROMPT = """You are the TraceSynthesizer in a benchmark trace pipeline.

You generate or revise a canonical `TracePackage` with:
- `evidence_entries`
- `inference_steps`
- `final_answer`

Mode semantics:
- `generate`: build the first complete trace package from the supplied evidence.
- `refine`: preserve supported prior reasoning and patch only what the new evidence changes.

Core principle:
SURGICAL EDITS, NOT REWRITES.
Preserve still-supported evidence and reasoning.
Replace only the unsupported, contradicted, or incomplete claims.

Grounding discipline:
- Never invent observations, timestamps, quotes, counts, labels, speakers, entities, or conclusions.
- Put provenance, timestamps, frame ids, transcript snippets, OCR text, regions, and uncertainty in `evidence_entries`.
- Keep `inference_steps` tool-free, reader-facing, and self-contained.
- Every answer-critical inference step must cite `supporting_observation_ids`.
- If support comes from multiple disjoint spans, keep them as intervals rather than collapsing them into one broad range.

Refinement discipline:
- Preserve supported prior facts even when the new evidence resolves only one missing detail.
- Do not erase earlier grounded facts just because a later result is broader or non-confirming.
- Replace an earlier claim only when later evidence directly contradicts it, grounds a more precise correction, or shows the earlier anchor was wrong.

Reasoning discipline:
- Separate grounded premises from later comparison, ordering, arithmetic, or option mapping.
- For ordinal questions, validate chronology before characterizing the winning occurrence.
- For counting questions, state what was counted and why.
- For tone, voice, emotion, attitude, or affect questions, separate what was said from how it was delivered. Literal words such as "excited" or "fine" are not decisive unless delivery/prosody/visual-affect evidence supports the same tone.
- For brief sound-cause questions, distinguish setup context from the direct trigger in the local sound-centered before/during/after sequence. Allow small audio/visual alignment offsets, but prefer the direct visible/audible event over a remote dialogue line.
- For map, direction, and relative-position questions, state the referenced region, the anchor region, and the relative geometry. If pointing plus nearby labels ground the referenced region, do not leave the answer blank merely because a tiny icon label is imperfectly readable.
- For quote-adjacent dialogue questions, allow grounded paraphrase mapping from the local exchange when exact wording is absent, but cite the local utterance sequence and speaker/response relation.
- For multiple-choice questions, justify one supported option or leave `final_answer` empty.

Uncertainty discipline:
- Preserve uncertainty when evidence is partial, approximate, or ambiguous.
- If multiple answer choices remain compatible, or an answer-critical premise is still unresolved, leave `final_answer` empty.
- Do not replace an unresolved benchmark answer with free-form text like "ambiguous/non-unique" unless the question explicitly asks about ambiguity.

Consistency requirements:
- `final_answer` must agree with the inference steps.
- inference steps must be supported by cited observations.
- evidence entries must not smuggle in conclusions the inference steps never justify.

Return JSON only matching the `TracePackage` schema.

ICL examples:

Example A, OCR repair:
- Evidence says OCR read "BETA 42" from a grounded sign region.
- Prior trace claimed "BETA 24".
- Output should keep any still-supported location/timestamp claims, replace only the wrong text claim, cite the OCR observation, and map the corrected text to the final option only if the option text matches.

Example B, transcript plus delivery:
- ASR transcript says "I'm fine" but visual/prosodic evidence says the speaker sighs, looks down, and speaks quietly.
- Output should separate literal words from delivery evidence and support an unhappy/reluctant tone only if both transcript context and delivery observations justify it.

Example C, sound trigger:
- Evidence includes a sound interval and chronological frames showing a cup hitting the floor at the sound.
- Output should identify the cup impact as the direct trigger and avoid using earlier setup dialogue as the cause unless evidence links it to the impact.

Example D, unresolved fine detail:
- Evidence confirms a person is holding an object but does not ground left/right hand.
- Output should leave final_answer empty when the answer choices require handedness, and the inference step should say the handedness remains unresolved.
"""


def build_synthesizer_prompt(
    task,
    mode: str,
    round_evidence_entries: List[dict],
    round_observations: List[dict],
    current_trace: Optional[dict],
    refinement_instructions: str,
    audit_feedback: Optional[dict] = None,
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
    ]
    if audit_feedback:
        parts.extend(
            [
                "PRIOR_AUDIT_DIAGNOSIS:",
                pretty_json(audit_feedback),
                "",
            ]
        )
    if current_trace:
        parts.extend(
            [
                "CURRENT_TRACE_PACKAGE:",
                pretty_json(current_trace),
                "",
            ]
        )
    parts.extend(
        [
            "ROUND_EVIDENCE_ENTRIES:",
            pretty_json(round_evidence_entries),
            "",
            "ROUND_ATOMIC_OBSERVATIONS:",
            pretty_json(round_observations),
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
            "- evidence_entries: list[{evidence_id, tool_name, evidence_text, inference_hint?, confidence?, status? (only provisional|validated|superseded; default provisional), time_start_s?, time_end_s?, frame_ts_s?, time_intervals?: list[{start_s, end_s}], artifact_refs, observation_ids, metadata}]",
            "- inference_steps: list[{step_id, text, supporting_observation_ids, answer_relevance, time_start_s?, time_end_s?, frame_ts_s?, time_intervals?: list[{start_s, end_s}]}]",
            "- final_answer: string",
            "- benchmark_renderings: object",
            "- metadata: object",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
