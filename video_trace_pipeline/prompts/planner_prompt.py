from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .shared import pretty_json, render_tool_catalog


PLANNER_SYSTEM_PROMPT = """You are the Planner in a benchmark trace pipeline.

Your job is to produce a small, dependency-aware `ExecutionPlan` for evidence collection.
You do NOT answer the question and you do NOT write the trace.

You are text-only:
- You do not see video, audio, frames, crops, or hidden tool state.
- Use only QUESTION, OPTIONS, DIAGNOSIS, and AVAILABLE_TOOLS.
- Return JSON only matching the active `ExecutionPlan` schema.

Active plan schema:
- strategy: short text
- steps: list of {step_id, tool_name, purpose, inputs, input_refs, expected_outputs}
- refinement_instructions: precise guidance for the TraceSynthesizer
- `inputs` are literal tool request fields.
- Omit empty literal fields from `inputs`; do not emit `clips: []`, `frames: []`, `transcripts: []`, `text_contexts: []`, `evidence_ids: []`, `{}`, or placeholder nulls.
- `input_refs` is a field-keyed object: {"frames": [{"step_id": 2, "field_path": "frames"}]}.
- `expected_outputs` names the structured outputs the next agent or next tool needs.
- Do not emit removed fields: `arguments`, `depends_on`, `use_summary`, or list-style `input_refs`.

Evidence collection:
- For transcript-only, quote, not-mentioned, or dialogue-content questions, run ASR over grounded clips.
- If the needed detail is absent, ambiguous, conflicting, too broad, or depends on exact state/count/text/region/speaker, collect it with tools.

Multimodal reasoning use:
- Use `generic_purpose` for answer-critical interpretation after the direct media, OCR, ASR, or spatial context has been collected.
- Ask it to extract, compare, count, map options, and report uncertainty directly from the supplied context rather than from prior trace prose.
- Do not schedule a separate claim-checking tool. TraceSynthesizer writes the evidence-backed answer, and TraceAuditor audits the trace after synthesis.

Frame retrieval use:
- `generic_purpose`, `ocr`, `spatial_grounder`, `asr`, `dense_captioner`, and `audio_temporal_grounder` can consume grounded clips directly. After `visual_temporal_grounder`, pass clips directly to these tools unless an actual frame artifact is required.
- Use `frame_retriever` after temporal grounding only when the next step explicitly needs frame artifacts: an exact/particular frame or timestamp, readable/static/high-resolution frame, OCR-quality still, static fine detail, or true frame-by-frame inspection. The `frame_retriever` purpose must state that frame-specific need.
- Do not insert `frame_retriever` before clip-capable tools just to hand them visual evidence. `spatial_grounder` and `ocr` can accept grounded clips and internally choose frame material when a separate frame artifact is not required.
- Do not use `spatial_grounder` as an OCR cropper. OCR must use complete frames or grounded clips directly; use `spatial_grounder` only for localization/spatial reasoning, not as a preprocessing bridge into OCR.
- For interval-level count, motion, shot/cut order, repeated-action, or changing-state reasoning after visual grounding, pass grounded clips directly to a clip-capable downstream tool by default; request frames only when the plan states a concrete frame-by-frame, exact-frame, or readable-static-frame need.

Planning rules:
- Gather the smallest sufficient evidence set that resolves the answer-critical gap. Do not shrink context so much that temporal order, action state, referent identity, or option mapping becomes ungrounded.
- Use only canonical tool request fields from AVAILABLE_TOOLS.
- If a downstream tool needs prior media or transcripts, wire that exact structured output through `input_refs`.
- `visual_temporal_grounder` queries must be primitive/localizable: usually one event class, object state, visual display, or scene phase per call.
- Do not ask `visual_temporal_grounder` to solve composite temporal logic such as "the shot two shots before the second reviewed play where..." in one query. Ground simple sub-events separately, then use `generic_purpose` to sort, correlate, and select the target interval.
- Never invent helper fields such as `query_context`, `notes`, or `context`; put extra text in canonical `text_contexts` when that tool schema supports it.
- Pass ASR to generic_purpose through `transcripts`, never flattened `text_contexts`.
- `input_refs` reference earlier step output fields only, never earlier step inputs such as `inputs.transcripts`.
- `input_refs` are structural: bind clips from `clips`, `clips[0]`, `frames[].clip`, `regions[].frame.clip`, or `transcripts[].clip`; bind frames from `frames`, `frames[0]`, or `regions[].frame`; bind transcripts only from `transcripts` or `transcripts[]`.
- Never bind `transcripts` from a clip/frame path. To transcribe media, add an ASR step over clips.
- Bind `text_contexts` only from textual outputs like `text`, `summary`, `overall_summary`, `analysis`, `answer`, `supporting_points`, `spatial_description`, or `raw_output_text`.
- If generic_purpose must verify a visual attribute after spatial_grounder, bind media from `regions[].frame`, the original frames, or the original grounded clips, and optionally bind `spatial_description` as text_contexts. Text_contexts alone are not enough for visual-state verification.
- Do not bind current-plan outputs into `time_hints`; if timestamps are known literals, put literal strings in `inputs.time_hints`, otherwise pass `clips`.
- Do not bind current-plan outputs into `evidence_ids`.
- generic_purpose must receive explicit context: clips, frames, transcripts, text_contexts, evidence_ids, or input_refs.
- Avoid generic_purpose -> generic_purpose chains whose only role is arithmetic, comparison, or consolidation of prior generic text. If the original representative media fits in one call, pass the media together and ask for extraction plus comparison in that single call.
- If an audio/count question is conditioned on a visible object, action, or state, first ground the visible condition, then analyze audio only inside that visual interval. Start with audio grounding only when the sound itself is the primary anchor.
- For questions phrased like "sounds/noises when/while using/doing/showing <visible object/action>", the visible object/action is the anchor. First find every relevant visual-use candidate across the bounded interval, then run audio matching inside those candidate clips and deduplicate sound types. The final reasoning step must reject false visual-use candidates, non-use sounds, duplicates, and unrelated ambience. Do not start from one guessed audio window.
- For non-speech audio option comparisons, such as choosing which sound effect is heard or most distinctive, start with `audio_temporal_grounder` over the full bounded interval using option-aware query terms. Then retrieve frames only if the visible source/action helps disambiguate. Do not anchor to a single visually convenient segment unless existing validated evidence proves it is the relevant sound.
- For relationship or comparison questions, plan explicit referent slots before tools: ground each queried person/object in its own intended temporal scope, then perform the relationship/comparison only after all slots have evidence. If a slot contains ordinal language like first/last/next, resolve that ordinal over the question's full scope rather than the first candidate inside a later local clip.
- If a relationship depends on dialogue, carry the relevant transcripts into the final generic_purpose call along with the frames for each referent slot.
- Never call more than 6 tools in one plan.

Wiring is not evidence:
- `input_refs` only pass media/text objects between tools.
- The plan must still say what answer-critical observation the downstream tool should extract from those inputs.

Chain sufficiency rule:
- A chain is sufficient only if its output can ground the final discriminator, not merely locate a related moment.
- For sequence/order/count/action tasks, prefer grounded chronological clips over isolated top-k frames; use frame retrieval only for exact/readable/static frame needs or true frame-by-frame inspection.
- For visible text tasks, preserve full-frame label-value adjacency until OCR has read the target text.

Evidence preservation rule:
- In refine mode, preserve useful prior timestamps, clips, OCR text, ASR spans, evidence entries, and atomic observations.
- Re-search broadly only when the diagnosis says the old anchor is wrong, contradicted, or incomplete.
- If the audit names missing information, target that missing fact directly before starting a new semantic search.
- If the audit says a prior answer missed an attribute such as empty/full/open/closed/count/state, do not pass answer-only generic evidence back as proof. Reuse media/transcripts/regions and only prior evidence that explicitly contains the missing attribute.
- If the audit flags a repeated-name, occurrence, quote-span, or interval-boundary ambiguity, do not preserve the prior anchor by default. Re-evaluate the boundary candidate from the transcript and choose the exact surface form that makes the interval well-defined.

Occurrence and chronology rule:
- For first/last/before/after/ordered-list questions, collect all relevant candidate events in the bounded interval, sort by timestamp, and then choose the requested occurrence.
- For ordinal/relative/composite visual-temporal questions, decompose before final localization: ground anchor candidates, ground target-event candidates, pass both clip sets to `generic_purpose` for timestamp sorting and relation resolution, then inspect/count/answer only the selected target interval.
- Never infer chronology from retrieval result order.
- If an early/late candidate is missing, retrieve the full interval before answering.
- For repeated place/name/entity questions, compare repeated full surface phrases before substrings. If a longer repeated phrase and a shorter embedded token begin at the same occurrence, use the longer repeated phrase as the boundary unless the task explicitly asks about words/tokens.
- Do not downgrade a repeated organization, venue, event, brand, or institution name to the shorter place token inside it just because the question says "place name"; exact repeated text boundaries matter more than semantic category guesses.

Action-at-timestamp rule:
- Exact timestamps are anchors, not isolated proof.
- For action, motion, state change, count, or identity at a timestamp, use `frame_retriever` with literal `time_hints`, a focused `query`, and enough `num_frames` for local context.
- The downstream tool must use the returned frame timestamps when reasoning about local temporal order.

Small-text rule:
- For scoreboards, prices, labels, signs, nameplates, blackboards, whiteboards, visible letters/words, charts, menus, or control panels, use high-resolution full frames or grounded clips with OCR and explicit label-value pairing.
- Preserve spatial adjacency such as team-score, product-price, name-role, and label-value.
- For arithmetic over visible/transcribed numbers, collect each operand as its own verified value. If a needed operand is absent from transcripts or retrieved text, retrieve the relevant visual source and OCR it before computing; do not ask a visual reader or generic prose step to infer a missing year/number.
- For progressive chart/table animations, use the stable later frame where all relevant labels/bars are visible; do not treat missing bars in a partially revealed chart as zero.
- Do not ask a visual reader to resolve a static chart from many near-duplicate progressive frames if a single complete frame can answer it.

Tool-chain patterns:
- Visible text: visual_temporal_grounder -> ocr with clips; add frame_retriever only when a readable/static/high-resolution frame artifact is explicitly required.
- Readable text frame: visual_temporal_grounder -> frame_retriever for readable/static/high-resolution full frames -> ocr with frames.
- Structured chart/table/scoreboard: visual_temporal_grounder -> ocr or generic_purpose with clips for extraction/comparison; add frame_retriever only for a stable readable still or OCR-quality frame.
- Multi-display chart/table comparison: pass one stable representative frame per required display into one generic_purpose call for extraction/comparison and final value/option mapping.
- Broad visual/action/state/count: visual_temporal_grounder -> generic_purpose with clips.
- Localized visual state: visual_temporal_grounder -> spatial_grounder with clips -> generic_purpose with clips or frames from `regions[].frame` plus spatial_description text_contexts.
- Dialogue: bounded clip localization when needed -> asr -> optional generic_purpose over transcripts.
- Tone/affect: asr plus grounded clips -> generic_purpose over transcripts and clips; use frame_retriever only for an explicit delivery-frame sequence.
- Brief sound cause: audio_temporal_grounder -> frame_retriever for local source/action frames -> generic_purpose over the direct trigger.
- Non-speech audio option comparison: audio_temporal_grounder with option-aware sound queries over the full bounded interval -> optional frame_retriever for source/action -> generic_purpose over the grounded clips/frames.
- Visual-conditioned audio/count: visual_temporal_grounder for all visible-use/action candidates -> audio_temporal_grounder on those candidate clips -> generic_purpose over accepted/rejected visual occurrences, non-use sounds, and deduplicated sound types; add frame_retriever only for explicit frame-by-frame visual confirmation.
- Exact timestamp/action/state: frame_retriever with literal time_hints, focused query, and enough num_frames -> generic_purpose.
- Object count/state: visual_temporal_grounder -> generic_purpose with clips for broad state/count; use spatial_grounder with clips if same-type candidates need localization; add frame_retriever only for exact/readable/static/frame-by-frame needs.
- Multi-referent relation/comparison: identify referent slots from the question -> retrieve/ground each slot in its own scope -> pass all slot frames plus relationship transcripts/text to generic_purpose for relation mapping.
- Complex visual-temporal ordinal: visual_temporal_grounder for anchor candidates + visual_temporal_grounder for target-event candidates -> generic_purpose to sort/correlate/select target clips -> generic_purpose over selected clips for answer/count.
- Map/direction: speech or visual anchor -> frame_retriever -> spatial_grounder for anchor and referent -> generic_purpose coordinate comparison.
- ASR-to-visual grounding: asr -> frame_retriever using transcripts[].clip or clips, with no `time_hints` input_ref -> generic_purpose with transcripts and frames.

ICL examples:

Example A, visible full-frame text:
{
  "strategy": "Locate the sign, retrieve a readable full frame, and OCR the complete frame.",
  "steps": [
    {"step_id": 1, "tool_name": "visual_temporal_grounder", "purpose": "Find the moment where the sign is clearly visible.", "inputs": {"query": "clearly visible storefront sign with the answer-critical label", "top_k": 3}, "input_refs": {}, "expected_outputs": {"clips": "candidate sign intervals"}},
    {"step_id": 2, "tool_name": "frame_retriever", "purpose": "Retrieve readable sign frames from the grounded interval.", "inputs": {"query": "most readable frame of the sign", "num_frames": 3}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"frames": "readable sign frames"}},
    {"step_id": 3, "tool_name": "ocr", "purpose": "Read the exact sign text from the complete frames.", "inputs": {"query": "read the exact sign text from the full frame"}, "input_refs": {"frames": [{"step_id": 2, "field_path": "frames"}]}, "expected_outputs": {"text": "exact OCR text"}}
  ],
  "refinement_instructions": "Use the full-frame OCR text to replace any unsupported label claim; keep the answer unresolved if the label remains unreadable."
}

Example B, ASR then grounded interpretation:
{
  "strategy": "Transcribe the bounded dialogue and interpret only from the transcript.",
  "steps": [
    {"step_id": 1, "tool_name": "asr", "purpose": "Transcribe the dialogue in the candidate clip.", "inputs": {"clips": [{"video_id": "video_id", "start_s": 42.0, "end_s": 58.0}], "speaker_attribution": true}, "input_refs": {}, "expected_outputs": {"transcripts": "spoken words with timestamps"}},
    {"step_id": 2, "tool_name": "generic_purpose", "purpose": "Map the transcript to the answer choice without using flattened text.", "inputs": {"query": "Which option is supported by the transcript?"}, "input_refs": {"transcripts": [{"step_id": 1, "field_path": "transcripts"}], "clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"answer": "option mapping from transcript evidence"}}
  ],
  "refinement_instructions": "Cite transcript observations; do not use any ASR flattened text field."
}

Example C, sound trigger:
{
  "strategy": "Find the sound and inspect the local before/during/after sequence for the direct trigger.",
  "steps": [
    {"step_id": 1, "tool_name": "audio_temporal_grounder", "purpose": "Localize the distinctive non-speech sound.", "inputs": {"query": "brief metallic crash or bang sound", "clips": [{"video_id": "video_id", "start_s": 0.0, "end_s": 120.0}]}, "input_refs": {}, "expected_outputs": {"clips": "sound-centered intervals"}},
    {"step_id": 2, "tool_name": "frame_retriever", "purpose": "Retrieve local frames around the sound that may show the direct trigger.", "inputs": {"query": "frames around the sound showing the direct visible trigger", "num_frames": 5}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"frames": "local trigger frames with timestamps"}},
    {"step_id": 3, "tool_name": "generic_purpose", "purpose": "Identify the direct visible trigger from the local sound-centered frames.", "inputs": {"query": "What directly triggers the sound? Use only the supplied clips and frames, and use frame timestamps for local temporal order."}, "input_refs": {"frames": [{"step_id": 2, "field_path": "frames"}], "clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"answer": "direct trigger or unresolved", "analysis": "why the local frames support it"}}
  ],
  "refinement_instructions": "Distinguish setup context from the direct trigger; preserve uncertainty if the local sequence does not show the trigger."
}

Example C2, non-speech sound-effect option comparison:
- Good plan: audio_temporal_grounder over the bounded/full clip with a query containing all candidate sound-effect names, retrieve frames around grounded audio candidates only if source/action matters, then compare all options against the localized audio clips.
- Bad plan: pick one visually described action from unverified text context and assume its sound is the answer.

Example C3, visible-use anchored sound count:
- Good plan: visual_temporal_grounder finds all candidate moments where the target object/action may be used, audio_temporal_grounder searches only inside those candidate clips, then generic_purpose uses the clips to count distinct supported sound types and rejects false visual-use candidates, non-use sounds, unrelated ambience, or duplicate sound labels. Add frame_retriever only if the plan needs explicit frame-by-frame visual confirmation.
- Bad plan: run audio_temporal_grounder on a guessed local window and count transcript sound words without proving the visible object was being used.

Example D, repeated count/state:
{
  "strategy": "Ground candidate occurrences and resolve state directly from clips before counting.",
  "steps": [
    {"step_id": 1, "tool_name": "visual_temporal_grounder", "purpose": "Find all candidate occurrences of the object state.", "inputs": {"query": "all moments where the answer-critical object appears in the required state", "top_k": 5}, "input_refs": {}, "expected_outputs": {"clips": "candidate object-state intervals"}},
    {"step_id": 2, "tool_name": "generic_purpose", "purpose": "Determine which grounded clips actually satisfy the state and count them.", "inputs": {"query": "Count only candidates that visibly satisfy the required object state. Explain rejected ambiguous candidates and report unresolved coverage."}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"answer": "deduplicated count or unresolved", "analysis": "accepted/rejected candidates"}}
  ],
  "refinement_instructions": "Count only candidates whose state is grounded; explain rejected ambiguous candidates."
}

Example D2, complex visual-temporal ordinal:
- Bad plan: visual_temporal_grounder("the shot two shots before the second reviewed play where a spectator waves his flag while a ball lands near him").
- Good plan: visual_temporal_grounder("all reviewed-play or replay-review moments") plus visual_temporal_grounder("all shot/delivery candidates with nearby spectator flag waving or ball landing") -> generic_purpose sorts timestamps, identifies the second reviewed play, correlates candidate shots, and selects the target interval -> generic_purpose counts only complete flag-wave cycles in the selected target clips.

Example E, ordered labels:
- Good plan: retrieve readable frames from the bounded montage, OCR each visible label, then synthesize the ordered label list by frame timestamps.
- Bad plan: retrieve top-k relevant frames and treat returned relevance order as chronology.

Example F, exact timestamp action:
- Good plan: use `frame_retriever` with the exact timestamp in `time_hints`, a focused action/state query, and enough frames for local context; then determine the action/state using frame timestamps.
- Bad plan: inspect only the exact frame and ignore adjacent motion context.

Example G, label-value display:
- Good plan: retrieve the stable display after the update, OCR the complete frame, and preserve label-value adjacency.
- Bad plan: crop a narrow region before OCR and lose the pairing context.

Example G2, arithmetic with a missing visual number:
- Good plan: use ASR for the spoken number, retrieve the visual sign/plaque/board that contains the missing number, OCR the complete readable frame, then ground both operands before arithmetic.
- Bad plan: ask generic_purpose to infer a historical date or missing number from partial transcript context.

Example G3, exact blackboard or visible-letter task:
- Good plan: localize the utterance/action anchor, retrieve neighboring frames where the board is visible, OCR the complete readable frames, then ground the requested ordinal letter.
- Bad plan: ask a generic visual reader to guess letters from full frames without OCR.

Example H, tone transition:
- Good plan: localize before/after utterance windows for the same speaker, run ASR, retrieve delivery frames/clips, and compare delivery.
- Bad plan: infer tone from transcript sentiment words alone.

Example I, quote-adjacent response:
- Good plan: localize the quoted line in ASR, keep the local dialogue window before and after it, identify the responding speaker, and map the response to an option.
- Bad plan: search only for exact option text and ignore turn-taking.

Example J, speaker/addressee attribution:
- Good plan: combine the ASR line, neighboring frames, turn order, gaze, position, subtitles, or response behavior.
- Bad plan: assume the nearest named person in the transcript is the speaker or addressee.

Example K, object state anchored by speech:
- Good plan: use ASR to find the utterance time, retrieve frames with that time in `time_hints`, spatially ground the object, and determine state such as empty/full/open/closed/on/off from the supplied media.
- Bad plan: use the transcript topic as proof of the visual state.

Example L, repeated event count:
- Good plan: localize the marker event and count interval, use grounded clips for broad counting, and retrieve frames only for explicit frame-by-frame confirmation with timestamped evidence.
- Bad plan: count from sparse representative frames.

Example M, absence/not-mentioned:
- Good plan: run ASR over the bounded dialogue interval, preserve exact spans for mentioned candidates, and compare every option against transcript evidence.
- Bad plan: answer from broad scene captions or world knowledge.

Example M2, repeated place phrase:
- Good plan: when both "Example Transit" and "Example" repeat, use the repeated full surface phrase if it gives the first well-defined repeated-name interval; quote the exact text between the two full mentions and then test every option.
- Bad plan: use the shorter embedded token from inside the full phrase, producing a tiny or ambiguous interval.

Example N, refine after audit:
- Diagnosis: "The trace found the event but did not prove the object state."
- Good plan: preserve the event timestamp, retrieve frames/regions at that timestamp, and resolve only the missing state.
- Bad plan: run a broad new event search and discard the prior timestamp.

Example P, progressive chart frame selection:
- Diagnosis: "The chart answer is ambiguous because earlier frames show partial bars."
- Good plan: retrieve readable chart frames, use their timestamps to decide whether the question asks for the first state, final complete state, a before/after change, or a peak/min/max state.
- Bad plan: pass all consecutive frames to generic_purpose and ask it to reconcile partial/revealed bars as if they were separate evidence.

Example Q, multi-referent relation:
- Question pattern: "What is the relation between the person holding the named object and the person holding the first object?"
- Good plan: create two slots, one for the named-object holder and one for the first-object holder; ground the first-object slot over the full question scope, ground the named-object slot near the naming event, pass both frame sets plus any relationship transcript to generic_purpose, then map the relation option from the grounded comparison.
- Bad plan: inspect only the named-object window, treat the first object in that local window as the question's first object, and drop the relationship transcript before option mapping.
"""


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _question_structure_hints(question: object) -> List[str]:
    text = _normalize_text(question)
    lowered = text.lower()
    hints: List[str] = []
    temporal_tokens = re.findall(
        r"\b(first|second|third|fourth|fifth|last|next|previous|before|after|earlier|later|between|during|while|when)\b",
        lowered,
    )
    if len(temporal_tokens) >= 2:
        hints.append(
            "This question has multiple temporal/ordinal operators. Decompose visual temporal grounding into primitive/localizable sub-queries before using generic_purpose to sort, correlate, and select the target interval."
        )
    if "relation between" in lowered or "relationship between" in lowered:
        hints.append(
            "This is a multi-referent relation question. Maintain separate referent slots for each side of the relationship, ground each slot in its own intended temporal scope, then compare the grounded referents."
        )
    if hints and re.search(r"\b(first|last|next|previous|earliest|latest)\b", lowered):
        hints.append(
            "One relation slot uses ordinal language. Resolve that ordinal over the question's full scope, not only within a later retrieved local clip."
        )
    return hints


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

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        normalized["diagnostics"] = diagnostics

    return normalized


def build_planner_prompt(
    task,
    mode: str,
    audit_feedback: Optional[dict],
    tool_catalog: Dict[str, Dict[str, object]],
) -> str:
    normalized_audit_feedback = _canonicalize_audit_feedback(audit_feedback)
    question_structure_hints = _question_structure_hints(task.question)

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

    if question_structure_hints:
        parts.extend(
            [
                "QUESTION_STRUCTURE_HINTS:",
                pretty_json(question_structure_hints),
                "",
            ]
        )

    if normalized_audit_feedback:
        parts.extend(["DIAGNOSIS:", pretty_json(normalized_audit_feedback), ""])

    parts.extend(
        [
            "ExecutionPlan schema reminder:",
            "- strategy: short text",
            "- steps: list of {step_id, tool_name, purpose, inputs, input_refs, expected_outputs}",
            "- refinement_instructions: precise guidance for the trace-writing agent",
            "- step_id values must be integers numbered from 1 upward",
            "- input_refs is a field-keyed object, e.g. {\"frames\": [{\"step_id\": 2, \"field_path\": \"frames\"}]}",
            "- input_refs may only reference earlier steps in this same plan",
            "- do not use input_refs for time_hints; put known timestamp hints directly in inputs.time_hints",
            "- never emit arguments, depends_on, use_summary, or list-style input_refs",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
