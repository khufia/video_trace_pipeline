from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .shared import pretty_json, render_tool_catalog


PLANNER_SYSTEM_PROMPT = """You are the Planner in a benchmark trace pipeline.

Your job is to produce a small, dependency-aware `ExecutionPlan` for evidence collection.
You do NOT answer the question and you do NOT write the trace.

You are text-only:
- You do not see video, audio, frames, crops, or hidden tool state.
- Use only QUESTION, OPTIONS, RICH_PREPROCESS_SEGMENTS, TASK_STATE, RETRIEVAL_CATALOG, RETRIEVED_CONTEXT, DIAGNOSIS, and AVAILABLE_TOOLS.
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

Preprocess use:
- RICH_PREPROCESS_SEGMENTS contain dense captions, attributes, clips, overall_summary, and ASR transcript spans.
- Use them as full first-round video context, not as final proof for answer-critical fine detail.
- For transcript-only, quote, not-mentioned, or dialogue-content questions, use PREPROCESS_TRANSCRIPTS_AVAILABLE as structured `inputs.transcripts` for generic_purpose when the spans cover the needed interval.
- Call ASR only when preprocessing has no transcript coverage, the existing transcript is contradicted/incomplete, or the task needs speaker attribution that is not present.
- If the needed detail is absent, ambiguous, conflicting, too broad, or depends on exact state/count/text/region/speaker, call tools.

Retrieval use:
- RETRIEVAL_CATALOG lists the available text stores before planning: preprocess windows, ASR/dense-caption stores, evidence entries, observations, and prior trace claims.
- RETRIEVED_CONTEXT is the text-only retrieval package selected before this final plan. It may include preprocess spans, prior atomic observations, evidence summaries, OCR text, spatial boxes, audit gaps, and prior trace claims.
- Prefer validated retrieved observations and evidence before starting broad searches.
- If using previous evidence directly, pass literal `evidence_ids` from RETRIEVED_CONTEXT in `inputs`; do not invent IDs.

Task-state use:
- TASK_STATE is the canonical compact memory for the task: claims, referents, coverage, OCR occurrences, counters, temporal events, evidence status updates, retrieval memory, and open questions.
- Treat TASK_STATE as the planner checklist. Plan only for unresolved, unknown, contradicted, or partially validated answer-critical claims.
- Candidate retrieved context is not proof. Prefer validated evidence; use candidate evidence as a source to verify or target.
- If TASK_STATE shows an answer-critical claim is already validated with matching coverage, do not recollect it.
- If TASK_STATE marks evidence refuted, irrelevant, stale, or superseded, do not use it as support.

Verifier use:
- Use `verifier` for answer-critical checks about small-object state/count, subtle visual state, exact OCR/scoreboard/price/table/chart values, event-moment action, identity/same-different/referent relation, speaker/addressee/tone, contradictions, and audit repairs.
- `generic_purpose` may extract or compare from rich clips/frames/transcripts, but it is not a final claim-status reducer. When a claim is answer-critical and not already validated, finish the chain with `verifier`.
- Verifier inputs must include claim objects plus relevant media/text/evidence context. Ask for supported/refuted/unknown verdicts, not a benchmark answer.

Planning rules:
- Gather the smallest sufficient evidence set that resolves the answer-critical gap. Do not shrink context so much that temporal order, action state, referent identity, or option mapping becomes ungrounded.
- Use only canonical tool request fields from AVAILABLE_TOOLS.
- If a downstream tool needs prior media or transcripts, wire that exact structured output through `input_refs`.
- Never invent helper fields such as `query_context`, `notes`, or `context`; put extra text in canonical `text_contexts` or `retrieved_context` when that tool schema supports it.
- Pass ASR to generic_purpose through `transcripts`, never flattened `text_contexts`.
- `input_refs` reference earlier step output fields only, never earlier step inputs such as `inputs.transcripts`.
- `input_refs` are structural: bind clips from `clips`, `clips[0]`, `frames[].clip`, `regions[].frame.clip`, or `transcripts[].clip`; bind frames from `frames`, `frames[0]`, or `regions[].frame`; bind transcripts only from `transcripts` or `transcripts[]`.
- Never bind `transcripts` from a clip/frame path. To transcribe media, add an ASR step over clips; to reuse preprocessing ASR, copy the relevant PREPROCESS_TRANSCRIPTS_AVAILABLE objects literally into each tool's `inputs.transcripts`.
- Bind `text_contexts` only from textual outputs like `text`, `summary`, `overall_summary`, `analysis`, `answer`, `supporting_points`, `spatial_description`, or `raw_output_text`.
- If generic_purpose must verify a visual attribute after spatial_grounder, bind frames from `regions[].frame` or pass the original frames, and optionally bind `spatial_description` as text_contexts. Text_contexts alone are not enough for visual-state verification.
- Do not bind current-plan outputs into `time_hints`; if timestamps are known from preprocess or retrieval, put literal strings in `inputs.time_hints`, otherwise pass `clips`.
- Do not bind current-plan outputs into `evidence_ids`.
- generic_purpose must receive explicit context: clips, frames, transcripts, text_contexts, evidence_ids, or input_refs.
- Avoid generic_purpose -> generic_purpose chains whose only role is arithmetic, comparison, or consolidation of prior generic text. If the original representative media fits in one call, pass the media together and ask for extraction plus comparison in that single call.
- If an audio/count question is conditioned on a visible object, action, or state, first ground the visible condition, then analyze audio only inside that visual interval. Start with audio grounding only when the sound itself is the primary anchor.
- For questions phrased like "sounds/noises when/while using/doing/showing <visible object/action>", the visible object/action is the anchor. First find every relevant visual-use candidate across the bounded interval, then run audio matching inside those candidate clips and deduplicate sound types. The verifier must reject false visual-use candidates, non-use sounds, duplicates, and unrelated ambience. Do not start from one guessed audio window.
- For non-speech audio option comparisons, such as choosing which sound effect is heard or most distinctive, start with `audio_temporal_grounder` over the full bounded interval using option-aware query terms. Then retrieve frames only if the visible source/action helps disambiguate. Do not anchor to a single visually convenient segment unless existing validated evidence proves it is the relevant sound.
- For relationship or comparison questions, plan explicit referent slots before tools: ground each queried person/object in its own intended temporal scope, then perform the relationship/comparison only after all slots have evidence. If a slot contains ordinal language like first/last/next, resolve that ordinal over the question's full scope rather than the first candidate inside a later local clip.
- If a relationship depends on dialogue, carry the relevant transcripts into the final generic_purpose call along with the frames for each referent slot.
- Never call more than 6 tools in one plan.

Wiring is not evidence:
- `input_refs` only pass media/text objects between tools.
- The plan must still say what answer-critical observation the downstream tool should extract from those inputs.

Chain sufficiency rule:
- A chain is sufficient only if its output can ground the final discriminator, not merely locate a related moment.
- For sequence/order/count/action tasks, prefer chronological clips or frame sequences over isolated top-k frames.
- For visible text tasks, preserve label-value adjacency and region context until OCR has read the target text.

Evidence preservation rule:
- In refine mode, preserve useful prior timestamps, clips, OCR text, ASR spans, evidence entries, and atomic observations.
- Re-search broadly only when the diagnosis says the old anchor is wrong, contradicted, or incomplete.
- If the audit names missing information, target that missing fact directly before starting a new semantic search.
- If the audit says a prior answer missed an attribute such as empty/full/open/closed/count/state, do not pass answer-only generic evidence back as proof. Reuse media/transcripts/regions and only prior evidence that explicitly contains the missing attribute.
- If the audit flags a repeated-name, occurrence, quote-span, or interval-boundary ambiguity, do not preserve the prior anchor by default. Re-evaluate the boundary candidate from the transcript and choose the exact surface form that makes the interval well-defined.

Occurrence and chronology rule:
- For first/last/before/after/ordered-list questions, collect all relevant candidate events in the bounded interval, sort by timestamp, and then choose the requested occurrence.
- Never infer chronology from retrieval result order.
- If an early/late candidate is missing, retrieve the full interval before answering.
- For repeated place/name/entity questions, compare repeated full surface phrases before substrings. If a longer repeated phrase and a shorter embedded token begin at the same occurrence, use the longer repeated phrase as the boundary unless the task explicitly asks about words/tokens.
- Do not downgrade a repeated organization, venue, event, brand, or institution name to the shorter place token inside it just because the question says "place name"; exact repeated text boundaries matter more than semantic category guesses.

Action-at-timestamp rule:
- Exact timestamps are anchors, not isolated proof.
- For action, motion, state change, count, or identity at a timestamp, retrieve the anchor frame plus chronological neighbors.
- The downstream tool must receive the structured frame sequence and answer from the local sequence.

Small-text rule:
- For scoreboards, prices, labels, signs, nameplates, blackboards, whiteboards, visible letters/words, charts, menus, or control panels, use high-resolution frames, region grounding, OCR, and explicit label-value pairing.
- Preserve spatial adjacency such as team-score, product-price, name-role, and label-value.
- For arithmetic over visible/transcribed numbers, collect each operand as its own verified value. If a needed operand is absent from transcripts or preprocess text, retrieve the relevant visual source and OCR it before computing; do not ask a visual reader or generic prose step to infer a missing year/number.
- For progressive chart/table animations, use the stable later frame where all relevant labels/bars are visible; do not treat missing bars in a partially revealed chart as zero.
- Do not ask a visual reader to resolve a static chart from many near-duplicate progressive frames if a single complete frame can answer it.

Tool-chain patterns:
- Visible text: visual_temporal_grounder -> frame_retriever -> ocr -> verifier when the read text selects an answer.
- Region text: visual_temporal_grounder -> frame_retriever -> spatial_grounder -> ocr -> verifier when exact text matters.
- Structured chart/table/scoreboard: visual_temporal_grounder -> frame_retriever -> generic_purpose for extraction/comparison, then verifier for the answer-critical extracted claim; use OCR for explicit labels or numbers.
- Multi-display chart/table comparison: pass one stable representative frame per required display into one generic_purpose call for extraction/comparison, then verifier for the final value/option claim.
- Localized visual state: frame_retriever or retrieved frames -> spatial_grounder -> verifier with frames from `regions[].frame` plus spatial_description text_contexts.
- Transcript already in preprocessing: generic_purpose over PREPROCESS_TRANSCRIPTS_AVAILABLE, with no ASR call.
- Dialogue: bounded clip localization when needed -> asr -> optional generic_purpose over transcripts.
- Tone/affect: asr plus frame_retriever sequence -> generic_purpose over transcripts and frames.
- Brief sound cause: audio_temporal_grounder -> frame_retriever chronological sequence -> verifier over the direct-trigger claim.
- Non-speech audio option comparison: audio_temporal_grounder with option-aware sound queries over the full bounded interval -> optional frame_retriever for source/action -> generic_purpose or verifier over the grounded clips/frames.
- Visual-conditioned audio/count: visual_temporal_grounder for all visible-use/action candidates -> frame_retriever for chronological visual confirmation -> audio_temporal_grounder on those candidate clips -> verifier over accepted/rejected visual occurrences, non-use sounds, and deduplicated sound types.
- Exact timestamp/action/state: frame_retriever with sequence_mode "anchor_window", include_anchor_neighbors true, sort_order "chronological" -> verifier.
- Object count/state: visual_temporal_grounder -> frame_retriever -> spatial_grounder if same-type candidates matter -> verifier.
- Multi-referent relation/comparison: identify referent slots from the question -> retrieve/ground each slot in its own scope -> pass all slot frames plus relationship transcripts/text to verifier for relation mapping.
- Map/direction: speech or visual anchor -> frame_retriever -> spatial_grounder for anchor and referent -> generic_purpose coordinate comparison.
- ASR-to-visual grounding: asr -> frame_retriever using transcripts[].clip or clips, with no `time_hints` input_ref -> generic_purpose with transcripts and frames.

ICL examples:

Example A, visible text region:
{
  "strategy": "Locate the sign, crop the relevant region, and read it.",
  "steps": [
    {"step_id": 1, "tool_name": "visual_temporal_grounder", "purpose": "Find the moment where the sign is clearly visible.", "inputs": {"query": "clearly visible storefront sign with the answer-critical label", "top_k": 3}, "input_refs": {}, "expected_outputs": {"clips": "candidate sign intervals"}},
    {"step_id": 2, "tool_name": "frame_retriever", "purpose": "Retrieve readable sign frames from the grounded interval.", "inputs": {"query": "most readable frame of the sign", "num_frames": 3}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"frames": "readable sign frames"}},
    {"step_id": 3, "tool_name": "spatial_grounder", "purpose": "Localize the answer-critical sign region.", "inputs": {"query": "the sign text region that answers the question"}, "input_refs": {"frames": [{"step_id": 2, "field_path": "frames"}]}, "expected_outputs": {"regions": "localized sign text"}},
    {"step_id": 4, "tool_name": "ocr", "purpose": "Read the localized sign text.", "inputs": {"query": "read the exact sign text"}, "input_refs": {"regions": [{"step_id": 3, "field_path": "regions"}]}, "expected_outputs": {"text": "exact OCR text"}},
    {"step_id": 5, "tool_name": "verifier", "purpose": "Verify the OCR-derived answer claim against the localized region and OCR output.", "inputs": {"query": "verify the exact sign-text claim", "claims": [{"claim_id": "claim_sign_text", "text": "The localized sign text supports the selected option.", "claim_type": "ocr"}]}, "input_refs": {"regions": [{"step_id": 3, "field_path": "regions"}], "ocr_results": [{"step_id": 4, "field_path": "reads"}]}, "expected_outputs": {"claim_results": "supported/refuted/unknown verdict for the sign-text claim"}}
  ],
  "refinement_instructions": "Use the verifier verdict over the OCR text to replace any unsupported label claim; keep the answer unresolved if the label remains unreadable."
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
    {"step_id": 2, "tool_name": "frame_retriever", "purpose": "Retrieve chronological neighboring frames around the sound.", "inputs": {"query": "frames before during and after the sound showing the direct trigger", "num_frames": 5, "sequence_mode": "anchor_window", "neighbor_radius_s": 2.0, "include_anchor_neighbors": true, "sort_order": "chronological"}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"frames": "chronological trigger sequence"}},
    {"step_id": 3, "tool_name": "verifier", "purpose": "Verify the direct visible trigger claim against the local sound-centered sequence.", "inputs": {"query": "verify the direct trigger of the sound", "claims": [{"claim_id": "claim_sound_trigger", "text": "The local before/during/after sequence directly shows the trigger of the sound.", "claim_type": "event_action"}]}, "input_refs": {"frames": [{"step_id": 2, "field_path": "frames"}], "clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"claim_results": "direct trigger claim verdict"}}
  ],
  "refinement_instructions": "Distinguish setup context from the direct trigger; preserve uncertainty if the local sequence does not show the trigger."
}

Example C2, non-speech sound-effect option comparison:
- Good plan: audio_temporal_grounder over the bounded/full clip with a query containing all candidate sound-effect names, retrieve frames around grounded audio candidates only if source/action matters, then verify all option claims against the localized audio clips.
- Bad plan: pick one visually described action from preprocessing and assume its sound is the answer.

Example C3, visible-use anchored sound count:
- Good plan: visual_temporal_grounder finds all candidate moments where the target object/action may be used, frame_retriever supplies chronological visual confirmation, audio_temporal_grounder searches only inside those candidate clips, then verifier counts distinct supported sound types and rejects false visual-use candidates, non-use sounds, unrelated ambience, or duplicate sound labels.
- Bad plan: run audio_temporal_grounder on a guessed local window and count transcript sound words without proving the visible object was being used.

Example D, repeated count/state:
{
  "strategy": "Ground candidate occurrences, retrieve frames, and verify state before counting.",
  "steps": [
    {"step_id": 1, "tool_name": "visual_temporal_grounder", "purpose": "Find all candidate occurrences of the object state.", "inputs": {"query": "all moments where the answer-critical object appears in the required state", "top_k": 5}, "input_refs": {}, "expected_outputs": {"clips": "candidate object-state intervals"}},
    {"step_id": 2, "tool_name": "frame_retriever", "purpose": "Retrieve representative frames for each candidate interval.", "inputs": {"query": "representative frames showing the object state", "num_frames": 5, "sort_order": "chronological"}, "input_refs": {"clips": [{"step_id": 1, "field_path": "clips"}]}, "expected_outputs": {"frames": "candidate state frames"}},
    {"step_id": 3, "tool_name": "verifier", "purpose": "Verify which candidates actually satisfy the state and count them.", "inputs": {"query": "verify the object-state count", "claims": [{"claim_id": "claim_state_count", "text": "The retrieved candidates satisfy the required object state and yield the claimed deduplicated count.", "claim_type": "count"}], "verification_policy": {"sampling": "chronological_dense", "require_direct_support": true, "report_unknown_when_uncovered": true}}, "input_refs": {"frames": [{"step_id": 2, "field_path": "frames"}]}, "expected_outputs": {"claim_results": "state/count verdict with rejected candidates"}}
  ],
  "refinement_instructions": "Count only candidates whose state is grounded; explain rejected ambiguous candidates."
}

Example E, ordered labels:
- Good plan: retrieve the bounded montage as a chronological frame sequence, OCR each visible label, then synthesize the ordered label list.
- Bad plan: retrieve top-k relevant frames and treat returned relevance order as chronology.

Example F, exact timestamp action:
- Good plan: retrieve the anchor frame plus +/- 2s chronological neighbors, then verify the action/state claim over the sequence.
- Bad plan: inspect only the exact frame and ignore adjacent motion context.

Example G, label-value display:
- Good plan: retrieve the stable display after the update, spatially ground the relevant region, OCR label-value pairs, and preserve adjacency.
- Bad plan: OCR an early full frame and infer the missing pairing.

Example G2, arithmetic with a missing visual number:
- Good plan: use preprocess transcripts for the spoken number, retrieve the visual sign/plaque/board that contains the missing number, spatially ground it, OCR it, then verify both operands before arithmetic.
- Bad plan: ask generic_purpose to infer a historical date or missing number from partial transcript context.

Example G3, exact blackboard or visible-letter task:
- Good plan: localize the utterance/action anchor, retrieve neighboring frames where the board is visible, spatially ground the board/target line, OCR the letters, then verify the requested ordinal letter.
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
- Good plan: use ASR to find the utterance time, retrieve frames around it, spatially ground the object, and verify state such as empty/full/open/closed/on/off.
- Bad plan: use the transcript topic as proof of the visual state.

Example L, repeated event count:
- Good plan: localize the marker event and count interval, retrieve a dense chronological sequence, count complete cycles, and state inclusion rules.
- Bad plan: count from sparse representative frames.

Example M, absence/not-mentioned:
- Good plan: use preprocessing transcript spans as `inputs.transcripts` when they cover the interval; run ASR only when transcript coverage is missing or insufficient, preserve exact spans for mentioned candidates, and compare every option against transcript evidence.
- Bad plan: answer from broad scene captions or world knowledge.

Example M2, repeated place phrase:
- Good plan: when both "Example Transit" and "Example" repeat, use the repeated full surface phrase if it gives the first well-defined repeated-name interval; quote the exact text between the two full mentions and then test every option.
- Bad plan: use the shorter embedded token from inside the full phrase, producing a tiny or ambiguous interval.

Example N, refine after audit:
- Diagnosis: "The trace found the event but did not prove the object state."
- Good plan: preserve the event timestamp, retrieve frames/regions at that timestamp, and verify only the missing state.
- Bad plan: run a broad new event search and discard the prior timestamp.

Example O, retrieved evidence in refine:
- Diagnosis: "A chart/table value is missing from the current trace."
- RETRIEVED_CONTEXT includes validated OCR/evidence for one metric but not the missing comparison metric.
- Good plan: reuse the validated evidence id, then retrieve or ground only the missing display/metric before comparing.
- Bad plan: trust a prior trace sentence as if it were validated evidence, or rerun every previous tool without targeting the missing value.

Example P, progressive chart frame selection:
- Diagnosis: "The chart answer is ambiguous because earlier frames show partial bars."
- Good plan: retrieve a chronological frame sequence and decide from the question whether it asks for the first state, final complete state, a before/after change, or a peak/min/max state.
- Bad plan: pass all consecutive frames to generic_purpose and ask it to reconcile partial/revealed bars as if they were separate evidence.

Example Q, multi-referent relation:
- Question pattern: "What is the relation between the person holding the named object and the person holding the first object?"
- Good plan: create two slots, one for the named-object holder and one for the first-object holder; ground the first-object slot over the full question scope, ground the named-object slot near the naming event, pass both frame sets plus any relationship transcript to verifier, then map the relation option from the verifier claim result.
- Bad plan: inspect only the named-object window, treat the first object in that local window as the question's first object, and drop the relationship transcript before option mapping.
"""


PLANNER_RETRIEVAL_SYSTEM_PROMPT = """You are the Planner retrieval controller in a benchmark trace pipeline.

Your job is to decide whether the final Planner already has enough text context, or whether it should retrieve more text records before making an `ExecutionPlan`.
You do NOT answer the task and you do NOT emit tool steps.

You are text-only:
- You do not see video, audio, frames, crops, or hidden tool state.
- Use only QUESTION, OPTIONS, TASK_STATE, RETRIEVAL_CATALOG, CURRENT_RETRIEVED_CONTEXT, DIAGNOSIS, PRIOR_TRACE, and AVAILABLE_TOOLS.
- Return JSON only matching the active `PlannerRetrievalDecision` schema.

Active retrieval decision schema:
- action: "ready" or "retrieve"
- rationale: short reason
- requests: list of retrieval requests when action is "retrieve"; empty when action is "ready"

Retrieval request schema:
- request_id: stable short id such as "need_object_state"
- target: one of "existing_evidence", "evidence", "observations", "preprocess", "asr_transcripts", "dense_captions", "prior_trace", "task_state", or "mixed"
- need: the precise missing answer-critical fact or context
- query: lexical query terms for deterministic retrieval
- modalities: optional text labels such as ["visual"], ["audio"], ["asr"], ["ocr"]
- time_range: optional {"start_s": number, "end_s": number}
- source_tools: optional exact tool names such as ["asr"], ["ocr"], ["generic_purpose"]
- evidence_status: optional status such as "validated", "candidate", "unknown", "refuted", "irrelevant", "stale", or "superseded"
- evidence_ids, observation_ids: optional exact ids from the catalog/current context
- limit: integer from 1 to 50

Decision rules:
- First inspect TASK_STATE, RETRIEVAL_CATALOG, and CURRENT_RETRIEVED_CONTEXT. If state or existing observations/evidence already answer the missing fact, return "ready".
- Prefer retrieving existing evidence, observations, ASR transcripts, or dense-caption/preprocess spans before planning new tool calls.
- Ask for narrow text records: relevant ids, time ranges, source tools, and statuses. Do not ask for raw pixels, audio, or video.
- If the catalog shows no relevant existing source, return "ready" so the final Planner can collect new evidence with tools.
- Stop retrieving once the current context contains enough text for the final Planner to either reuse evidence ids or target a narrow missing gap.

ICL examples:

Example A, evidence reuse:
Catalog says validated OCR evidence `ev_score_1` and observation `obs_score_1` mention the visible scoreboard value. Current context already includes both ids and text.
Return:
{"action":"ready","rationale":"The visible value is already present as validated OCR evidence.","requests":[]}

Example B, retrieve by audit gap and prior timestamp:
Audit says the trace found the event but missed object state at 132s. Catalog has generic_purpose observations and evidence near that time.
Return:
{"action":"retrieve","rationale":"Need existing visual observations for the audited timestamp before planning new frame tools.","requests":[{"request_id":"state_at_132s","target":"observations","need":"Object state at the audited 132s event","query":"object state audited event", "time_range":{"start_s":130.0,"end_s":134.0},"source_tools":["generic_purpose"],"limit":10}]}

Example C, retrieve ASR instead of re-transcribing:
Question asks what a speaker said, and catalog shows ASR transcripts exist for the whole candidate interval.
Return:
{"action":"retrieve","rationale":"The transcript may already contain the quote, so retrieve ASR spans before planning an ASR tool call.","requests":[{"request_id":"quote_transcript","target":"asr_transcripts","need":"Exact quoted speech and neighboring turns","query":"quoted speech neighboring response","modalities":["asr"],"time_range":{"start_s":40.0,"end_s":70.0},"source_tools":["asr"],"limit":20}]}

Example D, no relevant source:
Catalog has only generic opening-scene captions, no evidence entries, and no ASR for the relevant event.
Return:
{"action":"ready","rationale":"No existing text source appears relevant; the final Planner should collect new evidence with tools.","requests":[]}

Example E, conflicting prior trace timestamp:
Prior trace says the relevant display is at one timestamp, but catalog evidence and observations show nearby candidate timestamps. The missing fact may require checking those evidence-grounded times instead of the prior prose timestamp.
Return:
{"action":"retrieve","rationale":"Evidence timestamps conflict with the prior trace, so retrieve observations around the candidate times before planning.","requests":[{"request_id":"display_context","target":"observations","need":"Relevant display observations from candidate evidence timestamps","query":"display label value entity comparison metric","source_tools":["generic_purpose"],"limit":20}]}
"""


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _question_structure_hints(question: object) -> List[str]:
    text = _normalize_text(question)
    lowered = text.lower()
    hints: List[str] = []
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


def _collect_retrieved_evidence_ids(retrieved_context: dict) -> List[str]:
    evidence_ids = []
    seen = set()
    records = []
    if isinstance(retrieved_context, dict):
        records.extend(list(retrieved_context.get("observations") or []))
        records.extend(list(retrieved_context.get("evidence") or []))
    for item in records:
        if not isinstance(item, dict):
            continue
        evidence_id = _normalize_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        evidence_ids.append(evidence_id)
    return evidence_ids


def _collect_preprocess_transcript_refs(planner_segments: List[dict], *, video_id: str) -> List[dict]:
    refs: List[dict] = []
    seen = set()
    for index, segment in enumerate(list(planner_segments or []), start=1):
        if not isinstance(segment, dict):
            continue
        asr = segment.get("asr") if isinstance(segment.get("asr"), dict) else {}
        spans = [dict(item) for item in list(asr.get("transcript_spans") or []) if isinstance(item, dict)]
        if not spans:
            continue

        segment_id = _normalize_text(segment.get("segment_id")) or "seg_%03d" % index
        clip_payload = None
        clips = [dict(item) for item in list(segment.get("clips") or []) if isinstance(item, dict)]
        if clips:
            clip_payload = clips[0]
        else:
            try:
                clip_payload = {
                    "video_id": video_id,
                    "start_s": float(segment.get("start_s") or 0.0),
                    "end_s": float(segment.get("end_s") or segment.get("start_s") or 0.0),
                }
            except (TypeError, ValueError):
                clip_payload = None
        if clip_payload is not None:
            clip_payload = {
                key: value
                for key, value in dict(clip_payload).items()
                if key in {"video_id", "start_s", "end_s", "artifact_id", "relpath", "metadata"} and value not in (None, "", [], {})
            }
            clip_payload.setdefault("video_id", video_id)

        normalized_spans = []
        for span in spans:
            text = _normalize_text(span.get("text"))
            if not text:
                continue
            try:
                start_s = float(span.get("start_s") or 0.0)
                end_s = float(span.get("end_s") or start_s)
            except (TypeError, ValueError):
                continue
            normalized_span = {
                "start_s": start_s,
                "end_s": end_s,
                "text": text,
            }
            speaker_id = _normalize_text(span.get("speaker_id") or span.get("speaker"))
            if speaker_id:
                normalized_span["speaker_id"] = speaker_id
            if span.get("confidence") not in (None, ""):
                normalized_span["confidence"] = span.get("confidence")
            normalized_spans.append(normalized_span)
        if not normalized_spans:
            continue

        signature = tuple((item["start_s"], item["end_s"], item["text"]) for item in normalized_spans)
        if signature in seen:
            continue
        seen.add(signature)
        refs.append(
            {
                "transcript_id": "preprocess_%s" % segment_id,
                "clip": clip_payload,
                "segments": normalized_spans,
                "metadata": {
                    "source": "preprocess",
                    "segment_id": segment_id,
                    "span_count": len(normalized_spans),
                },
            }
        )
    return refs


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
    planner_segments: List[dict],
    retrieved_context: Dict[str, object],
    audit_feedback: Optional[dict],
    tool_catalog: Dict[str, Dict[str, object]],
    retrieval_catalog: Optional[Dict[str, object]] = None,
    task_state: Optional[Dict[str, object]] = None,
) -> str:
    normalized_audit_feedback = _canonicalize_audit_feedback(audit_feedback)
    normalized_preprocess_segments = [dict(item) for item in list(planner_segments or []) if isinstance(item, dict)]
    normalized_retrieved_context = {
        key: value
        for key, value in dict(retrieved_context or {}).items()
        if value not in (None, "", [], {})
    }
    normalized_retrieval_catalog = {
        key: value
        for key, value in dict(retrieval_catalog or {}).items()
        if value not in (None, "", [], {})
    }
    normalized_task_state = {
        key: value
        for key, value in dict(task_state or {}).items()
        if value not in (None, "", [], {})
    }
    available_retrieved_evidence_ids = _collect_retrieved_evidence_ids(normalized_retrieved_context)
    available_preprocess_transcripts = _collect_preprocess_transcript_refs(
        normalized_preprocess_segments,
        video_id=str(task.video_id or task.sample_key),
    )
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

    if normalized_preprocess_segments:
        parts.extend(
            [
                "RICH_PREPROCESS_SEGMENTS:",
                pretty_json(normalized_preprocess_segments),
                "",
                "RICH_PREPROCESS_SEGMENTS_USAGE_NOTE:",
                "These are full first-round preprocess segments with dense captions, attributes, clips, overall summaries, and ASR transcript spans. Use them for broad context, but verify answer-critical fine detail with tools.",
                "",
            ]
        )

    if available_preprocess_transcripts:
        parts.extend(
            [
                "PREPROCESS_TRANSCRIPTS_AVAILABLE:",
                pretty_json(available_preprocess_transcripts),
                "",
                "PREPROCESS_TRANSCRIPTS_USAGE_NOTE:",
                "These are structured TranscriptRef objects built from preprocessing ASR spans. For transcript-only, quote, not-mentioned, or dialogue-content questions, copy the relevant objects into generic_purpose inputs.transcripts instead of calling ASR again. Call ASR only when these transcripts do not cover the needed interval, are contradicted/incomplete, or the task requires missing speaker attribution.",
                "",
            ]
        )

    if normalized_task_state:
        parts.extend(
            [
                "TASK_STATE:",
                pretty_json(normalized_task_state),
                "",
                "TASK_STATE_USAGE_NOTE:",
                "Use this as the canonical checklist of claim statuses, open questions, coverage, OCR occurrences, counters, referents, retrieval memory, and evidence status updates. Do not recollect already validated matching claims; do not use refuted, irrelevant, stale, or superseded evidence as support.",
                "",
            ]
        )

    if normalized_retrieval_catalog:
        parts.extend(
            [
                "RETRIEVAL_CATALOG:",
                pretty_json(normalized_retrieval_catalog),
                "",
                "RETRIEVAL_CATALOG_USAGE_NOTE:",
                "This is the text catalog the retrieval controller inspected before final planning. Use it to avoid redundant tool calls when retrieved evidence already exists.",
                "",
            ]
        )

    if normalized_retrieved_context:
        parts.extend(
            [
                "RETRIEVED_CONTEXT:",
                pretty_json(normalized_retrieved_context),
                "",
                "RETRIEVED_CONTEXT_USAGE_NOTE:",
                "Use this text-only package before broad re-search. It may include observations, evidence summaries, prior trace claims, audit gaps, and selected preprocess spans.",
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
                "Only these exact evidence_ids may be copied into generic_purpose inputs.evidence_ids for reinterpretation of previous textual evidence.",
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


def build_planner_retrieval_prompt(
    task,
    mode: str,
    retrieval_catalog: Dict[str, object],
    retrieved_context: Dict[str, object],
    audit_feedback: Optional[dict],
    tool_catalog: Dict[str, Dict[str, object]],
    iteration: int,
    max_iterations: int,
    task_state: Optional[Dict[str, object]] = None,
) -> str:
    normalized_audit_feedback = _canonicalize_audit_feedback(audit_feedback)
    normalized_retrieval_catalog = {
        key: value
        for key, value in dict(retrieval_catalog or {}).items()
        if value not in (None, "", [], {})
    }
    normalized_retrieved_context = {
        key: value
        for key, value in dict(retrieved_context or {}).items()
        if value not in (None, "", [], {})
    }
    normalized_task_state = {
        key: value
        for key, value in dict(task_state or {}).items()
        if value not in (None, "", [], {})
    }

    parts = [
        "MODE: %s" % mode,
        "RETRIEVAL_ITERATION: %s of %s" % (int(iteration), int(max_iterations)),
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

    if normalized_task_state:
        parts.extend(
            [
                "TASK_STATE:",
                pretty_json(normalized_task_state),
                "",
            ]
        )

    parts.extend(
        [
            "RETRIEVAL_CATALOG:",
            pretty_json(normalized_retrieval_catalog),
            "",
        ]
    )

    if normalized_retrieved_context:
        parts.extend(
            [
                "CURRENT_RETRIEVED_CONTEXT:",
                pretty_json(normalized_retrieved_context),
                "",
            ]
        )

    if normalized_audit_feedback:
        parts.extend(["DIAGNOSIS:", pretty_json(normalized_audit_feedback), ""])

    parts.extend(
        [
            "PlannerRetrievalDecision schema reminder:",
            "- action: \"ready\" or \"retrieve\"",
            "- rationale: short reason",
            "- requests: [] when ready",
            "- requests items contain request_id, target, need, query, modalities, time_range, source_tools, evidence_status, evidence_ids, observation_ids, limit",
            "- request narrow existing text records only; do not request raw image, audio, or video modality",
            "- return ready when existing retrieved context is enough or no relevant existing text source is available",
            "",
            "Return JSON only.",
        ]
    )
    return "\n".join(parts).strip()
