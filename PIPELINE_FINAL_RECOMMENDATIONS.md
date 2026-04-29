# Pipeline Final Recommendations

This file is the implementation-ready recommendation set after:

- prompt-only planner probes,
- deterministic policy probes,
- verifier/OCR probes,
- task-state and synthesizer handoff probes,
- review scan over saved failures,
- inventory scan over local benchmark inputs, run manifests, and original trace steps.

Supporting raw notes remain in `VERIFIER_STATE_AND_OCR_RECOMMENDATIONS.md`.
The optional deterministic plan-validator design is separated into `PIPELINE_PLAN_VALIDATOR_FALLBACK.md`.

## 1. Final Recommendation

Implement the next pipeline version as:

```text
preprocess
  -> build TASK_STATE
  -> retrieve text context from preprocess/evidence/prior trace
  -> planner emits a normal ExecutionPlan
  -> executor runs tools
  -> verifier checks answer-critical claims when needed
  -> task-state reducer updates claims/evidence/counters/referents
  -> one-shot synthesizer consumes TASK_STATE + linked evidence
  -> auditor scores the final trace and feeds the next round
```

Do not implement the deterministic plan validator in the first pass. The prompt-only probe passed the focused risky families once `TASK_STATE`, verifier schema, and stronger routing ICL were shown to the planner. Keep the validator as a fallback if broad runs show repeated unsafe plans.

Required changes for the first pass:

1. Add canonical `TASK_STATE`.
2. Add a `verifier` tool with strict claim-checking output.
3. Pass `TASK_STATE` to planner, retrieval controller, synthesizer, and auditor.
4. Update planner prompts with generic policy flows from this file.
5. Keep the synthesizer one-shot, but make it state-aware.
6. Keep PP-OCRv5 as the OCR backend. Improve crop/upscale/occurrence handling instead of changing OCR models.
7. Reuse preprocess ASR and dense-caption windows when the requested tool window exactly matches or is contained by preprocessing coverage.

## 2. Evidence For The Recommendation

Probe reports:

- `workspace/probe_results/planner_prompt_only_probe/planner_prompt_only_probe_report.json`
- `workspace/probe_results/deterministic_policy_probe/deterministic_policy_probe_report.json`
- `workspace/probe_results/task_state_system_probe/task_state_system_probe_report.json`
- `workspace/probe_results/task_state_system_probe/task_state_replan_probe_report.json`
- `workspace/probe_results/task_state_system_probe/synthesizer_state_probe_report.json`
- `workspace/probe_results/ocr_probe/paddleocr_classic_probe_report.json`
- `workspace/probe_results/ocr_probe/qwen3vl_pricepoint_probe_report.json`
- `workspace/probe_results/direct_verifier_probe/direct_verifier_Qwen__Qwen3-VL-8B-Instruct.json`

Inventory report:

- `workspace/probe_results/policy_inventory/policy_inventory_report.json`
- `workspace/probe_results/policy_inventory/policy_inventory_summary.md`

Inventory scale:

- 335 unique local tasks from input JSONs and run manifests.
- 119 review markdown files.
- The most common task patterns were temporal/visual grounding, ordering, small-object state, transcript/dialogue, referent identity, chart/math, counting, OCR, audio-event alignment, sports/game state, and speaker/tone.
- The most common review failure themes were weak referent or speaker attribution, motion/counting, lost initial-trace anchors, OCR/layout failures, non-chronological retrieval use, temporal-grounding failure, over-narrow refinement, ambiguous evidence policy, and unsupported trace prose.

The focused prompt-only planner probe passed these cases:

- validated transcript task: empty plan,
- validated chart task: empty plan,
- small visual count: `spatial_grounder -> verifier`,
- event action around sound: `audio_temporal_grounder -> frame_retriever -> spatial_grounder -> verifier`,
- referent relation: `spatial_grounder -> verifier`,
- OCR/math: `frame_retriever -> spatial_grounder -> ocr -> verifier`.

The deterministic validator probe also passed 11/11 synthetic safety cases. Because prompt-only now passed the focused set, keep deterministic validation out of the first implementation unless broader runs show it is needed.

## 3. Current Design Problems To Fix

The current pipeline is structurally much better than the older version, but these issues remain:

1. `generic_purpose` can answer or describe, but it does not produce per-claim verdicts, coverage, refutations, or evidence-status updates.
2. Evidence is stored, but it is not task-relative enough. The planner cannot reliably tell whether an evidence item is candidate, validated, refuted, irrelevant, stale, or superseded for the current claim.
3. The planner can receive useful context but still needs a structured state summary of what is already solved, what is contradicted, and what remains open.
4. Refine rounds can over-narrow after audit feedback and lose useful broader evidence from earlier rounds.
5. Retrieval results are candidates, not proof. The pipeline needs to preserve this distinction.
6. Frame retrieval is relevance-ranked unless explicitly requested otherwise. Many tasks require chronological ordering.
7. Spatial grounding sometimes returns good regions, but sometimes returns zero regions for OCR/layout tasks. OCR needs a fallback policy.
8. OCR outputs do not yet preserve enough occurrence semantics for prices, charts, scoreboards, tables, and repeated visible values.
9. The synthesizer can only safely emit an empty-plan answer if it receives explicit task state showing validated answer-critical claims.

## 4. Canonical Task State

Add a canonical task-state schema. It can live in a new module such as `video_trace_pipeline/schemas/task_state.py`.

Recommended shape:

```json
{
  "schema_version": "task_state_v1",
  "task_key": "sample id",
  "claim_results": [],
  "referent_slots": [],
  "coverage_records": [],
  "counter_records": [],
  "ocr_occurrences": [],
  "temporal_events": [],
  "evidence_status_updates": [],
  "retired_evidence": [],
  "retrieval_memory": [],
  "answer_candidates": [],
  "open_questions": [],
  "ready_for_synthesis": false
}
```

Required claim result fields:

```json
{
  "claim_id": "opt_a_claim_1",
  "option": "A",
  "text": "The claim being tested.",
  "claim_type": "transcript | visual_state | count | ocr | event_action | referent_relation | temporal_order | chart_math | speaker_tone | audio_event | option_mapping",
  "required_modalities": ["visual", "asr"],
  "status": "unverified | validated | refuted | unknown | partially_validated",
  "supporting_evidence_ids": [],
  "supporting_observation_ids": [],
  "refuting_evidence_ids": [],
  "refuting_observation_ids": [],
  "coverage_ids": [],
  "notes": ""
}
```

Required evidence status values:

- `candidate`: collected but not checked against the current claim.
- `validated`: directly supports the current claim.
- `refuted`: contradicts the current claim.
- `irrelevant`: true, but not useful for the current claim.
- `superseded`: replaced by better or more precise evidence.
- `stale`: from an older round and not revalidated after a contradiction.
- `unknown`: insufficient coverage or ambiguous output.

Bad evidence should not be deleted. Mark it with task-relative status so later rounds do not reuse it as support.

## 5. Verifier Tool

Add `verifier` as a real tool. It may use the same multimodal runner family as `generic_purpose`, but the contract must be different.

Recommended backend:

- First choice: `Qwen/Qwen3-VL-8B-Instruct` for the verifier.
- Do not use Qwen3-VL as the OCR or arithmetic engine.
- Do not use Qwen3.5 as the main verifier unless a final-JSON parser rejects reasoning-only outputs. It can be tolerated only if the parser ignores thinking/prose and requires a final structured object.

Request schema:

```json
{
  "tool_name": "verifier",
  "query": "What should be verified?",
  "claims": [
    {
      "claim_id": "claim_1",
      "text": "There are zero empty bottles on the table during the quoted line.",
      "claim_type": "visual_state",
      "expected_answer_option": "A"
    }
  ],
  "clips": [],
  "frames": [],
  "regions": [],
  "transcripts": [],
  "ocr_results": [],
  "dense_captions": [],
  "evidence_ids": [],
  "observations": [],
  "retrieved_context": {},
  "verification_policy": {
    "sampling": "chronological_dense",
    "require_direct_support": true,
    "allow_absence_evidence": true,
    "report_unknown_when_uncovered": true
  }
}
```

Output schema:

```json
{
  "claim_results": [
    {
      "claim_id": "claim_1",
      "verdict": "supported | refuted | unknown | partially_supported",
      "confidence": 0.0,
      "answer_value": null,
      "supporting_observation_ids": [],
      "supporting_evidence_ids": [],
      "refuting_observation_ids": [],
      "refuting_evidence_ids": [],
      "time_intervals": [],
      "artifact_refs": [],
      "rationale": "Short grounded reason.",
      "coverage": {
        "checked_inputs": ["frames"],
        "missing_inputs": [],
        "sampling_summary": "Checked frames 129s, 131s, 133s chronologically."
      }
    }
  ],
  "new_observations": [],
  "evidence_updates": [],
  "checklist_updates": [],
  "counter_updates": [],
  "referent_updates": [],
  "ocr_occurrence_updates": [],
  "unresolved_gaps": []
}
```

Verifier is required for these claim types:

- small-object state/count,
- subtle visual state,
- exact OCR/scoreboard/price/table/chart values,
- event-moment action,
- identity, same/different, referent, relationship,
- speaker/addressee/tone when ASR alone is insufficient,
- contradictions or audit repairs.

Verifier is usually not required when:

- preprocess transcripts fully cover a transcript-only/not-mentioned/quote task,
- a previous verifier result already validated the same claim with the same artifact/time/modality coverage,
- the task is pure deterministic arithmetic over already validated normalized values.

## 6. Planner And Retrieval Flow

Keep the current two-step retrieval-before-planning loop, but make it state-aware.

Round flow:

1. Build or load rich preprocess.
2. Build initial `TASK_STATE` from question, options, initial trace steps, and preprocess coverage.
3. Build retrieval catalog from preprocess, evidence ledger, observations, prior trace, audit gaps, and task state.
4. Seed retrieval with deterministic text matches.
5. Let planner retrieval controller request additional text records up to the existing max iterations.
6. Final planner sees:
   - question/options,
   - rich preprocess segments,
   - `TASK_STATE`,
   - retrieval catalog,
   - retrieved context,
   - latest audit,
   - tool catalog.
7. Planner emits a normal `ExecutionPlan`.

Retrieval is not proof. Retrieved observations/evidence become candidate context unless their state says `validated` for the same claim.

Planner should emit an empty plan only when:

- all answer-critical claims are already validated, or
- the task state says the remaining claims are unresolved and no safe next tool action exists.

## 7. Policy Coverage From Sample Inventory

These policies should be encoded as planner prompt rules and ICL examples. They are not sample-specific.

### 7.1 Transcript And Dialogue

Policy:

- Use preprocess transcripts first.
- Do not call ASR again when preprocess transcript coverage is sufficient.
- Use ASR only for missing coverage, contradiction, or needed speaker attribution.
- For not-mentioned/repeated-phrase tasks, preserve exact transcript spans and compare every option against the span.
- Prefer the full repeated surface phrase over a shorter embedded token when defining interval boundaries.

### 7.2 Speaker, Addressee, And Tone

Policy:

- Speaker/tone tasks need speaker slots and evidence of turn-taking, mouth activity, gaze, position, subtitles, or response behavior.
- Transcript sentiment alone is not enough for visible speaker attribution.
- Tone comparisons should compare before/after delivery for the same speaker when possible.

### 7.3 Audio Event And Cause

Policy:

- Localize the sound.
- Retrieve before/during/after chronological frames around the sound.
- Verify the direct trigger, not just the scene context.
- If the sound is conditioned on a visible object or action, ground the visible condition first, then analyze audio within that interval.
- If the question asks about sounds/noises "when" or "while" a visible object/action is used, the visual-use event is the anchor. Ground every relevant visual-use candidate first, then run audio matching inside those candidate clips and deduplicate sound types. The verifier rejects false visual-use candidates, non-use sounds, duplicates, and unrelated ambience.
- For non-speech sound-effect option comparisons, localize audio over the full bounded interval using option-aware query terms before choosing a visual/source window. Do not choose a single visually convenient segment from preprocessing unless validated evidence already proves it is the relevant sound.

### 7.4 Action At Timestamp

Policy:

- Exact timestamps are anchors, not isolated proof.
- Retrieve the anchor frame plus neighboring frames.
- Use chronological sequences for actions, motion, state changes, identity, and count.

### 7.5 Temporal Order, First, Last, And Sequence

Policy:

- Retrieve candidate events across the bounded interval.
- Sort by timestamp.
- Never infer chronology from retrieval result order.
- Preserve initial-trace timestamps as hard candidate anchors unless contradicted.
- Do not let refine rounds discard broader prior context unless newer evidence refutes it.

### 7.6 Counting And Motion Cycles

Policy:

- Define inclusion and exclusion rules before counting.
- Store accepted and rejected candidates.
- Dense chronological clips are preferred over sparse top-k frames.
- Count complete cycles/events, not frames that merely contain the object.

### 7.7 Small Object State And Attribute

Policy:

- Use frames/regions/crops before verification.
- Treat broad dense captions as candidate context, not exact proof.
- Absence is valid only when coverage is sufficient for the interval and modality.

### 7.8 OCR, Scoreboard, Label, And Layout

Policy:

- Use region/crop/upscale where possible.
- Preserve label-value adjacency.
- For scoreboards, sample after the event update, not only at the first zero-clock frame.
- For signs, labels, nameplates, plates, blackboards, whiteboards, visible letters/words, boards, and control panels, keep raw OCR lines and source artifacts.
- If spatial grounding returns empty regions, do not call OCR with empty media. Retry broader regions or run full-frame OCR.

### 7.9 Chart, Table, Price, And Math

Policy:

- For progressive displays, decide if the task asks for initial state, final state, change, max/min, or stable complete display.
- Use one stable representative frame per required display when possible.
- Store OCR/value occurrences with source artifact, bbox, normalized value, and ambiguity.
- Product/item deduplication must be explicit. Some questions ask for occurrences, not distinct products.
- Arithmetic should be deterministic over normalized validated values, not free-form VLM math.
- Arithmetic/date tasks must verify each operand separately. If one operand is absent from transcript/preprocess text but likely visible on a sign, plaque, board, chart, scoreboard, or overlay, retrieve that source and OCR it before computing.

Recommended OCR occurrence object:

```json
{
  "occurrence_id": "occ_1",
  "kind": "price | score | chart_value | label | text",
  "raw_text": "$19.82",
  "normalized_value": 19.82,
  "nearby_label": "item name or team label",
  "source_artifact_id": "frame_81.00_crop",
  "bbox": [0, 0, 100, 50],
  "confidence": 0.0,
  "status": "candidate | verified | ambiguous | rejected",
  "dedupe_key": null
}
```

### 7.10 Referent Identity And Relationship

Policy:

- Create separate referent slots for each person/object/speaker mentioned in the question.
- Resolve ordinal referents over the full question scope.
- Ground same/different/relationship after all slots have evidence.
- For relationship tasks, carry relevant transcripts into the final verifier/generic call with the visual slots.

### 7.11 Sports And Game State

Policy:

- Prefer scoreboards, final celebration, handshake, trophy, replay, or official overlay for winner/result questions.
- If answer depends on team color, explicitly link uniform color to the winning/labeled team.
- For replay questions, use initial trace anchors and ordinal replay order rather than a broad semantic temporal search.

### 7.12 Ambiguous Option Mapping

Policy:

- Multiple-choice tasks sometimes use approximate wording. If the event and role match but object taxonomy is ambiguous, map to the closest option and state the ambiguity.
- Do not turn harmless synonym mismatch into unresolved output when one option is clearly better supported.
- Do not silently convert ambiguous values, such as a decade to an exact year, without recording the convention.

## 8. Prompt Changes

Add these planner prompt policies and examples. Keep them generic.

Planner rules to add or keep:

```text
For multiple-choice tasks, decompose each option into atomic claims before choosing tools. Plan evidence collection so every live option is supported, refuted, or left unknown for a stated coverage reason.

When retrieved evidence or preprocessing suggests an answer but does not directly prove it, call verifier with the candidate claim and the original media/transcript/OCR artifacts.

For first/last/before/after/count/order questions, collect candidate events across the bounded interval, sort by timestamp, and verify the requested occurrence.

Only treat absence as evidence if the relevant interval, modality, and sampling coverage are sufficient.

For relationship, identity, same/different, speaker, addressee, first/last object, or named-object questions, create separate referent slots and resolve each slot before comparing them.

For progressive charts, animated tables, scoreboards, or changing displays, decide whether the task asks for initial state, final state, peak/minimum, or change over time before selecting frames.

For repeated motion or event counts, define inclusion/exclusion rules before counting, retrieve dense chronological evidence, and store accepted/rejected candidate events.
```

Planner ICL patterns:

```text
Small visual count/state:
Good: retrieve chronological frames, localize/crop target objects, then verifier checks each option claim.
Bad: answer from broad dense-caption text or generic scene prose.

Sound cause:
Good: audio event -> before/during/after frames -> verifier of direct trigger.
Bad: inspect only frames after the sound and infer the cause from scene context.

Referent relation:
Good: resolve each referent slot in its own temporal scope, then compare relationship.
Bad: inspect only the named-object window and assume the first local object is the first object in the whole question.

OCR/math:
Good: preserve source artifact, region, raw lines, normalized values, ambiguity, and occurrence-vs-distinct semantics.
Bad: ask a VLM to read many near-duplicate frames and do arithmetic in prose.

Refine after audit:
Good: preserve the accepted timestamp/evidence and target only the missing attribute.
Bad: broad re-search that discards prior useful anchors.
```

Synthesizer rules:

```text
Consume TASK_STATE directly.
If all answer-critical claims are validated, synthesize from linked evidence even when the plan had no steps.
If any answer-critical claim remains unverified, unknown, or partially validated with open questions, leave final_answer empty.
Do not use candidate, stale, irrelevant, or refuted evidence as support.
Mention ambiguity or option-mapping convention explicitly when needed.
```

Auditor rules:

```text
Score whether the trace follows TASK_STATE, not only whether prose sounds plausible.
Flag unsupported answers that rely on candidate/stale/refuted evidence.
Do not fail a correct trace solely because a visual synonym is not exact when the option mapping is otherwise clearly best supported.
For action-at-timestamp tasks, allow neighboring-frame evidence because action spans time.
```

## 9. OCR Recommendation

Keep current PaddleOCR/PP-OCRv5 as the OCR backend.

Do not adopt these as production OCR backends from the current probes:

- PaddleOCR-VL,
- Nanonets-OCR-s,
- GOT-OCR-2.0,
- Qwen3-VL as a standalone OCR/math engine.

Improve OCR by:

1. Region crop and upscale.
2. Full-frame OCR fallback when regions are empty.
3. Tiling for high-resolution displays.
4. Numeric normalization for scores, prices, percentages, and chart values.
5. Occurrence schema.
6. Verifier check for ambiguous OCR-derived claims.

## 10. Concrete Implementation Plan

### 10.1 Schemas

Add:

- `video_trace_pipeline/schemas/task_state.py`
- `VerifierRequest` in `video_trace_pipeline/schemas/tool_requests.py`
- `VerifierOutput`, `VerifierClaimResult`, and OCR occurrence models in `video_trace_pipeline/schemas/tool_outputs.py`
- exports in `video_trace_pipeline/schemas/__init__.py`

Update existing trace/evidence schemas only as needed to store task-state links and richer evidence-status metadata.

### 10.2 Tool Registry And Config

Add `verifier` to:

- `configs/models.yaml`
- `video_trace_pipeline/tools/specs.py`
- `video_trace_pipeline/tools/registry.py`
- `video_trace_pipeline/tools/process_adapters.py`

Create a verifier runner, likely:

- `video_trace_pipeline/tool_wrappers/verifier_runner.py`

The runner should:

- accept all input modalities,
- format claims and evidence,
- require final JSON,
- parse `<final>...</final>` JSON if present,
- reject prose-only outputs,
- store raw output and parse status.

### 10.3 Task State Builder And Reducer

Add a task-state builder/reducer module, for example:

- `video_trace_pipeline/orchestration/task_state.py`

Responsibilities:

- initialize option claims from task/options/initial trace,
- detect claim types with deterministic heuristics plus optional LLM claim typing,
- incorporate preprocess transcript coverage,
- update state from tool outputs and verifier outputs,
- update evidence status after audit,
- compute `ready_for_synthesis`.

### 10.4 Pipeline Integration

Update `video_trace_pipeline/orchestration/pipeline.py`:

- build `TASK_STATE` after preprocess,
- pass task state into retrieval catalog and retrieval prompts,
- pass task state into planner prompt,
- after execution, feed verifier/tool outputs to reducer,
- pass task state into synthesizer prompt,
- pass task state into auditor prompt,
- persist per-round `task_state_before_plan.json` and `task_state_after_round.json`.

### 10.5 Retrieval

Update `video_trace_pipeline/orchestration/planner_retrieval.py`:

- index task-state claims, referents, coverage, retired evidence, and open questions,
- let retrieval requests target `task_state`,
- avoid returning retired/refuted evidence as support unless the need is contradiction analysis,
- preserve whether retrieved context is `candidate` or `validated`.

### 10.6 Prompts

Update:

- `video_trace_pipeline/prompts/planner_prompt.py`
- `video_trace_pipeline/prompts/trace_synthesizer_prompt.py`
- `video_trace_pipeline/prompts/trace_auditor_prompt.py`

Keep ICL generic. Do not use actual sample IDs or benchmark-specific answers in production prompts.

### 10.7 Executor And Evidence

Update:

- `video_trace_pipeline/orchestration/executor.py`
- `video_trace_pipeline/storage/evidence_store.py`
- `video_trace_pipeline/tools/extractors.py`

Needed behavior:

- verifier outputs become observations and evidence updates,
- invalid OCR media remains a state gap, not a crash,
- cached verifier/tool outputs are reused,
- evidence entries retain task-relative status in metadata or state exports.

## 11. Tests To Add

Add focused tests before running full benchmarks:

- schema rejects old plan fields and accepts verifier schemas,
- planner receives `TASK_STATE` and emits empty plan for already validated transcript/chart cases,
- planner emits verifier for small visual state/count, sound-event action, referent relation, and OCR/math families,
- preprocess transcripts prevent redundant ASR calls,
- ASR output is passed as `transcripts`, never flattened text,
- OCR with empty regions triggers fallback state, not downstream empty-media OCR,
- verifier updates claim status, counter records, referent slots, OCR occurrences, and evidence status,
- synthesizer answers from validated state with no new tools,
- synthesizer leaves `final_answer` empty when answer-critical state remains unknown,
- retrieved candidate evidence is not treated as validated support,
- prior validated evidence prevents redundant tool calls,
- chronological tasks sort frames/events by timestamp.

Recommended focused cases:

- transcript not-mentioned/repeated-phrase,
- small object empty/full/count,
- loud sound cause/action,
- multi-referent same/different/relation,
- scoreboard post-update,
- price/list occurrence math,
- replay/ordinal event,
- motion cycle count,
- exact timestamp action with neighbors,
- ambiguous synonym option mapping.

## 12. Expected Impact And Limits

This should fix most observed pipeline failures caused by:

- weak proof accepted as evidence,
- repeated redundant ASR/tool calls,
- lost initial-trace anchors,
- non-chronological frame reasoning,
- over-narrow refine rounds,
- small object state/count without crops,
- referent and speaker confusion,
- OCR occurrence/value ambiguity,
- stale or refuted evidence reused as support.

It will not guarantee every answer is correct. Remaining hard failures can still come from:

- temporal grounder never finding the relevant interval,
- spatial grounder failing to box tiny/blurred targets,
- OCR text being unreadable,
- verifier perception failure,
- wrong claim typing,
- tasks needing domain knowledge not in the video.

If broad runs still show unsafe planner plans after these changes, enable the separate plan-validator fallback in `PIPELINE_PLAN_VALIDATOR_FALLBACK.md`.
