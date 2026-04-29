# Verifier, Pipeline State, Prompt Flow, And OCR Recommendations

## 0. Probe-Backed Recommendation Summary

Recommendation: implement the verifier/state changes, but do not replace OCR with a new model backend yet.

Temporary probe reports:

- `workspace/probe_results/direct_verifier_probe/direct_verifier_Qwen__Qwen3-VL-8B-Instruct.json`
- `workspace/probe_results/direct_verifier_probe/qwen3vl_targeted_followups.json`
- `workspace/probe_results/direct_verifier_probe/qwen35_thinking_allowed_probe.json`
- `workspace/probe_results/ocr_probe/paddleocr_vl_probe_report.json`
- `workspace/probe_results/ocr_probe/paddleocr_vl_bf16_probe_report.json`
- `workspace/probe_results/ocr_probe/paddleocr_classic_probe_report.json`
- `workspace/probe_results/ocr_probe/qwen3vl_ocr_probe_report.json`
- `workspace/probe_results/ocr_probe/qwen3vl_pricepoint_probe_report.json`
- `workspace/probe_results/ocr_probe/nanonets_ocr_probe_report.json`
- `workspace/probe_results/ocr_probe/got_ocr_probe_report.json`
- `workspace/probe_results/planner_probe/planner_api_probe_report.json`

Recommended implementation order:

1. Add `verifier` with a strict structured contract.
2. Add task state for claims, referents, counters, coverage, contradictions, and evidence status.
3. Add modality routing before verification.
4. Improve OCR by deterministic frame tiling/cropping/upscaling plus structured OCR normalization.
5. Use a VL verifier to check OCR-derived claims, not as the sole OCR or arithmetic engine.
6. Do not use PaddleOCR-VL, Nanonets-OCR-s, or GOT-OCR-2.0 as the production OCR backend based on the current probes.

Probe-backed model recommendation:

- Use `Qwen/Qwen3-VL-8B-Instruct` as the first verifier backend candidate.
- Do not use `Qwen/Qwen3.5-9B` as the main verifier backend. It produced useful reasoning and can be parsed when it emits a final JSON block, but it did not reliably finish structured visual outputs.
- Do not use Qwen3-VL as a standalone exact OCR/math engine. It improved with zoomed crops, but still misread some prices and produced an incorrect arithmetic sum.
- Keep PP-OCRv5-style deterministic OCR as the primary OCR extractor for now, with better tiling/upscaling and numeric postprocessing.
- Treat Qwen thinking content as discardable reasoning. The parser should accept only the final structured answer and reject prose-only or reasoning-only outputs.

Probe-backed behavior requirements:

- Transcript-only claims must be verified with transcript-only inputs unless visual evidence is actually needed.
- Small visual-state/count questions must use zoomed crops or regions before verification.
- Identity, relationship, speaker, same/different, and first/last object questions need referent slots before final answer selection.
- Price/table/chart tasks need price-point or cell-level occurrence tracking, not only product-level deduplication.
- Evidence should not be deleted when wrong; it should be marked `refuted`, `irrelevant`, or `superseded` for the relevant claim.

## 1. Add A General Verifier Tool

Recommendation: add a verifier tool that can accept any pipeline input type and verify claims against the supplied evidence.

The verifier should not be a free-form generic answerer. It should be a strict claim-checking agent.

Recommended tool name:

```json
{
  "tool_name": "verifier"
}
```

Recommended request schema:

```json
{
  "query": "What should be verified?",
  "claims": [
    {
      "claim_id": "claim_1",
      "text": "The visible object is empty at the quoted line.",
      "expected_answer_option": "A",
      "claim_type": "visual_state"
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

Recommended output schema:

```json
{
  "claim_results": [
    {
      "claim_id": "claim_1",
      "verdict": "supported | refuted | unknown | partially_supported",
      "confidence": 0.0,
      "supporting_observation_ids": [],
      "supporting_evidence_ids": [],
      "refuting_observation_ids": [],
      "refuting_evidence_ids": [],
      "time_intervals": [],
      "artifact_refs": [],
      "rationale": "Short grounded reason.",
      "coverage": {
        "checked_inputs": ["frames", "transcripts"],
        "missing_inputs": [],
        "sampling_summary": "Checked frames from 120.0-150.0s chronologically."
      }
    }
  ],
  "new_observations": [],
  "evidence_updates": [],
  "checklist_updates": [],
  "counter_updates": [],
  "unresolved_gaps": []
}
```

Why this should exist:

- Many failures are not caused by missing tools; they are caused by accepting weak evidence as if it proved the final answer.
- The current `generic_purpose` tool can describe or answer, but it does not guarantee per-claim verdicts, coverage, explicit rejection, or unknown handling.
- A verifier can reuse the same underlying multimodal model as `generic_purpose`, but the contract must be different.
- The planner should use the verifier when it already has candidate evidence and needs to decide whether each answer option is actually supported.

Recommended first implementation:

- Make `verifier` a thin adapter over the same local multimodal runner interface used by the existing generic tool.
- Configure the first probe-backed backend as `Qwen/Qwen3-VL-8B-Instruct` when GPU memory allows.
- Change only the request/output contract and prompt.
- Require structured claim results.
- Require every supported/refuted verdict to cite concrete timestamps, input artifacts, observations, or evidence ids.
- Treat "not visible", "not covered", and "not enough frames" as `unknown`, not as refutation.
- Use a robust parser that accepts clean JSON and `<final>...</final>` wrapped JSON; reject prose-only outputs.
- Store the verifier's raw output, parsed result, and parse status separately.

Recommended later implementation:

- Swap the backend to a stronger video-capable verifier model if the generic backend remains weak at motion, counts, absence, or identity.
- Keep the same verifier schema so the rest of the pipeline does not change.

## 2. Add More Pipeline State Than Checklist And Counter

Checklist and counter are necessary, but not enough.

Recommended canonical task state:

```json
{
  "task_claims": [],
  "checklist": [],
  "counters": [],
  "referents": [],
  "temporal_map": [],
  "coverage_map": [],
  "evidence_status": [],
  "contradictions": [],
  "retrieval_memory": [],
  "answer_candidates": [],
  "open_questions": []
}
```

### 2.1 Task Claims

Recommendation: decompose each task into atomic claims before planning.

For multiple choice questions, each option should become one or more claims.

Example:

```json
{
  "claim_id": "opt_b_claim_1",
  "option": "B",
  "text": "The event happens before the person enters the room.",
  "claim_type": "temporal_order",
  "required_modalities": ["visual"],
  "status": "unverified"
}
```

Why:

- The planner often searches for a related event but does not test every option.
- A claim ledger makes the missing discriminator explicit.

### 2.2 Checklist

Recommendation: use checklist items for required evidence-gathering actions.

Example:

```json
{
  "item_id": "check_1",
  "need": "Verify the object state at the quoted line timestamp.",
  "status": "open",
  "linked_claim_ids": ["opt_a_claim_1", "opt_b_claim_1"]
}
```

Why:

- The checklist tracks work still needed.
- The claims track what answer facts need support.
- These are related but should not be the same object.

### 2.3 Counters

Recommendation: counters should be explicit state, not hidden in evidence prose.

Example:

```json
{
  "counter_id": "count_1",
  "target": "complete object appearances",
  "inclusion_rule": "count only fully visible completed appearances",
  "exclusion_rule": "exclude partial, repeated, or ambiguous frames",
  "observed_events": [],
  "count": null,
  "status": "open"
}
```

Why:

- VideoMathQA and Minerva-style tasks often need explicit counting rules.
- The system currently stores observations but does not maintain a count ledger with inclusion/exclusion decisions.

### 2.4 Referents

Recommendation: add referent slots for people, objects, locations, speakers, and answer-critical entities.

Example:

```json
{
  "referent_id": "person_1",
  "description": "the person holding the first object",
  "scope": "first object occurrence over the full video",
  "candidate_observations": [],
  "resolved_observation_id": null,
  "status": "unresolved"
}
```

Why:

- Several failures came from mixing up people, dogs, speakers, or "first" versus later entities.
- The planner should not answer relationship questions until each referent slot is resolved separately.

### 2.5 Temporal Map

Recommendation: maintain a temporal map of candidate events and intervals.

Example:

```json
{
  "event_id": "event_1",
  "description": "first time the object becomes visible",
  "start_s": 10.0,
  "end_s": 13.0,
  "source": "visual_temporal_grounder",
  "status": "candidate",
  "linked_claim_ids": []
}
```

Why:

- Retrieval order is not chronology.
- First/last/before/after questions require sorting grounded candidates by timestamp.

### 2.6 Coverage Map

Recommendation: track what was actually checked.

Example:

```json
{
  "coverage_id": "coverage_1",
  "input_type": "frames",
  "time_range": {"start_s": 120.0, "end_s": 150.0},
  "sampling": "1 fps chronological",
  "checked_by": "verifier",
  "coverage_status": "sufficient_for_local_state"
}
```

Why:

- A negative result only means something if the relevant interval was actually covered.
- This prevents "not seen in sparse frames" from becoming false evidence.

### 2.7 Evidence Status

Recommendation: evidence should have explicit task-relative status.

Use these statuses:

- `candidate`: collected but not checked against the current claim.
- `validated`: directly supports a claim.
- `refuted`: directly contradicts a claim.
- `superseded`: replaced by better or more precise evidence.
- `irrelevant`: true observation, but not useful for the current task.
- `stale`: from an earlier round and not yet revalidated after a contradiction.

Why:

- Evidence can be true but irrelevant.
- Evidence can be useful for one claim and useless for another.
- The planner needs to know whether evidence should be reused, challenged, or ignored.

### 2.8 Contradictions

Recommendation: store contradictions as first-class state.

Example:

```json
{
  "contradiction_id": "contra_1",
  "claim_id": "opt_c_claim_1",
  "old_evidence_id": "ev_12",
  "new_evidence_id": "ev_21",
  "reason": "New OCR reads 55-40, while old evidence read an earlier 55-38 scoreboard.",
  "resolution": "prefer_newer_stable_display",
  "status": "resolved"
}
```

Why:

- The pipeline should not simply accumulate evidence.
- It should remember when earlier evidence was too broad, stale, or contradicted.

### 2.9 Retrieval Memory

Recommendation: keep retrieval memory separate from evidence.

Example:

```json
{
  "retrieval_id": "ret_1",
  "query": "scoreboard value after third quarter ends",
  "results": [],
  "used_result_ids": [],
  "unused_result_ids": [],
  "reason_unused": "Results were before the final score update."
}
```

Why:

- Retrieval results are candidates, not evidence.
- The planner should know what it already searched so it does not repeat the same weak query.

### 2.10 Answer Candidates

Recommendation: maintain answer candidates with support/refutation.

Example:

```json
{
  "option": "B",
  "supporting_claim_ids": [],
  "refuting_claim_ids": [],
  "unknown_claim_ids": [],
  "status": "possible | ruled_out | selected"
}
```

Why:

- This helps the synthesizer avoid blank answers when only one option remains possible.
- It also prevents premature selection when multiple options remain compatible.

## 3. Evidence Should Be Updated, Not Only Checklist

Recommendation: when the system learns that evidence is useless or false for the current task, update evidence state.

Do not delete old evidence.

Instead:

- Mark it `irrelevant` if it is true but does not answer the claim.
- Mark it `superseded` if a later or more precise observation replaces it.
- Mark it `refuted` if another observation contradicts it.
- Link the status change to the claim/checklist item that caused the update.

Recommended evidence update record:

```json
{
  "evidence_id": "ev_old",
  "previous_status": "candidate",
  "new_status": "irrelevant",
  "claim_id": "opt_b_claim_1",
  "reason": "The evidence shows a related object, but not the object named in the option.",
  "updated_by": "verifier",
  "round": 2
}
```

Why:

- Checklist says what remains to be done.
- Evidence status says what can or cannot be trusted.
- Both are needed.

## 4. Planner Prompt Recommendations

Recommendation: add more generic flow examples, not sample-specific examples.

### 4.1 Option Claim Decomposition

Add a planner rule:

```text
For multiple-choice tasks, decompose each option into atomic claims before choosing tools. Plan evidence collection so every live option is either supported, refuted, or left unknown for a stated coverage reason.
```

Add ICL pattern:

```text
Good: Convert options into visual/audio/text claims, retrieve evidence for each claim, then compare claim_results.
Bad: Search for only the option that seems likely from the first caption.
```

### 4.2 Verifier After Retrieval

Add a planner rule:

```text
When retrieved evidence or preprocessing suggests an answer but does not directly prove it, call the verifier with the candidate claim and the original media/transcript artifacts.
```

Add ICL pattern:

```text
Good: retrieved caption says "a dog appears"; verifier checks whether it is the same dog/person/referent required by the question.
Bad: treat broad caption text as proof of exact identity.
```

### 4.3 Temporal Candidate Sweep

Add a planner rule:

```text
For first/last/before/after/count questions, retrieve candidate events across the full bounded interval, sort by timestamp, and verify the requested occurrence.
```

Add ICL pattern:

```text
Good: collect all candidate events, sort them chronologically, then verify the first/last.
Bad: use the first retrieved semantic match as the first chronological event.
```

### 4.4 Coverage-Aware Negative Evidence

Add a planner rule:

```text
Only treat absence as evidence if the relevant interval, modality, and sampling coverage are sufficient.
```

Add ICL pattern:

```text
Good: "unknown because only sparse frames were checked."
Bad: "not present" because one sampled frame did not show it.
```

### 4.5 Referent Slot Resolution

Add a planner rule:

```text
For relationship, identity, same/different, speaker, addressee, first/last object, or named-object questions, create separate referent slots and resolve each slot before comparing them.
```

Add ICL pattern:

```text
Good: resolve "person with first object" and "person with named object" as separate slots, then compare.
Bad: assume two visually nearby people are the same or different without slot evidence.
```

### 4.6 Progressive Chart/Table Displays

Add a planner rule:

```text
For progressive charts, animated tables, scoreboards, or changing displays, decide whether the task asks for initial state, final state, peak/minimum, or change over time before selecting frames.
```

Add ICL pattern:

```text
Good: retrieve chronological display frames, choose the stable final frame if the question asks for final values, then read label-value pairs.
Bad: use an early partially rendered display as if missing values are zero.
```

### 4.7 Motion And Event Counter

Add a planner rule:

```text
For repeated motion or event counts, define inclusion/exclusion rules before counting, retrieve dense chronological evidence, and store accepted/rejected candidate events.
```

Add ICL pattern:

```text
Good: count complete cycles and reject partial/duplicate candidates.
Bad: count every retrieved frame containing the object.
```

## 5. OCR Recommendation

Recommendation: do not adopt PaddleOCR-VL, Nanonets-OCR-s, or GOT-OCR-2.0 as the pipeline OCR backend based on the current probes.

Current recommendation:

- Use the current PP-OCRv5-style OCR path as the primary OCR extractor for now, but add deterministic tiling, crop/upscale passes, and stricter numeric normalization.
- Use the verifier to check OCR-derived answer claims after OCR extraction, especially for totals, option matching, and ambiguous digits.
- Keep OCR model replacement as an evaluation gate, not a prompt change.

Why:

- PaddleOCR-VL is likely a better model class for charts, tables, scoreboards, labels, and layout-heavy frames, but model quality is irrelevant unless inference is reliable and GPU-backed.
- PaddleOCR-VL failed in the target Slurm/Paddle runtime with a GPU kernel dtype issue and was removed from the Hugging Face cache after the probe.
- Nanonets-OCR-s loaded but produced gibberish on the sampled video frames/crops in this environment.
- GOT-OCR-2.0 loaded and ran, but its outputs on the price-point case were noisier and less useful than PP-OCRv5-style OCR.
- The probe on the Minerva price case showed that Qwen3-VL can help when given zoomed crops, but it still misread exact prices and produced a wrong sum. It should verify claims, not replace OCR extraction.
- The same probe showed that classic OCR on upscaled crops recovered many useful raw tokens (`$1982`, `$1126`, `$817`, `$1741`, `$612`, `$259%`), so the immediate improvement should be better frame preprocessing and price normalization rather than a new OCR backend.

Runtime requirements before any new OCR backend adoption:

- Install the candidate runtime in the actual pipeline environment, not only ad-hoc temp deps.
- Confirm the backend uses CUDA on the Slurm job.
- Confirm one-frame inference completes on GPU.
- Confirm output contains structured text or markdown that can be normalized into the pipeline OCR output schema.
- Compare against current OCR on known OCR failure frames.

Recommended acceptance test:

```text
Use 10-20 known OCR/layout failure frames.
For each frame, compare:
1. exact visible text,
2. label-value pairing,
3. table/chart structure,
4. scoreboard/state update correctness,
5. runtime latency,
6. GPU memory and CPU memory use.
Adopt only if the candidate wins on correctness and runs reliably.
```

Recommended OCR output schema addition:

```json
{
  "price_points": [
    {
      "occurrence_id": "price_1",
      "text": "$19.82",
      "normalized_value": 19.82,
      "nearby_item_text": "Dixie ITW Brands 25316 50PK#50 Stud Dry Anchor",
      "source_artifact_id": "frame_81.00_crop_grid",
      "bbox": null,
      "confidence": 0.0,
      "status": "candidate | verified | ambiguous | rejected"
    }
  ],
  "raw_ocr_lines": [],
  "ambiguous_tokens": [],
  "normalization_notes": []
}
```

Why:

- Product deduplication is not always correct. Some questions ask for every visible price-point occurrence, even if the same product appears again on a detail page.
- Exact arithmetic should be performed from normalized numeric fields, not from a VL model's free-form sum.
- Ambiguous OCR tokens should remain visible to the verifier and synthesizer instead of being silently coerced.

Recommended fallback while new OCR backends remain operationally weak:

- Use a strong general VLM as the OCR verifier for hard frames.
- Keep classic OCR for cheap/simple text detection.
- Route only difficult frames to the expensive VLM OCR verifier.

## 6. Latest Probe-Backed Adjustments

These are the changes I would make to the recommendations after testing failed samples, OCR candidates, and planner behavior.

### 6.1 What Was Tried

- Qwen3-VL verifier probes on failed OmniVideoBench-style cases: empty bottles, loud bang action, Magnolia/dog relationship, and premature-lie transcript reasoning.
- Qwen3.5 thinking-allowed verifier probe with a parser that ignores reasoning and accepts only final JSON.
- OCR probes on the Minerva price-point video `7eF4EjdYZ7k`: PP-OCRv5-style classic OCR, Qwen3-VL OCR, Qwen3-VL price-point extraction, PaddleOCR-VL, Nanonets-OCR-s, and GOT-OCR-2.0.
- Planner prompt/schema probe through the API on failed OmniVideoBench cases plus Minerva price/counter cases.

### 6.2 What Worked

- Qwen3-VL verifier correctly fixed several failure classes when it received the right evidence: transcript-only truth checking, zoomed crop checks for small objects, and pre/post-event visual frames.
- PP-OCRv5-style OCR with upscaled crops recovered useful price tokens from the Minerva price case, even when full-frame VLM OCR hallucinated exact values.
- The current planner schema changes worked structurally: the probe did not emit old `arguments`, `depends_on`, `use_summary`, list-style `input_refs`, or `planner_context`.
- Qwen3.5 thinking can be tolerated only if the runtime parser discards reasoning and requires a final structured payload.

### 6.3 What Failed

- PaddleOCR-VL failed in the target environment with a Paddle GPU dtype/kernel issue and should stay out of the pipeline.
- Nanonets-OCR-s produced unusable gibberish on the tested video frames/crops.
- GOT-OCR-2.0 produced partial tokens but was worse than PP-OCRv5-style OCR on the price-point case.
- Qwen3-VL should not be trusted as the arithmetic engine for OCR totals.
- Qwen3.5 should not be trusted as the main visual verifier because it may spend the output budget on reasoning and fail to emit final JSON.

### 6.4 Planner Gaps Still Present

The planner is schema-compliant, but not yet semantically strong enough.

Needed generic prompt/routing changes:

- Small visual-state/count questions should require crop, region, zoom, or verifier evidence before final answer selection.
- Event-centered questions should retrieve and verify both sides of a localized event, not only frames after the event timestamp.
- Price/table/math questions should track visible price-point or cell occurrences. Product deduplication should be an explicit task decision, not the default.
- Relationship, speaker, addressee, same/different, and first/last referent questions should resolve referent slots before comparing options.
- Counting tasks should maintain accepted/rejected candidates and coverage, not just a prose observation.
- Retrieved context should be treated as candidates until verified against the specific claim.

### 6.5 New State Recommendation

Add these task-state records in addition to checklist and counters:

- `claim_results`: per-option support/refutation/unknown state.
- `referent_slots`: people, objects, speakers, addressees, and answer-critical entities.
- `coverage_records`: what time ranges and modalities were actually checked.
- `evidence_status_updates`: candidate, validated, refuted, irrelevant, superseded, or stale.
- `price_point_occurrences`: visible OCR value occurrences with source artifact and normalization status.
- `retired_evidence`: evidence known to be false, stale, or irrelevant for the current claim.

The planner should see these states before each planning round. It should not delete bad evidence; it should mark it so later rounds do not reuse it as if it were still valid.

## 7. Expected Impact

These changes should fix many of the current failure classes:

- wrong option because only one option was searched,
- broad caption treated as exact proof,
- stale evidence reused after a better observation exists,
- referent/person/object confusion,
- first/last/order mistakes,
- count mistakes from sparse frames,
- false absence claims,
- OCR/layout mistakes,
- blank answers when one option is clearly best supported.

These changes will not fix everything by themselves.

Remaining hard cases:

- the underlying visual model cannot perceive the needed detail,
- visual temporal grounding never retrieves the relevant interval,
- OCR text is too small or blurred in all sampled frames,
- audio/speaker attribution is genuinely ambiguous,
- the task requires domain knowledge outside the video.

The highest-leverage order is:

1. Add task claims, referents, temporal map, coverage map, and evidence status.
2. Add verifier tool contract over the existing generic backend.
3. Add planner ICL/rules for option claims, coverage, referents, chronology, counters, and progressive displays.
4. Add evidence status updates and contradiction records.
5. Upgrade OCR only after the runtime is proven reliable.

## 8. Final Concrete Recommendations After Comparison Probe

Additional probe report:

- `workspace/probe_results/final_recommendation_probe/planner_comparison_report.json`

Additional validation:

- `pytest -q tests/test_schemas.py tests/test_plan_normalizer.py tests/test_executor_cache.py tests/test_planner_retrieval.py tests/test_prompt_builders.py`
- Result: 38 passed.

The comparison probe ran current planner prompts and temporary proposed rules side by side on correct transcript/chart/scoreboard/counter samples plus known failure samples. The result supports the changes, but also shows that verifier routing must be gated.

Final implementation recommendations:

1. Add `verifier` as a real tool, but make it claim-checking only.
   - It should verify supplied claims against supplied clips, frames, regions, transcripts, OCR results, observations, and evidence.
   - It should not replace `generic_purpose` as a general answerer.
   - It should output claim verdicts, coverage, supporting/refuting evidence, unresolved gaps, and evidence status updates.

2. Add task state before adding more prompt text.
   - Required state: `claim_results`, `referent_slots`, `coverage_records`, `counter_records`, `price_point_occurrences`, `evidence_status_updates`, `retired_evidence`, and `open_questions`.
   - This is more important than more free-form planner prose because the planner needs a structured memory of what was verified, rejected, stale, or still open.

3. Add verifier routing gates.
   - Always use verifier for small object state/count, subtle visual state, exact OCR/price/table/scoreboard values, event-moment action, identity/same-different/referent, speaker/addressee, and contradiction repair.
   - Skip verifier for transcript-only tasks when preprocess transcripts cover the needed interval and no speaker attribution or contradiction is needed.
   - Skip verifier when a previous evidence item is already validated for the same claim, same artifact/time, and same modality coverage.

4. Keep the planner simple for easy cases.
   - The probe showed the proposed rules did not over-tool the transcript-only correct case.
   - But proposed rules did add verifier to already-correct chart/scoreboard/counter cases, so the implementation needs explicit cost/risk gates rather than "always verify everything."

5. Add planner ICL/rules as routing examples, not sample-specific examples.
   - Small object/count: retrieve chronological frames, spatially ground/crop the target, then verify the exact claim.
   - Event action: localize audio/visual event, retrieve before/during/after frames around the localized interval, then verify option claims.
   - Referent relation: resolve each referent slot separately, then verify the relationship claim.
   - OCR/math: preserve source artifact, region, raw OCR lines, normalized values, ambiguity, and whether the task asks for occurrences or distinct entities.
   - Counting: maintain accepted/rejected candidates and coverage.

6. Do not replace OCR yet.
   - Keep PP-OCRv5-style OCR as primary.
   - Add deterministic tiling, crop/upscale passes, numeric normalization, and price/cell occurrence schema.
   - Use verifier for OCR-derived claims and ambiguous digits.
   - Do not use PaddleOCR-VL, Nanonets-OCR-s, or GOT-OCR-2.0 as production OCR from the current probes.

7. Add focused implementation tests before full benchmark runs.
   - Verifier request/output schema rejects prose-only and missing verdicts.
   - Planner emits verifier only for gated risky claim types.
   - Planner does not call ASR when preprocess transcript coverage is sufficient.
   - Event plans retrieve both before and after the localized event.
   - OCR/price plans preserve occurrence-vs-distinct semantics.
   - Executor passes verifier outputs into evidence status updates and synthesizer context.
   - Invalid verifier/tool wiring fails before downstream tool execution.

The final recommendation is to implement verifier + state + routing gates first, then prompt ICL, then OCR preprocessing/schema improvements. Do not start by swapping OCR models or adding more ungated prompt text.

## 9. Task-State Prototype And OCR Region Wiring Probe

Additional temporary probes:

- `workspace/probe_results/task_state_system_probe/task_state_system_probe_report.json`
- `workspace/probe_results/task_state_system_probe/task_state_replan_probe_report.json`

Additional OCR-region validation:

- `pytest -q tests/test_paddleocr_runner.py::test_prepare_single_request_uses_region_frame_source tests/test_executor_cache.py::test_tool_chain_passes_structured_outputs_between_tools tests/test_executor_cache.py::test_empty_resolved_media_records_invalid_request_before_tool_call tests/test_plan_normalizer.py::test_plan_normalizer_rejects_noncanonical_inputs_and_wiring`
- Result: 4 passed.

What the task-state prototype tested:

- Built a temporary canonical task state with `claim_results`, `referent_slots`, `coverage_records`, `counter_records`, `price_point_occurrences`, `evidence_status_updates`, `retired_evidence`, and `open_questions`.
- Ran two planning rounds per case:
  - round 1 with open or validated task state,
  - synthetic tool/verifier updates applied into task state,
  - round 2 with updated task state to check whether planner repeats unnecessary evidence collection.
- Cases: transcript-only `video_6`, empty-bottle `video_13`, loud-bang `video_340`, Magnolia referent relation `video_476`, Minerva price-point `7eF4EjdYZ7k`, and validated chart sample `875b24c9-a2ab-4965-8186-76495a5b553d`.

Findings:

- The task state correctly prevented redundant evidence collection in validated cases. `video_6` and the validated chart case produced empty plans, and after synthetic validation `video_13`, `video_340`, and `video_476` also produced empty follow-up plans.
- Prompt rules alone were not reliable enough in round 1. For `video_13`, the planner still produced a `generic_purpose`-only plan instead of region/crop + verifier. For `video_340`, it produced audio localization + frame retrieval + `generic_purpose`, but skipped verifier.
- A deterministic plan-validation feedback loop fixed those failures. After rejecting the invalid plans with explicit policy feedback:
  - `video_13` replanned to `spatial_grounder -> verifier`.
  - `video_340` replanned to `audio_temporal_grounder -> frame_retriever -> verifier`.
- Therefore the implementation should not rely on prompt compliance alone. It needs a task-state-aware plan validator that either rejects/retries invalid plans or deterministically requires missing verifier/coverage steps before execution.

OCR region wiring findings:

- The current pipeline can pass regions to OCR structurally:
  - `spatial_grounder` emits `regions`, each with `frame`, `bbox`, `label`, and metadata.
  - planner/executor can wire `input_refs={"regions": [{"step_id": N, "field_path": "regions"}]}` into `ocr`.
  - `OCRRequest` prefers `regions`/`frames` over clips.
  - PaddleOCR runner resolves `region.frame` via `metadata.source_path`, `frame_path`, `source_frame_path`, or workspace-relative `relpath`, then crops `region.bbox` before OCR.
- Existing tests confirm the crop path uses `region.frame` directly and does not regenerate frames when the frame source resolves.
- Remaining risk is quality, not wiring: spatial grounding must return accurate enough boxes. If it returns no regions, OCR correctly fails/skips before a bad downstream call, but the planner must recover by trying full-frame OCR, better frame retrieval, or a different crop query.

Updated final implementation recommendation:

1. Implement canonical task state.
2. Implement verifier.
3. Implement task-state-aware plan validation and replan feedback before execution.
4. Keep prompt ICL, but treat it as guidance rather than enforcement.
5. Keep OCR backend, improve OCR region/frame preparation, and add fallback policy for empty or low-confidence regions.

## 10. Empty-Plan And Synthesizer State Handoff Probe

Additional probe:

- `workspace/probe_results/task_state_system_probe/synthesizer_state_probe_report.json`

Definitions:

- Rejecting an invalid plan means rejecting a syntactically valid `ExecutionPlan` before execution because it violates task-state policy. Example: a small-object visual-state claim has status `unverified`, but the plan only calls `generic_purpose` instead of using region/crop evidence plus verifier.
- An empty plan means `ExecutionPlan.steps == []`. It does not mean the pipeline has no evidence. It means no new tool calls are needed because task state and existing evidence already validate the answer-critical claims, or because the state says the task is unresolved and more tool calls would be redundant without a different strategy.

Role separation:

- The plan validator validates plan safety and routing, not truth.
- The verifier validates claims against supplied media/transcripts/OCR/evidence.
- The task-state reducer records verifier results as claim status, evidence status, coverage, counters, referents, and open questions.
- The synthesizer converts validated task state plus linked observations/evidence into the final trace.
- The auditor checks the final trace quality after synthesis.

Synthesizer findings:

- For a validated `video_13` empty-plan state, the synthesizer correctly produced `A. 0` from the supplied validated verifier evidence.
- For a partially validated Minerva price-point state, the synthesizer only behaved correctly when an explicit `TASK_STATE` section was supplied. With task state, it left `final_answer` empty because the total was unresolved. Without task state, it produced schema-invalid step ids in this probe.
- Therefore empty plans require an explicit task-state handoff to the synthesizer. Relying only on evidence prose is not robust enough.

Additional implementation requirements:

1. Add `TASK_STATE` to synthesizer input.
2. Add a synthesizer rule: if answer-critical claims are `unverified`, `unknown`, or `partially_validated` with open questions, leave `final_answer` empty.
3. Add a synthesizer rule: if claims are `validated` and linked observations/evidence are present, synthesize from them even when no new tools ran.
4. Ensure task-state evidence links contain valid `ArtifactRef` objects for `evidence_entries.artifact_refs`; plain artifact id strings belong only in observation `source_artifact_refs`.
5. Add schema-repair or retry for synthesizer outputs that use string step ids or string artifact refs.

Remaining issues to fix before implementation is trustworthy:

- Plan policy validation must be deterministic and task-state-aware.
- Verifier outputs must update canonical state; otherwise the planner will not know what is validated, stale, refuted, or still open.
- Synthesizer must consume canonical task state directly.
- OCR fallback policy is needed when `spatial_grounder` returns empty/low-confidence regions:
  - retry with a broader region query,
  - run full-frame OCR if the task is text/number critical,
  - route ambiguous OCR values to verifier,
  - leave the claim open if coverage remains insufficient.
- Evidence statuses should allow task-relative statuses beyond the current trace-facing `validated/provisional/superseded`; internal state needs `candidate`, `refuted`, `irrelevant`, `stale`, and `unknown`.

## 11. Deterministic Control-Layer Probe

Additional probe:

- `workspace/probe_results/deterministic_policy_probe/deterministic_policy_probe_report.json`
- Result: 11/11 probe cases passed.

What deterministic means here:

- It does not mean the planner or verifier perception is guaranteed correct.
- It means the pipeline has deterministic control-flow invariants that the LLM cannot bypass.
- The pipeline can guarantee that risky claims are either verified with required coverage or left unresolved; it cannot guarantee that every video answer is correct if the visual/audio model fails to perceive the needed fact.

Recommended deterministic pipeline:

1. Build canonical task state before planning.
   - Store claims, claim types, required modalities, coverage, referents, counters, price/cell occurrences, evidence statuses, and open questions.
   - Claim typing can use an LLM, but the output must be normalized into a strict schema and corrected by deterministic heuristics.

2. Planner proposes a plan.
   - Planner sees task state, retrieval catalog, retrieved context, tools, and prior trace/audit.
   - Planner is allowed to return empty steps only when task state says no answer-critical claim remains open.

3. Plan policy validator checks the plan before execution.
   - This validator is deterministic code.
   - It rejects syntactically valid but unsafe plans.
   - Examples tested:
     - rejects ASR when transcript-only task already has sufficient preprocess transcript coverage,
     - rejects `generic_purpose`-only small visual count/state plans,
     - rejects event-action plans without verifier,
     - rejects OCR/math plans without occurrence semantics,
     - rejects use of refuted/stale evidence as support,
     - accepts validated empty plans,
     - accepts region/crop + verifier plans.

4. Repair or deterministic template fallback.
   - First repair path: feed deterministic validation errors back to the planner.
   - If repair still fails, use a small deterministic template for common risky claim types:
     - small visual state/count: `frame_retriever or retrieved frames -> spatial_grounder -> verifier`,
     - event action: `audio_temporal_grounder -> frame_retriever(before/during/after) -> verifier`,
     - OCR/math: `frame_retriever -> spatial_grounder or full-frame OCR -> ocr -> verifier`,
     - referent relation: `generic/transcript referent extraction -> visual referent verification -> verifier`.
   - If no safe template applies, leave the claim open instead of executing an unsafe plan.

5. Executor runs only validated plans.
   - Invalid wiring still fails before downstream tools.
   - Tool failures become state updates, not silent evidence.

6. Task-state reducer updates canonical state.
   - Verifier `supported` -> claim `validated`.
   - Verifier `refuted` -> claim/evidence `refuted`.
   - Verifier `unknown` or partial -> claim remains open with unresolved gaps.
   - Empty OCR regions or invalid OCR requests -> OCR fallback state.
   - Evidence can be true but `irrelevant`, `stale`, or `superseded` for the current claim.

7. OCR fallback policy.
   - If spatial grounding returns empty regions or OCR receives no valid media:
     - retry spatial grounding with a broader query,
     - if still empty, run OCR on original frames,
     - route OCR lines/ambiguous values to verifier,
     - keep the claim open if coverage remains insufficient.

8. Synthesizer consumes task state.
   - Empty plans are safe only if the synthesizer receives `TASK_STATE`.
   - If all answer-critical claims are validated, synthesize from linked evidence/observations.
   - If any answer-critical claim remains `unverified`, `unknown`, or `partially_validated` with open questions, leave `final_answer` empty.
   - Add schema repair/retry for string step ids and string artifact refs.

9. Auditor remains final quality check.
   - The auditor should score whether the trace followed task state, not just whether prose sounds plausible.

What we can claim after these probes:

- The control design can prevent the known failure class where the planner gives a plausible but under-verified plan.
- The system can stop redundant tool calls after validation.
- The OCR region path is structurally viable.
- Empty plans are viable only with explicit task-state handoff.
- Remaining unresolved claims can be carried forward safely rather than hallucinated into a final answer.

What we cannot claim:

- We cannot guarantee correctness for every sample.
- We cannot guarantee the verifier or spatial grounder will perceive tiny, blurry, or ambiguous details.
- We cannot guarantee the claim typer catches every possible benchmark pattern unless it is evaluated on a broad labeled set.

Final implementation recommendation:

- Build this as a deterministic control layer around the existing planner/executor/synthesizer, not as more prompt text.
- Treat benchmark success as: correct when evidence supports an answer, empty/unresolved when evidence is insufficient, never unsupported confident answers.

## 12. Prompt-Only Control And Spatial Grounder Reality Check

Why not put the deterministic layer only in the planner prompt/ICL:

- Prompt rules can be ignored or partially followed by the planner. In the task-state prototype, the planner still produced unsafe plans for known risky cases until deterministic validation rejected them.
- ICL improves the probability of good planning, but it does not create a hard invariant.
- The pipeline needs hard gates for cases where a bad plan is worse than no answer:
  - small visual-state/count claim solved with `generic_purpose` only,
  - event-action claim without before/during/after verification,
  - OCR/math claim without occurrence semantics,
  - reuse of refuted/stale evidence,
  - repeated ASR despite sufficient preprocess transcripts.
- The right design is prompt guidance plus deterministic enforcement. Prompt examples should teach the planner the expected plan shape; policy validation should prevent unsafe deviations.

Spatial grounder evidence from existing runs:

- It does emit structurally valid `regions` when it succeeds.
- Saved `video_13` beer-bottle result:
  - path: `workspace/runs/video_13/20260428T082841Z_6f4ba176/round_02/tools/01_spatial_grounder/result.json`
  - query: locate the two beer bottles on the foreground table.
  - output: 6 detections/regions over frames 130s and 132s.
  - example regions: left bottle `[407, 515, 479, 902]`, right bottle `[622, 545, 717, 902]`.
- Saved OCR/layout cases show failures:
  - `minerva_results/7eF4EjdYZ7k/.../round_02/tools/01_spatial_grounder/result.json` returned 0 regions for product cards.
  - cached chart-region queries for Store Cleanliness and Value for Dollar returned 0 regions.

Conclusion:

- Region wiring is real and tested.
- Spatial grounding quality is not reliable enough to be assumed.
- Therefore the OCR branch must be stateful:
  1. Try spatial grounding for precise regions.
  2. If regions are non-empty, pass them to OCR.
  3. If regions are empty, record `spatial_regions_empty`.
  4. Retry broader spatial query or run full-frame OCR.
  5. Send OCR outputs and ambiguity to verifier.
  6. If still ambiguous, keep the claim open rather than selecting an answer.

Additional issues to handle:

- Plan validator must allow empty plans only when task state is already validated or explicitly unresolved with no safe next action.
- Synthesizer must see task state, not just evidence prose.
- Internal evidence statuses should be richer than public trace statuses.
- Spatial/OCR confidence should influence routing: low-confidence regions should be verified or broadened, not blindly trusted.
- We need broad evaluation of claim typing, because if a task is misclassified as easy transcript-only or generic visual, the policy validator may apply the wrong invariant.

## 13. Planner Prompt-Only Probe

Additional probe:

- `workspace/probe_results/planner_prompt_only_probe/planner_prompt_only_probe_report.json`

What changed in the temporary prompt:

- Added explicit `TASK_STATE`.
- Added proposed `verifier` tool schema.
- Added hard routing rules directly to the planner prompt.
- Added ICL patterns for:
  - small visual count/state,
  - event action,
  - OCR/math occurrence tracking,
  - already-validated empty plans.
- Did not use deterministic plan-validator feedback or replan in this probe.

Probe cases and outcomes:

- validated transcript `video_6`: empty plan, passed.
- validated chart `875b24c9-a2ab-4965-8186-76495a5b553d`: empty plan, passed.
- small visual count `video_13`: `spatial_grounder -> verifier`, passed.
- event action `video_340`: `audio_temporal_grounder -> frame_retriever -> spatial_grounder -> verifier`, passed.
- referent relation `video_476`: `spatial_grounder -> verifier`, passed.
- OCR/math Minerva `7eF4EjdYZ7k`: `frame_retriever -> spatial_grounder -> ocr -> verifier`, passed.

Conclusion from this probe:

- A stronger planner prompt with `TASK_STATE` and verifier-aware ICL worked on this focused set.
- If the goal is to keep the implementation simpler, start with prompt-only enforcement and no plan validator.
- This still requires implementing `TASK_STATE`, the `verifier` tool, and synthesizer task-state handoff.
- Keep detailed planner telemetry so we can add a deterministic validator later only if broader runs show the planner still skips required steps.

Prompt-only implementation recommendation:

1. Implement task state and verifier.
2. Add the prompt-only routing rules and ICL from this probe to the planner prompt.
3. Do not implement the plan validator in the first pass.
4. Add tests that assert planner outputs the expected tool shapes for the six probe families above.
5. Run broader samples and only add deterministic validation if prompt-only planning fails repeatedly.
