# Plan Validator Fallback Design

This is not part of the first-pass implementation recommended in `PIPELINE_FINAL_RECOMMENDATIONS.md`.

Use this only if broad runs show that the planner still emits unsafe plans even after it receives `TASK_STATE`, verifier schema, and the stronger generic ICL rules.

## 1. Purpose

The plan validator is a deterministic control layer between planner and executor.

It does not decide truth. It only decides whether a syntactically valid `ExecutionPlan` is safe to execute for the current `TASK_STATE`.

Truth roles remain:

- verifier validates claims against supplied evidence,
- task-state reducer records verifier outputs,
- synthesizer writes the final trace,
- auditor scores trace quality.

## 2. When To Enable

Enable this fallback if any of these recur in broader runs:

- planner uses `generic_purpose` alone for small-object state/count,
- planner answers event-action tasks without before/during/after evidence,
- planner performs OCR/math without occurrence or label-value semantics,
- planner reuses refuted/stale/irrelevant evidence as support,
- planner calls ASR when preprocess transcript coverage is sufficient,
- planner emits empty plans while answer-critical claims remain unverified,
- planner repeatedly routes OCR to empty regions without fallback.

## 3. Inputs

Validator inputs:

```json
{
  "task": {},
  "task_state": {},
  "plan": {},
  "retrieved_context": {},
  "tool_catalog": {},
  "audit_feedback": {}
}
```

Output:

```json
{
  "valid": false,
  "issues": [
    {
      "code": "missing_verifier_small_visual",
      "message": "Small visual count/state claim is unverified without verifier.",
      "repair_hint": "Add spatial_grounder or explicit region/crop evidence, then verifier."
    }
  ],
  "repair_allowed": true,
  "fallback_template": "small_visual_state"
}
```

## 4. Deterministic Invariants

### 4.1 Empty Plan Gate

Reject empty plans unless:

- every answer-critical claim is `validated`, or
- remaining claims are explicitly `unknown`/unresolved with no safe next action and synthesizer is instructed to leave `final_answer` empty.

### 4.2 Transcript Reuse Gate

Reject ASR steps when:

- the task is transcript-only/dialogue/not-mentioned,
- preprocess transcript coverage is sufficient,
- no speaker attribution, contradiction, or missing interval requires ASR.

### 4.3 Small Visual State/Count Gate

Reject plans for small visual state/count when:

- no region/crop/local frame evidence is supplied,
- no verifier step checks the answer-critical count/state,
- absence is claimed without sufficient coverage.

Accepted pattern:

```text
frame_retriever or retrieved frames -> spatial_grounder -> verifier
```

### 4.4 Audio/Event Action Gate

Reject plans for sound/event-action tasks when:

- only after-event frames are inspected,
- no before/during/after chronological sequence is retrieved,
- no verifier checks the direct trigger/action claim.

Accepted pattern:

```text
audio_temporal_grounder -> frame_retriever(anchor_window chronological) -> verifier
```

Optional visual-conditioned pattern:

```text
visual_temporal_grounder -> frame_retriever -> audio_temporal_grounder inside grounded clip -> verifier
```

### 4.5 OCR/Math Gate

Reject OCR/math/table/chart/price/scoreboard plans when:

- OCR receives empty media,
- label-value adjacency is not preserved,
- occurrence-vs-distinct semantics are absent,
- arithmetic is delegated only to free-form VLM prose,
- progressive displays are not sampled at a stable state.

Accepted pattern:

```text
frame_retriever -> spatial_grounder or full-frame OCR -> ocr -> verifier -> deterministic arithmetic
```

### 4.6 Temporal Order Gate

Reject first/last/before/after/order/count plans when:

- retrieved frames/events are not sorted chronologically,
- top-k retrieval order is treated as event order,
- initial trace anchors are discarded without contradiction.

### 4.7 Referent/Speaker Gate

Reject relationship, same/different, speaker, addressee, and identity plans when:

- answer-critical referents are not represented as separate slots,
- ordinal referents are resolved only in a local clip rather than the full question scope,
- dialogue relationship is inferred without carrying transcripts into the final verification step.

### 4.8 Evidence Status Gate

Reject plans that use these evidence statuses as support:

- `candidate`,
- `refuted`,
- `irrelevant`,
- `stale`,
- `superseded` unless the plan explicitly uses it for contradiction analysis.

## 5. Repair Flow

First repair path:

1. Write `planner_invalid_policy.json`.
2. Feed validation errors and repair hints back to the planner.
3. Let planner produce one repaired plan.
4. Validate again.

Second repair path:

If the repaired plan fails and a safe template exists, use a deterministic template.

Template examples:

```text
small_visual_state:
  frame_retriever/retrieved frames -> spatial_grounder -> verifier

event_action:
  audio_temporal_grounder -> frame_retriever(anchor_window chronological) -> verifier

ocr_math:
  frame_retriever -> spatial_grounder -> ocr -> verifier
  if regions empty: frame_retriever -> ocr(full frames) -> verifier

referent_relation:
  retrieve/ground each slot -> verifier over all slot evidence
```

If no safe template applies, execute no tools and keep the claim open.

## 6. Where It Would Live

Suggested module:

- `video_trace_pipeline/orchestration/plan_policy.py`

Suggested integration point:

- after `ExecutionPlanNormalizer.normalize(...)`,
- before `PlanExecutor.execute_plan(...)`.

Suggested persisted files:

- `planner_policy_validation.json`,
- `planner_policy_repair_request.json`,
- `planner_policy_repair_raw.json`,
- `planner_policy_repaired_plan.json`.

## 7. Tests

Add tests for:

- validated empty plans pass,
- unverified empty plans fail,
- transcript-only sufficient preprocess rejects ASR,
- small visual `generic_purpose`-only plan fails,
- small visual region/crop plus verifier passes,
- event action without verifier fails,
- event action with before/during/after verifier passes,
- OCR/math without occurrence semantics fails,
- empty OCR regions trigger fallback,
- refuted/stale evidence cannot support final claims,
- deterministic template fallback creates executable plans,
- no downstream tool runs after an invalid plan unless repaired.

## 8. Why This Is Fallback Only

The latest prompt-only planner probe passed the focused risky cases without validator feedback. Adding deterministic validation now would increase implementation surface area and could block plans because of imperfect claim typing.

Keep this design ready, but first implement:

1. `TASK_STATE`,
2. `verifier`,
3. state-aware planner/synthesizer prompts,
4. broad regression tests.

If broad samples still show skipped verifier/crop/chronology requirements, enable this validator.
