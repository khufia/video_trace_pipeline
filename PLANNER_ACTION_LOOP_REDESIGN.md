# Planner Action Loop Redesign

Working design for the new pipeline branch. This file is the running plan and should be updated as the design changes.

## Branch Name

Recommended implementation branch name:

```text
feature/planner-action-loop-full-preprocess
```

Reason: the branch name captures the two main architectural changes: planner-directed single-action control and full combined preprocess context.

## Current Decisions

- Do not use task state for now.
- Replace multi-tool planner rounds with a planner action loop.
- The planner may choose exactly one external action at a time: one tool call, synthesize, or stop/fail.
- Synthesizer and auditor are control steps, not counted as tool rounds.
- If the planner chooses synthesis, the pipeline always runs the auditor after synthesis.
- If the auditor fails, its review is added to the next planner context and the planner continues.
- Keep the `max_rounds` / `--max-rounds` naming, but set its default to 15 in code and in `slurm/run_video_trace_pipeline.slurm`.
- Enable preprocessing with dense captions and ASR disk cache.
- Pass the full combined preprocess text/JSON content to planner and auditor, not only selected hints.
- The planner/auditor preprocess context should be one canonical combined view of dense captions plus ASR. Do not append dense captions separately again and duplicate the same information.
- Treat preprocess as candidate context, not trusted proof.
- Do not use the verifier tool in this redesign.
- Pass observation IDs together with their text to planner, synthesizer, and auditor. IDs alone are too lossy for reasoning.

## Remove Task State For Now

The current branch has task-state code, but this redesign should not depend on it.

Implementation should remove or bypass:

- `video_trace_pipeline/orchestration/task_state.py`
- `video_trace_pipeline/schemas/task_state.py`
- task-state tests
- task-state prompt/context fields
- task-state readiness logic

The replacement memory is simpler:

- evidence ledger: every tool result becomes durable evidence/observations
- action history: every planner action and control action is saved
- trace history: every synthesized trace is saved
- audit history: every audit report is saved
- preprocess bundle: full cached combined ASR+dense-caption JSON is available every planner/auditor call

## New Control Loop

The pipeline should run this loop:

1. Build planner context.
2. Call planner.
3. Planner returns one `PlannerAction`.
4. If action is `tool_call`, execute exactly that tool call.
5. Save raw request, raw output, normalized output, artifacts, evidence entry, observations.
6. Build the next planner context from full combined preprocess + action history + evidence ledger + latest audit/trace.
7. If action is `synthesize`, run synthesizer, then auditor.
8. If auditor passes, finish.
9. If auditor fails, save audit feedback and continue.
10. Stop when `max_rounds` reaches 15, then force one final synthesis/audit or return unresolved.

Synthesizer/auditor calls are not counted against `max_rounds`. They should have their own safety cap, for example 3 synthesis attempts, so a bad audit loop cannot run forever without new evidence.

## Planner Output Schema

The planner should output one action, not an `ExecutionPlan` with many steps.

```json
{
  "action_type": "tool_call",
  "rationale": "Need to localize the visible event before retrieving frames.",
  "tool_name": "visual_temporal_grounder",
  "tool_request": {
    "query": "person opens the red door",
    "top_k": 5
  },
  "expected_observation": "Candidate clips where the red door is opened."
}
```

For synthesis:

```json
{
  "action_type": "synthesize",
  "rationale": "The evidence now resolves the answer-critical visual state and transcript.",
  "synthesis_instructions": "Use validated OCR/ASR/tool evidence. Do not use dense captions as proof of answer-critical claims."
}
```

For stopping unresolved:

```json
{
  "action_type": "stop_unresolved",
  "rationale": "The required text is unreadable after OCR-quality frame retrieval.",
  "missing_information": ["exact scoreboard value for team A"]
}
```

## Planner Context

Every planner call should receive text only, with these sections:

1. `TASK`
2. `OPTIONS`
3. `TOOL_CATALOG`
4. `FULL_PREPROCESS_CONTENT`
5. `ACTION_HISTORY`
6. `EVIDENCE_LEDGER`
7. `LATEST_TRACE`, if one exists
8. `LATEST_AUDIT`, if one exists
9. `OUTPUT_SCHEMA`

`FULL_PREPROCESS_CONTENT` means the full cached combined text/JSON bundle:

- preprocess manifest
- full canonical combined preprocess segments
- dense captions inside each segment
- ASR transcripts/transcript spans inside each segment
- preprocess coverage metadata
- paths/ids for preprocess artifacts

It should not mean a few selected search hits. It should also not duplicate dense captions by providing both combined segments and a second dense-caption-only section. If sampled frames exist as image files, the planner receives the paths/metadata, not image pixels.

Preprocess must be clearly labeled:

```text
PREPROCESS_TRUST_POLICY:
- This is broad candidate context generated before task-specific reasoning.
- It may be incomplete, wrong, over-broad, or hallucinated.
- Use it as searchable background and candidate context, not as a trusted locator.
- If preprocess points to the wrong window, misses evidence, or conflicts with tool evidence, ignore or override it and call the appropriate grounding/retrieval tools over a broader, narrower, or different scope.
- Do not treat dense captions as final proof of answer-critical claims.
- ASR transcript spans can support transcript-only claims when coverage is adequate, but visual/audio state still needs direct tool evidence when answer-critical.
```

## Token Budget Concern

Giving full preprocess content every planner call is attractive because the planner can inspect all cached context itself. The risk is input context size, not planner output size.

Recommended behavior:

- Keep planner output small because it returns one action.
- Set planner `max_tokens` around 3000 to 6000. The current config already uses 6000.
- Before each planner call, estimate prompt tokens.
- Always include the full combined preprocess content.
- If the estimated context is large, raise a visible warning in the run logs/manifest, but do not stop only because of the estimate.
- If preprocessing itself is enabled and fails, fail loudly instead of silently running without preprocess.

For now, the cleanest version is full combined preprocess JSON in every planner call, with token-size warnings and loud failure only when preprocess generation/loading fails.

## Tool Inputs And Outputs

All tool calls should be saved under the run directory:

```text
round_or_action_XX/
  planner_request.json
  planner_action.json
  tool_request.json
  tool_raw_output.json
  tool_result.json
  evidence_entry.json
  observations.json
  summary.md
```

The planner can refer to previous outputs by evidence IDs and artifact IDs, but it should also see compact structured outputs in `EVIDENCE_LEDGER`.

Observation payloads passed to agents should include both identity and meaning:

```json
{
  "observation_id": "obs_03_01",
  "evidence_id": "ev_03_generic",
  "text": "At 42.1s, the object is being lifted from the table.",
  "source_tool": "generic_purpose",
  "time_intervals": [{"start_s": 41.3, "end_s": 43.2}],
  "confidence": 0.82
}
```

Use IDs for citation, trace linking, and audit status bookkeeping. Use text for reasoning. If an artifact such as a frame has not been inspected, the observation text must say that explicitly or no visual claim should be inferred from it.

Canonical planner-facing tool fields should stay small:

- `query`
- `clips`
- `frames`
- `regions`
- `transcripts`
- `time_hints`
- `evidence_ids`
- `text_contexts`
- `top_k`
- `num_frames`
- `sequence_mode`
- `neighbor_radius_s`
- `include_anchor_neighbors`
- `sort_order`

Raw tool output can remain richer on disk.

## Example Flow 1: ASR-Anchored Visual Question

Question: "What is the object doing when the speaker says 'now it is ready'?"

### Preprocess

Before planner loop:

Input:

```json
{
  "video_path": "...",
  "clip_duration_s": 30,
  "include_asr": true,
  "dense_captioner": true
}
```

Saved output on disk:

```text
workspace/preprocess/<video_id>/<fingerprint>/<settings_hash>/
  manifest.json
  raw_segments.json
  planner_segments.json
  dense_segments.json
  transcripts.json
```

Planner receives the full combined JSON content from `planner_segments.json` in `FULL_PREPROCESS_CONTENT`. `planner_segments.json` is already the planner-facing combination derived from raw dense-caption segments plus transcripts, so we should not introduce another `preprocess_context.json` file unless we later need a different canonical format. The planner does not receive a separate duplicate dense-caption block.

Combined preprocess excerpt:

```json
{
  "segment_id": "seg_002",
  "start_s": 30.0,
  "end_s": 60.0,
  "dense_caption": {
    "overall_summary": "A person handles an object at a table."
  },
  "asr": {
    "transcript_id": "pre_tx_002",
    "transcript_spans": [
      {"start_s": 42.1, "end_s": 43.4, "text": "now it is ready"}
    ]
  }
}
```

### Action 1: Frame Retriever

Planner already has the phrase timestamp from preprocessed ASR, so it should not call ASR again. It asks for chronological neighboring frames. The query is visual only; temporal information goes in `time_hints`.

Planner action:

```json
{
  "action_type": "tool_call",
  "tool_name": "frame_retriever",
  "tool_request": {
    "tool_name": "frame_retriever",
    "clips": [{"video_id": "video_1", "start_s": 39.0, "end_s": 46.0}],
    "time_hints": ["42.1s"],
    "query": "object state and action visible in the local scene",
    "num_frames": 7,
    "sequence_mode": "anchor_window",
    "neighbor_radius_s": 2.0,
    "include_anchor_neighbors": true,
    "sort_order": "chronological"
  },
  "expected_observation": "Chronological frames before/during/after the phrase."
}
```

Tool output:

```json
{
  "frames": [
    {"frame_id": "fr_001", "timestamp_s": 40.2, "relpath": "artifacts/fr_001.jpg"},
    {"frame_id": "fr_002", "timestamp_s": 41.3, "relpath": "artifacts/fr_002.jpg"},
    {"frame_id": "fr_003", "timestamp_s": 42.2, "relpath": "artifacts/fr_003.jpg"},
    {"frame_id": "fr_004", "timestamp_s": 43.2, "relpath": "artifacts/fr_004.jpg"}
  ]
}
```

Saved evidence card:

```json
{
  "evidence_id": "ev_02_frames",
  "tool_name": "frame_retriever",
  "evidence_text": "Retrieved chronological frames around 42.1s. The frames are artifacts only; contents are not yet inspected.",
  "artifact_refs": ["fr_001", "fr_002", "fr_003", "fr_004"],
  "time_intervals": [{"start_s": 40.2, "end_s": 43.2}],
  "status": "candidate"
}
```

Important: the planner cannot assume what the frames show yet.

### Action 2: Generic Purpose

Planner asks a visual reader to inspect the frames with the preprocessed ASR transcript.

Planner action:

```json
{
  "action_type": "tool_call",
  "tool_name": "generic_purpose",
  "tool_request": {
    "tool_name": "generic_purpose",
    "query": "Using the supplied transcript and chronological frames, what is the object doing when the speaker says 'now it is ready'? State uncertainty if the action is not visible.",
    "frames": [{"artifact_id": "fr_001"}, {"artifact_id": "fr_002"}, {"artifact_id": "fr_003"}, {"artifact_id": "fr_004"}],
    "transcripts": [{"transcript_id": "pre_tx_002"}],
    "evidence_ids": ["ev_02_frames"]
  },
  "expected_observation": "Grounded visual action at the ASR timestamp."
}
```

Tool output:

```json
{
  "answer": "The object is being lifted from the table.",
  "analysis": "The transcript phrase occurs at 42.1s; frames from 41.3-43.2s show the object moving upward from the table.",
  "confidence": 0.82
}
```

Saved evidence card:

```json
{
  "evidence_id": "ev_03_generic",
  "tool_name": "generic_purpose",
  "evidence_text": "At the ASR phrase timestamp, the object is lifted from the table.",
  "observation_ids": ["obs_03_01"],
  "status": "candidate"
}
```

### Action 3: Synthesize

Planner action:

```json
{
  "action_type": "synthesize",
  "synthesis_instructions": "Use preprocessed ASR for the phrase timestamp and generic visual evidence for the object action. Do not use dense captions as proof of the visual action."
}
```

Pipeline runs synthesizer automatically.

Synthesizer input:

```json
{
  "task": "...",
  "full_preprocess_content": "...",
  "evidence_entries": ["ev_02_frames", "ev_03_generic"],
  "observations": ["obs_03_01"],
  "latest_audit": null,
  "instructions": "..."
}
```

Synthesizer output:

```json
{
  "final_answer": "The object is being lifted from the table.",
  "evidence_entries": ["..."],
  "inference_steps": ["..."]
}
```

Pipeline runs auditor automatically.

Auditor input:

```json
{
  "task": "...",
  "trace_package": "...",
  "full_preprocess_content": "...",
  "evidence_summary": "...",
  "preprocess_trust_policy": "candidate context, not proof"
}
```

If auditor passes, run finishes. If it fails, the next planner context includes the failure and the planner chooses the next single tool call.

## Example Flow 2: Visible Text / OCR

Question: "What percentage is shown for Whole Foods?"

### Action 1: Visual Temporal Grounder

Input:

```json
{
  "tool_name": "visual_temporal_grounder",
  "query": "chart or table showing Whole Foods percentage",
  "top_k": 5
}
```

Output:

```json
{
  "clips": [
    {"video_id": "video_chart", "start_s": 61.0, "end_s": 67.0, "confidence": 0.71}
  ]
}
```

### Action 2: Frame Retriever

Because this is visible text, planner asks for readable stable frames.

Input:

```json
{
  "tool_name": "frame_retriever",
  "clips": [{"video_id": "video_chart", "start_s": 61.0, "end_s": 67.0}],
  "query": "stable readable complete chart frame with Whole Foods label and percentage",
  "num_frames": 4,
  "sequence_mode": "chronological",
  "sort_order": "chronological"
}
```

Output:

```json
{
  "frames": [
    {"frame_id": "fr_chart_1", "timestamp_s": 63.0, "relpath": "artifacts/fr_chart_1.jpg"},
    {"frame_id": "fr_chart_2", "timestamp_s": 64.0, "relpath": "artifacts/fr_chart_2.jpg"},
    {"frame_id": "fr_chart_3", "timestamp_s": 65.0, "relpath": "artifacts/fr_chart_3.jpg"},
    {"frame_id": "fr_chart_4", "timestamp_s": 66.0, "relpath": "artifacts/fr_chart_4.jpg"}
  ]
}
```

### Action 3: OCR

Input:

```json
{
  "tool_name": "ocr",
  "frames": [{"artifact_id": "fr_chart_1"}, {"artifact_id": "fr_chart_2"}, {"artifact_id": "fr_chart_3"}, {"artifact_id": "fr_chart_4"}],
  "query": "Read the Whole Foods label and its adjacent percentage from the complete chart."
}
```

Output:

```json
{
  "text": "Whole Foods 65%",
  "reads": [
    {"frame": {"artifact_id": "fr_chart_2"}, "text": "Whole Foods 65%"}
  ]
}
```

### Action 4: Synthesize And Audit

Planner chooses synthesis. Auditor gets both the OCR evidence and full preprocess content. If preprocess had a conflicting dense caption, auditor can flag the conflict, but OCR should be preferred for answer-critical visible text.

## Example Flow 3: Audit Fails And Planner Continues

Suppose the auditor fails:

```json
{
  "verdict": "FAIL",
  "missing_information": [
    "the OCR text does not prove label-value adjacency for Whole Foods and 65%"
  ],
  "feedback": "Retrieve a complete chart frame or use OCR reads that preserve adjacency."
}
```

Next planner context includes:

- full combined preprocess content again
- all previous tool calls
- OCR output
- trace output
- audit failure

Planner next action:

```json
{
  "action_type": "tool_call",
  "tool_name": "frame_retriever",
  "tool_request": {
    "tool_name": "frame_retriever",
    "clips": [{"video_id": "video_chart", "start_s": 63.0, "end_s": 67.0}],
    "query": "complete chart frame preserving Whole Foods label and adjacent percentage",
    "num_frames": 3,
    "sequence_mode": "chronological",
    "sort_order": "chronological"
  },
  "expected_observation": "Complete chart frames that preserve label-value adjacency."
}
```

Then planner can call OCR again, synthesize again, and audit again.

## Closed Decisions

- Use `planner_segments.json` as the combined planner/auditor preprocess context because it is already built from raw dense-caption segments plus transcripts.
- Do not implement chunked full-preprocess reading for now. Include full `planner_segments.json`, estimate tokens, and warn if the prompt is large.

## Implementation Checkpoint

- Branch: `pipeline-v2`.
- Planner now emits one `PlannerAction` at a time: `tool_call`, `synthesize`, or `stop_unresolved`.
- Tool calls consume `max_rounds`; synthesize/audit turns do not.
- The planner receives `ACTION_HISTORY` with compact summaries plus sanitized tool outputs, including clips/frames/transcripts/observations, so later actions can copy structured objects instead of relying on IDs alone.
- Evidence summaries include observation IDs and their corresponding observation text.
- The planner, synthesizer, and auditor receive full `FULL_PREPROCESS_CONTENT` from `planner_segments.json`.
- Preprocess is enabled in the main configs. If preprocess is enabled but `planner_segments` is missing, the pipeline fails loudly.
- The verifier tool is excluded from the v2 planner catalog and disabled in the main config.
- Task-state reducers are not wired into the v2 pipeline loop.
- `video_trace_pipeline_simple` Slurm entry is disabled.
- Verification run: `/fs/nexus-scratch/gnanesh/venv_vdr3/bin/python -m pytest -q tests` passed with `236 passed`.
