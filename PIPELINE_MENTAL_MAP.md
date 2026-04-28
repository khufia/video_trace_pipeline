# Video Trace Pipeline Mental Map

This document is a mental model for the full `video_trace_pipeline` runtime, especially what happens when you run:

```bash
vtp run --profile <machine.yaml> --models <models.yaml> ...
```

The short version:

```text
CLI args
  -> load profile + model config
  -> build TaskSpec(s)
  -> PipelineRunner.run_task(...)
  -> create run directory
  -> build/reuse dense-caption + ASR preprocess cache
  -> repeat planning/execution/synthesis/audit rounds
  -> write trace_package.json, benchmark_export.json, final_result.json, debug bundle
```

## 1. Entry Points

The installed console script is:

```toml
vtp = "video_trace_pipeline.cli.main:main"
```

`main()` invokes the Typer app in `video_trace_pipeline/cli/main.py`.

CLI commands:

- `vtp check-env`: validates packages, datasets, model/tool configuration, and readiness.
- `vtp preprocess`: builds or reuses the dense-caption/ASR preprocess cache for selected tasks.
- `vtp run`: runs the full benchmark trace pipeline.
- `vtp audit`: re-runs the auditor on an existing run directory.
- `vtp export`: regenerates `benchmark_export.json` from an existing `trace_package.json`.
- `vtp debug-run`: writes a compact debug bundle for an existing run directory.

`vtp run` is the main path.

## 2. Configuration Inputs

Two config files drive almost everything:

- Machine profile YAML:
  - workspace root
  - cache root
  - Hugging Face cache
  - dataset locations
  - API endpoints
  - GPU assignments
  - environment overrides

- Models config YAML:
  - agent models: planner, trace synthesizer, trace auditor, atomicizer
  - tool models: dense captioner, ASR, temporal grounders, frame retriever, OCR, spatial grounder, generic-purpose extractor
  - per-tool commands, prompt versions, model names, and extra runtime options

The CLI loads these through:

```text
load_machine_profile(profile)
load_models_config(models)
PipelineRunner(machine_profile, models_config, ...)
```

The profile can also override environment variables before the runner starts.

## 3. Task Selection

The CLI converts user input into one or more `TaskSpec` objects.

Task selection precedence:

1. `--inputs-json` plus `--input-index`
2. direct `--video-path` plus `--question`
3. benchmark adapter from `--benchmark`, optionally filtered by `--index` or `--limit`

A `TaskSpec` contains:

```text
benchmark
sample_key
question
options
video_path
video_id
question_id
gold_answer
initial_trace_steps
metadata
```

Direct video runs use `benchmark="adhoc"` unless you pass a benchmark name.

## 4. Runner Construction

`PipelineRunner` is the central object. It wires together:

```text
WorkspaceManager
OpenAIChatClient
ToolRegistry
DenseCaptionPreprocessor
PlanExecutor
ExecutionPlanNormalizer
PlannerAgent
TraceSynthesizerAgent
TraceAuditorAgent
AtomicFactAgent, optional
```

Think of it as the coordinator. It owns the high-level loop but delegates each specialized job to one of these components.

## 5. Workspace Layout

Each `vtp run` creates a run directory:

```text
<workspace_root>/runs/<video_id>/<run_id>/
```

Important files and folders:

```text
run_manifest.json
runtime_snapshot.json
trace_package.json
benchmark_export.json
final_result.json

evidence/
  evidence_index.jsonl
  atomic_observations.jsonl
  evidence.sqlite3
  evidence_memory.json

round_00/
  initial_trace_package.json
  auditor_request.json
  auditor_report.json

round_01/
  planner_request.json
  planner_plan.json
  synthesizer_request.json
  synthesizer_trace_package.json
  auditor_request.json
  auditor_report.json
  evidence_memory.json
  tools/
    01_<tool_name>/
      request.json
      runtime.json
      result.json
      timing.json
      artifact_refs.json
      observations.json

debug/
```

Separate shared caches live under the configured cache root:

```text
cache/
  preprocess/
  evidence/
  artifacts/
```

Mental distinction:

- Run directory: what happened in this specific run.
- Preprocess cache: reusable dense captions/transcripts for a video/model/settings combination.
- Evidence cache: reusable tool results for exact tool requests.
- Artifacts cache: stored frame/image/text artifacts produced by tools.

## 6. Preprocess Stage

Every full run starts by calling:

```text
DenseCaptionPreprocessor.get_or_build(task, clip_duration_s)
```

The preprocessor computes:

```text
video fingerprint
dense caption implementation
dense caption model name
clip duration
prompt version
preprocess settings signature
```

These form the preprocess cache path. If complete cache files already exist, they are loaded.

If cache is missing:

1. The video duration is measured.
2. The video is chunked into windows, normally 60 seconds unless overridden.
3. `dense_captioner` runs over each window.
4. If enabled, ASR runs once and transcript segments are assigned back into the dense-caption windows.
5. Planner-friendly summaries and memory are derived.
6. Cache files are written:

```text
manifest.json
segments.json
planner_segments.json
planner_context.json
```

The planner sees the compact `planner_segments` and `planner_context`, not the entire raw video.

## 7. Generate vs Refine Mode

`mode="generate"` starts with no trace.

`mode="refine"` can start from:

- `--initial-trace-path`
- `TaskSpec.initial_trace_steps`

If an initial trace exists, the runner performs an initial audit in `round_00`.

If that audit passes with no blocking findings and no missing information, the pipeline can stop immediately.

Otherwise the run enters the normal planning loop, but the planning mode becomes `refine` instead of `generate`.

## 8. Main Round Loop

The core loop runs up to `max_rounds`.

Each round follows this shape:

```text
retrieve previous observations
  -> planner builds ExecutionPlan
  -> normalizer validates/reorders plan
  -> executor runs tool steps
  -> synthesizer writes TracePackage
  -> auditor checks TracePackage
  -> evidence memory is updated
  -> evidence statuses are updated
  -> stop if audit accepted
```

The audit acceptance rule is stricter than just `verdict == "PASS"`:

- verdict must be `PASS`
- `missing_information` must be empty
- no blocking medium/high findings may remain

Blocking categories include answer errors, attribution gaps, counting gaps, incomplete traces, inference errors, reading gaps, and temporal gaps.

## 9. Planner Agent

The planner receives:

```text
task question/options
mode: generate or refine
preprocess planner segments
preprocess planning memory
compact summaries of prior rounds
retrieved observations from the evidence ledger
latest audit feedback, if any
tool catalog
compact evidence memory
```

It returns an `ExecutionPlan`:

```text
strategy
use_summary
steps[]
refinement_instructions
```

Each plan step has:

```text
step_id
tool_name
purpose
arguments
input_refs
depends_on
```

The planner does not execute tools. It only decides which tools should run and how outputs should flow between them.

## 10. Plan Normalization

The normalizer makes the planner output executable and stricter.

It validates:

- tool names exist
- argument names match each tool request schema
- `input_refs` point to earlier valid steps
- dependencies are real and acyclic
- structural refs are sensible, for example frames to `frames`, clips to `clips`
- `generic_purpose` has real context and is not context-free
- ASR transcript payloads are passed as `transcripts`, not as generic text blobs

Then it topologically sorts and resequences the steps so execution can be deterministic.

This step is important because the planner is an LLM and may produce messy JSON even when the schema is mostly correct.

## 11. Tool Execution

The executor walks the normalized plan in step order.

For each step:

1. Start with `step.arguments`.
2. Resolve `input_refs` from previous step outputs.
3. Merge list fields like `clips`, `frames`, `regions`, `transcripts`, `text_contexts`, `evidence_ids`, and `time_hints`.
4. Add task video context to clip/frame refs if missing.
5. Repair certain empty media arguments when dependencies contain obvious media outputs.
6. Parse the request through the tool adapter's Pydantic schema.
7. Hash the request for cache lookup.
8. Load a cached `ToolResult` if available.
9. Otherwise execute the adapter.
10. Extract atomic observations from the tool result.
11. Store result and observations in the shared evidence cache.
12. Append an `EvidenceEntry` and `AtomicObservation`s to the run ledger.
13. Write request/result/timing/artifact/observation files under the round tool directory.

If an input reference cannot be resolved, the executor records a structured failure evidence entry instead of silently skipping the problem.

## 12. Tool Registry and Tool Adapters

`ToolRegistry` builds enabled adapters from the models config.

Supported tool names:

- `dense_captioner`
- `visual_temporal_grounder`
- `frame_retriever`
- `ocr`
- `spatial_grounder`
- `asr`
- `audio_temporal_grounder`
- `generic_purpose`

Most tools are process-backed adapters. They send a JSON envelope to a configured command over stdin and expect JSON on stdout.

Envelope shape:

```text
tool_name
request
task
runtime
evidence_records
```

`runtime` includes:

```text
backend
model_name
device
device_mapping
hf_cache
resolved_model_path
model_resolution_status
workspace_root
scratch_dir
extra
```

Some tools can use persistent in-process runners via `--persist-tool-models`; `--preload-persisted-models` can eagerly load them at run startup.

## 13. Tool Purposes

Common roles:

- `dense_captioner`: creates visual/audio descriptions over bounded clips.
- `asr`: transcribes speech.
- `visual_temporal_grounder`: finds video intervals matching a visual query.
- `audio_temporal_grounder`: finds audio event intervals.
- `frame_retriever`: retrieves representative or query-relevant frames.
- `ocr`: reads text from clips, frames, or regions.
- `spatial_grounder`: localizes objects/regions in frames.
- `generic_purpose`: interprets clips, frames, transcripts, text contexts, or evidence records with a multimodal/general model.

The planner chooses among these based on the question and audit feedback.

## 14. Evidence Ledger

The evidence ledger is the run-local source of truth.

It stores evidence in three forms:

```text
evidence/evidence_index.jsonl
evidence/atomic_observations.jsonl
evidence/evidence.sqlite3
```

An `EvidenceEntry` is a tool-level result:

```text
evidence_id
tool_name
evidence_text
confidence
status: provisional | validated | superseded
time anchors
artifact refs
observation_ids
metadata
```

An `AtomicObservation` is a smaller fact extracted from a tool result:

```text
observation_id
subject
subject_type
predicate
object_text
object_type
time_start_s / time_end_s / frame_ts_s
bbox
speaker_id
confidence
source_tool
atomic_text
```

The auditor and later planner rounds use this ledger to retrieve prior facts.

## 15. Observation Extraction

After a tool returns a `ToolResult`, `ObservationExtractor` converts it into atomic observations.

Examples:

- temporal grounder clip -> `"query" is present from start to end`
- frame retriever frame -> candidate frame at timestamp
- ASR segment -> speaker said text from start to end
- dense caption -> visual facts, visible text, objects, actions, audio facts
- OCR -> detected text
- spatial grounder -> object located at bbox
- generic purpose -> derived supporting facts

If an atomicizer agent is configured, some longer text fields are decomposed by an LLM into cleaner atomic facts.

## 16. Synthesizer Agent

After tools run, the synthesizer receives:

```text
task
mode
round evidence entries
round observations
current trace, if refining
planner refinement instructions
latest audit feedback, if any
compact evidence memory
```

It returns a `TracePackage`:

```text
task_key
mode
evidence_entries
inference_steps
final_answer
benchmark_renderings
metadata
```

The runner then patches in:

```text
task_key = task.sample_key
mode = current planning mode
benchmark_renderings[task.benchmark]
```

The trace is written as:

```text
round_XX/synthesizer_trace_package.json
```

At the end of the full run it is also written to:

```text
trace_package.json
```

## 17. Auditor Agent

The auditor receives:

```text
task
trace package
evidence summary
compact evidence memory
```

It returns an `AuditReport`:

```text
verdict: PASS | FAIL
confidence
scores
findings
feedback
missing_information
diagnostics
```

The audit report does two jobs:

1. It decides whether the pipeline can stop.
2. If not accepted, it becomes feedback for the next planner round.

This creates the repair loop:

```text
auditor says what is missing
  -> planner asks better tool questions
  -> tools collect targeted evidence
  -> synthesizer repairs trace
  -> auditor checks again
```

## 18. Evidence Memory

Evidence memory is a compact cross-round state object.

It is seeded from the task and any initial trace, then updated after each audit.

It helps the next planner avoid starting from scratch. It may include:

- useful previous observations
- evidence ids available for reuse
- audit feedback and missing information
- hints about what was already checked

It is written both per round and under:

```text
evidence/evidence_memory.json
```

## 19. Finalization

When the loop stops, either because audit passed or `max_rounds` was reached, the runner persists:

```text
trace_package.json
benchmark_export.json
final_result.json
debug/
```

`benchmark_export.json` is benchmark-shaped:

- VideoMathQA: question id, video id, numbered steps, answer
- Minerva: key, video id, trace text, answer
- OmniVideoBench: evidence/inference triples, answer
- ad hoc/default: generic trace/evidence structure

The CLI finally prints:

```text
run <sample_key> -> <run_dir>
```

## 20. Reading a Run Directory

When debugging, read in this order:

1. `run_manifest.json`: task identity, mode, preprocess cache path.
2. `runtime_snapshot.json`: profile/model/tool runtime snapshot.
3. `round_XX/planner_request.json`: what the planner saw.
4. `round_XX/planner_plan.json`: what the planner decided.
5. `round_XX/tools/*/request.json`: exact tool request.
6. `round_XX/tools/*/runtime.json`: command/model/device used by the tool.
7. `round_XX/tools/*/result.json`: raw structured tool result.
8. `round_XX/tools/*/observations.json`: atomic facts extracted from the result.
9. `round_XX/synthesizer_trace_package.json`: trace after that round.
10. `round_XX/auditor_report.json`: why the run stopped or what must be repaired.
11. `final_result.json`: final trace, audit, export, preprocess summary, rounds executed.

## 21. The Entire Flow as One Diagram

```text
User command
  |
  v
Typer CLI: video_trace_pipeline.cli.main
  |
  +-- load machine profile
  +-- load models config
  +-- parse options / task source
  |
  v
TaskSpec(s)
  |
  v
PipelineRunner
  |
  +-- WorkspaceManager creates run directory
  +-- runtime snapshot written
  +-- optional persistent models preload
  |
  v
DenseCaptionPreprocessor
  |
  +-- hit preprocess cache?
  |     |
  |     +-- yes: load manifest/segments/planner context
  |     |
  |     +-- no:
  |          +-- chunk video
  |          +-- dense caption each chunk
  |          +-- optional ASR
  |          +-- align transcript to chunks
  |          +-- write preprocess cache
  |
  v
EvidenceLedger + ToolExecutionContext
  |
  v
Round loop
  |
  +-- retrieve prior observations
  +-- PlannerAgent -> ExecutionPlan
  +-- ExecutionPlanNormalizer -> valid ordered plan
  +-- PlanExecutor
  |     |
  |     +-- resolve dependencies
  |     +-- parse tool requests
  |     +-- shared evidence cache lookup
  |     +-- run tool adapter if cache miss
  |     +-- ObservationExtractor
  |     +-- EvidenceLedger append
  |
  +-- TraceSynthesizerAgent -> TracePackage
  +-- TraceAuditorAgent -> AuditReport
  +-- update evidence memory/statuses
  |
  +-- accepted?
        |
        +-- no: next round, with audit feedback
        |
        +-- yes or max_rounds reached:
              +-- persist final trace
              +-- export benchmark format
              +-- write final_result
              +-- write debug bundle
```

## 22. Key Mental Hooks

- The planner asks for evidence; it does not answer the question directly.
- Tools produce structured evidence; they do not own the final trace.
- The observation extractor turns bulky tool results into searchable atomic facts.
- The synthesizer writes the reasoning trace and final answer from evidence.
- The auditor is the quality gate and repair signal.
- Preprocess cache is video-level context; evidence cache is exact tool-call reuse.
- The run ledger is append-only history for the current run.
- `round_XX` folders are the best way to understand why the system did what it did.

