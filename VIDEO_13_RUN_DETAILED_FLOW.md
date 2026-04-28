# Video 13 Pipeline Code Walkthrough

This is a code-and-artifact map for this concrete run:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5
```

The task in that run was:

```text
Question:
When the sound 'come to bill's ammunition' appears in the video, how many empty beer bottles are there on the table in the picture?

Options:
A.0.
B.1.
C.2.
D.3.
```

The important thing to keep in mind:

```text
Preprocess output does not directly answer the question.
Preprocess output is sent to the planner.
The planner emits an ExecutionPlan.
The executor turns plan input_refs into real tool requests.
Tool results become step_outputs, evidence entries, and atomic observations.
The synthesizer writes a trace from the current round's evidence.
The auditor decides whether that trace is justified.
If not, audit feedback is sent to the next planner round.
```

## 0. The Run Identity

Run manifest:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5/run_manifest.json
```

Relevant fields:

```json
{
  "benchmark": "adhoc",
  "sample_key": "video_13__e6e95b328626",
  "video_id": "video_13",
  "run_id": "20260426T161430Z_085af9d5",
  "mode": "generate",
  "preprocess_cache": "cache/preprocess/647390d7c1fb5c394891830996a9323a00e899f2a525f5e8291fd34a181838de/dense_caption/local_process__yaolily_TimeChat-Captioner-GRPO-7B/30/tool_v2/3cdf60b86217",
  "clip_duration_s": 30.0
}
```

Notice that `mode` is `generate`.

The task has `initial_trace_steps`, but in this code those are only used when `mode == "refine"`:

- `video_trace_pipeline/orchestration/pipeline.py:496-501`

So this run did not create `round_00`; it starts with `round_01`.

## 1. CLI to Runner

Console entry:

- `pyproject.toml:15-16`

```toml
[project.scripts]
vtp = "video_trace_pipeline.cli.main:main"
```

`main()` invokes the Typer app:

- `video_trace_pipeline/cli/main.py:580-585`

The `run` command:

- declares CLI options at `video_trace_pipeline/cli/main.py:443-475`
- builds the runner at `video_trace_pipeline/cli/main.py:477-483`
- builds tasks at `video_trace_pipeline/cli/main.py:486-501`
- calls `runner.run_task(...)` at `video_trace_pipeline/cli/main.py:502-510`

Task loading rules live at:

- direct JSON sample: `video_trace_pipeline/cli/main.py:175-191`
- direct video path: `video_trace_pipeline/cli/main.py:192-205`
- benchmark adapter: `video_trace_pipeline/cli/main.py:206-211`

In this example the manifest says:

```json
"metadata": {"source": "direct_cli"}
```

so this was a direct CLI task, not a benchmark adapter task.

## 2. Runner Construction

`PipelineRunner` wires the pipeline together:

- `video_trace_pipeline/orchestration/pipeline.py:318-353`

It creates:

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

The tool registry builds enabled tool adapters from `models_config.tools`:

- adapter construction: `video_trace_pipeline/tools/registry.py:193-267`
- tool catalog for the planner: `video_trace_pipeline/tools/registry.py:438-461`

For this run, the key tools are:

```text
asr
frame_retriever
generic_purpose
```

## 3. Run Directory Creation

The run directory shape is defined by `RunContext`:

- `video_trace_pipeline/storage/workspace.py:22-43`

It creates:

```text
<workspace_root>/runs/<video_id>/<run_id>/
  run_manifest.json
  runtime_snapshot.json
  trace_package.json
  benchmark_export.json
  final_result.json
  evidence/
  debug/
  round_01/
  round_02/
  round_03/
```

The actual old run has:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5/
  benchmark_export.json
  debug/
  evidence/
  final_result.json
  round_01/
  round_02/
  round_03/
  run_manifest.json
  runtime_snapshot.json
  trace_package.json
```

The runner creates the run and writes the first manifest here:

- `video_trace_pipeline/orchestration/pipeline.py:450-464`

It then writes a runtime snapshot:

- `video_trace_pipeline/orchestration/pipeline.py:392-406`

## 4. Preprocessing: What It Does

The full run calls preprocess here:

- `video_trace_pipeline/orchestration/pipeline.py:473-484`

Code:

```text
preprocess_output = self.preprocessor.get_or_build(task, clip_duration_s=clip_duration_s)
video_fingerprint = preprocess_output["video_fingerprint"]
planner_segments = list(preprocess_output.get("planner_segments") or [])
preprocess_planning_memory = dict(preprocess_output.get("planner_context") or {})
manifest_payload["preprocess_cache"] = preprocess_output["cache_dir"]
```

The preprocessor implementation is:

- settings resolution: `video_trace_pipeline/orchestration/preprocess.py:522-540`
- cache path calculation: `video_trace_pipeline/orchestration/preprocess.py:542-558`
- cache completeness check: `video_trace_pipeline/orchestration/preprocess.py:564-600`
- dense caption cache build: `video_trace_pipeline/orchestration/preprocess.py:620-635`
- optional ASR preprocess alignment: `video_trace_pipeline/orchestration/preprocess.py:637-644`
- planner context and planner segment creation: `video_trace_pipeline/orchestration/preprocess.py:645-646`
- cache writes: `video_trace_pipeline/orchestration/preprocess.py:647-673`

For this run, the preprocess cache is:

```text
workspace/cache/preprocess/647390d7c1fb5c394891830996a9323a00e899f2a525f5e8291fd34a181838de/dense_caption/local_process__yaolily_TimeChat-Captioner-GRPO-7B/30/tool_v2/3cdf60b86217/
```

It contains:

```text
manifest.json
segments.json
planner_segments.json
planner_context.json
```

The Python object returned by `DenseCaptionPreprocessor.get_or_build(...)` has this exact top-level shape:

- `video_trace_pipeline/orchestration/preprocess.py:606-614` when the cache already exists
- `video_trace_pipeline/orchestration/preprocess.py:665-672` when the cache is freshly built

```json
{
  "cache_hit": true,
  "cache_dir": "cache/preprocess/647390d7c1fb5c394891830996a9323a00e899f2a525f5e8291fd34a181838de/dense_caption/local_process__yaolily_TimeChat-Captioner-GRPO-7B/30/tool_v2/3cdf60b86217",
  "manifest": {},
  "segments": [],
  "planner_segments": [],
  "planner_context": {},
  "video_fingerprint": "647390d7c1fb5c394891830996a9323a00e899f2a525f5e8291fd34a181838de"
}
```

In this old run, that full object was also copied into:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5/final_result.json
  .preprocess
```

Only two preprocess fields are sent into the planner prompt:

```text
preprocess_output["planner_segments"]
  -> planner_segments
  -> planner_request.user_prompt PREPROCESS_SEGMENTS

preprocess_output["planner_context"]
  -> preprocess_planning_memory
  -> planner_request.user_prompt PREPROCESS_PLANNING_MEMORY
```

The raw `segments` list is not directly placed into the planner prompt. It is retained in the cache/final result as the richer backing artifact.

## 5. Preprocessing: Actual Output

### 5.1 `manifest.json`

File:

```text
workspace/cache/preprocess/.../manifest.json
```

Actual content:

```json
{
  "video_fingerprint": "647390d7c1fb5c394891830996a9323a00e899f2a525f5e8291fd34a181838de",
  "clip_duration_s": 30.0,
  "model_id": "local_process__yaolily/TimeChat-Captioner-GRPO-7B",
  "prompt_version": "tool_v2",
  "preprocess_settings": {
    "clip_duration_s": 30.0,
    "sample_frames": 6,
    "fps": 1.0,
    "max_frames": 96,
    "use_audio_in_video": false,
    "include_asr": true,
    "collect_sampled_frames": false,
    "max_new_tokens": 700
  },
  "preprocess_signature": "3cdf60b86217",
  "include_asr": true,
  "segment_count": 6,
  "planner_segment_count": 6,
  "dense_caption_span_count": 14,
  "transcript_segment_count": 5,
  "identity_memory_count": 6,
  "audio_event_memory_count": 12
}
```

Meaning:

- Video was split into 6 chunks of 30 seconds.
- Dense captioning produced 14 caption spans.
- ASR preprocess produced 5 transcript spans.
- The planner receives compacted planner segments and planner context.

### 5.2 `segments.json`

`segments.json` is the raw-ish preprocess bundle. It keeps the dense captioner output in the original tool-like shape.

For the question-relevant chunk:

```json
{
  "start": 120.0,
  "end": 150.0,
  "dense_caption": {
    "clips": [
      {
        "video_id": "video_13",
        "start_s": 120.0,
        "end_s": 150.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      }
    ],
    "captions": [
      {
        "start": 120.0,
        "end": 125.0,
        "visual": "A montage of chaotic scenes unfolds: a woman in a black trench coat runs from a red helicopter on a rainy street; a man in a car looks distressed as he swerves; a group of men, including one in a purple shirt, are held captive with their hands behind their heads; a man in a neon-lit club dances with a woman in a bikini top; a man in a red shirt runs down a street with a gun in his hand.",
        "audio": "speech: None.; acoustics: 1) Tone of speech: N/A. 2) Background sounds or music: A fast, percussive, and high-energy electronic track plays, punctuated by sound effects like rain, a car engine, and a helicopter.",
        "on_screen_text": "",
        "actions": [],
        "objects": [],
        "attributes": [
          "camera_state: ...",
          "video_background: ...",
          "storyline: ...",
          "shooting_style: ..."
        ]
      },
      {
        "start": 126.0,
        "end": 130.0,
        "visual": "A man in a black t-shirt and sunglasses sits on a red motorcycle in a dark, industrial garage, holding a rifle. The scene cuts to a man in a black shirt walking across a rooftop at night, with the city lights in the background. This is followed by a shot of a helicopter flying low over a lake at sunset, and a man and woman on a jet ski, creating a large splash.",
        "audio": "speech: None.; acoustics: 1) Tone of speech: N/A. 2) Background sounds or music: The high-energy electronic music continues, with added sound effects of a motorcycle engine and splashing water.",
        "on_screen_text": "",
        "actions": [],
        "objects": [],
        "attributes": [
          "camera_state: ...",
          "video_background: ...",
          "storyline: ...",
          "shooting_style: ..."
        ]
      }
    ],
    "overall_summary": "...",
    "captioned_range": {
      "start_s": 120.0,
      "end_s": 150.0
    },
    "sampled_frames": [],
    "backend": "timechat_captioner_qwen25_omni"
  },
  "transcript_segments": [
    {
      "start_s": 129.125,
      "end_s": 158.167,
      "text": "Come to Phil's Ammu Nation today. We got more guns than the law allows. Hey. Hey, if you got friends, can you hook me up, please? Oh!",
      "speaker_id": null,
      "confidence": null
    }
  ]
}
```

Important:

- Preprocess found a rough transcript anchor at `129.125s`.
- It transcribed `Ammu Nation`.
- Later targeted ASR transcribed a different string: `Heavenly Nation`.
- This mismatch becomes important in the audit loop.

### 5.3 `planner_segments.json`

`planner_segments.json` is normalized and compacted for the planner.

Question-relevant segment:

```json
{
  "start_s": 120.0,
  "end_s": 150.0,
  "dense_caption_spans": [
    {
      "start_s": 120.0,
      "end_s": 125.0,
      "visual": "A montage of chaotic scenes unfolds: ...",
      "audio": [
        "1) Tone of speech: N/A. 2) Background sounds or music: A fast, percussive, and high-energy electronic track plays, punctuated by sound effects like rain, a car engine, and a helicopter."
      ]
    },
    {
      "start_s": 126.0,
      "end_s": 130.0,
      "visual": "A man in a black t-shirt and sunglasses sits on a red motorcycle in a dark, industrial garage, holding a rifle. ...",
      "audio": [
        "1) Tone of speech: N/A. 2) Background sounds or music: The high-energy electronic music continues, with added sound effects of a motorcycle engine and splashing water."
      ]
    }
  ],
  "transcript_spans": [
    {
      "start_s": 129.125,
      "end_s": 158.167,
      "text": "Come to Phil's Ammu Nation today. We got more guns than the law allows. Hey. Hey, if you got friends, can you hook me up, please? Oh!"
    }
  ]
}
```

This is exactly the broad context the planner sees.

The full `planner_segments.json` has 6 entries:

```json
[
  {
    "start_s": 0.0,
    "end_s": 30.0,
    "dense_caption_span_count": 2,
    "transcript_span_count": 1
  },
  {
    "start_s": 30.0,
    "end_s": 60.0,
    "dense_caption_span_count": 2,
    "transcript_span_count": 1
  },
  {
    "start_s": 60.0,
    "end_s": 90.0,
    "dense_caption_span_count": 2,
    "transcript_span_count": 1
  },
  {
    "start_s": 90.0,
    "end_s": 120.0,
    "dense_caption_span_count": 3,
    "transcript_span_count": 1
  },
  {
    "start_s": 120.0,
    "end_s": 150.0,
    "dense_caption_span_count": 2,
    "transcript_span_count": 1
  },
  {
    "start_s": 150.0,
    "end_s": 166.733,
    "dense_caption_span_count": 3,
    "transcript_span_count": 0
  }
]
```

So the planner can scan the whole video at coarse resolution, then choose a narrow tool call. For this question it picked the `120.0-150.0s` segment because that segment's transcript span contains the `Ammu Nation` line.

### 5.4 `planner_context.json`

`planner_context.json` contains deterministic memory derived from preprocess:

```json
{
  "identity_memory": [
    {
      "label": "Let Freedom Reign",
      "kind": "on_screen_text",
      "modalities": ["on_screen_text"],
      "time_ranges": [{"start_s": 60.0, "end_s": 65.0}],
      "mention_count": 1
    }
  ],
  "audio_event_memory": [
    {
      "label": "1) Tone of speech: N/A. 2) Background sounds or music: A fast, percussive, and high-energy electr...",
      "time_ranges": [{"start_s": 120.0, "end_s": 125.0}],
      "mention_count": 1
    }
  ]
}
```

The actual file has more entries, but this shows the structure.

## 6. How Preprocess Is Sent Forward

Preprocess output is passed to the planner, not directly to ASR/frame/generic tools.

The runner pulls these out:

- `video_trace_pipeline/orchestration/pipeline.py:476-481`

```text
planner_segments = preprocess_output["planner_segments"]
preprocess_planning_memory = preprocess_output["planner_context"]
```

Then it builds `planner_kwargs`:

- `video_trace_pipeline/orchestration/pipeline.py:581-590`

```text
planner_kwargs = dict(
  task=task,
  mode=planning_mode,
  planner_segments=planner_segments,
  compact_rounds=compact_rounds,
  retrieved_observations=retrieved_observations,
  preprocess_planning_memory=preprocess_planning_memory,
  audit_feedback=latest_audit.dict() if latest_audit is not None else None,
  tool_catalog=self.tool_registry.tool_catalog(),
  evidence_memory=compact_evidence_memory(evidence_memory),
)
```

The planner prompt builder inserts those fields:

- task/question/options/tool catalog: `video_trace_pipeline/prompts/planner_prompt.py:253-264`
- preprocess segments: `video_trace_pipeline/prompts/planner_prompt.py:266-276`
- preprocess planning memory: `video_trace_pipeline/prompts/planner_prompt.py:278-288`
- previous rounds: `video_trace_pipeline/prompts/planner_prompt.py:290-300`
- evidence memory: `video_trace_pipeline/prompts/planner_prompt.py:302-312`
- retrieved observations: `video_trace_pipeline/prompts/planner_prompt.py:314-324`
- retrieved evidence ids: `video_trace_pipeline/prompts/planner_prompt.py:326-336`
- audit diagnosis: `video_trace_pipeline/prompts/planner_prompt.py:338-339`

Actual `round_01/planner_request.json` contains a `user_prompt` with:

```text
MODE: generate

QUESTION:
When the sound 'come to bill's ammunition' appears in the video, how many empty beer bottles are there on the table in the picture?

OPTIONS:
[
  "A.0.",
  "B.1.",
  "C.2.",
  "D.3."
]

AVAILABLE_TOOLS:
...

PREPROCESS_SEGMENTS:
[
  ...
  {
    "start_s": 120.0,
    "end_s": 150.0,
    ...
    "transcript_spans": [
      {
        "start_s": 129.125,
        "end_s": 158.167,
        "text": "Come to Phil's Ammu Nation today. ..."
      }
    ]
  }
]
```

So the planner had a rough anchor: around `129s`, with text resembling the user quote.

## 7. Planner Output: Round 1

The planner request is created and saved here:

- build request: `video_trace_pipeline/orchestration/pipeline.py:592-594`
- complete request and parse plan: `video_trace_pipeline/orchestration/pipeline.py:595`
- normalize plan: `video_trace_pipeline/orchestration/pipeline.py:596-600`
- write plan: `video_trace_pipeline/orchestration/pipeline.py:601`

Planner agent code:

- prompt build: `video_trace_pipeline/agents/planner.py:54-84`
- JSON completion and `ExecutionPlan` validation: `video_trace_pipeline/agents/planner.py:86-89`

Round 1 plan file:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5/round_01/planner_plan.json
```

The plan has 3 steps:

```json
[
  {
    "step_id": 1,
    "tool_name": "asr",
    "purpose": "Verify the exact timing of the spoken line sounding like 'come to bill's ammunition' and bound the relevant clip around that utterance.",
    "arguments": {
      "clips": [
        {
          "video_id": "video",
          "start_s": 124.0,
          "end_s": 133.0,
          "artifact_id": null,
          "relpath": null,
          "metadata": {}
        }
      ],
      "speaker_attribution": false
    },
    "input_refs": [],
    "depends_on": []
  },
  {
    "step_id": 2,
    "tool_name": "frame_retriever",
    "arguments": {
      "clips": [],
      "num_frames": 6,
      "query": "table with beer bottles visible in the scene during the spoken line 'Come to Phil's Ammu Nation today'; retrieve frames where the number of empty beer bottles on the table can be counted",
      "time_hints": [
        "around the moment when the transcript says 'Come to Phil's Ammu Nation today'",
        "frames showing the pictured scene with a table and bottles",
        "prefer clear, stable views where the tabletop is visible"
      ]
    },
    "input_refs": [
      {
        "target_field": "clips",
        "source": {
          "step_id": 1,
          "field_path": "clips"
        }
      }
    ],
    "depends_on": [1]
  },
  {
    "step_id": 3,
    "tool_name": "generic_purpose",
    "arguments": {
      "clips": [],
      "evidence_ids": [],
      "frames": [],
      "query": "In these frames from the moment when the audio says 'Come to Phil's Ammu Nation today,' count the number of empty beer bottles that are on the table in the pictured scene. Only count bottles clearly resting on the table, and report the visible count among 0, 1, 2, or 3.",
      "text_contexts": [],
      "transcripts": []
    },
    "input_refs": [
      {
        "target_field": "frames",
        "source": {
          "step_id": 2,
          "field_path": "frames"
        }
      },
      {
        "target_field": "transcripts",
        "source": {
          "step_id": 1,
          "field_path": "transcripts"
        }
      }
    ],
    "depends_on": [1, 2]
  }
]
```

This is the key dataflow:

```text
step 1 ASR
  outputs clips + transcripts

step 2 frame_retriever
  starts with clips=[]
  input_ref fills clips from step 1 output field "clips"

step 3 generic_purpose
  starts with frames=[] and transcripts=[]
  input_ref fills frames from step 2 output field "frames"
  input_ref fills transcripts from step 1 output field "transcripts"
```

## 8. Plan Normalization

The LLM planner output is normalized before execution.

Relevant code:

- allowed fields from adapter schemas: `video_trace_pipeline/orchestration/plan_normalizer.py:117-123`
- normalize one step: `video_trace_pipeline/orchestration/plan_normalizer.py:166-236`
- validate refs and dependency rules: `video_trace_pipeline/orchestration/plan_normalizer.py:282-320`
- require context for `generic_purpose`: `video_trace_pipeline/orchestration/plan_normalizer.py:322-337`
- topological ordering: `video_trace_pipeline/orchestration/plan_normalizer.py:339-368`
- resequencing: `video_trace_pipeline/orchestration/plan_normalizer.py:370-393`
- normalize entry point: `video_trace_pipeline/orchestration/plan_normalizer.py:408-419`

For this run, the normalized plan keeps the chain:

```text
ASR -> frame_retriever -> generic_purpose
```

## 9. Executor: How Outputs Move to the Next Tool

The executor is where `input_refs` become real request JSON.

Core code:

- `PlanExecutor.execute_plan(...)`: `video_trace_pipeline/orchestration/executor.py:288-454`
- argument resolution: `video_trace_pipeline/orchestration/executor.py:173-187`

Argument resolution code:

```text
resolved = dict(step.arguments or {})
for binding in step.input_refs:
    source_obj = step_outputs.get(binding.source.step_id)
    value = traverse_path(source_obj, binding.source.field_path)
    existing_value = traverse_path(resolved, binding.target_field)
    value = _merge_dependency_values(existing_value, value, binding.target_field)
    assign_path(resolved, binding.target_field, value)
return resolved
```

The important runtime variable is `step_outputs`:

- after each tool result, executor builds `step_output_payload`: `video_trace_pipeline/orchestration/executor.py:424-432`
- then stores it as `step_outputs[step.step_id]`: `video_trace_pipeline/orchestration/executor.py:433`

So for step 1 ASR:

```text
step_outputs[1] = {
  clips: ...,
  text: ...,
  segments: ...,
  transcripts: ...,
  backend: ...,
  phrase_matches: ...,
  phrase_match_summary: ...,
  summary: ...,
  raw_output_text: ...
}
```

For step 2 frame retrieval:

```text
input_ref:
  target_field = clips
  source.step_id = 1
  source.field_path = clips

executor:
  source_obj = step_outputs[1]
  value = source_obj["clips"]
  resolved["clips"] = value
```

For step 3 generic-purpose:

```text
input_ref 1:
  target_field = frames
  source.step_id = 2
  source.field_path = frames

input_ref 2:
  target_field = transcripts
  source.step_id = 1
  source.field_path = transcripts

executor:
  resolved["frames"] = step_outputs[2]["frames"]
  resolved["transcripts"] = step_outputs[1]["transcripts"]
```

This is not implicit magic. It is exactly the `input_refs` loop in `executor.py:173-187`.

## 10. Tool Step 1: ASR

### 10.1 Plan step

From `round_01/planner_plan.json`:

```json
{
  "step_id": 1,
  "tool_name": "asr",
  "arguments": {
    "clips": [
      {
        "video_id": "video",
        "start_s": 124.0,
        "end_s": 133.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      }
    ],
    "speaker_attribution": false
  },
  "input_refs": []
}
```

### 10.2 Actual request sent

Saved at:

```text
round_01/tools/01_asr/request.json
```

Actual:

```json
{
  "tool_name": "asr",
  "clips": [
    {
      "video_id": "video",
      "start_s": 124.0,
      "end_s": 133.0,
      "artifact_id": null,
      "relpath": null,
      "metadata": {}
    }
  ],
  "speaker_attribution": false
}
```

This request is written by:

- `video_trace_pipeline/orchestration/executor.py:349-351`

### 10.3 ASR adapter code

ASR adapter builds output here:

- clip normalization and audio extraction: `video_trace_pipeline/tools/local_asr.py:383-406`
- WhisperX transcription call: `video_trace_pipeline/tools/local_asr.py:407-417`
- segment timestamps shifted into global video time: `video_trace_pipeline/tools/local_asr.py:446-458`
- phrase matching against quoted task text: `video_trace_pipeline/tools/local_asr.py:459-460`
- `data` payload with clips/text/segments/transcripts: `video_trace_pipeline/tools/local_asr.py:477-483`
- phrase match summary: `video_trace_pipeline/tools/local_asr.py:484-498`
- final `ToolResult`: `video_trace_pipeline/tools/local_asr.py:499-508`

### 10.4 Actual ASR result

Saved at:

```text
round_01/tools/01_asr/result.json
```

Important fields:

```json
{
  "data": {
    "clips": [
      {
        "video_id": "video",
        "start_s": 124.0,
        "end_s": 133.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      }
    ],
    "text": "Come to Phil's Heavenly Nation today. We got more guns.",
    "segments": [
      {
        "start_s": 124.031,
        "end_s": 132.975,
        "text": "Come to Phil's Heavenly Nation today. We got more guns.",
        "speaker_id": null,
        "confidence": null
      }
    ],
    "transcripts": [
      {
        "transcript_id": "tx_b621a75c149c",
        "clip": {
          "video_id": "video",
          "start_s": 124.0,
          "end_s": 133.0,
          "artifact_id": null,
          "relpath": null,
          "metadata": {}
        },
        "text": "Come to Phil's Heavenly Nation today. We got more guns.",
        "segments": [
          {
            "start_s": 124.031,
            "end_s": 132.975,
            "text": "Come to Phil's Heavenly Nation today. We got more guns.",
            "speaker_id": null,
            "confidence": null
          }
        ],
        "metadata": {
          "backend": "whisperx_local"
        }
      }
    ],
    "phrase_matches": [
      {
        "phrase": "come to bill",
        "matched_text": "Come to Phil's Heavenly Nation today. We got more guns.",
        "similarity": 0.3385
      }
    ],
    "phrase_match_summary": "Closest ASR match for quoted phrase \"come to bill\" is \"Come to Phil's Heavenly Nation today. We got more guns.\" (similarity=0.34)."
  },
  "summary": "Closest ASR match for quoted phrase \"come to bill\" is \"Come to Phil's Heavenly Nation today. We got more guns.\" (similarity=0.34). Come to Phil's Heavenly Nation today. We got more guns."
}
```

### 10.5 ASR observations

Observation extraction dispatch:

- `video_trace_pipeline/tools/extractors.py:72-98`

ASR observation construction:

- `video_trace_pipeline/tools/extractors.py:237-283`

Saved at:

```text
round_01/tools/01_asr/observations.json
```

Examples:

```json
[
  {
    "observation_id": "obs_19f0681cc07f6090",
    "subject": "unknown_speaker",
    "predicate": "said",
    "object_text": "Come to Phil's Heavenly Nation today.",
    "time_start_s": 124.031,
    "time_end_s": 132.975,
    "source_tool": "asr",
    "atomic_text": "unknown_speaker said \"Come to Phil's Heavenly Nation today.\" from 124.03s to 132.97s."
  },
  {
    "observation_id": "obs_9854e1e4b807855d",
    "subject": "unknown_speaker",
    "predicate": "said",
    "object_text": "We got more guns.",
    "time_start_s": 124.031,
    "time_end_s": 132.975,
    "source_tool": "asr",
    "atomic_text": "unknown_speaker said \"We got more guns.\" from 124.03s to 132.97s."
  }
]
```

### 10.6 ASR evidence entry

Executor creates the evidence entry here:

- `video_trace_pipeline/orchestration/executor.py:407-418`

It is appended to JSONL and SQLite here:

- `video_trace_pipeline/storage/evidence_store.py:155-178`

Actual raw evidence entry from:

```text
evidence/evidence_index.jsonl
```

```json
{
  "evidence_id": "ev_01_19acc2ce",
  "tool_name": "asr",
  "evidence_text": "Closest ASR match for quoted phrase \"come to bill\" is \"Come to Phil's Heavenly Nation today. We got more guns.\" (similarity=0.34). Come to Phil's Heavenly Nation today. We got more guns.",
  "status": "provisional",
  "time_start_s": 124.031,
  "time_end_s": 132.975,
  "observation_ids": [
    "obs_19f0681cc07f6090",
    "obs_9854e1e4b807855d",
    "obs_c99f8916384e4aa2",
    "obs_5885d9083a4f3986"
  ],
  "metadata": {
    "request_hash": "19acc2cea379f59ae926b0b9",
    "cache_hit": false
  }
}
```

## 11. Tool Step 2: Frame Retriever

### 11.1 Plan says `clips: []`

The round 1 plan step has:

```json
{
  "step_id": 2,
  "tool_name": "frame_retriever",
  "arguments": {
    "clips": [],
    "num_frames": 6,
    "query": "table with beer bottles visible in the scene during the spoken line 'Come to Phil's Ammu Nation today'; retrieve frames where the number of empty beer bottles on the table can be counted",
    "time_hints": [
      "around the moment when the transcript says 'Come to Phil's Ammu Nation today'",
      "frames showing the pictured scene with a table and bottles",
      "prefer clear, stable views where the tabletop is visible"
    ]
  },
  "input_refs": [
    {
      "target_field": "clips",
      "source": {
        "step_id": 1,
        "field_path": "clips"
      }
    }
  ]
}
```

The empty `clips` list is filled by the executor from `step_outputs[1]["clips"]`.

### 11.2 Actual frame retriever request

Saved at:

```text
round_01/tools/02_frame_retriever/request.json
```

Actual request after dependency resolution:

```json
{
  "tool_name": "frame_retriever",
  "clips": [
    {
      "video_id": "video",
      "start_s": 124.0,
      "end_s": 133.0,
      "artifact_id": null,
      "relpath": null,
      "metadata": {}
    }
  ],
  "time_hints": [
    "around the moment when the transcript says 'Come to Phil's Ammu Nation today'",
    "frames showing the pictured scene with a table and bottles",
    "prefer clear, stable views where the tabletop is visible"
  ],
  "query": "table with beer bottles visible in the scene during the spoken line 'Come to Phil's Ammu Nation today'; retrieve frames where the number of empty beer bottles on the table can be counted",
  "num_frames": 6
}
```

This is the first concrete "output of tool A becomes input of tool B" moment:

```text
ASR result data.clips
  -> executor step_outputs[1].clips
  -> frame_retriever request.clips
```

### 11.3 Frame retriever adapter code

Key code:

- process/persistent JSON wrapper: `video_trace_pipeline/tools/process_adapters.py:245-275`
- request envelope sent to subprocess or persistent runner: `video_trace_pipeline/tools/process_adapters.py:191-209`
- subprocess JSON execution: `video_trace_pipeline/tools/process_adapters.py:218-242`
- frame result conversion into `FrameRef`s: `video_trace_pipeline/tools/process_adapters.py:723-758`
- multi-clip/time-hint merge path: `video_trace_pipeline/tools/process_adapters.py:820-895`

### 11.4 Actual frame retriever result

Saved at:

```text
round_01/tools/02_frame_retriever/result.json
```

Important fields:

```json
{
  "data": {
    "query": "table with beer bottles visible in the scene during the spoken line 'Come to Phil's Ammu Nation today'; retrieve frames where the number of empty beer bottles on the table can be counted",
    "clips": [
      {
        "video_id": "video",
        "start_s": 124.0,
        "end_s": 133.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      }
    ],
    "frames": [
      {
        "video_id": "video_13",
        "timestamp_s": 132.0,
        "artifact_id": "b27ea19b17efb368d7e15f95",
        "relpath": "cache/artifacts/b27ea19b17efb368d7e15f95/frame_132.00.png",
        "clip": {
          "video_id": "video",
          "start_s": 124.0,
          "end_s": 133.0,
          "artifact_id": null,
          "relpath": null,
          "metadata": {}
        },
        "metadata": {
          "relevance_score": 0.4148074984550476,
          "device": "cuda:2",
          "clip_start_s": 124.0,
          "clip_end_s": 133.0,
          "temporal_score": 0.519699,
          "anchor_distance_s": null,
          "selection_reason": "structured_visual_plateau_center"
        }
      },
      {
        "video_id": "video_13",
        "timestamp_s": 131.0,
        "artifact_id": "548d093ab60ef377747d8e86",
        "relpath": "cache/artifacts/548d093ab60ef377747d8e86/frame_131.00.png",
        "clip": {
          "video_id": "video",
          "start_s": 124.0,
          "end_s": 133.0,
          "artifact_id": null,
          "relpath": null,
          "metadata": {}
        },
        "metadata": {
          "relevance_score": 0.40685462951660156,
          "device": "cuda:2",
          "clip_start_s": 124.0,
          "clip_end_s": 133.0,
          "temporal_score": 0.510244,
          "anchor_distance_s": null,
          "selection_reason": "structured_visual_temporal_rerank"
        }
      }
    ],
    "mode": "multi_clip_bounded",
    "cache_metadata": {
      "dense_frame_cache_hit": true,
      "dense_frame_count": 166,
      "bounded_frame_count": 10,
      "embedding_cache_ready": false
    },
    "rationale": "Frames were ranked within the requested clip using the configured Qwen visual embedder, with time-hint-aware temporal reranking."
  },
  "summary": "Retrieved 6 frame(s) across 1 input clip(s)."
}
```

There are 6 frames total:

```text
132.00s
131.00s
133.00s
130.00s
128.00s
129.00s
```

### 11.5 Frame retriever observations

Frame observation code:

- `video_trace_pipeline/tools/extractors.py:217-235`

Examples:

```json
[
  {
    "observation_id": "obs_ff4708481942ff5b",
    "subject": "requested frame",
    "predicate": "retrieved_frame_at",
    "object_text": "132.00s",
    "frame_ts_s": 132.0,
    "confidence": 0.4148074984550476,
    "source_tool": "frame_retriever",
    "atomic_text": "A candidate frame was retrieved at 132.00s."
  },
  {
    "observation_id": "obs_050900e226333aaa",
    "subject": "requested frame",
    "predicate": "retrieved_frame_at",
    "object_text": "131.00s",
    "frame_ts_s": 131.0,
    "confidence": 0.40685462951660156,
    "source_tool": "frame_retriever",
    "atomic_text": "A candidate frame was retrieved at 131.00s."
  }
]
```

## 12. Tool Step 3: Generic Purpose

### 12.1 Plan says `frames: []`, `transcripts: []`

From `round_01/planner_plan.json`:

```json
{
  "step_id": 3,
  "tool_name": "generic_purpose",
  "arguments": {
    "clips": [],
    "evidence_ids": [],
    "frames": [],
    "query": "In these frames from the moment when the audio says 'Come to Phil's Ammu Nation today,' count the number of empty beer bottles that are on the table in the pictured scene. Only count bottles clearly resting on the table, and report the visible count among 0, 1, 2, or 3.",
    "text_contexts": [],
    "transcripts": []
  },
  "input_refs": [
    {
      "target_field": "frames",
      "source": {
        "step_id": 2,
        "field_path": "frames"
      }
    },
    {
      "target_field": "transcripts",
      "source": {
        "step_id": 1,
        "field_path": "transcripts"
      }
    }
  ]
}
```

### 12.2 Actual generic-purpose request

Saved at:

```text
round_01/tools/03_generic_purpose/request.json
```

Actual request after dependency resolution:

```json
{
  "tool_name": "generic_purpose",
  "frames_count": 6,
  "transcripts_count": 1,
  "query": "In these frames from the moment when the audio says 'Come to Phil's Ammu Nation today,' count the number of empty beer bottles that are on the table in the pictured scene. Only count bottles clearly resting on the table, and report the visible count among 0, 1, 2, or 3.",
  "frames": [
    {
      "video_id": "video_13",
      "timestamp_s": 132.0,
      "artifact_id": "b27ea19b17efb368d7e15f95",
      "relpath": "cache/artifacts/b27ea19b17efb368d7e15f95/frame_132.00.png",
      "clip": {
        "video_id": "video",
        "start_s": 124.0,
        "end_s": 133.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      },
      "metadata": {
        "source_path": "/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline/workspace/cache/tool_wrappers/reference/video_13/dense_frames/video_13/frame_132.00.png",
        "relevance_score": 0.4148074984550476,
        "device": "cuda:2",
        "clip_start_s": 124.0,
        "clip_end_s": 133.0,
        "temporal_score": 0.519699,
        "anchor_distance_s": null,
        "selection_reason": "structured_visual_plateau_center"
      }
    }
  ],
  "transcripts": [
    {
      "transcript_id": "tx_b621a75c149c",
      "clip": {
        "video_id": "video",
        "start_s": 124.0,
        "end_s": 133.0,
        "artifact_id": null,
        "relpath": null,
        "metadata": {}
      },
      "relpath": null,
      "text": "Come to Phil's Heavenly Nation today. We got more guns.",
      "segments": [
        {
          "start_s": 124.031,
          "end_s": 132.975,
          "text": "Come to Phil's Heavenly Nation today. We got more guns.",
          "speaker_id": null,
          "confidence": null
        }
      ],
      "metadata": {
        "backend": "whisperx_local"
      }
    }
  ]
}
```

The file contains all 6 frames; the snippet above shows one frame to keep this readable.

This is the second concrete handoff:

```text
frame_retriever result data.frames
  -> executor step_outputs[2].frames
  -> generic_purpose request.frames

ASR result data.transcripts
  -> executor step_outputs[1].transcripts
  -> generic_purpose request.transcripts
```

### 12.3 Generic-purpose adapter code

Generic-purpose request parsing and execution:

- request schema parse: `video_trace_pipeline/tools/process_adapters.py:1399-1405`
- execute wrapper: `video_trace_pipeline/tools/process_adapters.py:1413-1416`
- output data fields: `video_trace_pipeline/tools/process_adapters.py:1423-1429`
- final `ToolResult`: `video_trace_pipeline/tools/process_adapters.py:1430-1446`

All process-backed tools use the same envelope:

- request envelope shape: `video_trace_pipeline/tools/process_adapters.py:191-209`
- command execution: `video_trace_pipeline/tools/process_adapters.py:218-242`

The envelope sent to the model runner contains:

```text
tool_name
request
task
runtime
evidence_records
```

The persisted `request.json` is just the `request` part. `runtime.json` stores the runtime part.

### 12.4 Actual generic-purpose result

Saved at:

```text
round_01/tools/03_generic_purpose/result.json
```

Important fields:

```json
{
  "data": {
    "answer": "The user wants to count empty beer bottles on a table during a specific audio segment.\n\n1.  **Identify the audio segment:** The transcript says \"Come to Phil's Heavenly Nation today. We got more guns.\" This corresponds to the first four images provided, which show a TV screen with a man in a tank top (presumably Phil) in a gun shop setting. The text on the TV screen says \"PHIL'S\" and \"AMMO NATION\" (or similar).\n\n2.  **Analyze the visual scene:**\n    *   The scene is a room with a TV in the center.\n    *   In the foreground, there is a table or counter.\n    *   On this table, there are objects.\n    *   I see a white game controller on the left.\n    *   I see two glass bottles in the center-right foreground.\n    *   ...\n    *   Bottle 1 looks like it has some liquid.\n    *   Bottle 2 looks like it has liquid in it.",
    "analysis": "same content as answer",
    "supporting_points": [],
    "confidence": null
  },
  "summary": "The user wants to count empty beer bottles on a table during a specific audio segment. ..."
}
```

### 12.5 Generic-purpose observations

Generic-purpose extraction code:

- dispatch: `video_trace_pipeline/tools/extractors.py:72-98`
- generic-purpose extraction: `video_trace_pipeline/tools/extractors.py:540-565`

Important observations from this step:

```json
[
  {
    "observation_id": "obs_7fa818f5045442ec",
    "subject": "table",
    "predicate": "has",
    "object_text": "two glass bottles in the center-right foreground",
    "atomic_text": "There are two glass bottles in the center-right foreground on the table."
  },
  {
    "observation_id": "obs_b88f93960f578545",
    "subject": "bottle 1",
    "predicate": "appears_to_be",
    "object_text": "a beer bottle",
    "atomic_text": "Bottle 1 appears to be a beer bottle."
  },
  {
    "observation_id": "obs_87b51b65f16d086e",
    "subject": "bottle 2",
    "predicate": "appears_to_be",
    "object_text": "a beer bottle",
    "atomic_text": "Bottle 2 appears to be a beer bottle."
  },
  {
    "observation_id": "obs_e0b00b5327942b43",
    "subject": "bottle 1",
    "predicate": "looks_like",
    "object_text": "it has some liquid",
    "atomic_text": "Bottle 1 looks like it has some liquid."
  },
  {
    "observation_id": "obs_3229a613e2d3484e",
    "subject": "bottle 2",
    "predicate": "looks_like",
    "object_text": "it has liquid in it",
    "atomic_text": "Bottle 2 looks like it has liquid in it."
  }
]
```

## 13. Shared Tool Cache and Request Hash

Before running a tool, executor hashes:

- tool name
- request payload
- video fingerprint
- tool implementation
- model name
- prompt version
- extraction version

Code:

- request hash: `video_trace_pipeline/orchestration/executor.py:338-348`
- shared cache lookup: `video_trace_pipeline/orchestration/executor.py:349`
- cache lock / execute / store: `video_trace_pipeline/orchestration/executor.py:364-404`

This run mostly executed fresh tool calls. Example ASR metadata:

```json
{
  "request_hash": "19acc2cea379f59ae926b0b9",
  "cache_hit": false
}
```

## 14. Evidence Ledger

The executor appends every tool result to the ledger:

- evidence entry creation: `video_trace_pipeline/orchestration/executor.py:407-418`
- writes per-step files: `video_trace_pipeline/orchestration/executor.py:419-422`
- appends JSONL and SQLite: `video_trace_pipeline/storage/evidence_store.py:155-178`

Ledger files:

```text
evidence/evidence_index.jsonl
evidence/atomic_observations.jsonl
evidence/evidence.sqlite3
```

SQLite schema:

- evidence entries table: `video_trace_pipeline/storage/evidence_store.py:89-98`
- atomic observations table: `video_trace_pipeline/storage/evidence_store.py:100-121`
- indexes: `video_trace_pipeline/storage/evidence_store.py:124-128`

The ledger is later searched with:

- `video_trace_pipeline/storage/evidence_store.py:335-390`

The lookup-by-evidence-id path used by `generic_purpose` `evidence_ids` is:

- `video_trace_pipeline/storage/evidence_store.py:261-333`

## 15. Synthesizer: What It Receives

After all round tools execute, the runner builds synthesis context:

- execution records returned from executor: `video_trace_pipeline/orchestration/pipeline.py:609-616`
- round evidence and observations extracted: `video_trace_pipeline/orchestration/pipeline.py:618`
- synthesizer request built: `video_trace_pipeline/orchestration/pipeline.py:619-628`
- request written: `video_trace_pipeline/orchestration/pipeline.py:629`
- LLM completes trace package: `video_trace_pipeline/orchestration/pipeline.py:630`
- runner patches task key, mode, benchmark rendering: `video_trace_pipeline/orchestration/pipeline.py:631-635`
- round trace written: `video_trace_pipeline/orchestration/pipeline.py:636`

Synthesizer prompt builder:

- `video_trace_pipeline/prompts/trace_synthesizer_prompt.py:60-134`

It includes:

```text
TASK_KEY
MODE
QUESTION
OPTIONS
PRIOR_AUDIT_DIAGNOSIS, if refining
CURRENT_TRACE_PACKAGE, if refining
EVIDENCE_MEMORY, if available
ROUND_EVIDENCE_ENTRIES
ROUND_ATOMIC_OBSERVATIONS
REFINEMENT_INSTRUCTIONS
TracePackage schema reminder
```

Actual `round_01/synthesizer_request.json` contains:

```text
ROUND_EVIDENCE_ENTRIES:
[
  {
    "evidence_id": "ev_01_19acc2ce",
    "tool_name": "asr",
    "evidence_text": "Closest ASR match for quoted phrase ..."
  },
  {
    "evidence_id": "ev_02_f4885d28",
    "tool_name": "frame_retriever",
    "evidence_text": "Retrieved 6 frame(s) across 1 input clip(s)."
  },
  {
    "evidence_id": "ev_03_7ab87ed7",
    "tool_name": "generic_purpose",
    "evidence_text": "The user wants to count empty beer bottles ..."
  }
]

ROUND_ATOMIC_OBSERVATIONS:
[
  {
    "observation_id": "obs_19f0681cc07f6090",
    "atomic_text": "unknown_speaker said \"Come to Phil's Heavenly Nation today.\" from 124.03s to 132.97s."
  },
  ...
  {
    "observation_id": "obs_7fa818f5045442ec",
    "atomic_text": "There are two glass bottles in the center-right foreground on the table."
  }
]
```

## 16. Synthesizer: Round 1 Output

Saved at:

```text
round_01/synthesizer_trace_package.json
```

The synthesizer rewrites raw evidence into trace-facing evidence entries and inference steps.

Round 1 evidence entries:

```json
[
  {
    "evidence_id": "ev_01_19acc2ce",
    "tool_name": "asr",
    "evidence_text": "Closest ASR match for quoted phrase \"come to bill\" is \"Come to Phil's Heavenly Nation today. We got more guns.\" (similarity=0.34). Come to Phil's Heavenly Nation today. We got more guns.",
    "time_start_s": 124.031,
    "time_end_s": 132.975,
    "observation_ids": [
      "obs_19f0681cc07f6090",
      "obs_9854e1e4b807855d"
    ]
  },
  {
    "evidence_id": "ev_02_f4885d28",
    "tool_name": "frame_retriever",
    "evidence_text": "Retrieved candidate frames at 128.00s, 129.00s, 130.00s, 131.00s, 132.00s, and 133.00s around the utterance window."
  },
  {
    "evidence_id": "ev_03_7ab87ed7",
    "tool_name": "generic_purpose",
    "evidence_text": "In the frames corresponding to the quoted line, the foreground contains a table or counter with two glass bottles in the center-right foreground. Both bottles appear to be beer bottles, but each also looks like it contains liquid."
  }
]
```

Round 1 inference steps:

```json
[
  {
    "step_id": 1,
    "text": "The relevant spoken line occurs in the interval from about 124.03s to 132.97s, and the retrieved frames at 129.00s and 130.00s fall within that utterance window, matching the instruction to anchor the event around 129-130 seconds.",
    "supporting_observation_ids": [
      "obs_19f0681cc07f6090",
      "obs_9854e1e4b807855d",
      "obs_cdf91975c35f8e4a",
      "obs_e0130ec777c753f6"
    ]
  },
  {
    "step_id": 2,
    "text": "In the frames tied to that spoken line, the foreground table/counter shows two glass bottles.",
    "supporting_observation_ids": [
      "obs_fe00a8acae9cc8fd",
      "obs_2186fd85216c420f",
      "obs_7fa818f5045442ec"
    ]
  },
  {
    "step_id": 3,
    "text": "Those two bottles appear to be beer bottles, but both also appear to contain liquid, so the grounded count of empty beer bottles on the table is zero.",
    "supporting_observation_ids": [
      "obs_b88f93960f578545",
      "obs_87b51b65f16d086e",
      "obs_e0b00b5327942b43",
      "obs_3229a613e2d3484e",
      "obs_40e04d4a064622da",
      "obs_13b48cd876d53fc3"
    ]
  }
]
```

Round 1 final answer:

```json
{
  "final_answer": "A.0."
}
```

## 17. Auditor: What It Receives

After synthesis:

- evidence summary built: `video_trace_pipeline/orchestration/pipeline.py:644`
- auditor request built: `video_trace_pipeline/orchestration/pipeline.py:645-650`
- request written: `video_trace_pipeline/orchestration/pipeline.py:651`
- LLM completes audit: `video_trace_pipeline/orchestration/pipeline.py:652`
- audit report written: `video_trace_pipeline/orchestration/pipeline.py:653`

Evidence summary code:

- `video_trace_pipeline/orchestration/pipeline.py:415-439`

Auditor prompt builder:

- `video_trace_pipeline/prompts/trace_auditor_prompt.py:195-220`

Auditor prompt includes:

```text
QUESTION
OPTIONS
TRACE_PACKAGE
EVIDENCE_SUMMARY
EVIDENCE_MEMORY, if available
AuditReport schema reminder
```

Actual `round_01/auditor_request.json` contains:

```text
TRACE_PACKAGE:
{
  ...
  "final_answer": "A.0.",
  ...
}

EVIDENCE_SUMMARY:
{
  "evidence_entry_count": 3,
  "observation_count": ...
  "evidence_entries": [...]
  "recent_observations": [...]
}
```

## 18. Auditor: Round 1 Output

Saved at:

```text
round_01/auditor_report.json
```

Actual:

```json
{
  "verdict": "FAIL",
  "confidence": 0.84,
  "scores": {
    "logical_coherence": 3,
    "completeness": 2,
    "factual_correctness": 3,
    "reasoning_order": 4
  },
  "findings": [
    {
      "severity": "HIGH",
      "category": "ATTRIBUTION_GAP",
      "message": "The trace does not firmly ground that the queried sound 'come to bill's ammunition' is the same event as the ASR match 'Come to Phil's Heavenly Nation today. We got more guns.' The ASR entry itself says this is only the closest match and gives low similarity (0.34), so the event anchor is provisional rather than securely identified.",
      "evidence_ids": ["ev_01_19acc2ce"]
    },
    {
      "severity": "MEDIUM",
      "category": "TEMPORAL_GAP",
      "message": "The bottle count is described for 'frames corresponding to the quoted line,' but the visual evidence is not timestamped to a specific frame within the utterance window. The trace narrows to 129-130s without textual support that these are the decisive frames for the answer.",
      "evidence_ids": ["ev_02_f4885d28", "ev_03_7ab87ed7"]
    }
  ],
  "feedback": "The answer choice A may be plausible from the bottle description, but the trace does not securely identify the queried audio phrase or tightly link the bottle observation to the exact moment of that phrase.",
  "missing_information": [
    "Whether the queried sound 'come to bill's ammunition' is textually grounded as the same utterance as the provisional ASR match",
    "Which specific frame(s) at the time of that grounded utterance show the table",
    "The count of empty beer bottles on the table at that exact grounded moment"
  ]
}
```

This FAIL drives round 2.

## 19. How Audit Feedback Reaches the Next Planner Round

The round loop sets:

- `latest_audit = auditor result`: `video_trace_pipeline/orchestration/pipeline.py:652`
- `current_trace = trace_package`: `video_trace_pipeline/orchestration/pipeline.py:661`
- `evidence_memory = update_evidence_memory(...)`: `video_trace_pipeline/orchestration/pipeline.py:662-669`
- `compact_rounds.append(...)`: `video_trace_pipeline/orchestration/pipeline.py:679`

Next round planner receives:

```text
compact_rounds
retrieved_observations
audit_feedback=latest_audit.dict()
evidence_memory
current mode becomes "refine"
```

This is assembled again at:

- `video_trace_pipeline/orchestration/pipeline.py:581-590`

Then `build_planner_prompt` inserts:

- previous iteration summary: `video_trace_pipeline/prompts/planner_prompt.py:290-300`
- retrieved observations: `video_trace_pipeline/prompts/planner_prompt.py:314-324`
- retrieved evidence ids: `video_trace_pipeline/prompts/planner_prompt.py:326-336`
- diagnosis: `video_trace_pipeline/prompts/planner_prompt.py:338-339`

Actual `round_02/planner_request.json` includes:

```text
PREVIOUS_ITERATIONS_SUMMARY:
[
  {
    "round": 1,
    "strategy": "Use the transcript anchor around 129-130s ...",
    "tools": ["asr", "frame_retriever", "generic_purpose"],
    "verdict": "FAIL",
    "feedback": "The answer choice A may be plausible ..."
  }
]

RETRIEVED_ATOMIC_OBSERVATIONS:
[
  ...
  {
    "atomic_text": "The transcript says, \"Come to Phil's Heavenly Nation today. We got more guns.\"",
    "evidence_id": "ev_03_7ab87ed7"
  },
  ...
]

RETRIEVED_EVIDENCE_IDS_AVAILABLE:
[
  "ev_03_7ab87ed7",
  "ev_01_19acc2ce",
  "ev_02_f4885d28"
]

DIAGNOSIS:
{
  "verdict": "FAIL",
  "missing_information": [
    "Whether the queried sound 'come to bill's ammunition' is textually grounded as the same utterance as the provisional ASR match",
    "Which specific frame(s) at the time of that grounded utterance show the table",
    "The count of empty beer bottles on the table at that exact grounded moment"
  ]
}
```

That is the repair loop.

## 20. Rounds 2 and 3 in This Run

The same chain repeats:

```text
ASR -> frame_retriever -> generic_purpose -> synthesizer -> auditor
```

Round 2 plan:

```text
round_02/planner_plan.json
```

It changes the goal to repair weak audio attribution and bottle emptiness:

```json
{
  "strategy": "Repair the weak audio attribution by grounding the exact spoken phrase in the known 124-133s window, then retrieve frames specifically at that grounded utterance and use grounded visual reasoning to determine how many empty beer bottles are on the table."
}
```

Round 2 final answer:

```json
{
  "final_answer": ""
}
```

Round 2 audit still fails because the trace does not select an option and does not establish empty vs near-empty.

Round 3 plan:

```text
round_03/planner_plan.json
```

It again tries to resolve:

```text
whether the user's quoted sound is securely the same spoken moment
how many bottles at that moment are definitively empty
```

Round 3 also ends unresolved.

## 21. Final Output

Saved at:

```text
final_result.json
```

Final status:

```json
{
  "rounds_executed": 3,
  "trace_final_answer": "",
  "audit_report": {
    "verdict": "FAIL",
    "confidence": 0.95,
    "findings": [
      {
        "category": "INCOMPLETE_TRACE",
        "message": "The trace does not produce a justified multiple-choice answer. It explicitly concludes that the evidence cannot be mapped reliably to options A-D and leaves final_answer blank, so the benchmark question remains unanswered."
      },
      {
        "category": "TEMPORAL_GAP",
        "message": "The answer-critical audio anchor is unresolved from the provided text. The quoted sound in the question is only provisionally linked to the transcript line, so the trace lacks secure grounding for which exact moment/frame should be used."
      },
      {
        "category": "COUNTING_GAP",
        "message": "Even within the candidate frames, the text supports at most that two beer bottles are visible, but not how many are definitively empty. Descriptions such as \"empty or near-empty\" do not justify selecting 0, 1, 2, or 3 empty bottles."
      }
    ]
  }
}
```

Final benchmark export answer is also empty:

```json
{
  "answer": ""
}
```

## 22. The Exact Data Lineage in Round 1

This is the most important mental map.

```text
preprocess planner_segments
  -> planner_request.user_prompt PREPROCESS_SEGMENTS
  -> planner_plan step 1 ASR clip [124, 133]

ASR request
  round_01/tools/01_asr/request.json
  -> ASR result data.clips
  -> ASR result data.transcripts
  -> executor step_outputs[1]
  -> evidence ev_01_19acc2ce
  -> observations obs_19f..., obs_985...

planner step 2 input_ref
  target_field: clips
  source: step 1 field_path clips
  -> executor resolves step_outputs[1].clips
  -> frame_retriever request.clips

frame_retriever result
  data.frames[0..5]
  -> executor step_outputs[2].frames
  -> evidence ev_02_f4885d28
  -> observations obs_ff..., obs_050..., ...

planner step 3 input_refs
  target_field: frames      source: step 2 field_path frames
  target_field: transcripts source: step 1 field_path transcripts
  -> executor resolves both
  -> generic_purpose request.frames
  -> generic_purpose request.transcripts

generic_purpose result
  data.answer / data.analysis
  -> observations about table, two bottles, liquid
  -> evidence ev_03_7ab87ed7

synthesizer request
  ROUND_EVIDENCE_ENTRIES = ev_01, ev_02, ev_03
  ROUND_ATOMIC_OBSERVATIONS = all observations from those tools
  -> trace package with final_answer "A.0."

auditor request
  TRACE_PACKAGE + EVIDENCE_SUMMARY
  -> audit FAIL
  -> round 2 planner receives DIAGNOSIS and previous iteration summary
```

## 23. Code Line Map by Responsibility

### CLI and task setup

```text
pyproject.toml:15-16
  console script maps vtp to video_trace_pipeline.cli.main:main

video_trace_pipeline/cli/main.py:443-475
  run command options

video_trace_pipeline/cli/main.py:477-483
  _load_runner(...)

video_trace_pipeline/cli/main.py:486-501
  _load_tasks(...)

video_trace_pipeline/cli/main.py:502-510
  runner.run_task(...)
```

### Runner phases

```text
video_trace_pipeline/orchestration/pipeline.py:450-464
  create run, runtime snapshot, manifest

video_trace_pipeline/orchestration/pipeline.py:473-484
  run preprocess and pull planner_segments/planner_context

video_trace_pipeline/orchestration/pipeline.py:485-494
  create EvidenceLedger and ToolExecutionContext

video_trace_pipeline/orchestration/pipeline.py:558-681
  main round loop

video_trace_pipeline/orchestration/pipeline.py:684-699
  persist final trace/export/result/debug
```

### Preprocess

```text
video_trace_pipeline/orchestration/preprocess.py:522-540
  resolve preprocess settings

video_trace_pipeline/orchestration/preprocess.py:542-558
  derive preprocess cache path

video_trace_pipeline/orchestration/preprocess.py:602-614
  load cache if complete

video_trace_pipeline/orchestration/preprocess.py:620-635
  build dense-caption cache

video_trace_pipeline/orchestration/preprocess.py:637-644
  optional ASR preprocess transcript and segment assignment

video_trace_pipeline/orchestration/preprocess.py:645-646
  build planner_context and planner_segments

video_trace_pipeline/orchestration/preprocess.py:661-664
  write preprocess cache files
```

### Planner

```text
video_trace_pipeline/agents/planner.py:54-84
  build planner request

video_trace_pipeline/prompts/planner_prompt.py:253-264
  add mode/question/options/tool catalog

video_trace_pipeline/prompts/planner_prompt.py:266-288
  add preprocess segments and planning memory

video_trace_pipeline/prompts/planner_prompt.py:290-339
  add previous rounds, evidence memory, retrieved observations, diagnosis

video_trace_pipeline/agents/planner.py:86-89
  complete JSON and parse ExecutionPlan
```

### Plan normalization

```text
video_trace_pipeline/orchestration/plan_normalizer.py:166-236
  normalize fields, arguments, input_refs, depends_on

video_trace_pipeline/orchestration/plan_normalizer.py:282-320
  validate references and dependencies

video_trace_pipeline/orchestration/plan_normalizer.py:322-337
  reject context-free generic_purpose

video_trace_pipeline/orchestration/plan_normalizer.py:339-393
  topological sort and resequence

video_trace_pipeline/orchestration/plan_normalizer.py:408-419
  normalize entry point
```

### Executor and data wiring

```text
video_trace_pipeline/orchestration/executor.py:173-187
  resolve input_refs from step_outputs

video_trace_pipeline/orchestration/executor.py:299-329
  iterate steps, resolve args, get adapter, validate request

video_trace_pipeline/orchestration/executor.py:338-348
  hash exact request for cache

video_trace_pipeline/orchestration/executor.py:349-404
  cache lookup, lock, execute, extract observations, store cache

video_trace_pipeline/orchestration/executor.py:407-418
  create and append EvidenceEntry

video_trace_pipeline/orchestration/executor.py:419-422
  write per-tool result files

video_trace_pipeline/orchestration/executor.py:424-433
  create step_outputs payload for downstream input_refs
```

### Tool process protocol

```text
video_trace_pipeline/tools/process_adapters.py:191-209
  build JSON envelope: tool_name, request, task, runtime, evidence_records

video_trace_pipeline/tools/process_adapters.py:218-242
  run command with JSON stdin and parse JSON stdout

video_trace_pipeline/tools/process_adapters.py:245-275
  base process adapter and persistent runner branch
```

### Tool-specific result shaping

```text
video_trace_pipeline/tools/local_asr.py:477-508
  ASR output data and ToolResult

video_trace_pipeline/tools/process_adapters.py:723-758
  frame_retriever single request result shaping

video_trace_pipeline/tools/process_adapters.py:820-895
  frame_retriever multi-clip/time-hint merge result shaping

video_trace_pipeline/tools/process_adapters.py:1399-1446
  generic_purpose request/result shaping
```

### Observation extraction

```text
video_trace_pipeline/tools/extractors.py:72-98
  dispatch by tool_name

video_trace_pipeline/tools/extractors.py:217-235
  frame observations

video_trace_pipeline/tools/extractors.py:237-283
  ASR observations

video_trace_pipeline/tools/extractors.py:540-565
  generic-purpose observations
```

### Evidence store

```text
video_trace_pipeline/storage/evidence_store.py:89-128
  SQLite schema and indexes

video_trace_pipeline/storage/evidence_store.py:155-178
  append evidence entries and observations

video_trace_pipeline/storage/evidence_store.py:261-333
  lookup_records for evidence_ids

video_trace_pipeline/storage/evidence_store.py:335-390
  retrieve observations for planner repair context
```

### Synthesizer

```text
video_trace_pipeline/orchestration/pipeline.py:618-636
  collect round evidence, build synthesizer request, complete trace package

video_trace_pipeline/prompts/trace_synthesizer_prompt.py:60-134
  prompt contents and TracePackage schema reminder
```

### Auditor and repair loop

```text
video_trace_pipeline/orchestration/pipeline.py:415-439
  build evidence summary

video_trace_pipeline/orchestration/pipeline.py:644-653
  build/write/complete auditor request

video_trace_pipeline/prompts/trace_auditor_prompt.py:195-220
  prompt contents for auditor

video_trace_pipeline/orchestration/pipeline.py:661-680
  update current_trace, evidence_memory, evidence statuses, compact_rounds
```

## 24. What To Open When Debugging This Run

Start here:

```text
workspace/runs_old/video_13/20260426T161430Z_085af9d5/run_manifest.json
workspace/runs_old/video_13/20260426T161430Z_085af9d5/final_result.json
workspace/runs_old/video_13/20260426T161430Z_085af9d5/debug/rounds.json
```

Then inspect round 1:

```text
round_01/planner_request.json
round_01/planner_plan.json
round_01/tools/01_asr/request.json
round_01/tools/01_asr/result.json
round_01/tools/01_asr/observations.json
round_01/tools/02_frame_retriever/request.json
round_01/tools/02_frame_retriever/result.json
round_01/tools/02_frame_retriever/observations.json
round_01/tools/03_generic_purpose/request.json
round_01/tools/03_generic_purpose/result.json
round_01/tools/03_generic_purpose/observations.json
round_01/synthesizer_request.json
round_01/synthesizer_trace_package.json
round_01/auditor_request.json
round_01/auditor_report.json
```

Then inspect repair:

```text
round_02/planner_request.json
round_02/planner_plan.json
round_02/auditor_report.json
round_03/planner_request.json
round_03/planner_plan.json
round_03/auditor_report.json
```

The key question to ask at each step:

```text
Did this file come from the planner, executor, tool adapter, observation extractor, synthesizer, or auditor?
```

Once you know that, the code line map above tells you where it was produced.
