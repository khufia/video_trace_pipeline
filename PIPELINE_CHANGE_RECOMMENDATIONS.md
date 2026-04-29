# Pipeline Flaws And Fixes

This is a concise design note only. No code changes are included here.

## 1. Preprocess Context Is Too Lossy

### Flaw

The pipeline builds rich raw preprocess `segments.json`, then passes thinner `planner_segments.json` plus `planner_context.json` to the planner.

Problems:

- `overall_summary` is lost.
- Dense-caption structure is flattened.
- ASR is present, but not grouped as first-class planner input.
- `planner_context` is extra noisy memory instead of solving the context-loss problem.
- Empty fields remain in preprocessed objects and add clutter.

### Fix

Remove `planner_context` entirely.

Keep `planner_segments`, but upgrade them into rich planner-facing segments. They should contain useful dense-caption output plus ASR transcript content in one consistent format.

Rules:

- Keep all attributes.
- Remove empty fields after preprocessing.
- Include `overall_summary`.
- Include dense-caption spans.
- Include ASR transcript spans.
- Do not include `backend`.
- Do not include `sampled_frames` if empty.

Target planner segment:

```json
{
  "segment_id": "seg_004",
  "start_s": 120.0,
  "end_s": 150.0,
  "dense_caption": {
    "clips": [
      {
        "video_id": "video_13",
        "start_s": 120.0,
        "end_s": 150.0,
        "artifact_id": "clip_120.00_150.00",
        "relpath": "workspace/artifacts/video_13/clips/clip_120.00_150.00.mp4"
      }
    ],
    "captioned_range": {
      "start_s": 120.0,
      "end_s": 150.0
    },
    "overall_summary": "...",
    "captions": [
      {
        "start_s": 120.0,
        "end_s": 125.0,
        "visual": "...",
        "audio": "...",
        "attributes": [
          "camera_state: ...",
          "video_background: ...",
          "storyline: ...",
          "shooting_style: ..."
        ]
      }
    ]
  },
  "asr": {
    "transcript_spans": [
      {
        "start_s": 129.125,
        "end_s": 158.167,
        "text": "Come to Phil's Ammu Nation today. ...",
        "speaker_id": null
      }
    ]
  }
}
```

Note: fields like `on_screen_text`, `actions`, `objects`, `confidence`, and `metadata` are kept only when non-empty.

## 2. Planner Needs Retrieval

### Flaw

The planner currently receives whatever context is pushed into the prompt and must emit a plan immediately. In refine mode, this makes it too dependent on stale summaries and whatever observations happened to be retrieved.

### Fix

Add a text-only retrieval system for the planner. This should run in both generate and refine rounds before the final execution plan is emitted.

Planner flow:

```text
task + options + audit diagnosis when present
  -> build retrieval catalog from preprocess, evidence, observations, artifacts, prior trace
  -> deterministic seed retrieval from task/audit terms
  -> planner/controller inspects catalog + current retrieved context
  -> planner/controller either returns ready or asks for narrower text context
  -> retriever returns text-only preprocess/evidence/artifact context
  -> repeat until ready or max retrieval iterations
  -> planner emits ExecutionPlan
```

The planner/controller retrieval decision should be schema-bound:

```json
{
  "action": "retrieve",
  "rationale": "Need existing ASR spans before deciding whether to call ASR again.",
  "requests": [
    {
      "request_id": "quote_transcript",
      "target": "asr_transcripts",
      "need": "Exact quoted speech and neighboring turns.",
      "query": "quoted speech response",
      "modalities": ["asr"],
      "time_range": {"start_s": 40.0, "end_s": 70.0},
      "source_tools": ["asr"],
      "evidence_status": "",
      "artifact_ids": [],
      "evidence_ids": [],
      "observation_ids": [],
      "limit": 20
    }
  ]
}
```

If existing retrieved evidence is enough, the controller should return:

```json
{"action": "ready", "rationale": "The exact OCR text is already available as validated evidence.", "requests": []}
```

The retriever can access:

- rich planner segments
- raw preprocess segments
- ASR transcript spans
- dense-caption summaries and captions
- atomic observations
- evidence entries
- artifact context records
- OCR text
- spatial boxes and labels
- prior trace claims
- audit gaps

Keep the synthesizer one-shot. It should receive a strong evidence package and write the trace once. Do not make synthesizer pause/retrieve.

## 3. Artifact Metadata Does Not Tell The Planner What The Artifact Contains

### Flaw

This is not enough:

```json
{
  "kind": "artifact_metadata",
  "artifact_id": "b27ea19b17efb368d7e15f95",
  "text": "Frame at 132.00s from clip 124.00-133.00s.",
  "relpath": "cache/artifacts/.../frame_132.00.png"
}
```

The planner cannot see the frame. A path and timestamp do not tell it what is in the frame.

### Fix

Artifact retrieval should return artifact context.

Target:

```json
{
  "kind": "artifact_context",
  "artifact_id": "frame_132.00",
  "artifact_type": "frame",
  "relpath": "workspace/artifacts/video_13/frames/frame_132.00.png",
  "time": {
    "timestamp_s": 132.0,
    "source_clip": {
      "start_s": 124.0,
      "end_s": 133.0
    }
  },
  "contains": [
    "The frame shows a TV/gun-shop scene.",
    "A foreground table/counter is visible.",
    "Two glass bottles are visible on the table.",
    "Both bottles appear to contain liquid."
  ],
  "linked_observations": [
    {
      "observation_id": "obs_...",
      "source_tool": "generic_purpose",
      "text": "There are two glass bottles in the center-right foreground on the table."
    },
    {
      "observation_id": "obs_...",
      "source_tool": "generic_purpose",
      "text": "Bottle 1 looks like it has some liquid."
    }
  ],
  "linked_evidence": [
    {
      "evidence_id": "ev_...",
      "tool_name": "frame_retriever",
      "summary": "Retrieved frames at 128.00s, 129.00s, 130.00s, 131.00s, 132.00s, and 133.00s.",
      "observation_texts": [
        "A candidate frame was retrieved at 132.00s."
      ]
    }
  ]
}
```

The important fields are `contains`, `linked_observations`, and `linked_evidence`. IDs are kept as handles, but every linked item must also include compact text saying what it contains.

## 4. ASR And Dense Captioner Are Recomputed Too Often

### Flaw

Preprocessing may already contain ASR and dense captions for a requested window, but the tools can run again just to create evidence.

### Fix

Reuse preprocess outputs but emit normal tool-shaped results.

ASR reuse:

```text
ASR request clip [start_s, end_s]
  -> check preprocess ASR spans
  -> if coverage matches, slice transcript spans
  -> emit normal ASR ToolResult
  -> attach clip artifact
  -> metadata.source = "preprocess_reuse"
```

Dense-caption reuse:

```text
dense_captioner request [start_s, end_s], focus_query=""
  -> exact matching preprocess window exists
  -> emit normal dense_captioner ToolResult
  -> metadata.source = "preprocess_reuse"
```

Run the actual tool only when:

- requested window differs
- focused query differs
- preprocess model/settings differ
- answer-critical detail needs a new targeted pass

## 5. ASR Output Should Use Transcripts Only

### Flaw

ASR currently has both flattened `text` and structured `transcripts`.

### Fix

Remove flattened ASR `text`.

Canonical ASR output:

```json
{
  "clips": [],
  "transcripts": [
    {
      "transcript_id": "tx_...",
      "clip": {},
      "segments": [
        {
          "start_s": 124.031,
          "end_s": 132.975,
          "text": "Come to Phil's Ammu Nation today.",
          "speaker_id": null
        }
      ]
    }
  ],
  "phrase_matches": []
}
```

If concatenated text is needed, derive it from transcript segments at render time.

## 6. Artifact And Preprocess Storage Is Too Hash-Heavy

### Flaw

Preprocess and artifact paths use hash-named folders. This is cache-safe, but hard for humans to inspect.

Current style:

```text
workspace/cache/preprocess/<video_hash>/dense_caption/<model_hashish>/30/tool_v2/<settings_hash>/
workspace/cache/artifacts/<artifact_hash>/frame_132.00.png
```

Problems:

- hard to find artifacts for a video
- hard to compare runs by video
- hard to inspect preprocess by hand
- stable video-level assets are hidden under cache hashes

### Fix

Use stable human-readable storage under `video_id`.

Target:

```text
workspace/preprocess/video_13/
  manifest.json
  planner_segments.json
  raw_segments.json
  asr/transcripts.json
  dense_caption/segments.json

workspace/artifacts/video_13/
  clips/clip_124.00_133.00.mp4
  frames/frame_132.00.png
  regions/frame_132.00_table_bbox_001.png
  artifact_context.jsonl
```

Pipeline/model/settings fingerprints can stay in `manifest.json` rather than path names.

If content changes because model/settings change, record it in the manifest:

```json
{
  "video_id": "video_13",
  "video_fingerprint": "...",
  "preprocess_schema_version": "rich_v1",
  "dense_caption_model": "...",
  "asr_model": "...",
  "settings_signature": "..."
}
```

The human entrypoint remains stable:

```text
workspace/preprocess/video_13/
workspace/artifacts/video_13/
```

## 7. Tool Field Contract Is Confusing

### Flaw

Current plan steps mix literal arguments, placeholder empty fields, and wiring:

```json
{
  "arguments": {
    "clips": [],
    "frames": [],
    "transcripts": []
  },
  "input_refs": [
    {
      "target_field": "frames",
      "source": {
        "step_id": 2,
        "field_path": "frames"
      }
    }
  ]
}
```

### Fix

Keep `input_refs` as the wiring mechanism, but separate planner-facing fields:

```json
{
  "step_id": 3,
  "tool_name": "generic_purpose",
  "purpose": "Count empty beer bottles on the table.",
  "inputs": {
    "query": "Count empty beer bottles on the table."
  },
  "input_refs": {
    "clips": [
      {"from_step": 1, "output": "clips"}
    ],
    "frames": [
      {"from_step": 2, "output": "frames"}
    ],
    "transcripts": [
      {"from_step": 1, "output": "transcripts"}
    ],
    "artifact_context": [
      {"from_step": 2, "output": "artifact_context"}
    ],
    "metadata": [
      {"from_step": 2, "output": "frame_metadata"}
    ]
  },
  "expected_outputs": {
    "observations": [
      "bottle_count",
      "empty_state",
      "supporting_frame_timestamps"
    ],
    "artifacts": [],
    "media": ["frames"],
    "text": ["analysis", "answer"]
  }
}
```

The normalizer/executor builds the concrete tool request and strips irrelevant empty fields.

Proposed generic plan step schema:

```json
{
  "step_id": 1,
  "tool_name": "tool_name",
  "purpose": "why this tool is being called",
  "inputs": {},
  "input_refs": {
    "clips": [],
    "frames": [],
    "regions": [],
    "transcripts": [],
    "artifact_context": [],
    "observations": [],
    "evidence_ids": [],
    "metadata": []
  },
  "expected_outputs": {
    "clips": [],
    "frames": [],
    "regions": [],
    "transcripts": [],
    "artifact_context": [],
    "observations": [],
    "text": [],
    "metadata": []
  }
}
```

Empty fields are removed before persistence/execution.

Use `expected_outputs` rather than `outputs` because the tool owns its real output schema. The planner is only saying what it needs from the tool call.

## 8. Observation Quality Depends Too Much On Generic-Purpose Text

### Flaw

Generic-purpose observations are derived from another model's prose. They are useful, but they are weaker than direct tool evidence for answer-critical details.

### Fix

Prefer narrower direct evidence when possible:

- ASR for speech
- OCR for visible text
- spatial grounder for object location
- frame metadata for timestamps
- dense caption for broad context
- generic purpose for interpretation after the relevant media/text has been grounded

Atomic observations should be the primary evidence surface. `EvidenceEntry.evidence_text` should remain only a summary/handle.

## 9. Prompt Cleanup And Robust Prompting From Reviews

### Review Signal

Sources used:

- `reviews/workspace_results`: 103 workspace runs summarized plus detailed per-run notes.
- `reviews/minerva_results`: detailed hand reviews for the Minerva examples and the Minerva inventory.
- Current prompt files checked against those reviews: `video_trace_pipeline/prompts/planner_prompt.py`, `video_trace_pipeline/prompts/trace_synthesizer_prompt.py`, and `video_trace_pipeline/prompts/trace_auditor_prompt.py`.

The review corpus shows the same prompt-level failures repeatedly:

- Useful evidence is found, then lost in later rounds after a narrower audit/planner repair.
- Exact timestamp questions get over-narrowed to one frame even when the answer is an action or state that needs neighbors.
- Frame retrieval is relevance-ranked, but ordered questions sometimes treat result order as chronology.
- Small visible text, scoreboards, prices, labels, and nameplates need structured crop/OCR behavior.
- Some wrong answers pass because the auditor checks final option text but not referent, timestamp, or intermediate ordered entities.
- Some correct or near-correct answers fail because the auditor asks for perfect taxonomy or exact media proof instead of a missing compact correction.
- Several traces leave `final_answer` blank even when the evidence supports one option better than all others.

The current local prompt changes help, but they are not enough. They add useful rules for exact timestamp anchors, delivery/tone, sound causes, map geometry, referent alignment, and evidence memory. What is still missing is calibrated few-shot behavior: the models need examples that demonstrate how to apply those rules, not only rule text.

### Current Prompt Parts To Remove Or Modify

`video_trace_pipeline/prompts/planner_prompt.py`, lines 24-25 and 278-288 still expose `PREPROCESS_PLANNING_MEMORY`.

Why: the design now removes `planner_context` entirely. Keeping this prompt surface after the data model changes would preserve the old lossy/noisy memory path.

Remove after the planner-context cleanup:

```text
PREPROCESS_PLANNING_MEMORY
PREPROCESS_PLANNING_MEMORY_USAGE_NOTE
```

`video_trace_pipeline/prompts/trace_auditor_prompt.py` used to have long duplicated text-only disclaimers. The current local diff already shortened them.

Why: this removal is good. The auditor needs the text-only boundary, but repeated warnings consume prompt budget and do not fix the review failures. Keep one concise operating-mode block and spend the saved budget on score ICL.

Keep the compact version; do not re-add the longer repeated disclaimer.

`video_trace_pipeline/prompts/planner_prompt.py`, lines 31-34 and 48-53 currently emphasize the smallest repair chain. Keep the intent, but soften it.

Why: reviews show that the planner sometimes repairs too narrowly, especially after the auditor asks for exact frames. "Smallest" should mean smallest sufficient evidence set, not one isolated frame or one weak current-round observation.

Replace:

```text
Gather the fewest tool calls that directly resolve the answer-critical gaps.
Build the smallest dependency chain that grounds that missing field.
```

With:

```text
Gather the smallest sufficient evidence set that resolves the answer-critical gaps. Do not shrink context so much that temporal order, action state, referent identity, or option mapping becomes ungrounded.
Build the shortest dependency chain that still preserves the needed context around the missing field.
```

`video_trace_pipeline/prompts/planner_prompt.py`, lines 89-99 list canonical chains.

Why: the chains are useful, but they read like generic recipes. The planner needs decision rules for when a chain is insufficient.

Modify by adding this immediately after the canonical chain list:

```text
Chain sufficiency rule:
- A chain is sufficient only if its output can ground the final discriminator, not merely locate a related moment.
- For sequence/order/count/action tasks, prefer chronological clips or frame sequences over isolated top-k frames.
- For visible text tasks, preserve label-value adjacency and region context until OCR has read the target text.
```

`video_trace_pipeline/prompts/planner_prompt.py`, lines 102-113 explain `input_refs`.

Why: this is necessary wiring language, but it competes with semantic planning. Keep it, but make clear that `input_refs` are not evidence by themselves.

Add:

```text
Wiring is not evidence:
- `input_refs` only pass media/text objects between tools.
- The plan must still say what answer-critical observation the downstream tool should extract from those inputs.
```

`video_trace_pipeline/prompts/trace_synthesizer_prompt.py`, lines 44-49 currently says to justify one option or leave `final_answer` empty.

Why: the reviews show many blank/unresolved outputs even when one option is best supported and the remaining issue is trace wording. The synthesizer should still leave blank when multiple options remain live, but it should not leave blank just because evidence is imperfect if the option mapping is clear.

Replace:

```text
For multiple-choice questions, justify one supported option or leave `final_answer` empty.
If multiple answer choices remain compatible, or an answer-critical premise is still unresolved, leave `final_answer` empty.
```

With:

```text
For multiple-choice questions, choose the uniquely best-supported option when the evidence rules out the alternatives or clearly maps to one option. Leave `final_answer` empty only when multiple options remain genuinely compatible or the missing premise could change the selected option.
If the best option is supported but one detail is weak, keep the answer and state the weakness in the evidence/inference text instead of erasing the answer.
```

`video_trace_pipeline/prompts/trace_auditor_prompt.py`, lines 155-166 define score semantics only abstractly.

Why: abstract score definitions are not enough. The reviews show wrong PASS, correct-but-under-supported FAIL, and blank-answer cases. The auditor needs score calibration examples.

Modify by adding the ICL block in the next subsection after the score definitions.

### Prompt Additions

Add to the Planner system prompt:

```text
Evidence preservation rule:
- When prior evidence already grounds a claim, do not replace it with weaker current-round evidence.
- Re-search broadly only when the diagnosis says the old anchor is wrong or incomplete.
- In refine mode, preserve useful prior timestamps, clips, OCR text, ASR spans, and atomic observations, then gather only the missing discriminator.

Occurrence and chronology rule:
- For first/last/before/after/ordered-list questions, collect all relevant candidate events in the bounded interval, sort by timestamp, and then choose the requested occurrence.
- Never infer chronology from retrieval result order.
- If an early/late candidate is missing, retrieve the full interval before answering.

Action-at-timestamp rule:
- Exact timestamps are anchors, not isolated proof.
- For action, motion, state change, count, or identity at a timestamp, retrieve the anchor frame plus chronological neighbors.
- The downstream tool must receive the structured frame sequence and answer from the local sequence.

Small-text rule:
- For scoreboards, prices, labels, signs, nameplates, charts, menus, or control panels, use high-resolution frames, region grounding, OCR, and explicit label-value pairing.
- Preserve spatial adjacency such as team-score, product-price, name-role, and label-value.

Audit-repair rule:
- When the audit names missing information, the next plan must directly target that missing fact.
- Do not broaden to a new semantic search unless the named anchor is contradicted or unavailable.
```

Add to the TraceSynthesizer system prompt:

```text
Chronology synthesis rule:
- Before answering sequence, first/last, before/after, or count-over-time questions, sort every frame/clip/transcript observation by timestamp.
- State the ordered candidates in the inference steps before mapping to the final option.

Atomic evidence rule:
- Prefer atomic observations as the citation surface for answer-critical claims.
- Use evidence summaries as handles only; do not treat a broad evidence summary as proof of details absent from the linked observations.

Prior-evidence rule:
- In refine mode, preserve prior supported evidence unless new evidence directly contradicts it, corrects it, or proves the old anchor wrong.
- If the new evidence is weaker or narrower, use it as a supplement, not a replacement.

Option-mapping rule:
- For multiple choice, explicitly state why the grounded observation maps to the selected option and why close alternatives are not better supported.
- Do not answer with a nearby referent, a related event, or a semantically similar option unless the evidence links that exact referent/event.
```

Add to the TraceAuditor system prompt:

```text
Answer-vs-trace calibration:
- PASS only when the final answer is both selected correctly from the trace text and supported by enough cited evidence.
- If the answer is likely correct but under-supported, FAIL with `INCOMPLETE_TRACE`, preserve the candidate answer in feedback, and name the exact missing proof.
- If the answer is wrong because the referent, timestamp, occurrence, or option mapping is wrong, FAIL with high severity even if the final option letter matches a gold label.

Intermediate-entity check:
- For ordered-list, count, scoreboard, label-value, and referent questions, audit the intermediate entities, not only the final option.
- A correct final option with wrong intermediate reasoning should not PASS.

Exact-frame calibration:
- Do not require a single exact frame to prove an action if neighboring chronological frames provide the action context.
- Do require the trace to connect the anchor timestamp to the neighboring frame sequence.

Option-taxonomy calibration:
- Do not fail a trace merely because the evidence uses a broader visual label when the question's options force a closest-category mapping and no alternative is better supported.
- Do fail when the broader label could map to multiple live options.
```

### Planner ICL Examples

Add general few-shot examples to the Planner prompt. These should be task-pattern examples, not sample-specific references.

```text
Example: ordered visible labels
Question pattern: "Which item appears before the instruction screen?"
Good plan:
1. retrieve frames across the bounded label montage with chronological sort
2. OCR each label region
3. synthesize the ordered label list
Bad plan:
1. retrieve top-k relevant frames and treat returned order as time order

Example: action at exact timestamp
Question pattern: "What is the person doing at 6.0s?"
Good plan:
1. retrieve anchor frame plus +/- 2 seconds chronological neighbors
2. ask generic visual reasoning over the sequence
Bad plan:
1. inspect only frame_006.00 and ignore neighbors

Example: scoreboard, price, nameplate, or control panel value
Question pattern: "What value is shown after the visible update?"
Good plan:
1. retrieve frames after the relevant visual update, not only before the boundary
2. spatially ground the relevant display region
3. OCR label-value pairs and preserve adjacency
Bad plan:
1. OCR a full frame or early frame and infer the missing mapping

Example: tone or emotional transition
Question pattern: "What is the speaker's tone before and after the reveal?"
Good plan:
1. localize the before and after utterance windows for the same speaker
2. run ASR for exact words and speaker continuity
3. retrieve neighboring frames or clips for delivery, facial expression, and behavior
4. ask generic_purpose to compare delivery across the two windows
Bad plan:
1. infer tone from transcript sentiment words alone

Example: quote-adjacent response
Question pattern: "How does the other person respond after the quoted line?"
Good plan:
1. localize the quoted line in ASR
2. keep the local dialogue window before and after the quote
3. identify the responding speaker and map the paraphrased response to an option
Bad plan:
1. search only for exact option text and ignore the local exchange

Example: speaker or addressee attribution
Question pattern: "Who says the line?" or "Who is being spoken to?"
Good plan:
1. retrieve the local ASR window containing the line
2. retrieve neighboring frames or clips that show the speakers/listeners
3. ground speaker/addressee identity from turn order, gaze, position, subtitles, or visual context
Bad plan:
1. assume the nearest named person in the transcript is the speaker or addressee

Example: brief sound cause
Question pattern: "What caused the short sound at the marked moment?"
Good plan:
1. localize the sound timestamp or use the given timestamp as an anchor
2. retrieve chronological frames before, during, and after the sound
3. use generic_purpose to identify the direct local trigger
Bad plan:
1. choose a remote setup event mentioned earlier in dialogue

Example: map, direction, or relative position
Question pattern: "Which direction is the highlighted area from the labeled anchor?"
Good plan:
1. retrieve the frame where the map/diagram is visible
2. spatially ground both the referenced region and anchor region
3. use OCR only for labels that must be read
4. compare coordinates/relative geometry
Bad plan:
1. answer from nearby label text without grounding the two regions

Example: object state anchored by speech
Question pattern: "When the speaker mentions the object, what state is it in?"
Good plan:
1. use ASR to find the utterance time
2. retrieve frames around that utterance
3. spatially ground the object
4. verify the state such as open/closed, empty/full, on/off, raised/lowered
Bad plan:
1. use the transcript topic as proof of the visual state

Example: closest-category visual option
Question pattern: "Which object category best describes the visible object?"
Good plan:
1. retrieve frames where the object is visible
2. spatially ground the object region
3. ask for visual attributes that distinguish the option categories
4. map to the closest supported option only after ruling out better matches
Bad plan:
1. fail because the tool used a broader synonym, or pick an option from taxonomy words alone

Example: repeated motion or event count
Question pattern: "How many times does the action happen before the marker event?"
Good plan:
1. localize the marker event and the count interval
2. retrieve a dense chronological clip/frame sequence over the interval
3. count complete cycles and state the inclusion rule
Bad plan:
1. count from a few sparse representative frames

Example: absence or not-mentioned question
Question pattern: "Which option is not mentioned in the explanation?"
Good plan:
1. run ASR over the relevant explanation window
2. preserve exact transcript spans for each mentioned candidate
3. compare every option against transcript evidence
Bad plan:
1. answer from broad scene captions or world knowledge

Example: chart, table, or worked visual calculation
Question pattern: "What value follows from the completed diagram/table?"
Good plan:
1. retrieve the completed/stable visual state
2. use generic_purpose for structured interpretation
3. add OCR only for labels or numbers that need explicit reading
4. verify arithmetic or option mapping separately
Bad plan:
1. OCR an incomplete visual state and compute from missing values

Example: refine after audit
Diagnosis pattern: "The trace found the event but did not prove the object state."
Good plan:
1. preserve the event timestamp from prior evidence
2. retrieve frames/regions at that timestamp
3. verify only the missing state
Bad plan:
1. run a broad new event search and discard the prior timestamp
```

### Synthesizer ICL Examples

```text
Example: chronology
Evidence observations:
- frame at 10s shows label "STEP_A"
- frame at 12s shows label "STEP_B"
- frame at 15s shows label "STEP_C"
Good synthesis:
- "Chronologically, the labels are STEP_A -> STEP_B -> STEP_C. Therefore the option matching that order is ..."
Bad synthesis:
- "The retrieved frames list STEP_B first, so STEP_B is first."

Example: option-induced convention
Evidence observations:
- transcript says "early 2000s"
- options contain "2000" but not "early 2000s"
Good synthesis:
- "The evidence states early 2000s; among the provided options, 2000 is the closest convention, but the exact year is not directly stated."
Bad synthesis:
- "The video directly says 2000."

Example: preserving prior evidence
Prior trace:
- frames around the anchor show a continuous throwing motion
New evidence:
- the single anchor frame is visually ambiguous
Good synthesis:
- "The exact frame is ambiguous alone, but the neighboring sequence around the anchor shows the throwing action."
Bad synthesis:
- "The exact frame is ambiguous, so discard the earlier neighboring sequence."

Example: label-value pairing
Evidence observations:
- OCR region 1: label "LEFT TEAM", value "12"
- OCR region 2: label "RIGHT TEAM", value "10"
Good synthesis:
- "The left label is paired with 12 and the right label is paired with 10, so the answer is the option matching LEFT TEAM 12, RIGHT TEAM 10."
Bad synthesis:
- "The visible values are 12 and 10, so choose any option containing those numbers."

Example: tone from delivery
Evidence observations:
- transcript: speaker says "That is fine"
- delivery observation: speaker sighs, pauses, and looks away
Good synthesis:
- "The words are neutral, but the delivery is reluctant; the supported tone is the option closest to reluctant."
Bad synthesis:
- "The speaker says 'fine', so the tone is approving."

Example: quote-adjacent response
Evidence observations:
- at 20s, Speaker 1 asks a question
- at 22s, Speaker 2 answers with a paraphrase of option C
Good synthesis:
- "The answer comes from Speaker 2's response after the quoted line, not from the quoted line itself."
Bad synthesis:
- "The quoted line contains the keyword, so choose the option with that keyword."

Example: speaker/addressee attribution
Evidence observations:
- transcript has the line "I can help"
- frames show Person A facing Person B while Person B responds immediately after
Good synthesis:
- "The line is attributed to Person A because the visual turn-taking and the next response identify Person A as the speaker."
Bad synthesis:
- "Person B is mentioned nearby in the transcript, so Person B said the line."

Example: sound cause
Evidence observations:
- at 30s, a short sound is heard
- frames at 29.5-30.5s show an object hitting the floor
- earlier dialogue mentions a different possible cause
Good synthesis:
- "The direct local trigger is the object hitting the floor; earlier dialogue is setup context, not the sound cause."
Bad synthesis:
- "The earlier dialogue mentioned a cause, so that cause made the sound."

Example: count over time
Evidence observations:
- interval 0-8s contains three complete repeated cycles
- interval 8-10s contains one partial cycle after the marker
Good synthesis:
- "Only complete cycles before the marker count, so the count is 3."
Bad synthesis:
- "There are four visible motions in nearby frames, so the count is 4."

Example: referent alignment
Evidence observations:
- the person with a red bag is seated
- a different person in a red shirt is standing
Good synthesis:
- "The question asks about the person with the red bag, so the answer must describe the seated person."
Bad synthesis:
- "A red item is associated with a standing person, so answer for the standing person."

Example: closest-category option mapping
Evidence observations:
- the grounded object is a small portable light source with a handle
- option A is "portable light"; option B is "book"; option C is "phone"
Good synthesis:
- "The evidence uses a broad description, but among the options it uniquely maps to portable light."
Bad synthesis:
- "The exact phrase 'portable light' was not observed, so leave the answer blank."

Example: ASR-to-visual anchor
Evidence observations:
- transcript at 40s names the object
- frames at 40-42s show the object is closed
Good synthesis:
- "The transcript anchors the moment; the visual frames ground the state as closed."
Bad synthesis:
- "Because the object is mentioned in speech, its state is known."
```

### Auditor Score ICL

Add this few-shot score calibration after the auditor score rubric. These are invented trace examples; they are not sample-specific.

```text
Audit scoring examples:

Example A: strong multimodal PASS
Question:
- "After the narrator says 'start', which status and number are shown?"
Evidence:
- obs_asr_1: at 5.2s, transcript says "start now"
- obs_frame_1: at 4.8s, display still shows old status "WAIT" and number "17"
- obs_ocr_1: at 6.1s, OCR reads label "STATUS" paired with value "READY"
- obs_ocr_2: at 6.1s, OCR reads label "COUNT" paired with value "18"
Trace:
- step 1 cites obs_asr_1 to define the post-start interval
- step 2 rejects obs_frame_1 because it is before the start cue
- step 3 cites obs_ocr_1 and obs_ocr_2, preserving label-value pairing
- final_answer: "B. READY, 18"
Expected audit:
- verdict: PASS
- scores: logical_coherence=5, completeness=5, factual_correctness=5, reasoning_order=5
- findings: []
- missing_information: []
- diagnostics: temporal_alignment=pass, option_alignment=pass, evidence_sufficiency=pass

Example B: answer likely correct, trace under-supported
Question:
- "Which container is empty after the lid is removed?"
Evidence:
- obs_1: at 14s, three containers are visible
- obs_2: at 15s, the lid is removed from the left container
- obs_3: at 16s, the left container interior is partly occluded; contents are not visible
- obs_4: at 16s, the middle and right containers visibly contain objects
Trace:
- step 1 correctly uses obs_2 as the post-lid-removal anchor
- step 2 says "the left container is empty" but cites only obs_3, which is occluded
- step 3 rules out middle and right using obs_4
- final_answer: "A. left container"
Expected audit:
- verdict: FAIL
- scores: logical_coherence=4, completeness=2, factual_correctness=3, reasoning_order=4
- findings:
  - severity=HIGH, category=INCOMPLETE_TRACE, message="The trace rules out the other containers but does not directly show the left container is empty after the lid is removed."
- missing_information:
  - "left container contents after the lid is removed"
- feedback: "Preserve the left-container candidate, but verify the container interior/state before PASS."
- diagnostics: temporal_alignment=pass, option_alignment=partial, evidence_sufficiency=fail

Example C: wrong temporal occurrence
Question:
- "What color is the first indicator light?"
Evidence:
- obs_1: at 5s, the first indicator is blue
- obs_2: at 18s, a later indicator is red
- obs_3: at 22s, another later indicator is yellow
Trace:
- step 1 cites obs_2 because it is visually clearest
- step 2 says the first indicator is red
- step 3 never discusses obs_1
- final_answer: "C. red"
Expected audit:
- verdict: FAIL
- scores: logical_coherence=2, completeness=2, factual_correctness=1, reasoning_order=2
- findings:
  - severity=HIGH, category=TEMPORAL_GAP, message="The trace answers from a later indicator while the evidence contains an earlier first indicator."
  - severity=HIGH, category=ANSWER_ERROR, message="The final answer is contradicted by the timestamped evidence for the first occurrence."
- missing_information: []
- feedback: "The trace used a later occurrence for a first-occurrence question."
- diagnostics: temporal_alignment=fail, option_alignment=fail, evidence_sufficiency=pass

Example D: correct final option but wrong intermediate reasoning
Question:
- "Which label appears second?"
Evidence:
- obs_1: at 3s, label ALPHA appears
- obs_2: at 6s, label BETA appears
- obs_3: at 9s, label GAMMA appears
Trace:
- step 1 lists ALPHA, GAMMA, BETA as the order
- step 2 says "GAMMA appears second"
- final_answer: "B. BETA"
Expected audit:
- verdict: FAIL
- scores: logical_coherence=1, completeness=2, factual_correctness=2, reasoning_order=2
- findings:
  - severity=HIGH, category=INFERENCE_ERROR, message="The final answer text names BETA, but the trace's own ordered reasoning says GAMMA is second."
  - severity=MEDIUM, category=TEMPORAL_GAP, message="The inference step does not preserve the evidence chronology."
- missing_information:
  - "chronological label order used consistently in inference and final answer"
- feedback: "The final answer text names the correct label, but the reasoning claims the wrong intermediate order."
- diagnostics: temporal_alignment=fail, option_alignment=contradictory, evidence_sufficiency=pass

Example E: unresolved despite enough evidence
Question:
- "Which word appears on the sign?"
Evidence:
- obs_1: OCR at 11s reads "CAUTION"
- options include "A. CAUTION", "B. START", "C. EXIT"
Trace:
- inference step says the sign reads "CAUTION"
- inference step says no other option text is observed
- final_answer: ""
Expected audit:
- verdict: FAIL
- scores: logical_coherence=3, completeness=2, factual_correctness=5, reasoning_order=4
- findings:
  - severity=HIGH, category=INCOMPLETE_TRACE, message="The trace has a grounded option mapping but leaves final_answer empty."
- missing_information:
  - "final answer selection from the already grounded OCR text"
- feedback: "The trace has enough evidence to select option A; the missing final answer should be repaired."
- diagnostics: temporal_alignment=pass, option_alignment=pass_but_not_selected, evidence_sufficiency=pass

Example F: exact timestamp action supported by neighbors
Question:
- "What action is happening at 12.0s?"
Evidence:
- obs_1: exact 12.0s frame is visually ambiguous
- obs_2: frames from 11.0s to 13.0s show a continuous pouring action
- obs_3: no alternative action is visible in the anchor window
Trace:
- step 1 says the exact frame alone is ambiguous
- step 2 cites obs_2 as neighboring temporal context around the anchor
- step 3 maps the action sequence to option D
- final_answer: "D. pouring"
Expected audit:
- verdict: PASS
- scores: logical_coherence=5, completeness=4, factual_correctness=5, reasoning_order=5
- findings: []
- missing_information: []
- feedback: "The trace correctly treats the exact timestamp as an anchor and uses neighboring frames for action context."
- diagnostics: temporal_alignment=pass, exact_anchor_handling=pass, evidence_sufficiency=pass

Example G: speaker/addressee mismatch
Question:
- "Who is being addressed after the quoted line?"
Evidence:
- obs_asr_1: Speaker A says the quoted line at 20s
- obs_frame_1: at 20-23s, Speaker A faces Person X
- obs_asr_2: Person X responds at 24s
- obs_frame_2: Person Y is nearby but does not respond
Trace:
- step 1 identifies Person Y as the addressee because Person Y is named earlier
- step 2 ignores Person X's visual response
- final_answer: "C. Person Y"
Expected audit:
- verdict: FAIL
- scores: logical_coherence=2, completeness=2, factual_correctness=1, reasoning_order=2
- findings:
  - severity=HIGH, category=ATTRIBUTION_GAP, message="The trace answers for the wrong addressee; the local response evidence points to Person X."
- missing_information:
  - "addressee identity grounded by local turn-taking and visual response"
- feedback: "Ground the addressee from the local quote/response sequence rather than an earlier name mention."
- diagnostics: referent_alignment=fail, speaker_attribution=fail, temporal_alignment=pass

Example H: closest-category option mapping should pass
Question:
- "Which object category best matches the item on the table?"
Evidence:
- obs_1: grounded region shows a small portable light source with a handle
- obs_2: no screen, pages, or phone-like rectangle is visible
- options: A portable light, B book, C phone, D plate
Trace:
- step 1 says the exact subtype is uncertain
- step 2 rules out book, phone, and plate from visible attributes
- step 3 maps the broad visual description to the closest option, portable light
- final_answer: "A. portable light"
Expected audit:
- verdict: PASS
- scores: logical_coherence=4, completeness=4, factual_correctness=4, reasoning_order=5
- findings:
  - severity=LOW, category=EVIDENCE_LIMITATION, message="The exact subtype is uncertain, but the option mapping is uniquely supported."
- missing_information: []
- feedback: "PASS because the trace does not overclaim the subtype and the closest-category option is uniquely supported."
- diagnostics: option_alignment=pass, evidence_sufficiency=pass_with_minor_uncertainty

Example I: option-induced convention overclaimed as direct evidence
Question:
- "What year does the narration imply?"
Evidence:
- obs_asr_1: transcript says "early 2000s"
- options: A 1995, B 2000, C 2010
Trace:
- step 1 says the transcript directly states "2000"
- step 2 computes the final answer from that exact year
- final_answer: "B. 2000"
Expected audit:
- verdict: FAIL
- scores: logical_coherence=3, completeness=3, factual_correctness=2, reasoning_order=4
- findings:
  - severity=MEDIUM, category=INFERENCE_ERROR, message="The trace converts 'early 2000s' into an exact year without marking it as an option-induced convention."
- missing_information:
  - "whether the task permits mapping a decade/period phrase to the closest year option"
- feedback: "The candidate answer may be the intended option, but the trace must not claim the exact year is directly stated."
- diagnostics: option_alignment=partial, evidence_sufficiency=partial, convention_handling=fail

Example J: blank or truncated benchmark row
Question:
- "Which sound occurs after the"
Evidence:
- obs_1: several sounds are listed, but the temporal boundary is missing from the question
Trace:
- step 1 says the task text is incomplete
- final_answer: ""
Expected audit:
- verdict: FAIL
- scores: logical_coherence=3, completeness=1, factual_correctness=3, reasoning_order=3
- findings:
  - severity=HIGH, category=TASK_DATA_ISSUE, message="The question is truncated and cannot define the requested temporal boundary."
- missing_information:
  - "complete question text or temporal boundary"
- feedback: "The benchmark row is truncated; quarantine or restore the task before evaluating model behavior."
- diagnostics: task_validity=fail, evidence_sufficiency=not_applicable
```

### Are The Current Local Prompt Changes Enough?

No. They are directionally good, but not sufficient.

What the local changes likely fix:

- The planner is less likely to inspect only a single exact frame for action/state questions.
- The auditor now has explicit checks for tone delivery, sound triggers, map geometry, and referent alignment.
- Evidence memory gives planner/synthesizer/auditor a way to avoid silently forgetting earlier useful observations.
- `frame_retriever` chronological/anchor-window language should reduce some order and timestamp regressions.

What remained weak before this implementation:

- No ICL examples are in the prompt code yet, so the current prompts rely on abstract rules.
- The planner still sees `PREPROCESS_PLANNING_MEMORY`, even though the design says to remove planner context entirely.
- The synthesizer can still overuse blank `final_answer` when one option is clearly best supported.
- The auditor score rubric is abstract and can still produce wrong PASS or over-strict FAIL.
- The planner still needs a retrieval/pause system for large preprocess/evidence context instead of relying on one prompt payload. The target implementation is the catalog + iterative retrieval controller described in Section 2.
- The prompt code does not yet explicitly teach scoreboard/price/nameplate label-value adjacency with examples.

## 10. Out-Of-Box Tooling Changes After Minerva/OmniVideoBench Failures

Prompt changes alone are not enough. The inspected failures show repeated capability and state-management gaps:

- Wrong temporal window: `video_485` missed fire/trapped evidence at 77-88s and searched the ending instead; Minerva `0XB0BWhzNc` stopped at the first 3rd-quarter 0:00.0 frame and missed the updated 55-40 scoreboard a few seconds later.
- Sparse-frame counting: flag waves, net hits, side steps, ball tosses, UNO cards, and ingredient appearances cannot be counted reliably from a handful of representative frames.
- Small/structured text: scoreboard, price grids, UI labels, chess clocks, ingredient lists, and ice text need OCR with region/layout preservation.
- Referent and speaker attribution: kidnapper, dog/mother/person relation, and dialogue-purpose questions need explicit identity slots and local turn-taking evidence.
- Auditor false positives: several traces were internally coherent but wrong against gold, meaning the auditor must check coverage and option discriminators, not just prose consistency.

### 10.1 Use `generic_purpose` As A Visual Verifier Only With A Stricter Contract

`generic_purpose` already accepts clips, frames, transcripts, and text context, so a separate visual-verifier backend is not required as the first implementation.

The flaw is that the current `generic_purpose` contract is too loose:

- It does not guarantee dense coverage of a clip.
- It does not expose the sampling policy used for a clip.
- It returns broad prose rather than claim-by-claim verification.
- It can answer from sparse frames while sounding certain.
- It is weak at proving absence, motion, counts, and before/after transitions.

Fix: add a verifier mode or a thin `visual_verifier` adapter that can initially reuse the `generic_purpose` model/backend but changes the request and output contract.

Target request:

```json
{
  "clips": [{"video_id": "video_id", "start_s": 70.0, "end_s": 90.0}],
  "sampling": {
    "mode": "dense",
    "fps": 2.0,
    "include_scene_changes": true,
    "max_frames": 48
  },
  "claims": [
    "fire is visible",
    "humans are trapped or unable to exit",
    "the robot/cat alliance precedes or causes the danger"
  ],
  "query": "Verify each claim from the sampled clip only."
}
```

Target output:

```json
{
  "coverage": {
    "clip_start_s": 70.0,
    "clip_end_s": 90.0,
    "sampled_timestamps_s": [70.0, 70.5, 71.0],
    "coverage_notes": "Dense frames and scene-change frames were inspected."
  },
  "claim_results": [
    {
      "claim": "fire is visible",
      "status": "supported",
      "time_ranges": [{"start_s": 77.0, "end_s": 79.0}],
      "evidence_text": "Visible flames appear in the room/kitchen.",
      "supporting_artifacts": ["frame_077.00", "frame_078.00"]
    },
    {
      "claim": "humans are trapped or unable to exit",
      "status": "supported",
      "time_ranges": [{"start_s": 84.0, "end_s": 88.0}],
      "evidence_text": "The humans are at the door and appear unable to get out.",
      "supporting_artifacts": ["frame_084.00", "frame_088.00"]
    }
  ]
}
```

Planner change:

- The planner must control `sampling.mode`, `fps`, `max_frames`, and temporal windows.
- For action/state/count/absence questions, the planner should prefer verifier mode over free-form `generic_purpose`.
- The synthesizer and auditor should trust verifier outputs only when `coverage` covers the answer-critical interval.

### 10.2 Option-Aware Temporal Search Should Be A Planning Subroutine, Not Preprocessing

`visual_temporal_grounder` already exists and should remain the low-level visual search tool. The missing piece is a controller that decomposes the question/options into atomic search claims and verifies coverage.

Do not make this pure preprocessing. The search targets are task-specific:

- `video_485` option A requires separate searches for service, betrayal, fire, and trapped humans.
- A scoreboard question requires end-of-quarter and post-buzzer update windows.
- A "last smartphone use" question requires all smartphone-use candidates and then the last one.
- A repeated-place question requires exact phrase occurrences, not generic scene captions.

Target flow:

```text
question + options + optional initial_trace_steps
  -> decompose each option into atomic visual/audio/text claims
  -> build search requests for unverified claims
  -> call visual_temporal_grounder, preprocess text retrieval, ASR retrieval, and artifact/evidence retrieval
  -> merge candidate clips by claim
  -> mark which option atoms have candidate support
  -> planner emits execution plan only for unresolved or high-risk atoms
```

This controller can call `visual_temporal_grounder`; it should not replace it.

Target option-claim search state:

```json
{
  "option_claims": [
    {
      "option_id": "A",
      "claim_id": "A.fire",
      "claim": "fire is visible",
      "modality": "visual",
      "status": "candidate_found",
      "candidate_clips": [{"start_s": 76.0, "end_s": 80.0}],
      "source": "visual_temporal_grounder"
    },
    {
      "option_id": "A",
      "claim_id": "A.trapped_humans",
      "claim": "humans are trapped or unable to exit",
      "modality": "visual",
      "status": "unverified",
      "candidate_clips": []
    }
  ]
}
```

Initial trace use:

- In generate mode, `initial_trace_steps` should be treated as hypotheses only.
- If an initial trace says "fire at 01:17", add a candidate search/verification window around 01:17.
- Never treat initial trace text as evidence without tool verification.

### 10.3 Checklist And Counter State Should Be Active Pipeline State

Evidence is passive memory: it stores what was observed. It does not say what remains to be proven.

Add an active `TaskChecklist` that is created before planning, updated after each tool call, and passed to planner, synthesizer, and auditor.

Observed benchmark question structures:

| Benchmark/source | Dominant structures seen | Required checklist modules |
| --- | --- | --- |
| OmniVideoBench/refiner inputs | occurrence/order, dialogue/audio anchors, relation/identity, exact timestamp actions, small text | temporal anchors, referent slots, ASR/quote spans, exact-timestamp verifier, OCR targets |
| Minerva random sample | scoreboard/text, count/motion, option event ordering, speaker/referent attribution, arithmetic over OCR values | option-claim checklist, counter ledger, OCR/layout ledger, referent tracker, arithmetic workspace |
| VideoMathQA MCQ/MBIN | chart/table reading, geometric/counting tasks, arithmetic, final diagram/graph state, step ordering | visual math workspace, chart/table extractor, final-frame selector, count ledger, formula/arithmetic ledger |

Target schema:

```json
{
  "task_type": ["multi_choice", "temporal", "count", "ocr", "relation"],
  "option_claims": [
    {
      "option_id": "A",
      "claim_id": "A.fire",
      "text": "fire is visible",
      "required": true,
      "status": "unverified",
      "supporting_evidence_ids": [],
      "blocking_reason": null
    }
  ],
  "temporal_anchors": [
    {
      "anchor_id": "old_trace_fire_0117",
      "source": "initial_trace_hypothesis",
      "time_s": 77.0,
      "status": "needs_verification"
    }
  ],
  "referent_slots": [
    {
      "slot_id": "accuser",
      "description": "man in the light gray jacket",
      "candidate_artifacts": [],
      "status": "unresolved"
    }
  ],
  "counter": {
    "event_definition": "complete flag wave cycle while ball lands near spectator",
    "candidates": [
      {
        "candidate_id": "wave_001",
        "time_range": {"start_s": 357.0, "end_s": 363.0},
        "status": "unverified",
        "cycle_count": null,
        "dedupe_group": null
      }
    ],
    "count": null
  },
  "ocr_targets": [
    {
      "target_id": "scoreboard_end_q3",
      "text_kind": "scoreboard",
      "needs": ["team labels", "scores", "quarter", "clock"],
      "status": "unverified"
    }
  ],
  "arithmetic": [
    {
      "expression": "sum visible prices",
      "operands": [],
      "status": "blocked_until_ocr_grounded"
    }
  ]
}
```

Can the current pipeline accommodate this?

Yes, but not as prompt-only state.

Needed integration points:

- Create the checklist before planner retrieval.
- Let planner retrieval query by checklist item, not only free-text task terms.
- Let tools update checklist items with support, rejection, or unresolved status.
- Store checklist snapshots in each round directory for debugging.
- Synthesize final answer from checklist-resolved atoms, not only from prose evidence.
- Audit against checklist coverage: do not PASS if required atoms are unverified.

This preserves the current flow while making the state explicit:

```text
preprocess -> task checklist -> planner retrieval -> execution plan -> tools/evidence -> checklist update -> one-shot synthesizer -> auditor
```

### 10.4 Single OCR Model Recommendation

If only one OCR model is allowed, use PaddleOCR-VL rather than classic PaddleOCR.

Reason:

- Many failures are layout-aware OCR failures, not just character recognition failures.
- Scoreboards, product grids, ingredient overlays, chess clocks, UI options, and chart/table values need text plus structure.
- PaddleOCR-VL is designed as a VLM-style document/OCR parser and is more aligned with these needs than plain OCR line detection.

Target OCR request:

```json
{
  "frames": [{"video_id": "video_id", "timestamp_s": 130.0}],
  "regions": [],
  "query": "Read the scoreboard. Return team labels, scores, quarter, and clock with spatial pairing.",
  "output_format": "structured"
}
```

Target OCR output:

```json
{
  "text": "BRISTOL CENT. 55 | E. CATHOLIC 40 | 3rd | 0:00.0",
  "blocks": [
    {"text": "BRISTOL CENT.", "bbox": [0.20, 0.88, 0.35, 0.94], "role": "team_label"},
    {"text": "55", "bbox": [0.36, 0.88, 0.40, 0.94], "role": "score"},
    {"text": "E. CATHOLIC", "bbox": [0.46, 0.88, 0.62, 0.94], "role": "team_label"},
    {"text": "40", "bbox": [0.63, 0.88, 0.67, 0.94], "role": "score"}
  ],
  "pairings": [
    {"label": "BRISTOL CENT.", "value": "55"},
    {"label": "E. CATHOLIC", "value": "40"}
  ],
  "confidence": 0.0
}
```

Download/cache target:

- Repository: `PaddlePaddle/PaddleOCR-VL`
- Cache root: `/fs/nexus-scratch/gnanesh/.cache/huggingface`
- Hub cache: `/fs/nexus-scratch/gnanesh/.cache/huggingface/hub`
- Convenience symlink target after download: `/fs/nexus-scratch/gnanesh/.cache/huggingface/hub/PaddleOCR-VL`

Adoption rule:

- First test it on known failures: Minerva scoreboard `0XB0BWhzNc`, price grid `7eF4EjdYZ7k`, ice text `0ZosUeqDg0`, UI option `GlecDUdZdkE`, ingredient overlay `ODflyuHFr0`, and chess clock/eval overlay `d8UESxnuVAY`.
- Adopt it only if it improves exact text plus label-value pairing on these samples.
