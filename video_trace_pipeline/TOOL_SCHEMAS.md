# Simple Pipeline Tool Schemas

This file documents the current schema design for `video_trace_pipeline_simple`.

## Model Parity

The simple package keeps its own request schema, and each tool module calls the same model-backed runner as the original pipeline:

- `visual_temporal_grounder`: `TencentARC/TimeLens-8B` through `timelens_runner`, with embedding prefilter disabled.
- `frame_retriever`: `Qwen/Qwen3-VL-Embedding-8B` plus `Qwen/Qwen3-VL-Reranker-8B`.
- `dense_captioner`: `yaolily/TimeChat-Captioner-GRPO-7B`.
- `ocr`: production `PaddleOCR` runner.
- `audio_temporal_grounder`: `Loie/SpotSound` with `nvidia/audio-flamingo-3-hf`.
- `spatial_grounder` and `multimodal_reasoner`: `Qwen/Qwen3.5-9B`.
- `asr`: original WhisperX-style `large-v3` path.

## Common Result Envelope

Every executable tool returns the same outer result envelope:

```json
{
  "ok": true,
  "tool": "frame_retriever",
  "output": {},
  "artifacts": [],
  "error": null,
  "metadata": {}
}
```

The common envelope is for the runner. The planner-specific content lives inside `output.plan`; media-tool content lives inside each tool's `output`.

## Planner Plan Schema

```json
{
  "strategy": "Find the relevant moment, sample frames, read text, then answer.",
  "steps": [
    {
      "id": "s1",
      "tool": "visual_temporal_grounder",
      "purpose": "Find when the scoreboard is visible.",
      "request": {
        "query": "find when the scoreboard is visible",
        "temporal_scope": {},
        "options": {}
      },
      "request_refs": {}
    }
  ]
}
```

Each planner step must have exactly these top-level keys: `id`, `tool`, `purpose`, `request`, `request_refs`.

Each `request` uses:

- `query`: natural-language instruction for the tool.
- `temporal_scope`: video-time limits and anchors.
- `media`: evidence objects passed to a tool, such as `frames`, `regions`, `captions`, `transcript_segments`, and `texts`.
- `options`: small tool-specific knobs, such as `num_frames`.

`request_refs` copies earlier tool outputs into later tool requests:

```json
{
  "request_refs": {
    "media.frames": [
      {
        "from_step": "s2",
        "output": "frames"
      }
    ]
  }
}
```

Do not use `input`, `input_refs`, `step_id`, `path`, or `output.frames`.

## Temporal Scope

All clip times are absolute timestamps in the original video.

```json
{
  "temporal_scope": {
    "clips": [
      {
        "start_s": 10.0,
        "end_s": 15.0
      }
    ],
    "anchors": [
      {
        "time_s": 12.0,
        "radius_s": 1.5,
        "reference": "video"
      }
    ]
  }
}
```

With the example above, the backend samples from `10.5s` to `13.5s`, clamped inside the `10s` to `15s` clip.

If an anchor is relative to the supplied clip, mark it explicitly:

```json
{
  "temporal_scope": {
    "clips": [
      {
        "start_s": 10.0,
        "end_s": 15.0
      }
    ],
    "anchors": [
      {
        "time_s": 2.0,
        "radius_s": 1.0,
        "reference": "clip"
      }
    ]
  }
}
```

That means `2s after the clip start`, so the absolute anchor is `12s`.

## Planner Output Examples

`generic_purpose` has been replaced by `multimodal_reasoner`.

### `dense_captioner`

```json
{
  "id": "s1",
  "tool": "dense_captioner",
  "purpose": "Describe visible actions in the selected span.",
  "request": {
    "query": "describe the visible actions needed to answer the question",
    "temporal_scope": {
      "clips": [
        {
          "start_s": 10.0,
          "end_s": 15.0
        }
      ]
    },
    "options": {}
  },
  "request_refs": {}
}
```

### `asr`

```json
{
  "id": "s1",
  "tool": "asr",
  "purpose": "Transcribe relevant speech in the selected span.",
  "request": {
    "query": "transcribe dialogue relevant to the question",
    "temporal_scope": {
      "clips": [
        {
          "start_s": 10.0,
          "end_s": 15.0
        }
      ]
    },
    "options": {}
  },
  "request_refs": {}
}
```

### `visual_temporal_grounder`

```json
{
  "id": "s1",
  "tool": "visual_temporal_grounder",
  "purpose": "Find moments where the relevant visual event occurs.",
  "request": {
    "query": "moments where the object is picked up",
    "temporal_scope": {},
    "options": {}
  },
  "request_refs": {}
}
```

### `frame_retriever`

```json
{
  "id": "s2",
  "tool": "frame_retriever",
  "purpose": "Sample readable frames from the grounded moment.",
  "request": {
    "query": "frames where the scoreboard text is readable",
    "temporal_scope": {},
    "options": {
      "num_frames": 6
    }
  },
  "request_refs": {
    "temporal_scope.clips": [
      {
        "from_step": "s1",
        "output": "segments"
      }
    ]
  }
}
```

### `ocr`

```json
{
  "id": "s3",
  "tool": "ocr",
  "purpose": "Read text from retrieved frames.",
  "request": {
    "query": "read all visible scoreboard and label text",
    "media": {},
    "options": {}
  },
  "request_refs": {
    "media.frames": [
      {
        "from_step": "s2",
        "output": "frames"
      }
    ]
  }
}
```

### `audio_temporal_grounder`

```json
{
  "id": "s1",
  "tool": "audio_temporal_grounder",
  "purpose": "Find moments containing the requested sound.",
  "request": {
    "query": "bell or alarm sound",
    "temporal_scope": {},
    "options": {}
  },
  "request_refs": {}
}
```

### `spatial_grounder`

```json
{
  "id": "s3",
  "tool": "spatial_grounder",
  "purpose": "Locate the target object in selected frames.",
  "request": {
    "query": "locate the red cup",
    "media": {},
    "temporal_scope": {},
    "options": {}
  },
  "request_refs": {
    "media.frames": [
      {
        "from_step": "s2",
        "output": "frames"
      }
    ]
  }
}
```

### `multimodal_reasoner`

```json
{
  "id": "s4",
  "tool": "multimodal_reasoner",
  "purpose": "Answer using the collected evidence.",
  "request": {
    "query": "answer the question from the supplied evidence",
    "media": {},
    "temporal_scope": {},
    "options": {}
  },
  "request_refs": {
    "media.frames": [
      {
        "from_step": "s2",
        "output": "frames"
      }
    ],
    "media.texts": [
      {
        "from_step": "s3",
        "output": "text"
      }
    ]
  }
}
```

## Tool Request And Output Models

`dense_captioner`

- Request: `{"query": "", "temporal_scope": {}, "options": {}}`
- Output: `{"captions": [], "artifacts": []}`

`asr`

- Request: `{"query": "", "temporal_scope": {}, "options": {}}`
- Output: `{"transcript_segments": []}`

`visual_temporal_grounder`

- Request: `{"query": "", "temporal_scope": {}, "options": {"top_k": 5}}`
- Output: `{"segments": [], "summary": ""}`

This tool is the broad visual temporal search tool. It should receive a strong visual `query`; it scans the video with TimeLens and should not be artificially clipped to weak text-only hints.

`frame_retriever`

- Request: `{"query": "", "temporal_scope": {}, "options": {"num_frames": 5}}`
- Output: `{"frames": []}`

This tool requires `temporal_scope.clips`, `temporal_scope.anchors`, or literal `options.time_hints`; it is not a broad temporal search tool.

`ocr`

- Request: `{"query": "", "media": {}, "options": {}}`
- Output: `{"text": "", "lines": [], "reads": [], "metadata": {}}`

`audio_temporal_grounder`

- Request: `{"query": "", "temporal_scope": {}, "options": {}}`
- Output: `{"segments": [], "summary": ""}`

`spatial_grounder`

- Request: `{"query": "", "media": {}, "temporal_scope": {}, "options": {}}`
- Output: `{"regions": [], "spatial_description": ""}`

`multimodal_reasoner`

- Request: `{"query": "", "media": {}, "temporal_scope": {}, "options": {}}`
- Output: `{"answer": "", "reasoning": "", "evidence": [], "confidence": null}`
