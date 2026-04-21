# Video Trace Pipeline

`video_trace_pipeline` is a standalone, file-first pipeline for:

- generating benchmark traces from scratch
- refining existing traces with structured evidence
- keeping append-only run outputs and reusable shared evidence caches

The pipeline is organized around:

- `Planner`
- `TraceSynthesizer`
- `TraceAuditor`
- a typed tool registry
- native in-repo tool backends
- a readable evidence ledger backed by JSON/JSONL/Markdown on disk

## Layout

```text
video_trace_pipeline/
  configs/
  video_trace_pipeline/
  tests/
  requirements.txt
  requirements-local-tools.txt
  requirements-whisperx.txt
  requirements-vllm.txt
```

## Quick Start

```bash
conda create -y -n video-trace-pipeline python=3.10
conda activate video-trace-pipeline
python -m pip install -r requirements.txt -r requirements-local-tools.txt
python -m pip install --no-build-isolation -e .
python -m pip install -r requirements-whisperx.txt
python -m video_trace_pipeline.cli.main --help
```

The environment created on this machine is:

```text
/home/ghazi.ahmad/.conda/envs/video-trace-pipeline
```

`requirements-vllm.txt` is optional and should only be installed when you want the separate `vllm` backend path.

## Setup Notes

- Use Python `3.10`.
- The pinned local tool stack targets CUDA `12.4`.
- `--no-build-isolation` is recommended for the editable install so `pip` does not try to rebuild the package toolchain in a fresh temporary environment.
- Machine-specific paths such as dataset roots, cache locations, and GPU assignments belong in the YAML machine profile, not in committed repo files.

Example commands:

```bash
vtp preprocess --profile configs/machine.example.yaml --benchmark videomathqa --index 0
vtp run --profile configs/machine.example.yaml --benchmark videomathqa --index 0 --mode generate
vtp run --profile configs/machine.example.yaml --benchmark minerva --index 0 --mode refine
```

## Notes

- Shared caches are immutable and reused across runs.
- Run directories are append-only and never cleaned in place.
- Machine-specific paths live in the YAML profile and are redacted from persisted run metadata.
- Backend implementations live inside `video_trace_pipeline/tools/`; the repo no longer imports or calls code from another project.
- The default internal stack uses in-repo frame sampling plus configurable multimodal/API backends for dense captioning, frame selection, OCR, spatial grounding, and generic extraction.
