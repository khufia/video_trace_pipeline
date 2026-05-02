# Video Trace Pipeline

`video_trace_pipeline` is a standalone, file-first pipeline for:

- generating benchmark traces from scratch
- refining existing traces with structured evidence
- keeping append-only run outputs and reusable shared evidence caches

The pipeline is organized around:

- `Planner`
- `TraceSynthesizer` (`refiner` replacement: it can generate or revise traces)
- `TraceAuditor` (audits traces and returns feedback)
- a typed tool registry
- native in-repo tool backends
- a readable evidence ledger backed by JSON/JSONL/Markdown plus a SQLite evidence database on disk

Prompt definitions now live in separate files under `video_trace_pipeline/prompts/`:

- `planner_prompt.py`
- `trace_synthesizer_prompt.py`
- `trace_auditor_prompt.py`
- `atomicizer_prompt.py`

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
/fs/nexus-scratch/gnanesh/venv_vdr3
```

`requirements-vllm.txt` is optional and should only be installed when you want an OpenAI-compatible local serving path for multimodal models. The pipeline itself does not require `vllm` if the agent LLMs run over API and the tool stack is provided via local wrappers/servers.

## Setup Notes

- Use Python `3.10`.
- The recommended runtime on this cluster uses the `cuda/12.8.1` and `cudnn/v9.10.2` modules with a Python 3.10 environment. The PyTorch wheel stack in `requirements-local-tools.txt` remains pinned to the CUDA 12.4 wheel index, which is compatible with the newer driver/runtime module setup used by Slurm.
- `--no-build-isolation` is recommended for the editable install so `pip` does not try to rebuild the package toolchain in a fresh temporary environment.
- Machine-specific paths such as dataset roots, cache locations, and GPU assignments belong in the YAML machine profile, not in committed repo files.

## Tool Backends

Two configs are provided:

- `configs/models.yaml`
  This is the planned production-style stack. It wires the pipeline to explicit backends for `TimeLens-8B`, `SpotSound`, `TimeChat-Captioner`, `PaddleOCR`, and the currently configured Qwen multimodal checkpoints through command-backed adapters.
- `configs/models.tool_servers.example.yaml`
  This mirrors the production tool-server wiring and is kept as a reference/example copy.

The command-backed adapters expect a wrapper to read a JSON request on stdin and return JSON on stdout. The repo now includes scaffold entrypoints under [`video_trace_pipeline/tool_wrappers/`](/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline/video_trace_pipeline/tool_wrappers) so the request/response contract is explicit; you can replace those scaffolds with direct launches into TimeLens, SpotSound, TimeChat-Captioner, PaddleOCR, or whichever Qwen multimodal checkpoint you standardize on.

## Evidence DB

Every run now persists evidence in three forms:

- append-only JSONL files for easy inspection
- a readable Markdown ledger grouped by subject and time
- `evidence/evidence.sqlite3` with indexed `evidence_entries` and `atomic_observations` tables

The SQLite DB is the canonical structured evidence store used for retrieval during refinement rounds.

Example commands:

```bash
vtp preprocess --profile configs/machine.example.yaml --benchmark videomathqa --index 0
vtp run --profile configs/machine.example.yaml --models configs/models.yaml --benchmark videomathqa --index 0 --mode generate
vtp run --profile configs/machine.example.yaml --models configs/models.yaml --benchmark minerva --index 0 --mode refine
vtp check-env --profile configs/machine.nexus.yaml --models configs/models.yaml --benchmark videomathqa
```

## Notes

- Shared caches are immutable and reused across runs.
- Run directories are append-only and never cleaned in place.
- Machine-specific paths live in the YAML profile and are redacted from persisted run metadata.
- Local tool implementations live inside `video_trace_pipeline/tools/`; the repo no longer imports or calls code from another project.
- `check-env` now reports both environment readiness and plan alignment against the production local-tool stack.
- Tool outputs are validated with Pydantic result schemas before they enter the evidence ledger.
- Command-backed wrappers receive `runtime.resolved_model_path` when the configured model is present in the shared Hugging Face cache, so wrappers can open the cached local snapshot directly instead of redownloading.
