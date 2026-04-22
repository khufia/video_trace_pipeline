#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec python -m video_trace_pipeline.cli.main run \
  --profile "${PROFILE:-configs/machine.example.yaml}" \
  --models "${MODELS:-configs/models.yaml}" \
  --benchmark "${BENCHMARK:-videomathqa}" \
  --index "${INDEX:-0}" \
  --mode "${MODE:-generate}" \
  "$@"
