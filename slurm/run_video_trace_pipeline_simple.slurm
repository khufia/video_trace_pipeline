#!/bin/bash
#SBATCH --job-name=video_trace_pipeline_simple
#SBATCH --output=/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline/logs/video_trace_pipeline_simple_%A_%a.out
#SBATCH --error=/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline/logs/video_trace_pipeline_simple_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=120G
#SBATCH --mail-user=gnanesh@umd.edu
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline}"
COT_ROOT="${COT_ROOT:-/fs/nexus-scratch/gnanesh/cot}"
VENV_PATH="${VENV_PATH:-/fs/nexus-scratch/gnanesh/venv_vdr3}"

PROFILE="${PROFILE:-$REPO_ROOT/configs/machine.nexus.yaml}"
MODELS="${MODELS:-$REPO_ROOT/video_trace_pipeline_simple/configs/models.yaml}"

# Benchmark mode. INPUT_INDEX accepts a single index or a comma/space-separated list.
# BENCHMARK="${BENCHMARK:-videomathqa}"
INPUT_INDEX="${INPUT_INDEX:-0}"

# Optional direct task modes. If INPUT_JSON is set, BENCHMARK/INPUT_INDEX are ignored.
INPUT_JSON="${INPUT_JSON:-$REPO_ROOT/inputs/refiner_inputs.json}"
# INPUT_JSON="${INPUT_JSON:-$REPO_ROOT/inputs/refiner_inputs_10pct_part1.json}"
QUESTION="${QUESTION:-}"
VIDEO_PATH="${VIDEO_PATH:-}"
OPTIONS_JSON="${OPTIONS_JSON:-}"

MAX_ROUNDS="${MAX_ROUNDS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
USE_TASK_STATE="${USE_TASK_STATE:-0}"
NO_PREPROCESS_CACHE="${NO_PREPROCESS_CACHE:-0}"
DRY_RUN_PLAN="${DRY_RUN_PLAN:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
SHOW_COLOR="${SHOW_COLOR:-1}"

OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$COT_ROOT/OPENAI_API_KEY.txt}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$COT_ROOT/HF_TOKEN.txt}"

if [[ $# -gt 0 ]]; then
  echo "This script does not parse command-line arguments. Edit the config block at the top instead." >&2
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module load gcc/11.2.0
  module load cuda/12.8.1
  module load cudnn/v9.10.2
  module load Python3/3.10.14
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(realpath "$(which nvcc)")")")"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

source "$VENV_PATH/bin/activate"
export PATH="/fs/nexus-scratch/gnanesh/ffmpeg-7.0.2-amd64-static:$PATH"

export HF_HOME="${HF_HOME:-/fs/nexus-scratch/gnanesh/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/fs/nexus-scratch/gnanesh/.cache/paddlex}"
export PADDLE_HOME="${PADDLE_HOME:-/fs/nexus-scratch/gnanesh/.cache/paddle}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/fs/nexus-scratch/gnanesh/.cache/triton}"
export TORCH_HOME="${TORCH_HOME:-/fs/nexus-scratch/gnanesh/.cache/torch}"
export TMPDIR="${TMPDIR:-/fs/nexus-scratch/gnanesh/.cache/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLAGS_enable_pir_api="${FLAGS_enable_pir_api:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SIMPLE_PIPELINE_COLOR="${SIMPLE_PIPELINE_COLOR:-$SHOW_COLOR}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:pyannote.audio.core.io,ignore::UserWarning:pyannote.audio.utils.reproducibility}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$PADDLE_PDX_CACHE_HOME" "$PADDLE_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$REPO_ROOT/logs"

if [[ -z "${OPENAI_API_KEY:-}" && -f "$OPENAI_KEY_FILE" ]]; then
  OPENAI_API_KEY="$(grep -E -v '^\s*(#|$)' "$OPENAI_KEY_FILE" | head -1 | xargs)"
  export OPENAI_API_KEY
fi

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(grep -E -v '^\s*(#|$)' "$HF_TOKEN_FILE" | head -1 | xargs)"
  export HF_TOKEN
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set and $OPENAI_KEY_FILE was not usable." >&2
  exit 1
fi

cd "$REPO_ROOT"

COMMON_ARGS=(
  --profile "$PROFILE"
  --models "$MODELS"
  --max-rounds "$MAX_ROUNDS"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  COMMON_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$USE_TASK_STATE" == "1" ]]; then
  COMMON_ARGS+=(--use-task-state)
fi
if [[ "$NO_PREPROCESS_CACHE" == "1" ]]; then
  COMMON_ARGS+=(--no-preprocess-cache)
fi
if [[ "$DRY_RUN_PLAN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run-plan)
fi
if [[ "$SHOW_PROGRESS" != "1" ]]; then
  COMMON_ARGS+=(--no-show-progress)
fi
if [[ "$SHOW_COLOR" != "1" ]]; then
  COMMON_ARGS+=(--no-color)
fi

if [[ -n "$INPUT_JSON" ]]; then
  if [[ ! -f "$INPUT_JSON" ]]; then
    echo "ERROR: INPUT_JSON not found: $INPUT_JSON" >&2
    exit 1
  fi
  INPUT_INDEX_LIST="${INPUT_INDEX//,/ }"
  declare -a INPUT_INDEX_VALUES=()
  for input_index in $INPUT_INDEX_LIST; do
    INPUT_INDEX_VALUES+=("$input_index")
  done
  if [[ "${#INPUT_INDEX_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: INPUT_INDEX is empty. Set it to a single index or a comma/space-separated list." >&2
    exit 1
  fi
  for input_index in "${INPUT_INDEX_VALUES[@]}"; do
    echo "INFO: running input_json=$INPUT_JSON index=$input_index"
    python -m video_trace_pipeline_simple.main "${COMMON_ARGS[@]}" --input-json "$INPUT_JSON" --index "$input_index"
  done
  exit 0
fi

if [[ -n "$QUESTION" || -n "$VIDEO_PATH" ]]; then
  if [[ -z "$QUESTION" || -z "$VIDEO_PATH" ]]; then
    echo "ERROR: QUESTION and VIDEO_PATH must both be set for direct task mode." >&2
    exit 1
  fi
  DIRECT_ARGS=(--question "$QUESTION" --video-path "$VIDEO_PATH")
  if [[ -n "$OPTIONS_JSON" ]]; then
    DIRECT_ARGS+=(--options-json "$OPTIONS_JSON")
  fi
  python -m video_trace_pipeline_simple.main "${COMMON_ARGS[@]}" "${DIRECT_ARGS[@]}"
  exit 0
fi

INPUT_INDEX_LIST="${INPUT_INDEX//,/ }"
declare -a INPUT_INDEX_VALUES=()
for input_index in $INPUT_INDEX_LIST; do
  INPUT_INDEX_VALUES+=("$input_index")
done

if [[ "${#INPUT_INDEX_VALUES[@]}" -eq 0 ]]; then
  echo "ERROR: INPUT_INDEX is empty. Set it to a single index or a comma/space-separated list." >&2
  exit 1
fi

for input_index in "${INPUT_INDEX_VALUES[@]}"; do
  echo "INFO: running benchmark=$BENCHMARK index=$input_index"
  python -m video_trace_pipeline_simple.main "${COMMON_ARGS[@]}" --benchmark "$BENCHMARK" --index "$input_index"
done
