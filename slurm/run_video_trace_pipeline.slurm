#!/bin/bash
#SBATCH --job-name=video_trace_pipeline
#SBATCH --output=/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline_preprocess_task_state/logs/video_trace_pipeline_%A_%a.out
#SBATCH --error=/fs/nexus-scratch/gnanesh/cot/video_trace_pipeline_preprocess_task_state/logs/video_trace_pipeline_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=120G
#SBATCH --mail-user=gnanesh@umd.edu
#SBATCH --mail-type=END,FAIL

set -Eeuo pipefail

# Edit these values directly before running the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
COT_ROOT="${COT_ROOT:-/fs/nexus-scratch/gnanesh/cot}"
VENV_PATH="${VENV_PATH:-/fs/nexus-scratch/gnanesh/venv_vdr3}"

PROFILE="${PROFILE:-$REPO_ROOT/configs/machine.nexus.yaml}"
MODELS="${MODELS:-$REPO_ROOT/configs/models.yaml}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$REPO_ROOT/workspace}"

MODE="${MODE:-generate}"
INPUTS_JSON="${INPUTS_JSON:-$REPO_ROOT/inputs/refiner_inputs.json}"
# INPUTS_JSON="${INPUTS_JSON:-$REPO_ROOT/inputs/refiner_inputs_10pct_part1.json}"
# INPUTS_JSON="${INPUTS_JSON:-$REPO_ROOT/inputs/refiner_inputs_10pct_part2.json}"
# INPUTS_JSON="${INPUTS_JSON:-$REPO_ROOT/inputs/refiner_inputs_10pct_part3.json}"
# Accept a single index or a comma/space-separated list, e.g. INPUT_INDEX="6 7 8" or "6,7,8".
# INPUT_INDEX="${INPUT_INDEX:-6 9}"
# INPUT_INDEX="${INPUT_INDEX:-1 4 7}"
# INPUT_INDEX="${INPUT_INDEX:-2 5 8}"
# INPUT_INDEX="${INPUT_INDEX:-10 11 12}"
# INPUT_INDEX="${INPUT_INDEX:-13 14 15}"
# INPUT_INDEX="${INPUT_INDEX:-18}"
# INPUT_INDEX="${INPUT_INDEX:-19 20 21}"
INPUT_INDEX="${INPUT_INDEX:-0 1 2 3 4 5 6 7 8 9 10}"
# INPUT_INDEX="${INPUT_INDEX:-3 4 5 6 7 8 9 10 11 12 13 14 15}"
# INPUT_INDEX="${INPUT_INDEX:-16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33}"
# INPUT_INDEX="${INPUT_INDEX:-17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
INITIAL_TRACE_PATH="${INITIAL_TRACE_PATH:-}"

CHECK_ENV_ONLY="${CHECK_ENV_ONLY:-0}"
RUN_CHECK_ENV="${RUN_CHECK_ENV:-0}"
STRICT_ENV_CHECK="${STRICT_ENV_CHECK:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
# Default to on-demand model loading for direct bash runs. Opt into
# persisted/preloaded tool models only when the allocation has enough GPU headroom.
#
# Four-GPU note:
# - The safest persistent set is empty: models load only when a tool is called.
# - If you want a warm shared model, persist generic_purpose||spatial_grounder
#   on cuda:3; they share the same Qwen3.5 runtime in configs/machine.nexus.yaml.
# - Consider adding dense_captioner only when dense captions/preprocess are frequent.
# - Keep visual_temporal_grounder on demand for full OmniVideoBench videos, since it
#   can spike memory and benefits from being released between calls.
# - Keep frame_retriever on demand unless frame retrieval dominates the run; by
#   default it uses cuda:2 from the machine profile and no multi-GPU device_map.
PERSIST_TOOL_MODELS="${PERSIST_TOOL_MODELS:-}"
PRELOAD_PERSISTED_MODELS="${PRELOAD_PERSISTED_MODELS:-0}"

OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$COT_ROOT/OPENAI_API_KEY.txt}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$COT_ROOT/HF_TOKEN.txt}"

if [[ $# -gt 0 ]]; then
  echo "This script no longer parses command-line arguments. Edit the config block at the top instead." >&2
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

if [[ ! -f "$INPUTS_JSON" ]]; then
  echo "ERROR: inputs file not found: $INPUTS_JSON" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set and $OPENAI_KEY_FILE was not usable." >&2
  exit 1
fi

cd "$REPO_ROOT"

CHECK_ENV_ARGS=(
  --profile "$PROFILE"
  --models "$MODELS"
  --workspace-root "$WORKSPACE_ROOT"
)
if [[ "$STRICT_ENV_CHECK" == "1" ]]; then
  CHECK_ENV_ARGS+=(--strict)
fi

if [[ "$RUN_CHECK_ENV" == "1" || "$CHECK_ENV_ONLY" == "1" ]]; then
  python -m video_trace_pipeline.cli.main check-env "${CHECK_ENV_ARGS[@]}"
fi

if [[ "$CHECK_ENV_ONLY" == "1" ]]; then
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
  echo "INFO: running input_index=$input_index"

  COMMON_ARGS=(
    --profile "$PROFILE"
    --models "$MODELS"
    --workspace-root "$WORKSPACE_ROOT"
    --inputs-json "$INPUTS_JSON"
    --input-index "$input_index"
  )

  if [[ -n "$PERSIST_TOOL_MODELS" ]]; then
    COMMON_ARGS+=(--persist-tool-models "$PERSIST_TOOL_MODELS")
  fi
  if [[ "$PRELOAD_PERSISTED_MODELS" == "1" ]]; then
    COMMON_ARGS+=(--preload-persisted-models)
  fi

  RUN_ARGS=(
    "${COMMON_ARGS[@]}"
    --mode "$MODE"
    --max-rounds "$MAX_ROUNDS"
  )

  if [[ -n "$INITIAL_TRACE_PATH" ]]; then
    RUN_ARGS+=(--initial-trace-path "$INITIAL_TRACE_PATH")
  fi
  if [[ "$SHOW_PROGRESS" != "1" ]]; then
    RUN_ARGS+=(--no-show-progress)
  fi

  python -m video_trace_pipeline.cli.main run "${RUN_ARGS[@]}"
done
