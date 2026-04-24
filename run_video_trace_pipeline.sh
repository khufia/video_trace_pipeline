#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${VENV_PATH:-/nfs-stor/ghazi.ahmad/.conda/envs/video-trace-pipeline}"
PYTHON_BIN="$VENV_PATH/bin/python"
MACHINE_PROFILE_PATH="${MACHINE_PROFILE_PATH:-$REPO_ROOT/configs/machine.example-2gpu.yaml}"
MODELS_CONFIG_PATH="${MODELS_CONFIG_PATH:-$REPO_ROOT/configs/models.yaml}"
INPUTS_JSON="${INPUTS_JSON:-$REPO_ROOT/inputs/refiner_inputs.json}"
RESULTS_NAME="${RESULTS_NAME:-run_refiner_inputs}"
PERSIST_TOOL_MODELS="${PERSIST_TOOL_MODELS:-visual_temporal_grounder}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
if [[ -n "${PROFILE:-}" ]]; then
  echo "INFO: ignoring PROFILE=$PROFILE; use MACHINE_PROFILE_PATH instead." >&2
fi

export CUDA_VISIBLE_DEVICES=0,1

SAMPLE_COUNT="$(
  "$PYTHON_BIN" - "$INPUTS_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    raise SystemExit(f"Expected a JSON list in {path}")
print(len(payload))
PY
)"

for ((input_index = 10; input_index < SAMPLE_COUNT; input_index++)); do
  echo "[$((input_index + 1))/$SAMPLE_COUNT] running input_index=$input_index"
  "$PYTHON_BIN" -m video_trace_pipeline.cli.main run \
    --profile "$MACHINE_PROFILE_PATH" \
    --models "$MODELS_CONFIG_PATH" \
    --mode refine \
    --max-rounds 2 \
    --inputs-json "$INPUTS_JSON" \
    --input-index "$input_index" \
    --persist-tool-models "$PERSIST_TOOL_MODELS" \
    --preload-persisted-models \
    --results-name "$RESULTS_NAME" \
    --mode generate
done
