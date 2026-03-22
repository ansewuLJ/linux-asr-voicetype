#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== Editable Defaults =====
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8789}"
MODEL="${MODEL:-/home/lijie/data/models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO}"
DEVICE="${DEVICE:-CPU}"
DTYPE="${DTYPE:-}"
MAX_INFERENCE_BATCH_SIZE="${MAX_INFERENCE_BATCH_SIZE:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CONFIG_FILE="${CONFIG_FILE:-$HOME/.config/asr-services/openvino.env}"
# =============================

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

cd "$WORKDIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

cmd=(
  "$UV_BIN" run voicetype serve
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL"
  --device "$DEVICE"
  --backend openvino
  --max-inference-batch-size "$MAX_INFERENCE_BATCH_SIZE"
)

if [[ -n "${DTYPE:-}" ]]; then
  cmd+=(--dtype "$DTYPE")
fi

exec "${cmd[@]}"
