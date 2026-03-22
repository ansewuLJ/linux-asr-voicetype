#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== Editable Defaults =====
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8789}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-0.6B}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-bfloat16}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_INFERENCE_BATCH_SIZE="${MAX_INFERENCE_BATCH_SIZE:-1}"
CONFIG_FILE="${CONFIG_FILE:-$HOME/.config/asr-services/transformers.env}"
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

exec "$UV_BIN" run voicetype serve \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --backend transformers \
  --dtype "$DTYPE" \
  --max-inference-batch-size "$MAX_INFERENCE_BATCH_SIZE"
