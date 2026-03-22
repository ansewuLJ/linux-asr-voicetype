#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== Editable Defaults =====
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
REPO_ID="${REPO_ID:-Qwen/Qwen3-ASR-0.6B}"
MODEL_ROOT="${MODEL_ROOT:-$WORKDIR/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REVISION="${REVISION:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"  # 1=true, 0=false
# =============================

repo_name="$(basename "$REPO_ID")"
target_dir="$MODEL_ROOT/$repo_name"

mkdir -p "$MODEL_ROOT"
cd "$WORKDIR"

args=(
  run voicetype model download "$REPO_ID"
  --local-dir "$target_dir"
  --max-workers "$MAX_WORKERS"
)

if [[ -n "$HF_ENDPOINT" ]]; then
  args+=(--hf-endpoint "$HF_ENDPOINT")
fi
if [[ -n "$REVISION" ]]; then
  args+=(--revision "$REVISION")
fi
if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
  args+=(--force)
fi

echo "[download_hf_model] repo_id=$REPO_ID"
echo "[download_hf_model] target_dir=$target_dir"
"$UV_BIN" "${args[@]}"
echo "[download_hf_model] done: $target_dir"
