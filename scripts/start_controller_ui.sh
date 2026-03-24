#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== Editable Defaults =====
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8790}"
CONFIG_FILE="${CONFIG_FILE:-$HOME/.config/asr-services/controller.env}"
# =============================

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

cd "$WORKDIR"
exec "$UV_BIN" run voicetype ui --host "$UI_HOST" --port "$UI_PORT"
