#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== Editable Defaults =====
REPO_DIR="${REPO_DIR:-$PROJECT_DIR}"
USER_UNIT_DIR="${USER_UNIT_DIR:-$HOME/.config/systemd/user}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/.config/asr-services}"
STACK_CONFIG_FILE="${STACK_CONFIG_FILE:-$CONFIG_DIR/stack.env}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8788}"
MANAGER_UI_CONFIG_FILE="${MANAGER_UI_CONFIG_FILE:-$CONFIG_DIR/manager-ui.env}"
# ===========================

if [[ -f "$STACK_CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STACK_CONFIG_FILE"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ui-host)
      UI_HOST="${2:-$UI_HOST}"
      shift 2
      ;;
    --ui-port)
      UI_PORT="${2:-$UI_PORT}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--ui-host 127.0.0.1|0.0.0.0] [--ui-port 8788]" >&2
      exit 1
      ;;
  esac
done

install_units() {
  mkdir -p "$USER_UNIT_DIR"
  install -m 0644 "$REPO_DIR/systemd/asr-transformers.service" "$USER_UNIT_DIR/asr-transformers.service"
  install -m 0644 "$REPO_DIR/systemd/asr-openvino.service" "$USER_UNIT_DIR/asr-openvino.service"
  install -m 0644 "$REPO_DIR/systemd/asr-manager-ui.service" "$USER_UNIT_DIR/asr-manager-ui.service"
  systemctl --user daemon-reload
}

cleanup_legacy_units() {
  # Migrate old service names to the new asr-* naming.
  systemctl --user stop voicetype.service || true
  systemctl --user disable voicetype.service || true
  rm -f "$USER_UNIT_DIR/voicetype.service"
  systemctl --user daemon-reload
}

enable_units() {
  systemctl --user enable asr-transformers.service
  systemctl --user enable asr-openvino.service
  systemctl --user enable asr-manager-ui.service
}

start_stack() {
  # Always require manual start from UI after saving config.
  systemctl --user stop asr-transformers.service || true
  systemctl --user stop asr-openvino.service || true
  systemctl --user restart asr-manager-ui.service
}

write_manager_ui_config() {
  mkdir -p "$CONFIG_DIR"
  cat > "$MANAGER_UI_CONFIG_FILE" <<EOF
UI_HOST=$UI_HOST
UI_PORT=$UI_PORT
EOF
}

print_port_hints() {
  if ss -ltnp 2>/dev/null | rg -q '127.0.0.1:8780|127.0.0.1:8781|127.0.0.1:8782|127.0.0.1:8787'; then
    cat <<'EOF'
[WARN] Detected legacy listeners on old ports (8780/8781/8782/8787).
[WARN] Usually caused by old manual processes.
[WARN] Check and stop with:
  ss -ltnp | rg '8780|8781|8782|8787'
  pkill -f "voicetype serve --host 127.0.0.1 --port 8780"
  pkill -f "voicetype asr-manager-ui --host 127.0.0.1 --port 8781"
  pkill -f "voicetype ui --host 127.0.0.1 --port 8782"
  pkill -f "voicetype serve --host 127.0.0.1 --port 8787"
EOF
  fi
}

cd "$REPO_DIR"
mkdir -p "$CONFIG_DIR"
write_manager_ui_config
cleanup_legacy_units
install_units
enable_units
start_stack
systemctl --user --no-pager --full status asr-transformers.service asr-openvino.service asr-manager-ui.service || true
print_port_hints
echo "[INFO] manager-ui: http://$UI_HOST:$UI_PORT"
