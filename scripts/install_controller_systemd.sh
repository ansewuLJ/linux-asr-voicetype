#!/usr/bin/env bash
# 安装控制面 systemd user unit
# 用法: ./scripts/install_controller_systemd.sh [--ui-host 127.0.0.1] [--ui-port 8790]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_SRC="$PROJECT_DIR/systemd"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
CONFIG_DIR="$HOME/.config/asr-services"

UI_HOST="127.0.0.1"
UI_PORT="8790"

log() {
  printf '[install-controller] %s\n' "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ui-host)
      UI_HOST="${2:-}"; shift 2 ;;
    --ui-port)
      UI_PORT="${2:-}"; shift 2 ;;
    *)
      log "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$USER_SYSTEMD_DIR"
mkdir -p "$CONFIG_DIR"

# 安装控制面 service（替换硬编码路径）
if [[ -f "$SYSTEMD_SRC/voicetype-ui.service" ]]; then
  sed "s|%h/code/linux-asr-voicetype|$PROJECT_DIR|g" \
    "$SYSTEMD_SRC/voicetype-ui.service" > "$USER_SYSTEMD_DIR/voicetype-ui.service"
  log "Installed: voicetype-ui.service"
else
  log "Missing template: $SYSTEMD_SRC/voicetype-ui.service"
  exit 1
fi

# 创建配置文件
cat > "$CONFIG_DIR/controller.env" << EOF
UI_HOST=${UI_HOST}
UI_PORT=${UI_PORT}
EOF
log "Wrote: $CONFIG_DIR/controller.env"

systemctl --user daemon-reload
log "systemd daemon-reload complete"

log ""
log "安装完成。启动命令:"
log "  systemctl --user enable --now voicetype-ui.service"
log "  访问 http://${UI_HOST}:${UI_PORT}"
