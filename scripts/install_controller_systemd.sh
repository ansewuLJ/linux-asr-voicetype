#!/usr/bin/env bash
# 安装控制面 systemd user unit
# 用法: ./scripts/install_controller_systemd.sh [--host 127.0.0.1] [--port 8790]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
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

# 创建启动脚本
cat > "$PROJECT_DIR/scripts/start_controller_ui.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
UI_HOST="\${UI_HOST:-${UI_HOST}}" UI_PORT="\${UI_PORT:-${UI_PORT}}" \\
  exec uv run voicetype ui --host "\${UI_HOST}" --port "\${UI_PORT}"
EOF
chmod +x "$PROJECT_DIR/scripts/start_controller_ui.sh"
log "Created: scripts/start_controller_ui.sh"

# 创建 systemd service
cat > "$USER_SYSTEMD_DIR/voicetype-ui.service" << EOF
[Unit]
Description=VoiceType Controller UI
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
Environment=UI_HOST=${UI_HOST}
Environment=UI_PORT=${UI_PORT}
Environment=DISPLAY=:0
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus
Environment=XDG_RUNTIME_DIR=%t
Environment=PULSE_SERVER=unix:%t/pulse/native
ExecStart=${PROJECT_DIR}/scripts/start_controller_ui.sh
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF
log "Created: $USER_SYSTEMD_DIR/voicetype-ui.service"

# 创建配置文件
if [[ ! -f "$CONFIG_DIR/controller.env" ]]; then
  cat > "$CONFIG_DIR/controller.env" << EOF
UI_HOST=${UI_HOST}
UI_PORT=${UI_PORT}
EOF
  log "Created: $CONFIG_DIR/controller.env"
fi

systemctl --user daemon-reload
log "systemd daemon-reload complete"

log ""
log "安装完成。启动命令:"
log "  systemctl --user enable --now voicetype-ui.service"
log "  访问 http://${UI_HOST}:${UI_PORT}"