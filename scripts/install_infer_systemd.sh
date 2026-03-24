#!/usr/bin/env bash
# 安装推理机 systemd user units
# 用法: ./scripts/install_infer_systemd.sh [--ui-host 0.0.0.0] [--ui-port 8788]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_SRC="$PROJECT_DIR/systemd"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
CONFIG_DIR="$HOME/.config/asr-services"

UI_HOST="0.0.0.0"
UI_PORT="8788"

log() {
  printf '[install-infer] %s\n' "$*"
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

# 安装推理服务（替换硬编码路径）
for svc in asr-openvino.service asr-transformers.service; do
  if [[ -f "$SYSTEMD_SRC/$svc" ]]; then
    sed "s|%h/code/linux-asr-voicetype|$PROJECT_DIR|g" "$SYSTEMD_SRC/$svc" > "$USER_SYSTEMD_DIR/$svc"
    log "Installed: $svc"
  fi
done

# 安装管理面 service
cat > "$USER_SYSTEMD_DIR/asr-manager-ui.service" << EOF
[Unit]
Description=ASR service manager UI
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
Environment=UV_CACHE_DIR=/tmp/uv-cache
ExecStart=${PROJECT_DIR}/scripts/start_asr_manager_ui.sh
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF
log "Installed: asr-manager-ui.service"

# 创建默认配置文件
if [[ ! -f "$CONFIG_DIR/openvino.env" ]]; then
  cat > "$CONFIG_DIR/openvino.env" << 'EOF'
HOST=0.0.0.0
PORT=8789
MODEL=
DEVICE=CPU
HF_ENDPOINT=https://hf-mirror.com
EOF
  log "Created: $CONFIG_DIR/openvino.env"
fi

if [[ ! -f "$CONFIG_DIR/transformers.env" ]]; then
  cat > "$CONFIG_DIR/transformers.env" << 'EOF'
HOST=0.0.0.0
PORT=8789
MODEL=Qwen/Qwen3-ASR-0.6B
DEVICE=cpu
DTYPE=bfloat16
HF_ENDPOINT=https://hf-mirror.com
EOF
  log "Created: $CONFIG_DIR/transformers.env"
fi

if [[ ! -f "$CONFIG_DIR/manager-ui.env" ]]; then
  cat > "$CONFIG_DIR/manager-ui.env" << EOF
UI_HOST=${UI_HOST}
UI_PORT=${UI_PORT}
EOF
  log "Created: $CONFIG_DIR/manager-ui.env"
fi

systemctl --user daemon-reload
log "systemd daemon-reload complete"

log ""
log "安装完成。可用服务："
log "  - asr-manager-ui.service    管理面 UI (${UI_HOST}:${UI_PORT})"
log "  - asr-openvino.service      推理服务 (OpenVINO)"
log "  - asr-transformers.service  推理服务"
log ""
log "启动命令:"
log "  systemctl --user enable --now asr-manager-ui.service"
log "  访问 http://${UI_HOST}:${UI_PORT} 配置并启动推理服务"
