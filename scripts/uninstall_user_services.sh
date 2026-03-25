#!/usr/bin/env bash
# 卸载当前用户下的 VoiceType/asr 相关 systemd 服务与配置

set -euo pipefail

log() {
  printf '[uninstall] %s\n' "$*"
}

log "1/3 停止并禁用服务..."
systemctl --user disable --now \
  voicetype-ui.service \
  asr-manager-ui.service \
  asr-openvino.service \
  asr-transformers.service || true

log "2/3 删除 systemd user units 与配置..."
rm -f "$HOME/.config/systemd/user/voicetype-ui.service"
rm -f "$HOME/.config/systemd/user/asr-manager-ui.service"
rm -f "$HOME/.config/systemd/user/asr-openvino.service"
rm -f "$HOME/.config/systemd/user/asr-transformers.service"
rm -rf "$HOME/.config/asr-services"

log "3/3 重新加载 systemd..."
systemctl --user daemon-reload
systemctl --user reset-failed

log "完成。"
