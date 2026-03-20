#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

HOST="127.0.0.1"
PORT="8787"
MODEL="Qwen/Qwen3-ASR-0.6B"
DEVICE="cpu"
MAX_INFERENCE_BATCH_SIZE="1"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
INSTALL_ADDON="1"
INSTALL_DEPS="1"
ENABLE_SERVICE="1"
ENABLE_UI_SERVICE="1"
UV_INDEX_URL_DEFAULT="${UV_INDEX_URL:-}"

UV_BIN=""

log() {
  printf '[voicetype-install] %s\n' "$*"
}

die() {
  printf '[voicetype-install] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --host <host>                            API bind host (default: 127.0.0.1)
  --port <port>                            API bind port (default: 8787)
  --model <model>                          Model repo/path (default: Qwen/Qwen3-ASR-0.6B)
  --device <device>                        Device map, e.g. cpu/cuda:0 (default: cpu)
  --max-inference-batch-size <n>           Inference batch upper bound (default: 1)
  --hf-endpoint <url>                      HF mirror endpoint (default: https://hf-mirror.com; pass empty to disable)
  --uv-index-url <url>                     Optional UV index mirror
  --no-addon                               Skip Fcitx5 addon build/install
  --no-deps                                Skip apt/dnf dependency installation
  --no-service                             Skip systemd user service install/start
  --no-ui-service                          Skip controller UI service install/start
  -h, --help                               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --host)
    HOST="${2:-}"; shift 2 ;;
  --port)
    PORT="${2:-}"; shift 2 ;;
  --model)
    MODEL="${2:-}"; shift 2 ;;
  --device)
    DEVICE="${2:-}"; shift 2 ;;
  --max-inference-batch-size)
    MAX_INFERENCE_BATCH_SIZE="${2:-}"; shift 2 ;;
  --hf-endpoint)
    HF_ENDPOINT="${2:-}"; shift 2 ;;
  --uv-index-url)
    UV_INDEX_URL_DEFAULT="${2:-}"; shift 2 ;;
  --no-addon)
    INSTALL_ADDON="0"; shift ;;
  --no-deps)
    INSTALL_DEPS="0"; shift ;;
  --no-service)
    ENABLE_SERVICE="0"; shift ;;
  --no-ui-service)
    ENABLE_UI_SERVICE="0"; shift ;;
  -h|--help)
    usage; exit 0 ;;
  *)
    die "Unknown option: $1"
    ;;
  esac
done

[[ "$(uname -m)" == "x86_64" ]] || die "Only x86_64 is supported."

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
else
  die "Cannot detect OS (/etc/os-release missing)."
fi

OS_ID="${ID:-unknown}"
OS_VERSION="${VERSION_ID:-unknown}"
PKG_MGR=""

detect_package_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt"
    return 0
  fi
  if command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
    return 0
  fi
  if command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
    return 0
  fi
  die "No supported package manager found (need apt-get, dnf, or yum)."
}

install_deps_ubuntu() {
  sudo apt-get update
  sudo apt-get install -y \
    curl ca-certificates git build-essential cmake pkg-config \
    libcurl4-openssl-dev nlohmann-json3-dev alsa-utils

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if sudo apt-get install -y fcitx5-dev; then
      :
    elif sudo apt-get install -y fcitx5-libs-dev fcitx5-modules-dev; then
      :
    else
      log "Fcitx5 development packages not found on this distro; skipping addon install."
      INSTALL_ADDON="0"
    fi
  fi
}

install_deps_dnf() {
  sudo dnf -y install epel-release || true
  sudo dnf -y install \
    curl ca-certificates git gcc gcc-c++ make cmake pkgconfig \
    libcurl-devel nlohmann-json-devel alsa-utils

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if ! sudo dnf -y install fcitx5-devel; then
      log "fcitx5-devel not found; skipping addon install."
      INSTALL_ADDON="0"
    fi
  fi
}

install_deps_yum() {
  sudo yum -y install epel-release || true
  sudo yum -y install \
    curl ca-certificates git gcc gcc-c++ make cmake pkgconfig \
    libcurl-devel nlohmann-json-devel alsa-utils

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if ! sudo yum -y install fcitx5-devel; then
      log "fcitx5-devel not found; skipping addon install."
      INSTALL_ADDON="0"
    fi
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi

  log "uv not found, installing to \$HOME/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  elif command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    die "uv installation failed."
  fi
}

run_uv_sync() {
  cd "$REPO_DIR"
  if [[ -n "$UV_INDEX_URL_DEFAULT" ]]; then
    UV_INDEX_URL="$UV_INDEX_URL_DEFAULT" "$UV_BIN" sync --extra asr
  else
    "$UV_BIN" sync --extra asr
  fi
}

install_addon() {
  log "Installing Fcitx5 addon ..."
  sudo rm -f /usr/share/fcitx5/addon/qfinput.conf \
             /usr/local/share/fcitx5/addon/qfinput.conf \
             /usr/local/lib/fcitx5/qfinput-fcitx5.so
  sudo find /usr/lib /usr/lib64 -type f -name "qfinput-fcitx5.so" -delete 2>/dev/null || true

  cd "$REPO_DIR/frontend/fcitx5-addon"
  rm -rf build
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
  cmake --build build -j"$(nproc)"
  sudo cmake --install build
}

install_user_service() {
  local user_service_dir="$HOME/.config/systemd/user"
  local unit_file="$user_service_dir/voicetype.service"
  local runtime_config_file="$HOME/.config/voicetype/runtime.json"
  local hf_endpoint_json="null"
  mkdir -p "$user_service_dir"
  mkdir -p "$(dirname "$runtime_config_file")"
  if [[ -n "$HF_ENDPOINT" ]]; then
    hf_endpoint_json="\"$HF_ENDPOINT\""
  fi
  cat >"$runtime_config_file" <<EOF
{
  "host": "${HOST}",
  "port": ${PORT},
  "model": "${MODEL}",
  "device": "${DEVICE}",
  "backend": "transformers",
  "default_language": "",
  "max_session_seconds": 120,
  "hotwords_file": null,
  "log_level": "info",
  "use_mock_when_unavailable": false,
  "hf_endpoint": ${hf_endpoint_json},
  "hf_probe_timeout_sec": 2.5,
  "hf_hub_etag_timeout_sec": 3,
  "hf_hub_download_timeout_sec": 30,
  "max_inference_batch_size": ${MAX_INFERENCE_BATCH_SIZE}
}
EOF

  {
    echo "[Unit]"
    echo "Description=VoiceType ASR backend"
    echo "After=network-online.target"
    echo
    echo "[Service]"
    echo "Type=simple"
    echo "WorkingDirectory=${REPO_DIR}"
    echo "Environment=UV_CACHE_DIR=/tmp/uv-cache"
    if [[ -n "$HF_ENDPOINT" ]]; then
      echo "Environment=HF_ENDPOINT=${HF_ENDPOINT}"
    fi
    echo "ExecStart=${UV_BIN} run voicetype serve-from-config --config-file ${runtime_config_file}"
    echo "Restart=always"
    echo "RestartSec=2"
    echo
    echo "[Install]"
    echo "WantedBy=default.target"
  } >"$unit_file"

  systemctl --user daemon-reload
  systemctl --user enable voicetype.service
}

install_ui_service() {
  local user_service_dir="$HOME/.config/systemd/user"
  local unit_file="$user_service_dir/voicetype-ui.service"
  mkdir -p "$user_service_dir"

  {
    echo "[Unit]"
    echo "Description=VoiceType Controller UI"
    echo "After=network-online.target"
    echo
    echo "[Service]"
    echo "Type=simple"
    echo "WorkingDirectory=${REPO_DIR}"
    echo "Environment=UV_CACHE_DIR=/tmp/uv-cache"
    echo "ExecStart=${UV_BIN} run voicetype ui --host 127.0.0.1 --port 8790"
    echo "Restart=always"
    echo "RestartSec=2"
    echo
    echo "[Install]"
    echo "WantedBy=default.target"
  } >"$unit_file"

  systemctl --user daemon-reload
  systemctl --user enable --now voicetype-ui.service
}

health_check() {
  local tries=20
  local url="http://${HOST}:${PORT}/health"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "Health check passed: $url"
      return 0
    fi
    sleep 1
  done
  log "Health check not ready yet: $url"
  return 1
}

main() {
  detect_package_manager
  log "Detected platform: ${OS_ID} ${OS_VERSION} ($(uname -m)), package manager: ${PKG_MGR}"

  if [[ "$INSTALL_DEPS" == "1" ]]; then
    log "Installing dependencies ..."
    if [[ "$PKG_MGR" == "apt" ]]; then
      install_deps_ubuntu
    elif [[ "$PKG_MGR" == "dnf" ]]; then
      install_deps_dnf
    else
      install_deps_yum
    fi
  else
    log "Skipping dependency installation (--no-deps)."
  fi

  ensure_uv
  log "Using uv: ${UV_BIN}"

  run_uv_sync

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    install_addon
  else
    log "Skipping Fcitx5 addon installation (--no-addon)."
  fi

  if [[ "$ENABLE_SERVICE" == "1" ]]; then
    install_user_service
    log "ASR service installed but not auto-started. Configure in UI then click start/restart."
  else
    log "Skipping systemd user service setup (--no-service)."
  fi

  if [[ "$ENABLE_UI_SERVICE" == "1" ]]; then
    install_ui_service
    log "UI is running at: http://127.0.0.1:8790/ui"
  else
    log "Skipping UI service setup (--no-ui-service)."
  fi

  log "Install complete."
  log "Service status: systemctl --user status voicetype.service"
  log "Manual start (ASR): systemctl --user start voicetype.service"
  log "Manual start (UI): ${UV_BIN} run voicetype ui --host 127.0.0.1 --port 8790"
}

main "$@"
