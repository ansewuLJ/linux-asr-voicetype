#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

INSTALL_ADDON="1"
ADDON_TARGET="auto"
RESOLVED_ADDON_TARGET=""
INSTALL_DEPS="1"
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
  cat <<'USAGE'
Usage: ./install.sh [options]

Options:
  --uv-index-url <url>                     Optional UV index mirror
  --no-addon                               Skip Fcitx addon build/install
  --addon-target <auto|fcitx4|fcitx5>      Force addon target (default: auto)
  --no-deps                                Skip apt/dnf dependency installation
  --no-ui-service                          Skip controller UI service install/start
  -h, --help                               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --uv-index-url)
    UV_INDEX_URL_DEFAULT="${2:-}"; shift 2 ;;
  --no-addon)
    INSTALL_ADDON="0"; shift ;;
  --addon-target)
    ADDON_TARGET="${2:-}"; shift 2 ;;
  --no-deps)
    INSTALL_DEPS="0"; shift ;;
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
    libcurl4-openssl-dev nlohmann-json3-dev alsa-utils xdotool

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if [[ "$ADDON_TARGET" == "fcitx5" || "$ADDON_TARGET" == "auto" ]]; then
      if sudo apt-get install -y fcitx5-dev; then
        RESOLVED_ADDON_TARGET="fcitx5"
      elif sudo apt-get install -y fcitx5-libs-dev fcitx5-modules-dev; then
        RESOLVED_ADDON_TARGET="fcitx5"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" && ( "$ADDON_TARGET" == "fcitx4" || "$ADDON_TARGET" == "auto" ) ]]; then
      if sudo apt-get install -y fcitx-libs-dev; then
        RESOLVED_ADDON_TARGET="fcitx4"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" ]]; then
      log "No usable fcitx addon development packages found; skipping addon install."
      INSTALL_ADDON="0"
    fi
  fi
}

install_deps_dnf() {
  sudo dnf -y install epel-release || true
  sudo dnf -y install \
    curl ca-certificates git gcc gcc-c++ make cmake pkgconfig \
    libcurl-devel nlohmann-json-devel alsa-utils xdotool

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if [[ "$ADDON_TARGET" == "fcitx5" || "$ADDON_TARGET" == "auto" ]]; then
      if sudo dnf -y install fcitx5-devel; then
        RESOLVED_ADDON_TARGET="fcitx5"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" && ( "$ADDON_TARGET" == "fcitx4" || "$ADDON_TARGET" == "auto" ) ]]; then
      if sudo dnf -y install fcitx-devel; then
        RESOLVED_ADDON_TARGET="fcitx4"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" ]]; then
      log "No usable fcitx addon development packages found; skipping addon install."
      INSTALL_ADDON="0"
    fi
  fi
}

install_deps_yum() {
  sudo yum -y install epel-release || true
  sudo yum -y install \
    curl ca-certificates git gcc gcc-c++ make cmake pkgconfig \
    libcurl-devel nlohmann-json-devel alsa-utils xdotool

  if [[ "$INSTALL_ADDON" == "1" ]]; then
    if [[ "$ADDON_TARGET" == "fcitx5" || "$ADDON_TARGET" == "auto" ]]; then
      if sudo yum -y install fcitx5-devel; then
        RESOLVED_ADDON_TARGET="fcitx5"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" && ( "$ADDON_TARGET" == "fcitx4" || "$ADDON_TARGET" == "auto" ) ]]; then
      if sudo yum -y install fcitx-devel; then
        RESOLVED_ADDON_TARGET="fcitx4"
      fi
    fi
    if [[ -z "$RESOLVED_ADDON_TARGET" ]]; then
      log "No usable fcitx addon development packages found; skipping addon install."
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
  "$UV_BIN" pip install --python "$REPO_DIR/.venv/bin/python" "pynput>=1.7.7"
}

resolve_addon_target() {
  if [[ -n "$RESOLVED_ADDON_TARGET" ]]; then
    return
  fi
  if [[ "$ADDON_TARGET" == "fcitx5" || "$ADDON_TARGET" == "auto" ]]; then
    if pkg-config --exists Fcitx5Core; then
      RESOLVED_ADDON_TARGET="fcitx5"
      return
    fi
  fi
  if [[ "$ADDON_TARGET" == "fcitx4" || "$ADDON_TARGET" == "auto" ]]; then
    if pkg-config --exists fcitx; then
      RESOLVED_ADDON_TARGET="fcitx4"
      return
    fi
  fi
}

install_addon() {
  resolve_addon_target
  if [[ "$RESOLVED_ADDON_TARGET" == "fcitx5" ]]; then
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
    return
  fi

  if [[ "$RESOLVED_ADDON_TARGET" == "fcitx4" ]]; then
    log "Installing Fcitx4 addon ..."
    cd "$REPO_DIR/frontend/fcitx4-addon"
    rm -rf build
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
    cmake --build build -j"$(nproc)"
    sudo cmake --install build
    return
  fi

  log "No addon target can be resolved; skipping addon install."
}

install_controller_ui_service() {
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
    echo "Environment=DISPLAY=:0"
    echo "Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus"
    echo "Environment=XDG_RUNTIME_DIR=%t"
    echo "Environment=PULSE_SERVER=unix:%t/pulse/native"
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
    log "Addon target: ${RESOLVED_ADDON_TARGET:-none}"
  else
    log "Skipping Fcitx addon installation (--no-addon)."
  fi

  if [[ "$ENABLE_UI_SERVICE" == "1" ]]; then
    install_controller_ui_service
    log "Controller UI is running at: http://127.0.0.1:8790"
  else
    log "Skipping UI service setup (--no-ui-service)."
  fi

  log "Install complete."
  log "Manual start (Controller UI): systemctl --user start voicetype-ui.service"
}

main "$@"
