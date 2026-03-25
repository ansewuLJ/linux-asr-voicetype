<h1 align="center">linux-asr-voicetype</h1>

<p align="center">Local offline speech input for Linux desktop.</p>

<p align="center">
  <a href="https://github.com/ansewuLJ/linux-asr-voicetype/stargazers"><img src="https://img.shields.io/github/stars/ansewuLJ/linux-asr-voicetype?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/ansewuLJ/linux-asr-voicetype/releases"><img src="https://img.shields.io/github/v/release/ansewuLJ/linux-asr-voicetype?style=flat-square" alt="Latest Release"></a>
  <a href="https://github.com/ansewuLJ/linux-asr-voicetype/releases"><img src="https://img.shields.io/github/downloads/ansewuLJ/linux-asr-voicetype/total?style=flat-square" alt="Downloads"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/ansewuLJ/linux-asr-voicetype?style=flat-square" alt="License"></a>
</p>

<p align="center"><strong>English</strong> | <a href="README_ZH.md">简体中文</a></p>

A local offline speech input plugin for Linux desktop environments (integrates with Fcitx or global hotkeys). It supports local inference by default, and also supports deploying inference services to a LAN server.

## ✦ Features

- Fully local deployment, supports CPU/CUDA, with typical memory/VRAM usage around 2GB+
- Powered by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), with strong performance for Chinese and mixed Chinese-English input
- Low latency in test environments (for example, speech of around 10+ seconds can be recognized in about 2 seconds, depending on hardware/model)
- Supports custom hotwords to reduce recognition mistakes
- Supports optional post-processing text models for better output quality
- Supports Fcitx4/Fcitx5 engine integration and global hotkey mode
- Supports both Transformers and OpenVINO inference backends
- Inference services and integration can be managed through web UIs

## ✦ Default Ports
- Inference manager UI: `8788`
- Inference API: `8789`
- Controller UI: `8790`

---

## ✦ System Dependencies

### Check Fcitx Version
```bash
fcitx --version  # if version appears, this is Fcitx4
fcitx5 --version # if version appears, this is Fcitx5
```

### Debian/Ubuntu

**Fcitx5**
```bash
sudo apt install alsa-utils xdotool build-essential cmake pkg-config \
  libcurl4-openssl-dev nlohmann-json3-dev fcitx5-dev

cd frontend/fcitx5-addon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$(nproc)
sudo cmake --install build
fcitx5-remote -r
```

**Fcitx4**
```bash
sudo apt install alsa-utils xdotool build-essential cmake pkg-config \
  libcurl4-openssl-dev nlohmann-json3-dev fcitx-libs-dev

cd frontend/fcitx4-addon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$(nproc)
sudo cmake --install build
fcitx-remote -r
```

### Fedora/RHEL

**Fcitx5**
```bash
sudo dnf install alsa-utils xdotool gcc gcc-c++ make cmake pkgconf-pkg-config \
  libcurl-devel nlohmann-json-devel fcitx5-devel

cd frontend/fcitx5-addon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$(nproc)
sudo cmake --install build
fcitx5-remote -r
```

**Fcitx4**
```bash
sudo dnf install alsa-utils xdotool gcc gcc-c++ make cmake pkgconf-pkg-config \
  libcurl-devel nlohmann-json-devel fcitx-devel

cd frontend/fcitx4-addon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$(nproc)
sudo cmake --install build
fcitx-remote -r
```

---

## ✦ Service Deployment
This project depends on uv. Install uv first from official docs: `https://docs.astral.sh/uv/getting-started/installation/`

### Single-Machine Deployment

```bash
# 1. Python dependencies
cd linux-asr-voicetype
# Ensure uv is installed first (official link above)
# uv sync auto-creates and uses the project virtual environment
uv sync --all-extras
# Activate only if you want to run python/pip directly in current shell
source .venv/bin/activate

# 2. Download model (choose one)

# OpenVINO (recommended for CPU)
uv run voicetype model download dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
  --local-dir models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO

# Required for OpenVINO: generate prompt_template.json and mel_filters.npy
MODEL_DIR="models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
uv run python scripts/generate_prompt_template.py --model-dir "$MODEL_DIR" --out-dir "$MODEL_DIR"

# Or Transformers (CPU/GPU)
uv run voicetype model download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/Qwen3-ASR-0.6B

# 3. Install systemd services
./scripts/install_controller_systemd.sh --ui-host 127.0.0.1 --ui-port 8790
./scripts/install_infer_systemd.sh --ui-host 0.0.0.0 --ui-port 8788

# 4. Start manager UI and configure inference service
systemctl --user enable --now asr-manager-ui.service
# Open http://127.0.0.1:8788 (local) or http://<local-ip>:8788 (remote)

# 5. Start controller UI
systemctl --user enable --now voicetype-ui.service
# Open http://127.0.0.1:8790 to manage integration/hotwords
```

Inference manager UI:
![alt text](asset/infer-ui.png)

On first use, select backend, save config, then start inference service.

Controller UI:
![alt text](asset/control-ui.png)

**Fcitx Integration (Recommended Checks)**

- In Fcitx config -> Addons, find `voicetype` and confirm it is enabled
- Ensure `Host/Port` points to controller UI (`127.0.0.1:8790` by default)
- Default hotkey: hold `Right ALT` to record, release to start recognition
- `Toggle Recording Key` (press once to start, press again to stop) is disabled/empty by default; for long dictation, set a custom hotkey manually

![alt text](asset/fcitx.png)

Post-deployment self-check:
- Open `http://127.0.0.1:8790` and confirm inference service status is online
- In any text input window, hold right `ALT` to speak, release, then confirm recognized text is inserted

### Two-Machine Deployment

Role definitions:
- Inference Node (ASR Server): runs inference manager UI + inference service (`8788/8789`), can be a headless server
- Input Node (Desktop Client): your desktop machine, runs controller UI + input method integration (`8790`)

#### Inference Node (ASR Server)

```bash
# Python dependencies (inference only)
cd linux-asr-voicetype
# Ensure uv is installed first (official link above)
# uv sync auto-creates and uses the project virtual environment
uv sync --extra infer
# Activate only if you want to run python/pip directly in current shell
source .venv/bin/activate

# Download model (choose one)
uv run voicetype model download dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
  --local-dir models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO

# Required for OpenVINO
MODEL_DIR="models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
uv run python scripts/generate_prompt_template.py --model-dir "$MODEL_DIR" --out-dir "$MODEL_DIR"

# Or Transformers
uv run voicetype model download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/Qwen3-ASR-0.6B

# Install systemd (allow remote access to manager UI)
./scripts/install_infer_systemd.sh --ui-host 0.0.0.0 --ui-port 8788
systemctl --user enable --now asr-manager-ui.service
# Open http://<inference-node-ip>:8788 to configure model/service
```

#### Input Node (Desktop Client)

```bash
# Python dependencies (controller only)
cd linux-asr-voicetype
# Ensure uv is installed first (official link above)
# uv sync auto-creates and uses the project virtual environment
uv sync --extra controller
# Activate only if you want to run python/pip directly in current shell
source .venv/bin/activate

# Install and start
./scripts/install_controller_systemd.sh
systemctl --user enable --now voicetype-ui.service
# Open http://127.0.0.1:8790 and set inference node address
```

---

## ✦ Service Control

- Controller UI (`voicetype-ui.service`, port `8790`)
```bash
systemctl --user status voicetype-ui.service   # check status
systemctl --user restart voicetype-ui.service  # restart (common after config changes)
systemctl --user stop voicetype-ui.service     # stop service
```

- Manager UI (`asr-manager-ui.service`, port `8788`)
```bash
systemctl --user status asr-manager-ui.service   # check status
systemctl --user restart asr-manager-ui.service  # restart (common after config changes)
systemctl --user stop asr-manager-ui.service     # stop service
```

Inference service (`8789`) is started/reloaded from manager UI, so no separate long-term operations guide is needed.

To clean all related user services and configs, run: `./scripts/uninstall_user_services.sh`

---

## ✦ Related Files

- `~/.config/systemd/user/`: user service definitions
- `~/.config/asr-services/`: ASR service configs
- `~/.config/voicetype/`: controller runtime configs

---

## ✦ Acknowledgements

- The OpenVINO processing pipeline in this repo references [QwenASRMiniTool](https://github.com/dseditor/QwenASRMiniTool).
