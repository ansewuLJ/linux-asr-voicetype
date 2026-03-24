# VoiceType

Linux 桌面语音输入方案，基于 Qwen3-ASR，面向中文/中英混合输入场景。
支持 Fcitx4/Fcitx5 接入，也支持全局热键模式；支持 Transformers / OpenVINO 两种推理后端。

默认端口：
- 管理面：`8788`
- 推理面：`8789`
- 控制面：`8790`

---

## 系统依赖安装

### 判断 Fcitx 版本
```bash
fcitx --version  # 显示版本号则为 Fcitx4
fcitx5 --version # 显示版本号则为 Fcitx5
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

## 服务部署
依赖uv，先按 uv 官方文档安装：`https://docs.astral.sh/uv/getting-started/installation/`

### 单机部署

```bash
# 1. Python 依赖
cd linux-asr-voicetype
# 先确保 uv 已安装（官方安装文档见上）
# uv sync 会自动创建并使用项目虚拟环境
uv sync --all-extras
# 需要在当前 shell 里直接运行 python/pip 时，手动激活
source .venv/bin/activate

# 2. 下载模型（二选一）

# OpenVINO（CPU 推荐）
uv run voicetype model download dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
  --local-dir models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO

# OpenVINO 必做：生成 prompt_template.json 和 mel_filters.npy
MODEL_DIR="models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
uv run python scripts/generate_prompt_template.py --model-dir "$MODEL_DIR" --out-dir "$MODEL_DIR"

# 或 Transformers（CPU/GPU）
uv run voicetype model download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/Qwen3-ASR-0.6B

# 3. 安装 systemd 服务
./scripts/install_controller_systemd.sh --ui-host 127.0.0.1 --ui-port 8790
./scripts/install_infer_systemd.sh --ui-host 0.0.0.0 --ui-port 8788
# 上面两个脚本主要会写入：
# ~/.config/systemd/user/voicetype-ui.service      # 控制 UI 的 user service 定义
# ~/.config/systemd/user/asr-manager-ui.service    # 管理 UI 的 user service 定义
# ~/.config/systemd/user/asr-openvino.service      # OpenVINO 推理服务定义
# ~/.config/systemd/user/asr-transformers.service  # Transformers 推理服务定义
# ~/.config/asr-services/controller.env            # 控制 UI 的 host/port 配置
# ~/.config/asr-services/manager-ui.env            # 管理 UI 的 host/port 配置
# ~/.config/asr-services/openvino.env              # OpenVINO 推理参数配置
# ~/.config/asr-services/transformers.env          # Transformers 推理参数配置

# 4. 启动管理面，配置推理服务
systemctl --user enable --now asr-manager-ui.service
# 打开 http://127.0.0.1:8788（本机）或 http://<本机IP>:8788（远程）配置模型路径、推理服务

# 5. 启动控制面
systemctl --user enable --now voicetype-ui.service
# 打开 http://127.0.0.1:8790 管理接入、热词等
```

### 双机部署

#### 推理机

```bash
# Python 依赖（仅推理）
cd linux-asr-voicetype
# 先确保 uv 已安装（官方安装文档见上）
# uv sync 会自动创建并使用项目虚拟环境
uv sync --extra infer
# 需要在当前 shell 里直接运行 python/pip 时，手动激活
source .venv/bin/activate

# 下载模型（二选一）
uv run voicetype model download dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
  --local-dir models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO

# OpenVINO 必做
MODEL_DIR="models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
uv run python scripts/generate_prompt_template.py --model-dir "$MODEL_DIR" --out-dir "$MODEL_DIR"

# 或 Transformers
uv run voicetype model download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/Qwen3-ASR-0.6B

# 安装 systemd（允许外部访问管理 UI）
./scripts/install_infer_systemd.sh --ui-host 0.0.0.0 --ui-port 8788
systemctl --user enable --now asr-manager-ui.service
# 打开 http://<推理机ip>:8788 配置模型路径、推理服务
```

#### 控制机

```bash
# Python 依赖（仅控制）
cd linux-asr-voicetype
# 先确保 uv 已安装（官方安装文档见上）
# uv sync 会自动创建并使用项目虚拟环境
uv sync --extra controller
# 需要在当前 shell 里直接运行 python/pip 时，手动激活
source .venv/bin/activate

# 安装并启动
./scripts/install_controller_systemd.sh
systemctl --user enable --now voicetype-ui.service
# http://127.0.0.1:8790 配置推理机地址
```

---

## 服务控制

- 控制 UI（`voicetype-ui.service`，端口 `8790`）
```bash
systemctl --user status voicetype-ui.service   # 查看当前状态
systemctl --user restart voicetype-ui.service  # 重启服务（改配置后常用）
systemctl --user stop voicetype-ui.service     # 停止服务
```

- 管理 UI（`asr-manager-ui.service`，端口 `8788`）
```bash
systemctl --user status asr-manager-ui.service   # 查看当前状态
systemctl --user restart asr-manager-ui.service  # 重启服务（改配置后常用）
systemctl --user stop asr-manager-ui.service     # 停止服务
```

推理服务（`8789`）由管理 UI 统一启动/重载，不需要单独长期维护一个第三套操作流程。

---

## 致谢

- 本仓库 OpenVINO 处理链路参考了 `QwenASRMiniTool` 项目：
  `https://github.com/dseditor/QwenASRMiniTool.git`
