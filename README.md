# linux-asr-voicetype

Linux 桌面语音输入方案，基于 Qwen3-ASR，面向中文/中英混合输入场景。  
支持 Fcitx4/Fcitx5 接入，也支持全局热键模式；支持 Transformers / OpenVINO 两种推理后端，并提供“推理管理 UI + 最终接入 UI”的双控制面，便于单机或双机部署。  

默认端口如下：
- 推理管理 UI：`8788`
- 推理服务：`8789`
- 最终接入 UI：`8790`

三者可以独立运行。你可以单机部署，也可以分成两台机器部署。

## 一、单机部署（全部在一台机器）

### 1) 安装（接入侧 + 依赖）

```bash
cd linux-asr-voicetype
./install.sh
```

这一步会启动最终接入 UI（`http://127.0.0.1:8790`）。

### 2) 下载模型（按需二选一）

OpenVINO（CPU 推荐）：

```bash
cd linux-asr-voicetype
REPO_ID=dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
bash scripts/download_hf_model.sh
```

Transformers（CPU/GPU）：

```bash
cd linux-asr-voicetype
REPO_ID=Qwen/Qwen3-ASR-0.6B \
bash scripts/download_hf_model.sh
```

### 3) 启动推理侧（管理 UI + systemd 服务）

```bash
cd linux-asr-voicetype
bash scripts/start_infer_service.sh --ui-host 127.0.0.1 --ui-port 8788
```

打开推理管理 UI：`http://127.0.0.1:8788`

在推理管理 UI 中按顺序：
1. 选后端（`openvino` / `transformers`）
2. 填模型、host、port、device、HF 镜像
3. 点“保存配置”
4. 点“启动/重载推理服务”

### 4) 在最终接入 UI 接入推理服务

打开 `http://127.0.0.1:8790`，把推理地址填为：
- host: `127.0.0.1`
- port: `8789`

按你的使用方式继续配置：

### 4.1 Fcitx5 配置

1. 打开 `fcitx5-configtool`。
2. 进入“附加组件”，找到 `VoiceType`（或 `VoiceType 语音输入`），确保已启用。
3. 打开该组件配置，填写：
   - `ASR Host`：最终接入 UI 主机（一般是本机 `127.0.0.1`）
   - `ASR Port`：最终接入 UI 端口（默认 `8790`）
   - `Hold-To-Talk Key` / `Toggle Recording Key`：按你的习惯设置热键
4. 应用配置后执行：

```bash
fcitx5 -r
```

### 4.2 Fcitx4 配置

1. 打开 `fcitx-configtool`。
2. 进入“附加组件”，找到 `VoiceType`，确保已启用。
3. 在 VoiceType 配置里填写：
   - `ASR Host`：最终接入 UI 主机（一般是本机 `127.0.0.1`）
   - `ASR Port`：最终接入 UI 端口（默认 `8790`）
   - `HoldKey` / `ToggleKey`：设置录音热键
4. 应用配置后执行：

```bash
fcitx -r
```

### 4.3 全局热键模式（Fcitx 不可用时）

1. 打开最终接入 UI：`http://127.0.0.1:8790`。
2. 在“接入配置”里填写推理服务 `host` 和 `port` 并保存。
3. 在“全局热键”里设置按住说话或切换录音热键并保存。
4. 点击健康检查/保存后，确认服务状态为可用再使用。

## 二、双机部署（A 有图形，B 无图形）

- A 机：桌面机，只负责“最终接入 + Fcitx”（`8790`）
- B 机：推理机，只负责“推理管理 UI + 推理服务”（`8788/8789`）

### A 机（有图形）

```bash
cd linux-asr-voicetype
./install.sh
```

打开最终接入 UI：`http://127.0.0.1:8790`

### B 机（无图形，纯推理）

1) 安装最小集（不装 Fcitx，不启最终 UI）：

```bash
cd linux-asr-voicetype
./install.sh --no-addon --no-ui-service
```

2) 下载模型（按需二选一）：

```bash
cd linux-asr-voicetype
REPO_ID=dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO bash scripts/download_hf_model.sh
# 或
REPO_ID=Qwen/Qwen3-ASR-0.6B bash scripts/download_hf_model.sh
```

3) 启动推理侧：

```bash
cd linux-asr-voicetype
bash scripts/start_infer_service.sh --ui-host 0.0.0.0 --ui-port 8788
```

4) 在 B 机推理管理 UI（`http://<B机IP>:8788`）中把推理服务配置为：
- host: `0.0.0.0`
- port: `8789`

5) 回到 A 机最终接入 UI（`8790`）里填写：
- host: `<B机IP>`
- port: `8789`

6) A 机上的 Fcitx4/Fcitx5 插件请配置到本机最终接入 UI：
- host: `127.0.0.1`
- port: `8790`

## 三、常用 systemd 命令


全部停止：

```bash
systemctl --user stop asr-transformers.service
systemctl --user stop asr-openvino.service
systemctl --user stop asr-manager-ui.service
systemctl --user stop voicetype-ui.service
```
