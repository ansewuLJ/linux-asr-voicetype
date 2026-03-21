# linux-asr-voicetype

独立语音输入项目：`fcitx5` 负责输入法前端，核心识别逻辑由 Python 服务承载，仅聚焦 `Qwen3-ASR-0.6B/1.7B`。

## 当前里程碑

- Python ASR 服务（FastAPI）
- 会话式音频分块接口（start/chunk/finish）
- 热词加载与清空
- `uv` 项目管理，可直接 `uv sync`
- Qwen 引擎封装（未装 `qwen-asr` 时自动使用 mock 模式，便于联调）

## 目录

- `src/voicetype/` Python 核心服务
- `systemd/` 用户服务文件
- `examples/` 示例配置

## 快速开始

一键安装：

```bash
cd linux-asr-voicetype
./install.sh
```

常用参数示例：

```bash
./install.sh --device cuda:0 --hf-endpoint https://hf-mirror.com
```

默认会使用 `https://hf-mirror.com` 作为 HF 镜像地址。  
如果你要改回官方源：在 UI 中把 HF 镜像输入框清空并保存（或安装时传空值）。

可运行条件（满足即可）：

- `x86_64` Linux
- 可用 `sudo`
- 系统具备可用的基础运行/编译环境：`curl`、`git`、`cmake`、C/C++ 编译工具链、`fcitx5` 开发库、`libcurl`、`nlohmann-json`、`alsa-utils`
- 可访问网络用于安装依赖与首次下载模型（首次完成后可离线使用）

安装脚本说明：

- `install.sh` 默认会自动安装依赖，当前内置了 `apt`/`dnf`/`yum` 三种安装路径（这是脚本实现细节，不是系统能力本质）
- 如果你的系统是定制发行版，且上述依赖已具备，可使用 `./install.sh --no-deps`

验证通过环境（持续补充）：

- Ubuntu（apt 系）
- CentOS/兼容发行版（dnf/yum 系）

手动方式：

```bash
cd linux-asr-voicetype
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

启动服务：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run voicetype serve \
  --host 127.0.0.1 \
  --port 8787 \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype bfloat16
```

国内网络建议直接带镜像：

```bash
uv run voicetype serve \
  --model Qwen/Qwen3-ASR-0.6B \
  --dtype bfloat16 \
  --hf-endpoint https://hf-mirror.com \
  --hf-probe-timeout-sec 2
```

检查健康状态：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run voicetype health
```

打开本地控制 UI（先配置，再启动/重载 ASR）：

```text
http://127.0.0.1:8790/ui
```

日志文件位置：

- ASR 服务日志：`~/.local/state/voicetype/asr.log`
- 控制台 UI 日志：`~/.local/state/voicetype/ui.log`

手动仅启动控制 UI：

```bash
uv run voicetype ui --host 127.0.0.1 --port 8790
```

## 可选：安装真实 Qwen 依赖

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra asr
```

如果未安装 `qwen-asr`，服务会以 mock 输出运行，便于先打通端到端链路。
默认不会 fallback 到 mock，模型不可用会直接启动失败退出。只有显式传 `--allow-mock` 才启用 mock。

## Hugging Face 模型下载（无符号链接）

基于 `huggingface_hub` 的下载接口实现，保证落地目录是完整真实文件，不使用符号链接。

```bash
uv run voicetype model download Qwen/Qwen3-ASR-0.6B \
  --local-dir ~/.local/share/voicetype/models/Qwen3-ASR-0.6B \
  --hf-endpoint https://hf-mirror.com
```

说明：

- 下载后目录结构与仓库根目录一致。
- 命令会进行符号链接校验，不允许残留链接。
- 默认会移除 `--local-dir` 下的 `.cache/huggingface` 元数据目录；如需保留可加 `--keep-hf-metadata`。
- 私有仓库可加 `--token <hf_token>`。

## HTTP API

- `GET /health`
- `POST /v1/session/start`
- `POST /v1/session/{id}/chunk`
- `POST /v1/session/{id}/finish`
- `POST /v1/transcribe`
- `POST /v1/hotwords/load`
- `DELETE /v1/hotwords`
- `GET /v1/recording/status`
- `POST /v1/recording/start`
- `POST /v1/recording/stop`

## Fcitx5 Thin Frontend (MVP)

提供了一个极简 `fcitx5` addon：`frontend/fcitx5-addon`。

- 热键：`Right Alt`（右 Alt）
- 按住右 Alt：开始录音（调用 `/v1/recording/start`）
- 松开右 Alt：停止录音并识别（调用 `/v1/recording/stop`），识别文本直接 commit

构建与安装：

```bash
cd /home/lijie/code/linux-asr-voicetype/frontend/fcitx5-addon
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
```

重启 `fcitx5` 后，在 addon 列表里启用 `VoiceType` 模块即可。

## 下一步

- 把热键、服务 URL、默认语言做成 addon 可配置项
- 增加流式 partial result 推送（WebSocket）
- 增加音频捕获子进程（PipeWire）
