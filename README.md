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

```bash
cd linux-asr-voicetype
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

启动服务：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run voicetype serve \
  --host 127.0.0.1 \
  --port 8787 \
  --model Qwen/Qwen3-ASR-0.6B
```

国内网络建议直接带镜像：

```bash
uv run voicetype serve \
  --model Qwen/Qwen3-ASR-0.6B \
  --hf-endpoint https://hf-mirror.com \
  --hf-probe-timeout-sec 2
```

检查健康状态：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run voicetype health
```

## 可选：安装真实 Qwen 依赖

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra asr
```

如果未安装 `qwen-asr`，服务会以 mock 输出运行，便于先打通端到端链路。
默认不会 fallback 到 mock，模型不可用会直接启动失败退出。只有显式传 `--allow-mock` 才启用 mock。

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
