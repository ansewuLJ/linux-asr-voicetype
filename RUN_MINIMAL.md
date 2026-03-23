```bash
# 进入项目
cd linux-asr-voicetype

# 一键安装（依赖 + 服务）
./install.sh

# 下载模型（OpenVINO 示例）
REPO_ID=dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO \
bash scripts/download_hf_model.sh

# 说明：若模型目录缺少 prompt_template.json / mel_filters.npy，请先生成
MODEL_DIR="models/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
uv run python scripts/generate_prompt_template.py --model-dir "$MODEL_DIR" --out-dir "$MODEL_DIR"

# 一键拉起推理服务管理 UI（不会自动启动推理）
bash scripts/start_infer_service.sh --ui-host 127.0.0.1 --ui-port 8788

# 管理 UI 服务（安装后默认已启动）
systemctl --user status asr-manager-ui.service

# 打开控制 UI（浏览器访问）
# http://127.0.0.1:8788

# 在 UI 里保存配置后，点“启动推理服务”
# 看 OpenVINO 推理服务状态
systemctl --user status asr-openvino.service

# 健康检查（成功应返回 status=ok）
curl -sS http://127.0.0.1:8789/health
```
