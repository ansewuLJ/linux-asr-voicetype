cd /home/lijie/code/linux-asr-voicetype

UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --extra asr

# 激活 uv 虚拟环境
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com

cd /home/lijie/code/linux-asr-voicetype/frontend/fcitx5-addon
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j
sudo cmake --install build

cd /home/lijie/code/linux-asr-voicetype
uv run voicetype serve --host 127.0.0.1 --port 8787 --model Qwen/Qwen3-ASR-0.6B --device cpu

fcitx5 -r -d
