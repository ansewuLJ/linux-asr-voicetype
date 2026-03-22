# Benchmark Guide (CPU/GPU Compare)

Use the same fixed wav set every time, so optimization results are comparable.

## 1) Prepare fixed wav files in your own folder

Use your own absolute directory path, for example:

`/home/lijie/data/asr_bench_audio`

Use 16k mono wav.

### Quickest Way to Create WAV Samples

```bash
mkdir -p /home/lijie/data/asr_bench_audio
arecord -f S16_LE -c 1 -r 16000 -t wav /home/lijie/data/asr_bench_audio/test_01.wav
```

## 2) Start service

```bash
systemctl --user restart voicetype.service
```

## 3) Run benchmark

```bash
cd /home/lijie/code/linux-asr-voicetype
uv run python benchmarks/bench_transcribe.py \
  --audio-dir /home/lijie/data/asr_bench_audio \
  --base-url http://127.0.0.1:8787 \
  --warmup 1 \
  --repeat 5 \
  --output-json /tmp/voicetype-bench.json
```

By default, output JSON contains only `summary`.
If you want per-run rows, add `--include-rows`.

## 4) Compare key metrics

- `latency_ms_p50` / `latency_ms_p90`
- `rtf_p50` / `rtf_p90`
- `success_rate`

Use the same machine + same wav set + same repeat count for fair comparison.




cd /home/lijie/code/linux-asr-voicetype
uv run python benchmarks/bench_transcribe.py \
  --audio-dir /home/lijie/code/audio \
  --base-url http://127.0.0.1:8787 \
  --warmup 1 \
  --repeat 5 \
  --output-json /home/lijie/code/audio-bench-results/trans_gpu.json



cd /home/lijie/code/linux-asr-voicetype
uv run python benchmarks/bench_transcribe.py \
  --audio-dir /home/lijie/code/audio \
  --base-url http://127.0.0.1:8787 \
  --warmup 1 \
  --repeat 5 \
  --output-json /home/lijie/code/audio-bench-results/trans_cpu_retry.json




cd /home/lijie/code/linux-asr-voicetype
python benchmarks/bench_qwen3_gguf_transcribe.py \
  --audio-dir /home/lijie/code/audio \
  --qwen-repo-dir /home/lijie/code/Qwen3-ASR-GGUF \
  --model-dir /home/lijie/data/models/Qwen3-ASR-0.6B-gguf \
  --repeat 5 \
  --warmup 1 \
  --output-json /home/lijie/code/audio-bench-results/qwen3_gguf_cpu.json


cd /home/lijie/code/linux-asr-voicetype
python benchmarks/bench_onnx_folder.py \
  --model-dir /home/lijie/data/models/Qwen3-ASR-0.6B-ONNX-CPU \
  --audio-dir /home/lijie/code/audio \
  --quantize int8 \
  --warmup 1 \
  --repeat 5 \
  --output-json /home/lijie/code/audio-bench-results/onnx_cpu_bench_int8_summary.json


## OpenVINO Qwen3-ASR

完整步骤（独立 uv 环境、模型转换、CPU/GPU benchmark）见：

- `benchmarks/OPENVINO_QWEN3_ASR_BENCH.md`

快速跑法（默认你已完成模型转换）：

```bash
cd /home/lijie/code/linux-asr-voicetype
source /home/lijie/code/.venv-openvino-asr/bin/activate

python benchmarks/bench_openvino_qwen3_asr.py \
  --audio-dir /home/lijie/code/audio \
  --model-dir /home/lijie/data/models/Qwen3-ASR-0.6B-OV \
  --device CPU \
  --max-inference-batch-size 32 \
  --helper-py /home/lijie/code/Qwen3-ASR/qwen_3_asr_helper.py \
  --warmup 1 \
  --repeat 5 \
  --output-json /home/lijie/code/audio-bench-results/openvino_qwen3_asr_cpu.json
```
