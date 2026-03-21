#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
import shutil

import soundfile as sf


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ONNX ASR model on all wav files in a folder.")
    ap.add_argument("--model-dir", type=Path, required=True, help="Model root dir, e.g. .../Qwen3-ASR-0.6B-ONNX-CPU")
    ap.add_argument("--audio-dir", type=Path, required=True, help="Folder containing wav files")
    ap.add_argument("--output-json", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each file N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per file")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime threads (0=default)")
    ap.add_argument("--quantize", type=str, default="int8", choices=["int8", "none"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-run rows in output JSON (default: summary only)",
    )
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()
    out_json = args.output_json.expanduser().resolve()
    source_onnx_dir = model_dir / "onnx_models"

    spec = importlib.util.spec_from_file_location("onnx_inference", model_dir / "onnx_inference.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load onnx_inference.py from {model_dir}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.WAV"))
    if not files:
        raise SystemExit(f"no wav files found in {audio_dir}")

    runtime_onnx_dir = source_onnx_dir
    tokenizer_in_onnx = source_onnx_dir / "tokenizer.json"
    tokenizer_root = model_dir / "tokenizer.json"
    if not tokenizer_in_onnx.exists() and tokenizer_root.exists():
        tmp_runtime = Path("/tmp/onnx_runtime_bench")
        if tmp_runtime.exists():
            shutil.rmtree(tmp_runtime)
        runtime_onnx_dir = tmp_runtime / "onnx_models"
        runtime_onnx_dir.mkdir(parents=True, exist_ok=True)
        for p in source_onnx_dir.iterdir():
            (runtime_onnx_dir / p.name).symlink_to(p)
        (runtime_onnx_dir / "tokenizer.json").symlink_to(tokenizer_root)

    pipe = mod.OnnxAsrPipeline(
        onnx_dir=str(runtime_onnx_dir),
        num_threads=args.threads,
        quantize=args.quantize,
    )

    rows: list[dict[str, object]] = []
    for wav in files:
        samples, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples[:, 0]
        if sr != 16000:
            raise SystemExit(f"{wav} sample_rate={sr}, expected 16000")
        duration_sec = float(len(samples)) / 16000.0

        for _ in range(max(0, args.warmup)):
            _ = pipe._transcribe_chunk(samples, language=None, max_new_tokens=args.max_new_tokens)

        for i in range(args.repeat):
            t0 = time.perf_counter()
            result = pipe._transcribe_chunk(samples, language=None, max_new_tokens=args.max_new_tokens)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            text = str(result.get("text", "") or "")
            ok = bool(text.strip() or str(result.get("raw_output", "")).strip())
            rtf = (latency_ms / 1000.0) / max(duration_sec, 1e-6)
            row = {
                "file": str(wav),
                "run": i + 1,
                "duration_sec": duration_sec,
                "latency_ms": latency_ms,
                "rtf": rtf,
                "ok": ok,
                "text_len": len(text),
                "tokens_generated": int(result.get("timing", {}).get("tokens_generated", 0)),
            }
            rows.append(row)
            print(f"[{i+1}/{args.repeat}] {wav.name} lat={latency_ms:.1f}ms rtf={rtf:.3f} ok={ok}")

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success_runs = sum(1 for r in rows if r["ok"])
    total_runs = len(rows)

    summary = {
        "model_dir": str(model_dir),
        "audio_dir": str(audio_dir),
        "samples": len(files),
        "runs_per_sample": args.repeat,
        "warmup_per_sample": args.warmup,
        "threads": args.threads,
        "quantize": args.quantize,
        "total_runs": total_runs,
        "success_runs": success_runs,
        "success_rate": (success_runs / total_runs) if total_runs else 0.0,
        "latency_ms_mean": statistics.mean(lat_ok) if lat_ok else 0.0,
        "latency_ms_p50": pct(lat_ok, 0.50),
        "latency_ms_p90": pct(lat_ok, 0.90),
        "rtf_mean": statistics.mean(rtf_ok) if rtf_ok else 0.0,
        "rtf_p50": pct(rtf_ok, 0.50),
        "rtf_p90": pct(rtf_ok, 0.90),
    }

    payload: dict[str, object] = {"summary": summary}
    if args.include_rows:
        payload["rows"] = rows

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
