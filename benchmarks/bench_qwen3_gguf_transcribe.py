from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf


def collect_wavs(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    files = sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.WAV"))
    return [p.resolve() for p in files]


def load_wav_file(path: Path):
    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != 16000:
        raise SystemExit(f"{path} sample_rate={sr}, expected 16000")
    if getattr(samples, "ndim", 1) > 1:
        samples = samples[:, 0]
    return samples


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def round_summary(summary: dict[str, object], ndigits: int = 3) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for k, v in summary.items():
        if isinstance(v, float):
            rounded[k] = round(v, ndigits)
        else:
            rounded[k] = v
    return rounded


def get_model_filenames(precision: str, is_aligner: bool = False) -> dict[str, str]:
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return {
        "frontend": f"{prefix}_encoder_frontend.{precision}.onnx",
        "backend": f"{prefix}_encoder_backend.{precision}.onnx",
    }


def check_model_files(
    model_dir: Path,
    asr_llm_fn: str,
    asr_frontend_fn: str,
    asr_backend_fn: str,
    timestamp: bool,
    align_llm_fn: str,
    align_frontend_fn: str,
    align_backend_fn: str,
) -> None:
    missing: list[str] = []

    for fn in [asr_llm_fn, asr_frontend_fn, asr_backend_fn]:
        p = model_dir / fn
        if not p.exists():
            missing.append(str(p))

    if timestamp:
        for fn in [align_llm_fn, align_frontend_fn, align_backend_fn]:
            p = model_dir / fn
            if not p.exists():
                missing.append(str(p))

    if missing:
        raise SystemExit("Missing model files:\n" + "\n".join(f"  - {x}" for x in missing))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark local Qwen3-ASR-GGUF engine latency on fixed wav set (same summary style as bench_transcribe.py)."
    )
    ap.add_argument("--audio-dir", type=Path, required=True, help="Absolute directory path to auto-scan wav files")
    ap.add_argument("--qwen-repo-dir", type=Path, required=True, help="Path to Qwen3-ASR-GGUF repo root")
    ap.add_argument("--model-dir", type=Path, required=True, help="Path to Qwen3-ASR model files directory")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each sample N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup rounds per sample")
    ap.add_argument("--language", default=None, help="Optional language hint, e.g. Chinese/English")
    ap.add_argument("--context", default="", help="Optional prompt/context")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--precision", default="int4", choices=["fp32", "fp16", "int8", "int4"])
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--chunk-size", type=float, default=40.0)
    ap.add_argument("--memory-num", type=int, default=1)
    ap.add_argument("--dml", action="store_true", help="Enable DirectML (mostly Windows)")
    ap.add_argument("--vulkan", action="store_true", help="Enable Vulkan")
    ap.add_argument("--timestamp", action="store_true", help="Enable aligner/timestamp")
    ap.add_argument("--seek-start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--output-json", type=Path, default=None, help="Optional result json path")
    ap.add_argument("--include-rows", action="store_true", help="Include per-run detailed rows in output json")
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute path, got: {args.audio_dir}")
    if not args.model_dir.is_absolute():
        raise SystemExit(f"--model-dir must be absolute path, got: {args.model_dir}")

    qwen_repo_dir = args.qwen_repo_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if not qwen_repo_dir.exists():
        raise SystemExit(f"--qwen-repo-dir not found: {qwen_repo_dir}")
    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")

    files = collect_wavs(args.audio_dir)
    if not files:
        raise SystemExit(f"No wav files found from audio-dir: {args.audio_dir}")

    # Keep behavior aligned with upstream transcribe.py
    if not args.vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"

    # Import upstream engine from local repo path
    sys.path.insert(0, str(qwen_repo_dir))
    try:
        from qwen_asr_gguf.inference import AlignerConfig, ASREngineConfig, QwenASREngine  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to import qwen_asr_gguf from {qwen_repo_dir}: {exc}")

    asr_files = get_model_filenames(args.precision, is_aligner=False)
    align_files = get_model_filenames(args.precision, is_aligner=True)

    align_cfg = None
    if args.timestamp:
        align_cfg = AlignerConfig(
            model_dir=str(model_dir),
            use_dml=args.dml,
            encoder_frontend_fn=align_files["frontend"],
            encoder_backend_fn=align_files["backend"],
            n_ctx=args.n_ctx,
        )

    cfg = ASREngineConfig(
        model_dir=str(model_dir),
        use_dml=args.dml,
        encoder_frontend_fn=asr_files["frontend"],
        encoder_backend_fn=asr_files["backend"],
        n_ctx=args.n_ctx,
        chunk_size=args.chunk_size,
        memory_num=args.memory_num,
        enable_aligner=args.timestamp,
        align_config=align_cfg,
        verbose=False,
    )

    check_model_files(
        model_dir=model_dir,
        asr_llm_fn=cfg.llm_fn,
        asr_frontend_fn=cfg.encoder_frontend_fn,
        asr_backend_fn=cfg.encoder_backend_fn,
        timestamp=args.timestamp,
        align_llm_fn=cfg.align_config.llm_fn,
        align_frontend_fn=cfg.align_config.encoder_frontend_fn,
        align_backend_fn=cfg.align_config.encoder_backend_fn,
    )

    rows: list[dict[str, object]] = []
    t_init0 = time.perf_counter()
    engine = QwenASREngine(config=cfg)
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    try:
        for wav_path in files:
            if not wav_path.exists():
                print(f"[skip] missing: {wav_path}")
                continue

            samples = load_wav_file(wav_path)
            duration_sec = float(samples.shape[0]) / 16000.0

            for _ in range(max(0, args.warmup)):
                _ = engine.transcribe(
                    audio_file=str(wav_path),
                    language=args.language,
                    context=args.context,
                    start_second=args.seek_start,
                    duration=args.duration,
                    temperature=args.temperature,
                )

            for i in range(args.repeat):
                t0 = time.perf_counter()
                ok = True
                text = ""
                err = ""
                try:
                    res = engine.transcribe(
                        audio_file=str(wav_path),
                        language=args.language,
                        context=args.context,
                        start_second=args.seek_start,
                        duration=args.duration,
                        temperature=args.temperature,
                    )
                    text = str(getattr(res, "text", "") or "")
                    ok = bool(text.strip())
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    err = str(exc)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                row = {
                    "file": str(wav_path),
                    "run": i + 1,
                    "duration_sec": duration_sec,
                    "latency_ms": elapsed_ms,
                    "rtf": (elapsed_ms / 1000.0) / max(duration_sec, 1e-6),
                    "ok": ok,
                    "text_len": len(text),
                    "error": err,
                }
                rows.append(row)
                print(
                    f"[{i+1}/{args.repeat}] {wav_path.name} "
                    f"lat={elapsed_ms:.1f}ms rtf={row['rtf']:.3f} ok={ok}"
                )
                if not ok and err:
                    print(f"    err: {err[:200]}")
    finally:
        engine.shutdown()

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "base_url": "local://qwen3-asr-gguf",
        "audio_dir": str(args.audio_dir),
        "samples": len(files),
        "runs_per_sample": args.repeat,
        "warmup_per_sample": args.warmup,
        "total_runs": total,
        "success_runs": success,
        "success_rate": (success / total) if total else 0.0,
        "latency_ms_mean": statistics.mean(lat_ok) if lat_ok else 0.0,
        "latency_ms_p50": pct(lat_ok, 0.50),
        "latency_ms_p90": pct(lat_ok, 0.90),
        "rtf_mean": statistics.mean(rtf_ok) if rtf_ok else 0.0,
        "rtf_p50": pct(rtf_ok, 0.50),
        "rtf_p90": pct(rtf_ok, 0.90),
        "engine_init_ms": init_ms,
        "model_dir": str(model_dir),
        "qwen_repo_dir": str(qwen_repo_dir),
        "precision": args.precision,
        "timestamp": args.timestamp,
        "dml": args.dml,
        "vulkan": args.vulkan,
    }
    summary = round_summary(summary, ndigits=3)

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"summary": summary}
        if args.include_rows:
            payload["rows"] = rows
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
