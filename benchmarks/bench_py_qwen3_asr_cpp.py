from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
from py_qwen3_asr_cpp.model import Qwen3ASRModel


def collect_audio(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    exts = ("*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC", "*.m4a", "*.M4A")
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(audio_dir.rglob(ext)))
    return [p.resolve() for p in files]


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


def audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark py-qwen3-asr-cpp latency on fixed audio set (same summary style as bench_transcribe.py)."
    )
    ap.add_argument("--audio-dir", type=Path, required=True, help="Absolute directory path to auto-scan audio files")
    ap.add_argument("--asr-model", required=True, help="Model name or absolute local .gguf path")
    ap.add_argument("--threads", type=int, default=8, help="Inference thread count")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each sample N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup rounds per sample")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional result json path")
    ap.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-run detailed rows in output json",
    )
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute path, got: {args.audio_dir}")

    files = collect_audio(args.audio_dir)
    if not files:
        raise SystemExit(f"No audio files found from audio-dir: {args.audio_dir}")

    t0 = time.perf_counter()
    model = Qwen3ASRModel(asr_model=args.asr_model, n_threads=args.threads)
    init_ms = (time.perf_counter() - t0) * 1000.0

    rows: list[dict[str, object]] = []
    for audio_path in files:
        if not audio_path.exists():
            print(f"[skip] missing: {audio_path}")
            continue

        duration_sec = audio_duration_sec(audio_path)

        for _ in range(max(0, args.warmup)):
            _ = model.transcribe(str(audio_path))

        for i in range(args.repeat):
            t0 = time.perf_counter()
            ok = True
            text = ""
            lang = ""
            err = ""
            try:
                res = model.transcribe(str(audio_path))
                text = str(getattr(res, "text", "") or "")
                lang = str(getattr(res, "language", "") or "")
                ok = bool(text.strip())
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = str(exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (elapsed_ms / 1000.0) / max(duration_sec, 1e-6)
            row = {
                "file": str(audio_path),
                "run": i + 1,
                "duration_sec": duration_sec,
                "latency_ms": elapsed_ms,
                "rtf": rtf,
                "ok": ok,
                "text_len": len(text),
                "language": lang,
                "error": err,
            }
            rows.append(row)
            print(f"[{i+1}/{args.repeat}] {audio_path.name} lat={elapsed_ms:.1f}ms rtf={rtf:.3f} ok={ok}")
            if not ok and err:
                print(f"    err: {err[:200]}")

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "base_url": "local://py-qwen3-asr-cpp",
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
        "asr_model": args.asr_model,
        "threads": args.threads,
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
