from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

from voicetype.audio import load_wav_file


INFER_RE = re.compile(r"Inference:\s*([0-9.]+)\s*ms")


def collect_wavs(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    files = sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.WAV"))
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


def parse_infer_ms(stderr_text: str) -> float | None:
    m = INFER_RE.search(stderr_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_one(bin_path: Path, model_dir: Path, wav_path: Path, threads: int) -> tuple[bool, float, float | None, str]:
    cmd = [
        str(bin_path),
        "-d",
        str(model_dir),
        "-i",
        str(wav_path),
        "-t",
        str(threads),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    infer_ms = parse_infer_ms(proc.stderr or "")
    ok = proc.returncode == 0
    err = ""
    if not ok:
        err = (proc.stderr or proc.stdout or f"returncode={proc.returncode}")[:300]
    return ok, elapsed_ms, infer_ms, err


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark qwen_asr CLI latency on fixed wav set (same style as bench_transcribe.py)."
    )
    ap.add_argument("--audio-dir", type=Path, required=True, help="Absolute directory path to auto-scan wav files")
    ap.add_argument("--qwen-bin", type=Path, required=True, help="Path to qwen_asr binary")
    ap.add_argument("--model-dir", type=Path, required=True, help="Path to qwen3-asr model dir")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each sample N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup rounds per sample")
    ap.add_argument("--threads", type=int, default=16, help="qwen_asr -t thread count")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional result json path")
    ap.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-run detailed rows in output json",
    )
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute path, got: {args.audio_dir}")
    if not args.qwen_bin.exists():
        raise SystemExit(f"--qwen-bin not found: {args.qwen_bin}")
    if not args.model_dir.exists():
        raise SystemExit(f"--model-dir not found: {args.model_dir}")

    files = collect_wavs(args.audio_dir)
    if not files:
        raise SystemExit(f"No wav files found from audio-dir: {args.audio_dir}")

    rows: list[dict[str, object]] = []

    for wav_path in files:
        if not wav_path.exists():
            print(f"[skip] missing: {wav_path}")
            continue

        samples = load_wav_file(wav_path)
        duration_sec = float(samples.shape[0]) / 16000.0

        for _ in range(max(0, args.warmup)):
            run_one(args.qwen_bin, args.model_dir, wav_path, args.threads)

        for i in range(args.repeat):
            ok, elapsed_ms, infer_ms, err = run_one(args.qwen_bin, args.model_dir, wav_path, args.threads)
            row = {
                "file": str(wav_path),
                "run": i + 1,
                "duration_sec": duration_sec,
                "latency_ms": elapsed_ms,
                "rtf": (elapsed_ms / 1000.0) / max(duration_sec, 1e-6),
                "inference_ms": infer_ms,
                "inference_rtf": ((infer_ms / 1000.0) / max(duration_sec, 1e-6)) if infer_ms is not None else None,
                "ok": ok,
                "error": err,
            }
            rows.append(row)
            infer_show = f"{infer_ms:.1f}ms" if infer_ms is not None else "n/a"
            print(
                f"[{i+1}/{args.repeat}] {wav_path.name} "
                f"lat={elapsed_ms:.1f}ms infer={infer_show} rtf={row['rtf']:.3f} ok={ok}"
            )

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    infer_ok = [float(r["inference_ms"]) for r in rows if r["ok"] and r["inference_ms"] is not None]
    infer_rtf_ok = [float(r["inference_rtf"]) for r in rows if r["ok"] and r["inference_rtf"] is not None]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "qwen_bin": str(args.qwen_bin),
        "model_dir": str(args.model_dir),
        "audio_dir": str(args.audio_dir),
        "samples": len(files),
        "runs_per_sample": args.repeat,
        "warmup_per_sample": args.warmup,
        "threads": args.threads,
        "total_runs": total,
        "success_runs": success,
        "success_rate": (success / total) if total else 0.0,
        "latency_ms_mean": statistics.mean(lat_ok) if lat_ok else 0.0,
        "latency_ms_p50": pct(lat_ok, 0.50),
        "latency_ms_p90": pct(lat_ok, 0.90),
        "rtf_mean": statistics.mean(rtf_ok) if rtf_ok else 0.0,
        "rtf_p50": pct(rtf_ok, 0.50),
        "rtf_p90": pct(rtf_ok, 0.90),
        "inference_ms_mean": statistics.mean(infer_ok) if infer_ok else 0.0,
        "inference_ms_p50": pct(infer_ok, 0.50),
        "inference_ms_p90": pct(infer_ok, 0.90),
        "inference_rtf_mean": statistics.mean(infer_rtf_ok) if infer_rtf_ok else 0.0,
        "inference_rtf_p50": pct(infer_rtf_ok, 0.50),
        "inference_rtf_p90": pct(infer_rtf_ok, 0.90),
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

