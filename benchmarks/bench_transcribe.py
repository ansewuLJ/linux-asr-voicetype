from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

from voicetype.audio import encode_pcm16_wav_base64, load_wav_file


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark /v1/transcribe latency on fixed wav set.")
    ap.add_argument("--audio-dir", type=Path, required=True, help="Absolute directory path to auto-scan wav files")
    ap.add_argument("--base-url", default="http://127.0.0.1:8789", help="ASR server base URL")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each sample N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup rounds per sample")
    ap.add_argument("--language", default=None, help="Optional language hint, e.g. zh/en")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional result json path")
    ap.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-run detailed rows in output json",
    )
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute path, got: {args.audio_dir}")
    files = collect_wavs(args.audio_dir)
    if not files:
        raise SystemExit(f"No wav files found from audio-dir: {args.audio_dir}")

    rows: list[dict[str, object]] = []

    with httpx.Client(timeout=120.0) as client:
        for wav_path in files:
            if not wav_path.exists():
                print(f"[skip] missing: {wav_path}")
                continue

            samples = load_wav_file(wav_path)
            duration_sec = float(samples.shape[0]) / 16000.0
            audio_b64 = encode_pcm16_wav_base64(samples, sample_rate=16000)
            payload = {"audio_base64": audio_b64, "language": args.language}

            for _ in range(max(0, args.warmup)):
                r = client.post(f"{args.base_url}/v1/transcribe", json=payload)
                r.raise_for_status()

            for i in range(args.repeat):
                t0 = time.perf_counter()
                r = client.post(f"{args.base_url}/v1/transcribe", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                ok = r.status_code == 200
                text = ""
                err = ""
                if ok:
                    data = r.json()
                    ok = bool(data.get("success", True))
                    text = str(data.get("text", ""))
                    err = str(data.get("error", "")) if data.get("error") else ""
                else:
                    err = f"http {r.status_code}: {r.text[:120]}"

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

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "base_url": args.base_url,
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
