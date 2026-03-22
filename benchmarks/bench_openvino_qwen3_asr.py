from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import soundfile as sf


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


def audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def load_qwen3_helper(helper_py: Path | None):
    if helper_py is None:
        try:
            from qwen_3_asr_helper import OVQwen3ASRModel, convert_qwen3_asr_model  # type: ignore

            return OVQwen3ASRModel, convert_qwen3_asr_model
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "Failed to import qwen_3_asr_helper from PYTHONPATH. "
                "Use --helper-py to point to qwen_3_asr_helper.py explicitly. "
                f"error={exc}"
            )

    hp = helper_py.expanduser().resolve()
    if not hp.exists():
        raise SystemExit(f"--helper-py not found: {hp}")

    spec = importlib.util.spec_from_file_location("qwen_3_asr_helper", hp)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load helper module from: {hp}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    OVQwen3ASRModel = getattr(mod, "OVQwen3ASRModel", None)
    convert_qwen3_asr_model = getattr(mod, "convert_qwen3_asr_model", None)
    if OVQwen3ASRModel is None or convert_qwen3_asr_model is None:
        raise SystemExit(
            f"{hp} does not expose required symbols: OVQwen3ASRModel / convert_qwen3_asr_model"
        )
    return OVQwen3ASRModel, convert_qwen3_asr_model


def parse_text_and_lang(result_obj: object) -> tuple[str, str]:
    text = ""
    language = ""

    if isinstance(result_obj, list) and result_obj:
        result_obj = result_obj[0]

    if isinstance(result_obj, dict):
        text = str(result_obj.get("text", "") or "")
        language = str(result_obj.get("language", "") or "")
        return text, language

    text = str(getattr(result_obj, "text", "") or "")
    language = str(getattr(result_obj, "language", "") or "")
    return text, language


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark local OpenVINO Qwen3-ASR latency on fixed wav set "
            "(same summary style as bench_transcribe.py)."
        )
    )
    ap.add_argument("--audio-dir", type=Path, required=True, help="Absolute directory path to auto-scan wav files")
    ap.add_argument("--model-dir", type=Path, required=True, help="Converted OpenVINO model dir")
    ap.add_argument("--device", default="CPU", help='OpenVINO device, e.g. "CPU" / "GPU" / "NPU"')
    ap.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=32,
        help="OVQwen3ASRModel.from_pretrained max_inference_batch_size",
    )
    ap.add_argument("--language", default=None, help="Optional language hint, e.g. zh/en")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each sample N times")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup rounds per sample")
    ap.add_argument(
        "--helper-py",
        type=Path,
        default=None,
        help="Optional absolute path to qwen_3_asr_helper.py (if not importable from env)",
    )
    ap.add_argument("--output-json", type=Path, default=None, help="Optional result json path")
    ap.add_argument(
        "--print-text",
        action="store_true",
        help="Print recognized text for each run",
    )
    ap.add_argument(
        "--print-text-max-len",
        type=int,
        default=200,
        help="Max text length to print when --print-text is enabled",
    )
    ap.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-run detailed rows in output json",
    )
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute path, got: {args.audio_dir}")

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_absolute():
        raise SystemExit(f"--model-dir must be absolute path, got: {model_dir}")
    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")

    files = collect_wavs(args.audio_dir)
    if not files:
        raise SystemExit(f"No wav files found from audio-dir: {args.audio_dir}")

    OVQwen3ASRModel, _ = load_qwen3_helper(args.helper_py)

    t_init0 = time.perf_counter()
    ov_model = OVQwen3ASRModel.from_pretrained(
        model_dir=str(model_dir),
        device=args.device,
        max_inference_batch_size=args.max_inference_batch_size,
    )
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    rows: list[dict[str, object]] = []
    for wav_path in files:
        if not wav_path.exists():
            print(f"[skip] missing: {wav_path}")
            continue

        duration_sec = audio_duration_sec(wav_path)

        for _ in range(max(0, args.warmup)):
            _ = ov_model.transcribe(audio=str(wav_path), language=args.language)

        for i in range(args.repeat):
            t0 = time.perf_counter()
            ok = True
            text = ""
            language = ""
            err = ""
            try:
                result_obj = ov_model.transcribe(audio=str(wav_path), language=args.language)
                text, language = parse_text_and_lang(result_obj)
                ok = bool(text.strip())
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = str(exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (elapsed_ms / 1000.0) / max(duration_sec, 1e-6)

            row = {
                "file": str(wav_path),
                "run": i + 1,
                "duration_sec": duration_sec,
                "latency_ms": elapsed_ms,
                "rtf": rtf,
                "ok": ok,
                "text_len": len(text),
                "language": language,
                "error": err,
            }
            rows.append(row)
            print(f"[{i+1}/{args.repeat}] {wav_path.name} lat={elapsed_ms:.1f}ms rtf={rtf:.3f} ok={ok}")
            if args.print_text:
                text_show = text.strip().replace("\n", " ")
                if len(text_show) > max(0, args.print_text_max_len):
                    text_show = text_show[: max(0, args.print_text_max_len)] + "..."
                print(f"    text: {text_show}")
            if not ok and err:
                print(f"    err: {err[:200]}")

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "base_url": "local://openvino-qwen3-asr",
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
        "device": args.device,
        "max_inference_batch_size": args.max_inference_batch_size,
        "helper_py": str(args.helper_py.expanduser().resolve()) if args.helper_py else None,
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
