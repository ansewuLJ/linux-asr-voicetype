from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import librosa
import numpy as np
import openvino as ov
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


def audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def load_light_processor(bench_dir: Path):
    proc_py = bench_dir / "processor_numpy_qwenmini.py"
    if not proc_py.exists():
        raise SystemExit(f"processor file not found: {proc_py}")

    spec = importlib.util.spec_from_file_location("processor_numpy_qwenmini", proc_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load processor module from: {proc_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    LightProcessor = getattr(mod, "LightProcessor", None)
    if LightProcessor is None:
        raise SystemExit(f"LightProcessor not found in: {proc_py}")
    return LightProcessor


class MiniToolOVASR:
    def __init__(self, model_dir: Path, device: str, bench_dir: Path, max_tokens: int) -> None:
        LightProcessor = load_light_processor(bench_dir)
        self.max_tokens = max_tokens

        core = ov.Core()
        self.audio_enc = core.compile_model(str(model_dir / "audio_encoder_model.xml"), device)
        self.embedder = core.compile_model(str(model_dir / "thinker_embeddings_model.xml"), device)
        dec_comp = core.compile_model(str(model_dir / "decoder_model.xml"), device)
        self.dec_req = dec_comp.create_infer_request()

        self.proc = LightProcessor(model_dir)

    def transcribe(self, wav_path: Path, language: str | None) -> str:
        audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        audio = audio.astype(np.float32)

        mel, ids = self.proc.prepare(audio, language=language, context=None)

        ae = list(self.audio_enc({"mel": mel}).values())[0]
        te = list(self.embedder({"input_ids": ids}).values())[0]

        combined = te.copy()
        mask = ids[0] == self.proc.pad_id
        npad = int(mask.sum())
        naud = ae.shape[1]
        if npad != naud:
            m = min(npad, naud)
            combined[0, np.where(mask)[0][:m]] = ae[0, :m]
        else:
            combined[0, mask] = ae[0]

        length = combined.shape[1]
        pos = np.arange(length, dtype=np.int64)[np.newaxis, :]
        self.dec_req.reset_state()
        out = self.dec_req.infer({0: combined, "position_ids": pos})
        logits = list(out.values())[0]

        eos = self.proc.eos_id
        eot = self.proc.eot_id
        gen: list[int] = []
        nxt = int(np.argmax(logits[0, -1, :]))
        cur = length

        while nxt not in (eos, eot) and len(gen) < self.max_tokens:
            gen.append(nxt)
            emb = list(self.embedder({"input_ids": np.array([[nxt]], dtype=np.int64)}).values())[0]
            out = self.dec_req.infer({0: emb, "position_ids": np.array([[cur]], dtype=np.int64)})
            logits = list(out.values())[0]
            nxt = int(np.argmax(logits[0, -1, :]))
            cur += 1

        raw = self.proc.decode(gen)
        if "<asr_text>" in raw:
            raw = raw.split("<asr_text>", 1)[1]
        return raw.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch benchmark using QwenASRMiniTool-style OpenVINO inference")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--language", default=None)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--print-text", action="store_true")
    ap.add_argument("--print-text-max-len", type=int, default=300)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if not args.audio_dir.is_absolute():
        raise SystemExit(f"--audio-dir must be absolute: {args.audio_dir}")

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")
    for req in [
        "audio_encoder_model.xml",
        "thinker_embeddings_model.xml",
        "decoder_model.xml",
        "prompt_template.json",
        "mel_filters.npy",
    ]:
        if not (model_dir / req).exists():
            raise SystemExit(f"missing required model file: {model_dir / req}")

    files = collect_wavs(args.audio_dir)
    if not files:
        raise SystemExit(f"No wav files found from: {args.audio_dir}")

    bench_dir = Path(__file__).resolve().parent
    t0 = time.perf_counter()
    engine = MiniToolOVASR(model_dir=model_dir, device=args.device, bench_dir=bench_dir, max_tokens=args.max_tokens)
    init_ms = (time.perf_counter() - t0) * 1000.0

    rows: list[dict[str, object]] = []

    for wav_path in files:
        dur = audio_duration_sec(wav_path)

        for _ in range(max(0, args.warmup)):
            _ = engine.transcribe(wav_path, args.language)

        for i in range(max(1, args.repeat)):
            st = time.perf_counter()
            ok = True
            err = ""
            text = ""
            try:
                text = engine.transcribe(wav_path, args.language)
                ok = bool(text.strip())
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = str(exc)

            latency_ms = (time.perf_counter() - st) * 1000.0
            rtf = (latency_ms / 1000.0) / max(dur, 1e-6)

            rows.append(
                {
                    "file": str(wav_path),
                    "run": i + 1,
                    "duration_sec": dur,
                    "latency_ms": latency_ms,
                    "rtf": rtf,
                    "ok": ok,
                    "text": text,
                    "text_len": len(text),
                    "error": err,
                }
            )

            print(f"[{i+1}/{args.repeat}] {wav_path.name} lat={latency_ms:.1f}ms rtf={rtf:.3f} ok={ok}")
            if args.print_text:
                show = text.strip().replace("\n", " ")
                if len(show) > max(0, args.print_text_max_len):
                    show = show[: max(0, args.print_text_max_len)] + "..."
                print(f"    text: {show}")
            if not ok and err:
                print(f"    err: {err[:400]}")

    lat_ok = [float(r["latency_ms"]) for r in rows if r["ok"]]
    rtf_ok = [float(r["rtf"]) for r in rows if r["ok"]]
    success = sum(1 for r in rows if r["ok"])
    total = len(rows)

    summary = {
        "base_url": "local://openvino-minitool-style",
        "audio_dir": str(args.audio_dir),
        "samples": len(files),
        "runs_per_sample": args.repeat,
        "warmup_per_sample": args.warmup,
        "total_runs": total,
        "success_runs": success,
        "success_rate": (success / total) if total else 0.0,
        "latency_ms_mean": round(statistics.mean(lat_ok), 3) if lat_ok else 0.0,
        "latency_ms_p50": round(pct(lat_ok, 0.50), 3),
        "latency_ms_p90": round(pct(lat_ok, 0.90), 3),
        "rtf_mean": round(statistics.mean(rtf_ok), 3) if rtf_ok else 0.0,
        "rtf_p50": round(pct(rtf_ok, 0.50), 3),
        "rtf_p90": round(pct(rtf_ok, 0.90), 3),
        "engine_init_ms": round(init_ms, 3),
        "model_dir": str(model_dir),
        "device": args.device,
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "rows": rows}
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
