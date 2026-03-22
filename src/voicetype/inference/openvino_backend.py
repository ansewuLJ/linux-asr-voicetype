from __future__ import annotations

from pathlib import Path

import numpy as np

from ..asr_types import AsrResult
from ..config import AppConfig
from ..processor_numpy_minitool import LightProcessor


class OpenVinoBackend:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._audio_enc = None
        self._embedder = None
        self._dec_req = None
        self._proc = None

    @property
    def loaded(self) -> bool:
        return self._audio_enc is not None and self._embedder is not None and self._dec_req is not None and self._proc is not None

    def load(self) -> None:
        model_dir = Path(self._config.model).expanduser().resolve()
        required = [
            "audio_encoder_model.xml",
            "thinker_embeddings_model.xml",
            "decoder_model.xml",
            "prompt_template.json",
            "mel_filters.npy",
        ]
        for name in required:
            p = model_dir / name
            if not p.exists():
                raise FileNotFoundError(f"OpenVINO model file not found: {p}")

        import openvino as ov  # type: ignore

        core = ov.Core()
        device = str(self._config.device or "CPU")

        self._audio_enc = core.compile_model(str(model_dir / "audio_encoder_model.xml"), device)
        self._embedder = core.compile_model(str(model_dir / "thinker_embeddings_model.xml"), device)
        dec_comp = core.compile_model(str(model_dir / "decoder_model.xml"), device)
        self._dec_req = dec_comp.create_infer_request()
        self._proc = LightProcessor(model_dir)

    def transcribe(self, samples: np.ndarray, language: str | None, context: str | None) -> AsrResult:
        if not self.loaded:
            return AsrResult(text="", language="", success=False, error="backend is not loaded")

        audio = samples.astype(np.float32, copy=False).reshape(-1)

        try:
            assert self._proc is not None
            assert self._audio_enc is not None
            assert self._embedder is not None
            assert self._dec_req is not None

            mel, ids = self._proc.prepare(audio, language=language, context=context)

            ae = list(self._audio_enc({"mel": mel}).values())[0]
            te = list(self._embedder({"input_ids": ids}).values())[0]

            combined = te.copy()
            mask = ids[0] == self._proc.pad_id
            npad = int(mask.sum())
            naud = int(ae.shape[1])
            if npad != naud:
                m = min(npad, naud)
                combined[0, np.where(mask)[0][:m]] = ae[0, :m]
            else:
                combined[0, mask] = ae[0]

            length = int(combined.shape[1])
            pos = np.arange(length, dtype=np.int64)[np.newaxis, :]
            self._dec_req.reset_state()
            out = self._dec_req.infer({0: combined, "position_ids": pos})
            logits = list(out.values())[0]

            eos = self._proc.eos_id
            eot = self._proc.eot_id
            gen: list[int] = []
            nxt = int(np.argmax(logits[0, -1, :]))
            cur = length
            max_tokens = 300

            while nxt not in (eos, eot) and len(gen) < max_tokens:
                gen.append(nxt)
                emb = list(self._embedder({"input_ids": np.array([[nxt]], dtype=np.int64)}).values())[0]
                out = self._dec_req.infer({0: emb, "position_ids": np.array([[cur]], dtype=np.int64)})
                logits = list(out.values())[0]
                nxt = int(np.argmax(logits[0, -1, :]))
                cur += 1

            raw = self._proc.decode(gen)
            if "<asr_text>" in raw:
                raw = raw.split("<asr_text>", 1)[1]
            text = raw.strip()
            return AsrResult(text=text, language=language or "auto", success=bool(text))
        except Exception as exc:  # noqa: BLE001
            return AsrResult(text="", language="", success=False, error=str(exc))
