from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from .config import AppConfig
from .hotwords import Hotwords

LOGGER = logging.getLogger(__name__)


LANGUAGE_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "yue": "Cantonese",
    "en": "English",
}


@dataclass(slots=True)
class AsrResult:
    text: str
    language: str
    success: bool
    error: str | None = None


class QwenEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None
        self._backend = "mock"

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend_name(self) -> str:
        return self._backend

    def load(self) -> None:
        self._prepare_hf_env()

        if not self._model_repo_reachable():
            msg = (
                "Model repository is unreachable. "
                "Set hf_endpoint/HF_ENDPOINT to mirror or use local model path."
            )
            if self._config.use_mock_when_unavailable:
                LOGGER.warning("%s Fallback to mock engine.", msg)
                self._model = None
                self._backend = "mock"
                return
            raise RuntimeError(msg)

        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore

            model_kwargs: dict[str, object] = {
                "device_map": self._config.device,
                "max_inference_batch_size": self._config.max_inference_batch_size,
            }
            dtype_obj = self._resolve_torch_dtype(self._config.dtype)
            if dtype_obj is not None:
                model_kwargs["dtype"] = dtype_obj

            self._model = Qwen3ASRModel.from_pretrained(
                self._config.model,
                **model_kwargs,
            )
            self._backend = self._config.backend
            LOGGER.info("Loaded model: %s (dtype=%s)", self._config.model, self._config.dtype)
        except Exception as exc:  # noqa: BLE001
            if self._config.use_mock_when_unavailable:
                LOGGER.warning("Model load failed, fallback to mock engine: %s", exc)
                self._model = None
                self._backend = "mock"
                return
            raise

    def _resolve_torch_dtype(self, dtype_name: str | None):
        if not dtype_name:
            return None
        try:
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            LOGGER.warning("torch unavailable, ignore dtype=%s", dtype_name)
            return None
        lowered = str(dtype_name).strip().lower()
        mapping = {
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float16": "float16",
            "fp16": "float16",
            "float32": "float32",
            "fp32": "float32",
        }
        attr = mapping.get(lowered)
        if not attr or not hasattr(torch, attr):
            LOGGER.warning("unsupported dtype=%s, ignore it", dtype_name)
            return None
        return getattr(torch, attr)

    def _prepare_hf_env(self) -> None:
        if self._config.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self._config.hf_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
        os.environ.setdefault(
            "HF_HUB_ETAG_TIMEOUT", str(self._config.hf_hub_etag_timeout_sec)
        )
        os.environ.setdefault(
            "HF_HUB_DOWNLOAD_TIMEOUT", str(self._config.hf_hub_download_timeout_sec)
        )

    def _model_repo_reachable(self) -> bool:
        model = self._config.model
        if Path(model).exists():
            return True

        endpoint = (
            self._config.hf_endpoint
            or os.environ.get("HF_ENDPOINT")
            or "https://huggingface.co"
        ).rstrip("/")
        url = f"{endpoint}/{model}/resolve/main/config.json"
        try:
            with httpx.Client(timeout=self._config.hf_probe_timeout_sec) as client:
                resp = client.get(url, follow_redirects=True)
                return resp.status_code < 500
        except Exception:
            return False

    def transcribe(
        self,
        samples: np.ndarray,
        language: str | None,
        hotwords: Hotwords,
    ) -> AsrResult:
        if samples.size == 0:
            return AsrResult(text="", language="", success=False, error="empty audio")

        normalized_lang = self._normalize_language(language)
        context = self._hotwords_to_context(hotwords)

        if self._model is None:
            seconds = samples.size / 16000.0
            return AsrResult(
                text=f"[mock:{self._config.model}] duration={seconds:.2f}s context={context}",
                language=normalized_lang or "auto",
                success=True,
            )

        try:
            kwargs = {"audio": (samples, 16000)}
            if normalized_lang:
                kwargs["language"] = normalized_lang
            if context:
                kwargs["context"] = context
            result = self._model.transcribe(**kwargs)
            first = result[0]
            text = first.text if hasattr(first, "text") else first.get("text", "")
            lang = first.language if hasattr(first, "language") else first.get("language", "auto")
            return AsrResult(text=text, language=lang, success=True)
        except Exception as exc:  # noqa: BLE001
            return AsrResult(text="", language="", success=False, error=str(exc))

    def _normalize_language(self, language: str | None) -> str | None:
        if not language:
            return None
        lowered = language.strip().lower()
        return LANGUAGE_MAP.get(lowered, language)

    def _hotwords_to_context(self, hotwords: Hotwords) -> str:
        flat: list[str] = []
        for words in hotwords.values():
            flat.extend(words.keys())
        return ", ".join(flat)
