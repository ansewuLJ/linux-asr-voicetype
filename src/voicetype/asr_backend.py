from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .asr_types import AsrResult, hotwords_to_context, normalize_language
from .config import AppConfig
from .hotwords import Hotwords
from .inference import OpenVinoBackend, TransformersBackend

LOGGER = logging.getLogger(__name__)


class _BackendProtocol(Protocol):
    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def transcribe(self, samples: np.ndarray, language: str | None, context: str | None) -> AsrResult: ...


class AsrEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._backend_impl: _BackendProtocol | None = None
        self._backend = "mock"

    @property
    def model_loaded(self) -> bool:
        return self._backend_impl is not None and bool(self._backend_impl.loaded)

    @property
    def backend_name(self) -> str:
        return self._backend

    def load(self) -> None:
        backend = str(self._config.backend or "transformers").strip().lower()

        if backend in {"openvino", "openvino_minitool", "openvino-minitool"}:
            impl: _BackendProtocol = OpenVinoBackend(self._config)
            backend_name = "openvino"
        else:
            impl = TransformersBackend(self._config)
            backend_name = "transformers"

        try:
            impl.load()
            self._backend_impl = impl
            self._backend = backend_name
            LOGGER.info("Loaded ASR backend=%s model=%s device=%s", backend_name, self._config.model, self._config.device)
        except Exception as exc:  # noqa: BLE001
            if self._config.use_mock_when_unavailable:
                LOGGER.warning("ASR backend load failed, fallback to mock backend=%s err=%s", backend_name, exc)
                self._backend_impl = None
                self._backend = "mock"
                return
            raise

    def transcribe(self, samples: np.ndarray, language: str | None, hotwords: Hotwords) -> AsrResult:
        if samples.size == 0:
            return AsrResult(text="", language="", success=False, error="empty audio")

        normalized_lang = normalize_language(language)
        context = hotwords_to_context(hotwords)

        if self._backend_impl is None:
            seconds = samples.size / 16000.0
            return AsrResult(
                text=f"[mock:{self._config.model}] duration={seconds:.2f}s context={context}",
                language=normalized_lang or "auto",
                success=True,
            )

        return self._backend_impl.transcribe(samples=samples, language=normalized_lang, context=context)


# Backward-compatible alias for existing imports.
QwenEngine = AsrEngine
