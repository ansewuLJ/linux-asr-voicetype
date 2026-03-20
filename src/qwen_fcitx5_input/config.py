from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    model: str = "Qwen/Qwen3-ASR-0.6B"
    device: str = "cuda:0"
    backend: str = "transformers"
    default_language: str = "zh"
    max_session_seconds: int = 120
    hotwords_file: Path | None = None
    log_level: str = "info"
    use_mock_when_unavailable: bool = False
    hf_endpoint: str | None = None
    hf_probe_timeout_sec: float = 2.5
    hf_hub_etag_timeout_sec: int = 3
    hf_hub_download_timeout_sec: int = 30


class RuntimeState(BaseModel):
    model: str
    backend: str
    device: str
    hotwords_count: int = 0


DEFAULT_CONFIG = AppConfig()
