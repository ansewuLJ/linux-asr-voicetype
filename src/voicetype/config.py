from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8789
    model: str = "Qwen/Qwen3-ASR-0.6B"
    device: str = "cuda:0"
    backend: str = "transformers"
    dtype: str | None = "bfloat16"
    default_language: str = ""
    max_session_seconds: int = 120
    hotwords_file: Path | None = None
    log_level: str = "info"
    use_mock_when_unavailable: bool = False
    hf_endpoint: str | None = "https://hf-mirror.com"
    hf_probe_timeout_sec: float = 2.5
    hf_hub_etag_timeout_sec: int = 3
    hf_hub_download_timeout_sec: int = 30
    max_inference_batch_size: int = 1
    global_hotkey_enabled: bool = False
    global_hotkey_key: str = "right_alt"


class RuntimeState(BaseModel):
    model: str
    backend: str
    device: str
    hotwords_count: int = 0


DEFAULT_CONFIG = AppConfig()


def default_runtime_config_path() -> Path:
    return Path.home() / ".config" / "voicetype" / "runtime.json"


def load_runtime_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or default_runtime_config_path()
    if not cfg_path.exists():
        return AppConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return AppConfig(**data)


def save_runtime_config(config: AppConfig, path: Path | None = None) -> Path:
    cfg_path = path or default_runtime_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cfg_path


def default_log_dir_path() -> Path:
    return Path.home() / ".local" / "state" / "voicetype"


def asr_log_file_path() -> Path:
    return default_log_dir_path() / "asr.log"


def ui_log_file_path() -> Path:
    return default_log_dir_path() / "ui.log"
