from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer

from .audio import encode_pcm16_wav_base64
from .config import AppConfig
from .server import run_server

app = typer.Typer(help="voicetype command line")


def _config_from_options(
    host: str,
    port: int,
    model: str,
    device: str,
    backend: str,
    hotwords_file: Path | None,
    hf_endpoint: str | None,
    hf_probe_timeout_sec: float,
    use_mock_when_unavailable: bool,
) -> AppConfig:
    return AppConfig(
        host=host,
        port=port,
        model=model,
        device=device,
        backend=backend,
        hotwords_file=hotwords_file,
        hf_endpoint=hf_endpoint,
        hf_probe_timeout_sec=hf_probe_timeout_sec,
        use_mock_when_unavailable=use_mock_when_unavailable,
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    model: str = "Qwen/Qwen3-ASR-0.6B",
    device: str = "cuda:0",
    backend: str = "transformers",
    hotwords_file: Path | None = None,
    hf_endpoint: str | None = None,
    hf_probe_timeout_sec: float = 2.5,
    allow_mock: bool = False,
) -> None:
    """Start local ASR service."""
    config = _config_from_options(
        host,
        port,
        model,
        device,
        backend,
        hotwords_file,
        hf_endpoint,
        hf_probe_timeout_sec,
        allow_mock,
    )
    run_server(config)


@app.command()
def health(base_url: str = "http://127.0.0.1:8787") -> None:
    """Check server health."""
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{base_url}/health")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), ensure_ascii=False, indent=2))


@app.command()
def mock_audio(
    seconds: float = 1.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
) -> None:
    """Generate test WAV base64 to stdout."""
    import numpy as np

    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    signal = 0.1 * np.sin(2.0 * np.pi * freq * t)
    typer.echo(encode_pcm16_wav_base64(signal.astype(np.float32), sample_rate=sample_rate))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
