from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer
import uvicorn

from .audio import encode_pcm16_wav_base64
from .config import AppConfig, default_runtime_config_path, load_runtime_config, save_runtime_config
from .controller_ui import create_controller_app
from .fcitx_bridge import BridgeConfig, run_fcitx_x11_bridge
from .model_downloader import download_hf_repo_snapshot
from .server import run_server

app = typer.Typer(help="voicetype command line")
model_app = typer.Typer(help="Model management commands")
app.add_typer(model_app, name="model")


def _config_from_options(
    host: str,
    port: int,
    model: str,
    device: str,
    backend: str,
    dtype: str | None,
    hotwords_file: Path | None,
    hf_endpoint: str | None,
    hf_probe_timeout_sec: float,
    max_inference_batch_size: int,
    use_mock_when_unavailable: bool,
) -> AppConfig:
    return AppConfig(
        host=host,
        port=port,
        model=model,
        device=device,
        backend=backend,
        dtype=dtype,
        hotwords_file=hotwords_file,
        hf_endpoint=hf_endpoint,
        hf_probe_timeout_sec=hf_probe_timeout_sec,
        max_inference_batch_size=max_inference_batch_size,
        use_mock_when_unavailable=use_mock_when_unavailable,
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    model: str = "Qwen/Qwen3-ASR-0.6B",
    device: str = "cuda:0",
    backend: str = "transformers",
    dtype: str | None = "bfloat16",
    hotwords_file: Path | None = None,
    hf_endpoint: str | None = "https://hf-mirror.com",
    hf_probe_timeout_sec: float = 2.5,
    max_inference_batch_size: int = 1,
    allow_mock: bool = False,
) -> None:
    """Start local ASR service."""
    config = _config_from_options(
        host,
        port,
        model,
        device,
        backend,
        dtype,
        hotwords_file,
        hf_endpoint,
        hf_probe_timeout_sec,
        max_inference_batch_size,
        allow_mock,
    )
    run_server(config)


@app.command("serve-from-config")
def serve_from_config(
    config_file: Path = typer.Option(
        default_runtime_config_path(),
        help="Runtime config JSON path",
    ),
) -> None:
    """Start ASR service from persisted runtime config."""
    config = load_runtime_config(config_file)
    run_server(config)


@app.command("ui")
def ui(
    host: str = "127.0.0.1",
    port: int = 8790,
    config_file: Path = typer.Option(
        default_runtime_config_path(),
        help="Runtime config JSON path",
    ),
) -> None:
    """Start lightweight controller UI (no model loading)."""
    if not config_file.exists():
        save_runtime_config(AppConfig(), config_file)
    app_ui = create_controller_app(config_file=config_file)
    uvicorn.run(app_ui, host=host, port=port, log_level="info")


@app.command("fcitx-bridge")
def fcitx_bridge(
    base_url: str = "http://127.0.0.1:8787",
    hold_key: str = "right_alt",
    toggle_key: str = "left_alt+z",
    sample_rate: int = 16000,
    type_delay_ms: int = 1,
) -> None:
    """Start X11 hotkey bridge (works on Fcitx4/Fcitx5 environments)."""
    cfg = BridgeConfig(
        base_url=base_url,
        hold_key=hold_key,
        toggle_key=toggle_key,
        sample_rate=sample_rate,
        type_delay_ms=type_delay_ms,
    )
    run_fcitx_x11_bridge(cfg)


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


@model_app.command("download")
def model_download(
    repo_id: str = typer.Argument(..., help="HF repo id, e.g. Qwen/Qwen3-ASR-0.6B"),
    local_dir: Path = typer.Option(..., "--local-dir", help="Target local directory"),
    revision: str | None = typer.Option(
        None,
        "--revision",
        help="Branch/tag/commit. Default is repo default branch.",
    ),
    hf_endpoint: str | None = typer.Option(
        "https://hf-mirror.com",
        "--hf-endpoint",
        help="HF endpoint or mirror. Pass empty string to use official endpoint.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="HF access token (optional for public repos).",
    ),
    force_download: bool = typer.Option(
        False,
        "--force",
        help="Force redownload even if local cache exists.",
    ),
    max_workers: int = typer.Option(
        8,
        "--max-workers",
        min=1,
        help="Parallel workers used by downloader.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Optional Hugging Face cache directory.",
    ),
    keep_hf_metadata: bool = typer.Option(
        False,
        "--keep-hf-metadata",
        help="Keep local .cache/huggingface metadata folder inside --local-dir.",
    ),
) -> None:
    """Download full model repo as real files (no symlinks)."""
    endpoint = (hf_endpoint or "").strip() or None
    out_dir = download_hf_repo_snapshot(
        repo_id=repo_id,
        local_dir=local_dir.expanduser().resolve(),
        endpoint=endpoint,
        revision=revision,
        token=token,
        force_download=force_download,
        max_workers=max_workers,
        cache_dir=cache_dir.expanduser().resolve() if cache_dir else None,
        strip_hf_metadata_dir=not keep_hf_metadata,
    )
    typer.echo(f"Download completed: {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
