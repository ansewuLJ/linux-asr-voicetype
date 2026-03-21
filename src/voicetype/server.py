from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from .asr_backend import QwenEngine
from .audio import decode_base64_wav, load_wav_file
from .config import AppConfig, asr_log_file_path
from .hotwords import Hotwords, count_hotwords, merge_hotwords, parse_hotwords_text
from .recorder import ArecordManager
from .schema import (
    DirectTranscribeRequest,
    FinishSessionRequest,
    HealthResponse,
    LoadHotwordsRequest,
    PushChunkRequest,
    PushChunkResponse,
    RecordingStartRequest,
    RecordingStartResponse,
    RecordingStatusResponse,
    StartSessionRequest,
    StartSessionResponse,
    TranscribeResponse,
    UiConfigUpdateRequest,
    UiHotwordsFileRequest,
    UiHotwordsTextRequest,
)
from .session_manager import SessionManager


def _setup_asr_file_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    log_file = asr_log_file_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    target = str(log_file)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            h.setLevel(level)
            return

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)


def create_app(config: AppConfig) -> FastAPI:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    _setup_asr_file_logging(config.log_level)

    app = FastAPI(title="VoiceType", version="0.1.0")
    engine = QwenEngine(config)
    sessions = SessionManager(max_seconds=config.max_session_seconds)
    recorder = ArecordManager()
    hotwords: Hotwords = {}
    allowed_models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]

    if config.hotwords_file:
        hotwords = parse_hotwords_text(Path(config.hotwords_file))

    def export_hotwords_text(hw: Hotwords) -> str:
        lines: list[str] = []
        for _, words in hw.items():
            for word, weight in words.items():
                lines.append(f"{word} {weight:g}")
        return "\n".join(lines)

    @app.on_event("startup")
    def startup() -> None:
        engine.load()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=engine.model_loaded,
            model=config.model,
            backend=engine.backend_name,
            device=config.device,
            hotwords_count=count_hotwords(hotwords),
        )

    @app.post("/v1/session/start", response_model=StartSessionResponse)
    def start_session(req: StartSessionRequest) -> StartSessionResponse:
        session = sessions.start(language=req.language or config.default_language)
        return StartSessionResponse(session_id=session.session_id)

    @app.post("/v1/session/{session_id}/chunk", response_model=PushChunkResponse)
    def push_chunk(session_id: str, req: PushChunkRequest) -> PushChunkResponse:
        try:
            samples = decode_base64_wav(req.audio_base64, expected_sample_rate=req.sample_rate)
            accepted, total = sessions.append(session_id, samples)
            return PushChunkResponse(accepted_samples=accepted, total_samples=total)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/session/{session_id}/finish", response_model=TranscribeResponse)
    def finish_session(session_id: str, req: FinishSessionRequest) -> TranscribeResponse:
        try:
            session = sessions.get(session_id)
            samples = sessions.finish(session_id)
            result = engine.transcribe(
                samples=samples,
                language=req.language or session.language,
                hotwords=hotwords,
            )
            return TranscribeResponse(
                text=result.text,
                language=result.language,
                success=result.success,
                error=result.error,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/transcribe", response_model=TranscribeResponse)
    def transcribe(req: DirectTranscribeRequest) -> TranscribeResponse:
        try:
            samples = decode_base64_wav(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = engine.transcribe(samples=samples, language=req.language, hotwords=hotwords)
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            success=result.success,
            error=result.error,
        )

    @app.post("/v1/hotwords/load")
    def load_hotwords(req: LoadHotwordsRequest) -> dict[str, object]:
        nonlocal hotwords
        hotwords = merge_hotwords(hotwords, req.hotwords) if req.merge else req.hotwords
        return {
            "success": True,
            "hotwords_count": count_hotwords(hotwords),
            "categories": list(hotwords.keys()),
        }

    @app.delete("/v1/hotwords")
    def clear_hotwords() -> dict[str, object]:
        nonlocal hotwords
        hotwords = {}
        return {"success": True}

    @app.get("/v1/ui/state")
    def ui_state() -> dict[str, object]:
        return {
            "config": config.model_dump(),
            "hotwords": hotwords,
            "hotwords_count": count_hotwords(hotwords),
            "model_options": allowed_models,
        }

    @app.post("/v1/ui/config")
    def ui_update_config(req: UiConfigUpdateRequest) -> dict[str, object]:
        nonlocal config
        nonlocal engine

        if req.max_inference_batch_size < 1:
            raise HTTPException(status_code=400, detail="max_inference_batch_size must be >= 1")
        if req.model not in allowed_models:
            raise HTTPException(status_code=400, detail=f"unsupported model: {req.model}")

        new_config = config.model_copy(
            update={
                "model": req.model,
                "device": req.device,
                "default_language": req.default_language,
                "max_inference_batch_size": req.max_inference_batch_size,
                "hf_endpoint": req.hf_endpoint,
            }
        )
        new_engine = QwenEngine(new_config)
        try:
            new_engine.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"reload model failed: {exc}") from exc

        config = new_config
        engine = new_engine
        return {"success": True, "config": config.model_dump()}

    @app.post("/v1/ui/hotwords/file")
    def ui_load_hotwords_file(req: UiHotwordsFileRequest) -> dict[str, object]:
        nonlocal hotwords
        path = Path(req.path).expanduser()
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")
        parsed = parse_hotwords_text(path)
        hotwords = parsed
        return {
            "success": True,
            "hotwords_count": count_hotwords(hotwords),
            "categories": list(hotwords.keys()),
            "effective_text": export_hotwords_text(hotwords),
        }

    @app.post("/v1/ui/hotwords/text")
    def ui_load_hotwords_text(req: UiHotwordsTextRequest) -> dict[str, object]:
        nonlocal hotwords
        parsed: Hotwords = {"custom": {}}
        for raw in req.text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            word = parts[0]
            weight = 30.0
            if len(parts) > 1:
                try:
                    weight = float(parts[1])
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"invalid weight in line: {line}",
                    ) from exc
            parsed["custom"][word] = weight

        hotwords = parsed
        return {
            "success": True,
            "hotwords_count": count_hotwords(hotwords),
            "categories": list(hotwords.keys()),
            "effective_text": export_hotwords_text(hotwords),
        }

    @app.post("/v1/ui/service/restart")
    def ui_restart_service() -> dict[str, object]:
        cmd = ["systemctl", "--user", "restart", "voicetype.service"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"restart failed: {exc}") from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"restart failed (code={proc.returncode}): {stderr or stdout or 'unknown error'}",
            )
        return {"success": True}

    @app.get("/v1/recording/status", response_model=RecordingStatusResponse)
    def recording_status() -> RecordingStatusResponse:
        return RecordingStatusResponse(recording=recorder.is_recording())

    @app.post("/v1/recording/start", response_model=RecordingStartResponse)
    def recording_start(req: RecordingStartRequest) -> RecordingStartResponse:
        try:
            recorder.start(
                language=req.language or config.default_language,
                sample_rate=req.sample_rate,
                device=req.device,
            )
            return RecordingStartResponse(success=True, recording=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/recording/stop", response_model=TranscribeResponse)
    def recording_stop(req: FinishSessionRequest) -> TranscribeResponse:
        try:
            wav_path, default_lang, sample_rate = recorder.stop()
            try:
                samples = load_wav_file(wav_path, expected_sample_rate=sample_rate)
            finally:
                wav_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = engine.transcribe(
            samples=samples,
            language=req.language or default_lang,
            hotwords=hotwords,
        )
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            success=result.success,
            error=result.error,
        )

    return app


def run_server(config: AppConfig) -> None:
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)
