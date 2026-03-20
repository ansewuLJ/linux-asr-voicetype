from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from .asr_backend import QwenEngine
from .audio import decode_base64_wav, load_wav_file
from .config import AppConfig
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
)
from .session_manager import SessionManager


def create_app(config: AppConfig) -> FastAPI:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    app = FastAPI(title="qwen-fcitx5-input", version="0.1.0")
    engine = QwenEngine(config)
    sessions = SessionManager(max_seconds=config.max_session_seconds)
    recorder = ArecordManager()
    hotwords: Hotwords = {}

    if config.hotwords_file:
        hotwords = parse_hotwords_text(Path(config.hotwords_file))

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
