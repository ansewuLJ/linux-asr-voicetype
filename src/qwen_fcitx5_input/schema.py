from __future__ import annotations

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    language: str | None = None


class StartSessionResponse(BaseModel):
    session_id: str


class PushChunkRequest(BaseModel):
    audio_base64: str = Field(description="Base64-encoded mono PCM16 WAV")
    sample_rate: int = 16000


class PushChunkResponse(BaseModel):
    accepted_samples: int
    total_samples: int


class FinishSessionRequest(BaseModel):
    language: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str
    success: bool
    error: str | None = None


class DirectTranscribeRequest(BaseModel):
    audio_base64: str
    language: str | None = None


class LoadHotwordsRequest(BaseModel):
    hotwords: dict[str, dict[str, float]]
    merge: bool = False


class RecordingStartRequest(BaseModel):
    language: str | None = None
    device: str | None = None
    sample_rate: int = 16000


class RecordingStartResponse(BaseModel):
    success: bool
    recording: bool
    error: str | None = None


class RecordingStatusResponse(BaseModel):
    recording: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    backend: str
    device: str
    hotwords_count: int
