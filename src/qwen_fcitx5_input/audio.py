from __future__ import annotations

import base64
import io
from pathlib import Path
import wave

import numpy as np


def decode_base64_wav(audio_base64: str, expected_sample_rate: int = 16000) -> np.ndarray:
    raw = base64.b64decode(audio_base64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.getnframes()
        data = wf.readframes(frames)

    pcm = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if sample_rate == expected_sample_rate:
        pass
    else:
        raise ValueError(
            f"Unsupported sample rate: {sample_rate}, expected {expected_sample_rate}"
        )

    return pcm.astype(np.float32) / 32768.0


def encode_pcm16_wav_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    pcm16 = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_wav_file(path: str | Path, expected_sample_rate: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.getnframes()
        data = wf.readframes(frames)

    pcm = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Unsupported sample rate: {sample_rate}, expected {expected_sample_rate}"
        )

    return pcm.astype(np.float32) / 32768.0
