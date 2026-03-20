from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Session:
    session_id: str
    language: str | None
    created_at: float
    chunks: list[np.ndarray] = field(default_factory=list)

    def append(self, chunk: np.ndarray) -> int:
        self.chunks.append(chunk)
        return int(sum(x.size for x in self.chunks))

    def joined(self) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.chunks)


class SessionManager:
    def __init__(self, max_seconds: int = 120, sample_rate: int = 16000) -> None:
        self._max_samples = max_seconds * sample_rate
        self._sessions: dict[str, Session] = {}

    def start(self, language: str | None = None) -> Session:
        session = Session(
            session_id=uuid.uuid4().hex,
            language=language,
            created_at=time.time(),
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError("session not found")
        return self._sessions[session_id]

    def append(self, session_id: str, chunk: np.ndarray) -> tuple[int, int]:
        session = self.get(session_id)
        total = session.append(chunk)
        if total > self._max_samples:
            raise ValueError("session audio exceeds limit")
        return chunk.size, total

    def finish(self, session_id: str) -> np.ndarray:
        session = self.get(session_id)
        audio = session.joined()
        del self._sessions[session_id]
        return audio

    def cancel(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
