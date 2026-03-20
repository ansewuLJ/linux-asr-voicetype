from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RecordingSession:
    process: subprocess.Popen[bytes]
    wav_path: Path
    language: str | None
    sample_rate: int


class ArecordManager:
    def __init__(self) -> None:
        self._session: RecordingSession | None = None

    def is_recording(self) -> bool:
        return self._session is not None and self._session.process.poll() is None

    def start(
        self,
        language: str | None,
        sample_rate: int = 16000,
        device: str | None = None,
    ) -> None:
        if self.is_recording():
            raise RuntimeError("already recording")

        arecord = shutil.which("arecord")
        if not arecord:
            raise RuntimeError("arecord not found; install alsa-utils")

        fd, wav_path_str = tempfile.mkstemp(prefix="qfinput_", suffix=".wav")
        os.close(fd)
        Path(wav_path_str).unlink(missing_ok=True)

        cmd = [arecord, "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]
        if device:
            cmd += ["-D", device]
        cmd.append(wav_path_str)

        process = subprocess.Popen(cmd)
        self._session = RecordingSession(
            process=process,
            wav_path=Path(wav_path_str),
            language=language,
            sample_rate=sample_rate,
        )

    def stop(self) -> tuple[Path, str | None, int]:
        if not self._session:
            raise RuntimeError("not recording")

        session = self._session
        self._session = None

        if session.process.poll() is None:
            session.process.terminate()
            try:
                session.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=2)

        return session.wav_path, session.language, session.sample_rate
