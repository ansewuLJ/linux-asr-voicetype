from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class BridgeConfig:
    base_url: str = "http://127.0.0.1:8787"
    hold_key: str = "right_alt"
    toggle_key: str = "left_alt+z"
    sample_rate: int = 16000
    type_delay_ms: int = 1


class FcitxX11Bridge:
    _ALIASES = {
        "rightalt": "right_alt",
        "alt_r": "right_alt",
        "leftalt": "left_alt",
        "alt_l": "left_alt",
        "rightctrl": "right_ctrl",
        "ctrl_r": "right_ctrl",
        "leftctrl": "left_ctrl",
        "ctrl_l": "left_ctrl",
        "rightshift": "right_shift",
        "shift_r": "right_shift",
        "leftshift": "left_shift",
        "shift_l": "left_shift",
        "esc": "escape",
    }
    _MODIFIERS = {
        "left_alt",
        "right_alt",
        "left_ctrl",
        "right_ctrl",
        "left_shift",
        "right_shift",
    }

    def __init__(self, cfg: BridgeConfig, logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._pressed: set[str] = set()
        self._toggle_latched = False
        self._recording = False
        self._recording_owner: str | None = None
        self._hold_blocked = False
        self._stop_event = threading.Event()
        self._hold_token = self._normalize_token(cfg.hold_key)
        self._toggle_tokens = self._parse_toggle_tokens(cfg.toggle_key)

    def _normalize_token(self, token: str) -> str:
        t = (token or "").strip().lower().replace("-", "_")
        if not t:
            raise ValueError("empty key token")
        return self._ALIASES.get(t, t)

    def _parse_toggle_tokens(self, spec: str) -> set[str]:
        raw = (spec or "").strip()
        if not raw:
            return set()
        tokens = {self._normalize_token(x) for x in raw.split("+") if x.strip()}
        if len(tokens) < 2:
            raise ValueError("toggle_key must be a combo, e.g. left_alt+z")
        return tokens

    def _asr_request(self, method: str, path: str, payload: dict) -> dict:
        url = f"{self._cfg.base_url.rstrip('/')}{path}"
        timeout = 120.0 if path.endswith("/stop") else 15.0
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _inject_text(self, text: str) -> None:
        content = text.strip()
        if not content:
            return
        xdotool = shutil.which("xdotool")
        if not xdotool:
            raise RuntimeError("xdotool not found, cannot inject text on X11.")
        proc = subprocess.run(
            [
                xdotool,
                "type",
                "--clearmodifiers",
                "--delay",
                str(self._cfg.type_delay_ms),
                content,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "xdotool failed").strip())

    def _start_recording(self, owner: str) -> None:
        self._logger.info("bridge: start recording (%s)", owner)
        self._asr_request("POST", "/v1/recording/start", {"sample_rate": self._cfg.sample_rate})
        self._recording = True
        self._recording_owner = owner
        self._hold_blocked = False

    def _stop_recording(self, inject: bool = True) -> None:
        self._logger.info("bridge: stop recording (inject=%s)", inject)
        out = self._asr_request("POST", "/v1/recording/stop", {})
        if inject:
            text = str(out.get("text", "")).strip()
            if text:
                self._inject_text(text)
        self._recording = False
        self._recording_owner = None
        self._hold_blocked = False

    def _key_to_token(self, key, keyboard) -> str | None:
        if key == keyboard.Key.alt_l:
            return "left_alt"
        if key == keyboard.Key.alt_r:
            return "right_alt"
        if key == keyboard.Key.ctrl_l:
            return "left_ctrl"
        if key == keyboard.Key.ctrl_r:
            return "right_ctrl"
        if key == keyboard.Key.shift_l:
            return "left_shift"
        if key == keyboard.Key.shift_r:
            return "right_shift"
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.enter:
            return "enter"
        if key == keyboard.Key.tab:
            return "tab"
        if key == keyboard.Key.esc:
            return "escape"
        for i in range(1, 13):
            if key == getattr(keyboard.Key, f"f{i}"):
                return f"f{i}"

        ch = getattr(key, "char", None)
        if ch and len(ch) == 1 and ch.isprintable():
            return ch.lower()
        return None

    def _on_press(self, token: str) -> None:
        self._pressed.add(token)

        if self._toggle_tokens and self._toggle_tokens.issubset(self._pressed) and not self._toggle_latched:
            self._toggle_latched = True
            try:
                if self._recording and self._recording_owner == "toggle":
                    self._stop_recording(inject=True)
                elif not self._recording:
                    self._start_recording("toggle")
            except Exception as exc:  # noqa: BLE001
                self._logger.error("toggle handling failed: %s", exc)
            return

        if token == self._hold_token and not self._recording:
            try:
                self._start_recording("hold")
            except Exception as exc:  # noqa: BLE001
                self._logger.error("hold start failed: %s", exc)
            return

        if (
            self._recording
            and self._recording_owner == "hold"
            and self._hold_token in self._MODIFIERS
            and token != self._hold_token
        ):
            # Modifier + other key is treated as normal shortcut chord, not speech trigger.
            self._hold_blocked = True

    def _on_release(self, token: str) -> None:
        self._pressed.discard(token)
        if self._toggle_latched and (not self._toggle_tokens.issubset(self._pressed)):
            self._toggle_latched = False

        if token == self._hold_token and self._recording and self._recording_owner == "hold":
            try:
                self._stop_recording(inject=not self._hold_blocked)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("hold stop failed: %s", exc)

    def run(self) -> None:
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type and session_type != "x11":
            raise RuntimeError(f"X11 only, current session={session_type}")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAY is not set, cannot start hotkey bridge.")

        try:
            from pynput import keyboard  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"pynput unavailable: {exc}") from exc

        self._logger.info(
            "bridge ready: hold_key=%s, toggle_key=%s, base_url=%s",
            self._hold_token,
            "+".join(sorted(self._toggle_tokens)) if self._toggle_tokens else "(disabled)",
            self._cfg.base_url,
        )

        def on_press(k) -> None:
            token = self._key_to_token(k, keyboard)
            if token is None:
                return
            self._on_press(token)

        def on_release(k) -> None:
            token = self._key_to_token(k, keyboard)
            if token is None:
                return
            self._on_release(token)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

        def _shutdown(*_args) -> None:
            self._stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        self._stop_event.wait()
        listener.stop()
        self._logger.info("bridge stopped")


def run_fcitx_x11_bridge(cfg: BridgeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[voicetype-bridge] %(message)s")
    logger = logging.getLogger("voicetype.bridge")
    bridge = FcitxX11Bridge(cfg, logger)
    bridge.run()
