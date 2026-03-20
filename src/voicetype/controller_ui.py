from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import (
    AppConfig,
    asr_log_file_path,
    load_runtime_config,
    save_runtime_config,
    ui_log_file_path,
)


class RuntimeConfigUpdateRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    model: str
    device: str
    hf_endpoint: str | None = None


class HotwordsTextRequest(BaseModel):
    text: str


class HotwordsFileRequest(BaseModel):
    path: str


class HotkeyConfigRequest(BaseModel):
    enabled: bool
    hotkey: str = "right_alt"


class X11GlobalHotkeyBridge:
    _HOTKEY_ALIASES = {
        "rightalt": "right_alt",
        "alt_r": "right_alt",
        "leftalt": "left_alt",
        "alt_l": "left_alt",
        "rightctrl": "right_ctrl",
        "ctrl_r": "right_ctrl",
        "leftctrl": "left_ctrl",
        "ctrl_l": "left_ctrl",
        "space": "space",
    }

    def __init__(self, asr_request, logger: logging.Logger) -> None:
        self._asr_request = asr_request
        self._logger = logger
        self._lock = threading.Lock()
        self._listener = None
        self._configured_enabled = False
        self._hotkey = "right_alt"
        self._pressed = False
        self._phase = "idle"
        self._last_error: str | None = None

    def _normalize_hotkey(self, hotkey: str) -> str:
        key = (hotkey or "right_alt").strip().lower().replace("-", "_")
        return self._HOTKEY_ALIASES.get(key, key)

    def _resolve_hotkey(self, keyboard, hotkey: str):
        key_name = self._normalize_hotkey(hotkey)
        mapping = {
            "right_alt": keyboard.Key.alt_r,
            "left_alt": keyboard.Key.alt_l,
            "right_ctrl": keyboard.Key.ctrl_r,
            "left_ctrl": keyboard.Key.ctrl_l,
            "right_shift": keyboard.Key.shift_r,
            "left_shift": keyboard.Key.shift_l,
            "space": keyboard.Key.space,
            "f7": keyboard.Key.f7,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
            "f11": keyboard.Key.f11,
            "f12": keyboard.Key.f12,
        }
        pause_key = getattr(keyboard.Key, "pause", None)
        if pause_key is not None:
            mapping["pause"] = pause_key
        scroll_lock_key = getattr(keyboard.Key, "scroll_lock", None)
        if scroll_lock_key is not None:
            mapping["scroll_lock"] = scroll_lock_key
        if key_name in mapping:
            return ("special", mapping[key_name], key_name)
        if len(key_name) == 1 and key_name.isprintable():
            return ("char", key_name, key_name)
        raise ValueError(f"unsupported hotkey: {hotkey}")


    def _inject_text(self, text: str) -> None:
        content = text.strip()
        if not content:
            return
        xdotool = shutil.which("xdotool")
        if not xdotool:
            raise RuntimeError("xdotool not found, cannot inject text on X11.")
        proc = subprocess.run(
            [xdotool, "type", "--clearmodifiers", "--delay", "1", content],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "xdotool failed").strip())

    def _start_recording(self) -> None:
        try:
            with self._lock:
                self._phase = "recording"
            self._logger.info("hotkey: start recording")
            self._asr_request("POST", "/v1/recording/start", {"sample_rate": 16000})
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._pressed = False
                self._phase = "error"
                self._last_error = f"start recording failed: {exc}"
            self._logger.error("global hotkey start failed: %s", exc)

    def _stop_recording_and_inject(self) -> None:
        try:
            with self._lock:
                self._phase = "transcribing"
            self._logger.info("hotkey: start transcribing")
            out = self._asr_request("POST", "/v1/recording/stop", {})
            text = str(out.get("text", "")).strip()
            if text:
                self._inject_text(text)
            with self._lock:
                self._phase = "idle"
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._phase = "error"
                self._last_error = f"stop recording failed: {exc}"
            self._logger.error("global hotkey stop failed: %s", exc)

    def apply(self, enabled: bool, hotkey: str) -> None:
        with self._lock:
            self._configured_enabled = enabled
            self._hotkey = self._normalize_hotkey(hotkey)
            self._last_error = None
            self._pressed = False
            self._phase = "idle"
            listener = self._listener
            self._listener = None

        if listener is not None:
            listener.stop()

        if not enabled:
            return

        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type and session_type != "x11":
            with self._lock:
                self._last_error = f"X11 only, current session={session_type}"
            return
        if not os.environ.get("DISPLAY"):
            with self._lock:
                self._last_error = "DISPLAY is not set, cannot start X11 hotkey listener."
            return

        try:
            from pynput import keyboard  # type: ignore
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f"pynput unavailable: {exc}"
            return

        try:
            kind, target, normalized = self._resolve_hotkey(keyboard, hotkey)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = str(exc)
            return

        with self._lock:
            self._hotkey = normalized

        def matches(k) -> bool:
            if kind == "special":
                return k == target
            return getattr(k, "char", None) == target

        def on_press(k) -> None:
            if not matches(k):
                return
            with self._lock:
                if self._pressed:
                    return
                self._pressed = True
            self._start_recording()

        def on_release(k) -> None:
            if not matches(k):
                return
            should_stop = False
            with self._lock:
                if not self._pressed:
                    return
                self._pressed = False
                should_stop = self._phase == "recording"
            if should_stop:
                self._stop_recording_and_inject()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        with self._lock:
            self._listener = listener
        self._logger.info("global hotkey listener started, key=%s", normalized)

    def stop(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
            self._pressed = False
            self._phase = "idle"
        if listener is not None:
            listener.stop()

    def state(self) -> dict[str, object]:
        with self._lock:
            running = self._listener is not None
            return {
                "configured_enabled": self._configured_enabled,
                "running": running,
                "hotkey": self._hotkey,
                "phase": self._phase,
                "last_error": self._last_error,
                "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
            }

def _run_systemctl_user(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def _kill_unmanaged_asr_processes() -> list[str]:
    killed: list[str] = []
    # Only kill manually launched "voicetype serve ..." processes.
    proc = subprocess.run(
        ["pgrep", "-a", "-f", r"voicetype serve( |$)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return killed
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, cmd = line.partition(" ")
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        if "serve-from-config" in cmd:
            continue
        # best effort terminate.
        subprocess.run(["kill", "-TERM", str(pid)], check=False)
        killed.append(f"{pid} {cmd}")
    return killed


def _setup_ui_file_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_file = ui_log_file_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_file)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            return
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)


def _tail_text(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def _read_full_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _detect_devices() -> list[dict[str, str]]:
    devices: list[dict[str, str]] = [{"value": "cpu", "label": "cpu"}]
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return devices
    proc = subprocess.run(
        [nvidia_smi, "--query-gpu=index,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return devices
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        left, sep, right = line.partition(",")
        idx = left.strip()
        name = right.strip() if sep else ""
        if not idx.isdigit():
            continue
        value = f"cuda:{idx}"
        label = f"{value} ({name})" if name else value
        devices.append({"value": value, "label": label})
    return devices


def render_controller_ui() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>VoiceType 控制台</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --line:#334155; --txt:#e2e8f0; --muted:#94a3b8; --ok:#16a34a; --bad:#ef4444; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:linear-gradient(160deg,#020617,#111827 45%,#1f2937); color:var(--txt); }
    .wrap { max-width: 980px; margin: 28px auto; padding: 0 14px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    .card { border:1px solid var(--line); background:rgba(17,24,39,.9); border-radius: 14px; padding:14px; }
    h1 { margin:0 0 16px; }
    h2 { margin:0 0 10px; font-size:16px; color:#cbd5e1; }
    label { display:block; font-size:13px; color:var(--muted); margin:10px 0 6px; }
    input, select, textarea { width:100%; border:1px solid #475569; background:#0b1220; color:#e5e7eb; border-radius:10px; padding:10px; font-size:14px; }
    textarea { min-height: 120px; resize: vertical; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .row { display:flex; gap:10px; }
    .row > * { flex:1; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; margin-top: 12px; }
    button { border:1px solid #475569; background:#1e293b; color:#e2e8f0; border-radius:10px; padding:9px 12px; cursor:pointer; }
    button.primary { background:#2563eb; border-color:#2563eb; }
    button.warn { background:#d97706; border-color:#d97706; }
    .status { margin-top:10px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
    .ok { color:#4ade80; } .bad { color:#f87171; }
    .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>VoiceType 控制台</h1>
    <div class="grid">
      <section class="card">
        <h2>运行配置</h2>
        <label>模型</label>
        <select id="model">
          <option value="Qwen/Qwen3-ASR-0.6B">Qwen/Qwen3-ASR-0.6B</option>
          <option value="Qwen/Qwen3-ASR-1.7B">Qwen/Qwen3-ASR-1.7B</option>
        </select>
        <div class="row">
          <div>
            <label>设备</label>
            <select id="device"></select>
          </div>
        </div>
        <div class="row">
          <div><label>服务 Host</label><input id="host" /></div>
          <div><label>服务 Port</label><input id="port" type="number" /></div>
        </div>
        <div class="row">
          <div><label>HF 镜像（清空=官方）</label><input id="hf_endpoint" placeholder="https://hf-mirror.com" /></div>
        </div>
        <div class="btns">
          <button class="primary" id="saveCfg">保存配置</button>
          <button id="startSvc">启动服务</button>
          <button id="restartSvc">重载服务</button>
          <button class="warn" id="stopSvc">停止服务</button>
        </div>
        <div class="status">先保存配置，再启动/重载服务</div>
        <div id="cfgStatus" class="status"></div>
        <details style="margin-top:10px;">
          <summary style="cursor:pointer;color:#cbd5e1;">X11 全局热键（替代 Fcitx 可选）</summary>
          <div style="margin-top:8px;">
            <label style="display:flex;align-items:center;gap:8px;">
              <input id="hotkeyEnabled" type="checkbox" style="width:auto;" />
              启用全局按住说话（X11）
            </label>
            <label>快捷键</label>
            <select id="hotkeyKey">
              <option value="f7">f7</option>
              <option value="f8">f8</option>
              <option value="f9">f9</option>
              <option value="f10">f10</option>
              <option value="f11">f11</option>
              <option value="f12">f12</option>
              <option value="pause">pause</option>
              <option value="scroll_lock">scroll_lock</option>
              <option value="right_alt">right_alt</option>
              <option value="left_alt">left_alt</option>
              <option value="right_ctrl">right_ctrl</option>
              <option value="left_ctrl">left_ctrl</option>
            </select>
            <div class="status">推荐优先：f10 / f9 / f8 / f7（通常比 Alt/Ctrl 稳定）</div>
            <div class="btns">
              <button id="saveHotkey">保存热键配置</button>
            </div>
            <div id="hotkeyStatus" class="status"></div>
          </div>
        </details>
      </section>

      <section class="card">
        <h2>热词维护（作用于正在运行的 ASR 服务）</h2>
        <label>输入框热词（每行：词 或 词 权重）</label>
        <textarea id="hotText">pytorch 50
centos 20</textarea>
        <label>热词文件路径</label>
        <input id="hotFile" placeholder="/path/to/hotwords.txt" />
        <div class="btns">
          <button class="primary" id="loadHotText">加载输入框热词</button>
          <button id="loadHotFile">从文件加载热词</button>
          <button id="clearHot">清空热词</button>
        </div>
        <div id="hotStatus" class="status"></div>
      </section>
    </div>
    <section class="card" style="margin-top:14px">
      <h2>日志</h2>
      <div class="btns">
        <button id="refreshLog">刷新日志</button>
        <button id="openAsrLogView">查看 ASR 服务日志详情</button>
        <button id="openUiLogView">查看控制台服务日志详情</button>
      </div>
      <details style="margin-top:8px;">
        <summary style="cursor:pointer;color:#cbd5e1;">ASR 服务日志（最近 200 行预览）</summary>
        <textarea id="asrLog" readonly></textarea>
      </details>
      <details style="margin-top:8px;">
        <summary style="cursor:pointer;color:#cbd5e1;">控制台服务日志（最近 200 行预览）</summary>
        <textarea id="uiLog" readonly></textarea>
      </details>
      <div id="logStatus" class="status"></div>
    </section>
  </div>
  <script>
    async function req(url, opt) {
      const r = await fetch(url, opt);
      const text = await r.text();
      let json; try { json = JSON.parse(text); } catch { json = {raw:text}; }
      if (!r.ok) throw new Error(json.detail || text || ("HTTP " + r.status));
      return json;
    }
    function setStatus(id, msg, ok=true) {
      const el = document.getElementById(id);
      el.textContent = msg;
      el.className = "status " + (ok ? "ok" : "bad");
    }
    function restoreHotwordFilePath() {
      const saved = localStorage.getItem("voicetype.hotword_file_path");
      if (saved) hotFile.value = saved;
    }
    function persistHotwordFilePath() {
      localStorage.setItem("voicetype.hotword_file_path", hotFile.value || "");
    }
    async function refreshDevices(selectedValue) {
      const out = await req("/api/devices");
      device.innerHTML = "";
      const values = new Set();
      for (const d of (out.devices || [])) {
        const opt = document.createElement("option");
        opt.value = d.value;
        opt.textContent = d.label || d.value;
        device.appendChild(opt);
        values.add(d.value);
      }
      if (selectedValue && !values.has(selectedValue)) {
        const opt = document.createElement("option");
        opt.value = selectedValue;
        opt.textContent = selectedValue + " (自定义)";
        device.appendChild(opt);
      }
      device.value = selectedValue || "cpu";
    }
    async function refreshState() {
      const s = await req("/api/state");
      const c = s.config;
      model.value = c.model;
      await refreshDevices(c.device);
      host.value = c.host; port.value = c.port;
      hf_endpoint.value = c.hf_endpoint || "";
      hotkeyEnabled.checked = !!c.global_hotkey_enabled;
      hotkeyKey.value = c.global_hotkey_key || "right_alt";
      await refreshHotkeyState();
    }
    async function refreshHotkeyState() {
      try {
        const hk = await req("/api/hotkey/state");
        const phaseMap = { idle: "空闲", recording: "说话中", transcribing: "识别中", error: "异常" };
        let msg = "状态: " + (hk.running ? "运行中" : "未运行");
        msg += " | 阶段: " + (phaseMap[hk.phase] || hk.phase || "未知");
        msg += " | 会话: " + (hk.session_type || "unknown");
        msg += " | 快捷键: " + (hk.hotkey || "right_alt");
        if (hk.last_error) {
          msg += "\\n错误: " + hk.last_error;
        }
        setStatus("hotkeyStatus", msg, !hk.last_error);
      } catch (e) {
        setStatus("hotkeyStatus", String(e), false);
      }
    }
    async function refreshLogs() {
      try {
        const data = await req("/api/logs?lines=200");
        asrLog.value = data.asr_log || "";
        uiLog.value = data.ui_log || "";
        setStatus("logStatus", "日志路径\\nASR: " + data.asr_log_path + "\\nUI: " + data.ui_log_path, true);
      } catch (e) {
        setStatus("logStatus", String(e), false);
      }
    }
    saveCfg.onclick = async () => {
      try {
        const body = {
          model: model.value, device: device.value,
          host: host.value, port: Number(port.value),
          hf_endpoint: hf_endpoint.value || null
        };
        const out = await req("/api/config", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
        setStatus("cfgStatus", "配置已保存\\n" + JSON.stringify(out, null, 2), true);
        await refreshState();
      } catch (e) { setStatus("cfgStatus", String(e), false); }
    };
    startSvc.onclick = async () => { try { const out = await req("/api/service/start", {method:"POST"}); setStatus("cfgStatus", JSON.stringify(out, null, 2), true); await refreshState(); } catch(e){ setStatus("cfgStatus", String(e), false);} };
    restartSvc.onclick = async () => { try { const out = await req("/api/service/restart", {method:"POST"}); setStatus("cfgStatus", JSON.stringify(out, null, 2), true); await refreshState(); } catch(e){ setStatus("cfgStatus", String(e), false);} };
    stopSvc.onclick = async () => { try { const out = await req("/api/service/stop", {method:"POST"}); setStatus("cfgStatus", JSON.stringify(out, null, 2), true); await refreshState(); } catch(e){ setStatus("cfgStatus", String(e), false);} };
    saveHotkey.onclick = async () => {
      try {
        const body = {
          enabled: !!hotkeyEnabled.checked,
          hotkey: hotkeyKey.value || "right_alt"
        };
        const out = await req("/api/hotkey/config", {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify(body)
        });
        setStatus("hotkeyStatus", JSON.stringify(out.hotkey || out, null, 2), true);
        await refreshState();
      } catch (e) { setStatus("hotkeyStatus", String(e), false); }
    };
    loadHotText.onclick = async () => {
      try {
        const out = await req("/api/hotwords/text", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({text: hotText.value})
        });
        if (out.effective_text !== undefined) hotText.value = out.effective_text;
        setStatus("hotStatus", JSON.stringify(out, null, 2), true);
      } catch (e) { setStatus("hotStatus", String(e), false); }
    };
    loadHotFile.onclick = async () => {
      try {
        persistHotwordFilePath();
        const out = await req("/api/hotwords/file", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({path: hotFile.value})
        });
        if (out.effective_text !== undefined) hotText.value = out.effective_text;
        setStatus("hotStatus", JSON.stringify(out, null, 2), true);
      } catch (e) { setStatus("hotStatus", String(e), false); }
    };
    clearHot.onclick = async () => {
      try {
        const out = await req("/api/hotwords", {method:"DELETE"});
        hotText.value = "";
        hotFile.value = "";
        persistHotwordFilePath();
        setStatus("hotStatus", JSON.stringify(out, null, 2), true);
      } catch (e) { setStatus("hotStatus", String(e), false); }
    };
    hotFile.addEventListener("change", persistHotwordFilePath);
    hotFile.addEventListener("input", persistHotwordFilePath);
    restoreHotwordFilePath();
    refreshLog.onclick = () => refreshLogs();
    openAsrLogView.onclick = () => window.open("/ui/logs/asr", "_blank");
    openUiLogView.onclick = () => window.open("/ui/logs/ui", "_blank");
    refreshState().catch(e => setStatus("cfgStatus", String(e), false));
    refreshLogs();
    setInterval(refreshLogs, 3000);
    setInterval(refreshHotkeyState, 600);
  </script>
</body>
</html>
"""


def render_log_view_ui(log_type: str) -> str:
    if log_type == "ui":
        page_title = "VoiceType UI 日志详情"
        heading = "控制台 UI 日志（完整）"
        api_path = "/api/logs/full/ui"
    else:
        page_title = "VoiceType ASR 日志详情"
        heading = "ASR 服务日志（完整）"
        api_path = "/api/logs/full/asr"
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>""" + page_title + """</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --line:#334155; --txt:#e2e8f0; --muted:#94a3b8; --ok:#16a34a; --bad:#ef4444; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:linear-gradient(160deg,#020617,#111827 45%,#1f2937); color:var(--txt); }
    .wrap { max-width: 1200px; margin: 22px auto; padding: 0 14px; }
    .card { border:1px solid var(--line); background:rgba(17,24,39,.9); border-radius: 14px; padding:14px; }
    h1 { margin:0 0 14px; }
    h2 { margin:0 0 10px; font-size:16px; color:#cbd5e1; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; margin: 0 0 12px; }
    button { border:1px solid #475569; background:#1e293b; color:#e2e8f0; border-radius:10px; padding:9px 12px; cursor:pointer; }
    textarea { width:100%; min-height: 38vh; border:1px solid #475569; background:#0b1220; color:#e5e7eb; border-radius:10px; padding:10px; font-size:13px; resize: vertical; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .status { margin-top:8px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
    .ok { color:#4ade80; } .bad { color:#f87171; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>""" + page_title + """</h1>
    <section class="card">
      <div class="btns">
        <button id="refresh">刷新全量日志</button>
        <button id="toggleAuto">自动刷新：开</button>
      </div>
      <h2>""" + heading + """</h2>
      <textarea id="logText" readonly></textarea>
      <div id="status" class="status"></div>
    </section>
  </div>
  <script>
    let timer = null;
    let auto = true;
    async function req(url, opt) {
      const r = await fetch(url, opt);
      const text = await r.text();
      let json; try { json = JSON.parse(text); } catch { json = {raw:text}; }
      if (!r.ok) throw new Error(json.detail || text || ("HTTP " + r.status));
      return json;
    }
    function setStatus(msg, ok=true) {
      const el = document.getElementById("status");
      el.textContent = msg;
      el.className = "status " + (ok ? "ok" : "bad");
    }
    async function refresh() {
      try {
        const out = await req(\"""" + api_path + """\");
        logText.value = out.log || "";
        setStatus("日志路径\\n" + out.log_path, true);
      } catch (e) {
        setStatus(String(e), false);
      }
    }
    function setAuto(next) {
      auto = next;
      toggleAuto.textContent = "自动刷新：" + (auto ? "开" : "关");
      if (timer) clearInterval(timer);
      timer = auto ? setInterval(refresh, 3000) : null;
    }
    refresh.onclick = () => refresh();
    toggleAuto.onclick = () => setAuto(!auto);
    refresh();
    setAuto(true);
  </script>
</body>
</html>
"""


def create_controller_app(config_file: Path, service_name: str = "voicetype.service") -> FastAPI:
    app = FastAPI(title="VoiceType Controller", version="0.1.0")
    _setup_ui_file_logging()
    logger = logging.getLogger("voicetype.controller")

    def _asr_base_url() -> str:
        cfg = load_runtime_config(config_file)
        return f"http://{cfg.host}:{cfg.port}"

    def _asr_request(method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        url = f"{_asr_base_url()}{path}"
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.request(method, url, json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ASR service unreachable: {exc}") from exc
        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            raise HTTPException(status_code=resp.status_code, detail=detail or str(data))
        return data if isinstance(data, dict) else {"data": data}

    hotkey_bridge = X11GlobalHotkeyBridge(_asr_request, logger)

    @app.on_event("startup")
    def startup() -> None:
        cfg = load_runtime_config(config_file)
        hotkey_bridge.apply(cfg.global_hotkey_enabled, cfg.global_hotkey_key)

    @app.on_event("shutdown")
    def shutdown() -> None:
        hotkey_bridge.stop()

    @app.get("/ui", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        return HTMLResponse(render_controller_ui())

    @app.get("/ui/logs/asr", response_class=HTMLResponse)
    def ui_logs_asr() -> HTMLResponse:
        return HTMLResponse(render_log_view_ui("asr"))

    @app.get("/ui/logs/ui", response_class=HTMLResponse)
    def ui_logs_ui() -> HTMLResponse:
        return HTMLResponse(render_log_view_ui("ui"))

    @app.get("/api/state")
    def state() -> dict[str, object]:
        cfg = load_runtime_config(config_file)
        active_code, _ = _run_systemctl_user("is-active", service_name)
        enabled_code, _ = _run_systemctl_user("is-enabled", service_name)
        return {
            "config": cfg.model_dump(),
            "service": {
                "active": active_code == 0,
                "enabled": enabled_code == 0,
            },
            "hotkey": hotkey_bridge.state(),
        }

    @app.get("/api/devices")
    def devices() -> dict[str, object]:
        return {"devices": _detect_devices()}

    @app.get("/api/asr-base")
    def asr_base() -> dict[str, str]:
        return {"url": _asr_base_url()}

    @app.get("/api/logs")
    def logs(lines: int = 200) -> dict[str, str]:
        if lines < 10:
            lines = 10
        if lines > 1000:
            lines = 1000
        asr_path = asr_log_file_path()
        ui_path = ui_log_file_path()
        return {
            "asr_log_path": str(asr_path),
            "ui_log_path": str(ui_path),
            "asr_log": _tail_text(asr_path, lines=lines),
            "ui_log": _tail_text(ui_path, lines=lines),
        }

    @app.get("/api/logs/full")
    def logs_full() -> dict[str, str]:
        asr_path = asr_log_file_path()
        ui_path = ui_log_file_path()
        return {
            "asr_log_path": str(asr_path),
            "ui_log_path": str(ui_path),
            "asr_log": _read_full_text(asr_path),
            "ui_log": _read_full_text(ui_path),
        }

    @app.get("/api/logs/full/asr")
    def logs_full_asr() -> dict[str, str]:
        asr_path = asr_log_file_path()
        return {
            "log_path": str(asr_path),
            "log": _read_full_text(asr_path),
        }

    @app.get("/api/logs/full/ui")
    def logs_full_ui() -> dict[str, str]:
        ui_path = ui_log_file_path()
        return {
            "log_path": str(ui_path),
            "log": _read_full_text(ui_path),
        }

    @app.post("/api/config")
    def update_config(req: RuntimeConfigUpdateRequest) -> dict[str, object]:
        prev_cfg = load_runtime_config(config_file)
        cfg = prev_cfg.model_copy(
            update={
                "host": req.host,
                "port": req.port,
                "model": req.model,
                "device": req.device,
                "hf_endpoint": req.hf_endpoint,
            }
        )
        saved = save_runtime_config(cfg, config_file)
        logger.info("runtime config updated: model=%s device=%s port=%s", cfg.model, cfg.device, cfg.port)
        return {"success": True, "path": str(saved)}

    @app.get("/api/hotkey/state")
    def hotkey_state() -> dict[str, object]:
        return hotkey_bridge.state()

    @app.post("/api/hotkey/config")
    def hotkey_config(req: HotkeyConfigRequest) -> dict[str, object]:
        cfg = load_runtime_config(config_file).model_copy(
            update={
                "global_hotkey_enabled": req.enabled,
                "global_hotkey_key": req.hotkey,
            }
        )
        saved = save_runtime_config(cfg, config_file)
        hotkey_bridge.apply(req.enabled, req.hotkey)
        return {"success": True, "path": str(saved), "hotkey": hotkey_bridge.state()}


    @app.post("/api/service/start")
    def start_service() -> dict[str, object]:
        killed = _kill_unmanaged_asr_processes()
        code, output = _run_systemctl_user("start", service_name)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "start failed")
        logger.info("service started: %s", service_name)
        return {"success": True, "killed_unmanaged": killed}

    @app.post("/api/service/restart")
    def restart_service() -> dict[str, object]:
        killed = _kill_unmanaged_asr_processes()
        code, output = _run_systemctl_user("restart", service_name)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "restart failed")
        logger.info("service restarted: %s", service_name)
        return {"success": True, "killed_unmanaged": killed}

    @app.post("/api/service/stop")
    def stop_service() -> dict[str, object]:
        code, output = _run_systemctl_user("stop", service_name)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "stop failed")
        logger.info("service stopped: %s", service_name)
        return {"success": True}

    @app.post("/api/hotwords/text")
    def hotwords_from_text(req: HotwordsTextRequest) -> dict[str, object]:
        return _asr_request("POST", "/v1/ui/hotwords/text", {"text": req.text})

    @app.post("/api/hotwords/file")
    def hotwords_from_file(req: HotwordsFileRequest) -> dict[str, object]:
        return _asr_request("POST", "/v1/ui/hotwords/file", {"path": req.path})

    @app.delete("/api/hotwords")
    def hotwords_clear() -> dict[str, object]:
        return _asr_request("DELETE", "/v1/hotwords")

    return app
