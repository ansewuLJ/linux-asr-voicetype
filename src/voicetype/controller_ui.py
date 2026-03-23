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

from .audio import encode_pcm16_wav_base64, load_wav_file
from .config import (
    AppConfig,
    asr_log_file_path,
    DEFAULT_POSTPROCESS_SYSTEM_PROMPT,
    load_runtime_config,
    save_runtime_config,
    ui_log_file_path,
)
from .recorder import ArecordManager
from .schema import RecordingStartRequest

ASR_TRANSFORMERS_SERVICE = "asr-transformers.service"
ASR_OPENVINO_SERVICE = "asr-openvino.service"
ASR_MANAGER_UI_SERVICE = "asr-manager-ui.service"
MANAGED_ASR_SERVICES = {
    ASR_TRANSFORMERS_SERVICE,
    ASR_OPENVINO_SERVICE,
    ASR_MANAGER_UI_SERVICE,
}

POSTPROCESS_SYSTEM_PROMPT = DEFAULT_POSTPROCESS_SYSTEM_PROMPT


class RuntimeConfigUpdateRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8789
    model: str
    device: str
    dtype: str | None = "bfloat16"
    hf_endpoint: str | None = None


class ConnectConfigRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8789


class HotwordsTextRequest(BaseModel):
    text: str


class HotwordsFileRequest(BaseModel):
    path: str


class HotkeyConfigRequest(BaseModel):
    enabled: bool
    hotkey: str = "right_alt"


class PostprocessConfigRequest(BaseModel):
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    system_prompt: str = POSTPROCESS_SYSTEM_PROMPT


class PostprocessTestRequest(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    text: str = "连接测试"


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
  <title>VoiceType 接入控制台</title>
  <style>
    :root { --line:#d5deea; --txt:#1f2937; --muted:#6b7280; --card:#ffffff; --bg1:#f7f9fc; --bg2:#edf3fb; --bg3:#e8eef8; }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body { margin:0; min-height:100vh; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:linear-gradient(160deg,var(--bg1) 0%, var(--bg2) 55%, var(--bg3) 100%); color:var(--txt); }
    .wrap { max-width: 980px; margin: 28px auto; padding: 0 14px 24px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    .layout { display:grid; grid-template-columns: 1fr 1fr; grid-template-areas: "connect hotwords" "hotkey hotwords" "postprocess hotwords"; gap:14px; align-items:stretch; }
    .card-connect { grid-area: connect; }
    .card-hotkey { grid-area: hotkey; }
    .card-postprocess { grid-area: postprocess; }
    .card-hotwords { grid-area: hotwords; }
    .card { border:1px solid var(--line); background:var(--card); border-radius: 14px; padding:14px; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06); }
    h1 { margin:0 0 16px; }
    h2 { margin:0 0 10px; font-size:16px; color:#1f2937; }
    label { display:block; font-size:13px; color:var(--muted); margin:10px 0 6px; }
    input, select, textarea { width:100%; border:1px solid #c6d0df; background:#ffffff; color:#1f2937; border-radius:10px; padding:10px; font-size:14px; }
    textarea { min-height: 140px; resize: vertical; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .input-with-action { display:flex; align-items:center; gap:8px; }
    .input-with-action input { flex:1; }
    .icon-btn { width:40px; height:40px; display:inline-flex; align-items:center; justify-content:center; border:1px solid #c6d0df; background:#f8fafc; color:#334155; border-radius:10px; cursor:pointer; }
    .icon-btn svg { width:18px; height:18px; }
    details.advanced { margin-top: 10px; border: 1px dashed #c6d0df; border-radius: 10px; background: #f8fafc; padding: 8px 10px; }
    details.advanced summary { cursor: pointer; color: #475569; font-size: 13px; user-select: none; }
    details.advanced[open] summary { margin-bottom: 8px; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; margin-top: 12px; }
    button { border:1px solid #c6d0df; background:#f8fafc; color:#1f2937; border-radius:10px; padding:9px 12px; cursor:pointer; }
    button.primary { background:#2563eb; border-color:#2563eb; color:#ffffff; }
    .status { margin-top:10px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
    .ok { color:#059669; } .bad { color:#dc2626; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .layout { grid-template-columns: 1fr; grid-template-areas: "connect" "hotkey" "postprocess" "hotwords"; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>VoiceType 接入控制台</h1>
    <div class="layout">
      <section class="card card-connect">
        <h2>接入配置</h2>
        <div class="grid" style="grid-template-columns: 1fr 1fr;">
          <div><label>推理服务 Host</label><input id="host" placeholder="127.0.0.1" /></div>
          <div><label>推理服务 Port</label><input id="port" type="number" placeholder="8789" /></div>
        </div>
        <div class="btns"><button class="primary" id="saveConnect">保存接入配置并检查健康</button></div>
        <div id="cfgStatus" class="status"></div>
      </section>

      <section class="card card-hotkey">
        <h2>全局热键（X11）</h2>
        <div class="status">说明：建议优先使用 Fcitx5；仅在 Fcitx 不可用时再启用全局热键。</div>
        <label style="display:flex;align-items:center;gap:8px;">
          <input id="hotkeyEnabled" type="checkbox" style="width:auto;" />
          启用全局按住说话
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
        <div class="btns"><button class="primary" id="saveHotkey">保存热键配置</button></div>
        <div id="hotkeyStatus" class="status"></div>
      </section>

      <section class="card card-postprocess">
        <h2>文本后处理（可选）</h2>
        <div class="status">说明：转写结果可再经文本模型清洗。关闭时保持原始 ASR 文本；开启后失败会自动回退原文。</div>
        <label style="display:flex;align-items:center;gap:8px;">
          <input id="ppEnabled" type="checkbox" style="width:auto;" />
          启用后处理
        </label>
        <label>Base URL</label>
        <input id="ppBaseUrl" placeholder="https://api.openai.com/v1" />
        <label>API Key</label>
        <div class="input-with-action">
          <input id="ppApiKey" type="password" />
          <button id="togglePpApiKey" class="icon-btn" type="button" aria-label="显示或隐藏 API Key" title="显示/隐藏 API Key">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
          </button>
        </div>
        <label>Model</label>
        <input id="ppModel" />
        <details class="advanced">
          <summary>高级选项</summary>
          <label>系统提示词</label>
          <textarea id="ppSystemPrompt" style="min-height: 120px;"></textarea>
        </details>
        <div class="btns">
          <button id="testPostprocess">测试连接</button>
          <button class="primary" id="savePostprocess">保存配置</button>
        </div>
        <div id="ppStatus" class="status"></div>
      </section>

      <section class="card card-hotwords">
        <h2>热词维护</h2>
        <div class="status">说明：热词用于提升专有名词/术语（如人名、地名、项目名）的识别命中率，减少错字。你新增或修改词表后，点击“加载输入框热词”或“从文件加载热词”即可立刻生效。</div>
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
  </div>

  <script>
    const host = document.getElementById('host');
    const port = document.getElementById('port');
    const saveConnect = document.getElementById('saveConnect');
    const hotkeyEnabled = document.getElementById('hotkeyEnabled');
    const hotkeyKey = document.getElementById('hotkeyKey');
    const saveHotkey = document.getElementById('saveHotkey');
    const ppEnabled = document.getElementById('ppEnabled');
    const ppBaseUrl = document.getElementById('ppBaseUrl');
    const ppModel = document.getElementById('ppModel');
    const ppApiKey = document.getElementById('ppApiKey');
    const ppSystemPrompt = document.getElementById('ppSystemPrompt');
    const togglePpApiKey = document.getElementById('togglePpApiKey');
    const savePostprocess = document.getElementById('savePostprocess');
    const testPostprocess = document.getElementById('testPostprocess');
    const hotText = document.getElementById('hotText');
    const hotFile = document.getElementById('hotFile');
    const loadHotText = document.getElementById('loadHotText');
    const loadHotFile = document.getElementById('loadHotFile');
    const clearHot = document.getElementById('clearHot');
    let ppConnectionVerified = false;

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

    function readPostprocessApiKeyCache() {
      return localStorage.getItem("voicetype.postprocess_api_key_cache") || '';
    }

    function writePostprocessApiKeyCache(value) {
      if (value) localStorage.setItem("voicetype.postprocess_api_key_cache", value);
    }

    function clearPostprocessApiKeyCache() {
      localStorage.removeItem("voicetype.postprocess_api_key_cache");
    }

    async function refreshState() {
      const s = await req('/api/state');
      const c = s.config;
      host.value = c.host || '127.0.0.1';
      port.value = c.port || 8789;
      hotkeyEnabled.checked = !!c.global_hotkey_enabled;
      hotkeyKey.value = c.global_hotkey_key || 'right_alt';
      ppEnabled.checked = !!c.postprocess_enabled;
      ppBaseUrl.value = c.postprocess_base_url || '';
      ppModel.value = c.postprocess_model || '';
      ppSystemPrompt.value = c.postprocess_system_prompt || '';
      if (s.postprocess_api_key_set) {
        ppApiKey.value = readPostprocessApiKeyCache();
      } else {
        ppApiKey.value = '';
        clearPostprocessApiKeyCache();
      }
      ppApiKey.type = 'password';
      togglePpApiKey.setAttribute('title', '显示 API Key');
      togglePpApiKey.setAttribute('aria-label', '显示 API Key');
      ppConnectionVerified = !!c.postprocess_enabled;
      await refreshHotkeyState();
      await refreshConnectHealth();
      setStatus('ppStatus', s.postprocess_api_key_set ? '后处理API Key：已配置' : '后处理API Key：未配置', true);
    }

    function markPostprocessDirty() {
      ppConnectionVerified = false;
    }

    function connectStatusText(ok, err='') {
      if (ok) return '推理服务状态：在线';
      if (!err) return '推理服务状态：离线';
      return '推理服务状态：离线\\n' + err;
    }

    async function refreshConnectHealth() {
      try {
        const out = await req('/api/connect-health');
        setStatus('cfgStatus', connectStatusText(!!out.health_ok, out.health_error || ''), !!out.health_ok);
      } catch (e) {
        setStatus('cfgStatus', '推理服务状态：离线\\n' + String(e), false);
      }
    }

    async function refreshHotkeyState() {
      try {
        const hk = await req('/api/hotkey/state');
        const phaseMap = { idle: '空闲', recording: '说话中', transcribing: '识别中', error: '异常' };
        let msg = '状态: ' + (hk.running ? '运行中' : '未运行');
        msg += ' | 阶段: ' + (phaseMap[hk.phase] || hk.phase || '未知');
        msg += ' | 快捷键: ' + (hk.hotkey || 'right_alt');
        if (hk.last_error) msg += '\\n错误: ' + hk.last_error;
        setStatus('hotkeyStatus', msg, !hk.last_error);
      } catch (e) {
        setStatus('hotkeyStatus', String(e), false);
      }
    }

    saveConnect.addEventListener('click', async () => {
      try {
        const out = await req('/api/connect-config', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ host: host.value, port: Number(port.value) })
        });
        setStatus('cfgStatus', connectStatusText(!!out.health_ok, out.health_error || ''), !!out.health_ok);
      } catch (e) {
        setStatus('cfgStatus', '推理服务状态：离线\\n' + String(e), false);
      }
    });

    saveHotkey.addEventListener('click', async () => {
      try {
        const body = { enabled: !!hotkeyEnabled.checked, hotkey: hotkeyKey.value || 'right_alt' };
        const out = await req('/api/hotkey/config', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(body)
        });
        setStatus('hotkeyStatus', JSON.stringify(out.hotkey || out, null, 2), true);
        await refreshState();
      } catch (e) {
        setStatus('hotkeyStatus', String(e), false);
      }
    });

    savePostprocess.addEventListener('click', async () => {
      try {
        if (ppEnabled.checked && !ppConnectionVerified) {
          throw new Error('请先测试连接，连接通过后才能启用。');
        }
        const body = {
          enabled: !!ppEnabled.checked,
          base_url: ppBaseUrl.value || '',
          model: ppModel.value || '',
          api_key: ppApiKey.value || '',
          system_prompt: ppSystemPrompt.value || ''
        };
        const out = await req('/api/postprocess/config', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(body)
        });
        if (ppApiKey.value) {
          writePostprocessApiKeyCache(ppApiKey.value);
        } else if (!out.api_key_set) {
          clearPostprocessApiKeyCache();
        }
        setStatus('ppStatus', '配置已保存\\n' + (out.enabled ? '后处理：已启用' : '后处理：未启用') + '\\n' + (out.api_key_set ? 'API Key: 已配置' : 'API Key: 未配置'), true);
      } catch (e) {
        setStatus('ppStatus', String(e), false);
      }
    });

    testPostprocess.addEventListener('click', async () => {
      try {
        const out = await req('/api/postprocess/test', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            base_url: ppBaseUrl.value || '',
            model: ppModel.value || '',
            api_key: ppApiKey.value || '',
            text: '连接测试'
          })
        });
        ppConnectionVerified = true;
        setStatus('ppStatus', '连接测试通过，可以启用并保存。', true);
      } catch (e) {
        ppConnectionVerified = false;
        setStatus('ppStatus', String(e), false);
      }
    });

    ppBaseUrl.addEventListener('input', markPostprocessDirty);
    ppModel.addEventListener('input', markPostprocessDirty);
    ppApiKey.addEventListener('input', markPostprocessDirty);
    ppSystemPrompt.addEventListener('input', markPostprocessDirty);
    ppEnabled.addEventListener('change', () => {
      if (ppEnabled.checked && !ppConnectionVerified) {
        setStatus('ppStatus', '请先测试连接，连接通过后再启用。', false);
      }
    });
    togglePpApiKey.addEventListener('click', () => {
      const show = ppApiKey.type === 'password';
      ppApiKey.type = show ? 'text' : 'password';
      togglePpApiKey.setAttribute('title', show ? '隐藏 API Key' : '显示 API Key');
      togglePpApiKey.setAttribute('aria-label', show ? '隐藏 API Key' : '显示 API Key');
    });

    loadHotText.addEventListener('click', async () => {
      try {
        const out = await req('/api/hotwords/text', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({text: hotText.value})
        });
        if (out.effective_text !== undefined) hotText.value = out.effective_text;
        setStatus('hotStatus', JSON.stringify(out, null, 2), true);
      } catch (e) {
        setStatus('hotStatus', String(e), false);
      }
    });

    loadHotFile.addEventListener('click', async () => {
      try {
        persistHotwordFilePath();
        const out = await req('/api/hotwords/file', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({path: hotFile.value})
        });
        if (out.effective_text !== undefined) hotText.value = out.effective_text;
        setStatus('hotStatus', JSON.stringify(out, null, 2), true);
      } catch (e) {
        setStatus('hotStatus', String(e), false);
      }
    });

    clearHot.addEventListener('click', async () => {
      try {
        const out = await req('/api/hotwords', {method:'DELETE'});
        hotText.value = '';
        hotFile.value = '';
        persistHotwordFilePath();
        setStatus('hotStatus', JSON.stringify(out, null, 2), true);
      } catch (e) {
        setStatus('hotStatus', String(e), false);
      }
    });

    hotFile.addEventListener('change', persistHotwordFilePath);
    hotFile.addEventListener('input', persistHotwordFilePath);
    restoreHotwordFilePath();

    refreshState();
    setInterval(refreshHotkeyState, 1000);
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


def create_controller_app(config_file: Path) -> FastAPI:
    app = FastAPI(title="VoiceType Controller", version="0.1.0")
    _setup_ui_file_logging()
    logger = logging.getLogger("voicetype.controller")
    recorder = ArecordManager()

    def _asr_base_url() -> str:
        cfg = load_runtime_config(config_file)
        return f"http://{cfg.host}:{cfg.port}"

    def _check_health(base_url: str) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code < 400:
                    return True, ""
                return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _extract_postprocess_text(resp_json: dict[str, object]) -> str:
        choices = resp_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("invalid response: missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("invalid response: invalid choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("invalid response: missing message")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(str(item["text"]))
            return "".join(parts).strip()
        raise ValueError("invalid response: missing content")

    def _postprocess_connectivity_check(base_url: str, model: str, api_key: str) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, headers=headers, json=payload)
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid JSON response: {exc}") from exc

        if resp.status_code != 200:
            err = ""
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    err = str(error.get("message", "")).strip()
                elif isinstance(error, str):
                    err = error.strip()
            raise ValueError(err or f"HTTP {resp.status_code}")

        if not isinstance(data, dict):
            raise ValueError("接口异常：响应格式错误")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("接口异常：缺少 choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("接口异常：choice 格式错误")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("接口异常：缺少 message")
        return "OK"

    def _postprocess_request_strict(
        base_url: str, model: str, api_key: str, system_prompt: str, text: str, language: str
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"原始识别文本：\n{text}\n语言参考：{language or 'zh'}",
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        processed = _extract_postprocess_text(data if isinstance(data, dict) else {})
        if not processed:
            raise ValueError("postprocess returned empty text")
        return processed

    def _run_postprocess(text: str, language: str) -> str:
        cfg = load_runtime_config(config_file)
        if not cfg.postprocess_enabled:
            return text
        base_url = cfg.postprocess_base_url.strip()
        model = cfg.postprocess_model.strip()
        api_key = cfg.postprocess_api_key.strip()
        system_prompt = (cfg.postprocess_system_prompt or "").strip() or POSTPROCESS_SYSTEM_PROMPT
        if not base_url or not model or not api_key:
            logger.warning("postprocess enabled but config incomplete; fallback to original text")
            return text

        try:
            return _postprocess_request_strict(base_url, model, api_key, system_prompt, text, language)
        except Exception as exc:  # noqa: BLE001
            logger.warning("postprocess failed, fallback to original text: %s", exc)
            return text

    def _asr_request(method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        # Keep recording on the local input machine, then forward audio to remote ASR.
        if method.upper() == "POST" and path == "/v1/recording/start":
            sample_rate = 16000
            language = "zh"
            if payload:
                if isinstance(payload.get("sample_rate"), int):
                    sample_rate = int(payload["sample_rate"])
                if isinstance(payload.get("language"), str):
                    language = str(payload["language"]).strip() or "zh"
            return _local_recording_start(sample_rate=sample_rate, language=language)
        if method.upper() == "POST" and path == "/v1/recording/stop":
            language = "zh"
            if payload and isinstance(payload.get("language"), str):
                language = str(payload["language"]).strip() or "zh"
            return _local_recording_stop_and_transcribe(language=language)

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

    def _local_recording_start(sample_rate: int = 16000, language: str = "zh") -> dict[str, object]:
        try:
            recorder.start(sample_rate=sample_rate, language=language)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"local recording start failed: {exc}") from exc
        return {"success": True, "recording": True, "sample_rate": sample_rate}

    def _local_recording_stop_and_transcribe(language: str = "zh") -> dict[str, object]:
        try:
            wav_path, default_lang, sample_rate = recorder.stop()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"local recording stop failed: {exc}") from exc

        try:
            samples = load_wav_file(wav_path, expected_sample_rate=sample_rate)
            audio_b64 = encode_pcm16_wav_base64(samples, sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"local wav load failed: {exc}") from exc

        req = {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "language": (language or default_lang or "zh"),
        }
        out = _asr_request("POST", "/v1/transcribe", req)
        if bool(out.get("success")) and isinstance(out.get("text"), str):
            raw_text = str(out.get("text", "")).strip()
            if raw_text:
                out["text"] = _run_postprocess(raw_text, str(out.get("language", language)))
        return out

    hotkey_bridge = X11GlobalHotkeyBridge(_asr_request, logger)

    @app.on_event("startup")
    def startup() -> None:
        cfg = load_runtime_config(config_file)
        hotkey_bridge.apply(cfg.global_hotkey_enabled, cfg.global_hotkey_key)

    @app.on_event("shutdown")
    def shutdown() -> None:
        hotkey_bridge.stop()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(render_controller_ui())

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
        safe_cfg = cfg.model_copy(
            update={
                "postprocess_api_key": "",
            }
        )
        active_service = (
            ASR_OPENVINO_SERVICE
            if str(cfg.backend).strip().lower() == "openvino"
            else ASR_TRANSFORMERS_SERVICE
        )
        active_code, _ = _run_systemctl_user("is-active", active_service)
        enabled_code, _ = _run_systemctl_user("is-enabled", active_service)
        managed: dict[str, dict[str, bool]] = {}
        for svc in sorted(MANAGED_ASR_SERVICES):
            a, _ = _run_systemctl_user("is-active", svc)
            e, _ = _run_systemctl_user("is-enabled", svc)
            managed[svc] = {"active": a == 0, "enabled": e == 0}
        return {
            "config": safe_cfg.model_dump(),
            "postprocess_api_key_set": bool(cfg.postprocess_api_key.strip()),
            "service": {
                "active": active_code == 0,
                "enabled": enabled_code == 0,
                "name": active_service,
                "active_service": active_service,
                "managed": managed,
            },
            "hotkey": hotkey_bridge.state(),
        }

    @app.get("/api/services")
    def services() -> dict[str, object]:
        managed: dict[str, dict[str, bool]] = {}
        for svc in sorted(MANAGED_ASR_SERVICES):
            a, _ = _run_systemctl_user("is-active", svc)
            e, _ = _run_systemctl_user("is-enabled", svc)
            managed[svc] = {"active": a == 0, "enabled": e == 0}
        return {"services": managed}

    @app.get("/api/devices")
    def devices() -> dict[str, object]:
        return {"devices": _detect_devices()}

    @app.get("/api/asr-base")
    def asr_base() -> dict[str, str]:
        return {"url": _asr_base_url()}

    @app.get("/api/connect-health")
    def connect_health() -> dict[str, object]:
        base_url = _asr_base_url()
        ok, err = _check_health(base_url)
        return {"base_url": base_url, "health_ok": ok, "health_error": err}

    @app.post("/v1/recording/start")
    def local_recording_start(req: RecordingStartRequest) -> dict[str, object]:
        # Compatibility endpoint for fcitx4/fcitx5 addon and global hotkey bridge.
        # Recording always happens on this local machine (input side).
        return _local_recording_start(sample_rate=req.sample_rate, language=req.language)

    @app.post("/v1/recording/stop")
    def local_recording_stop(req: dict[str, object] | None = None) -> dict[str, object]:
        # Compatibility endpoint for fcitx4/fcitx5 addon and global hotkey bridge.
        # Stop local recording and forward audio to remote ASR /v1/transcribe.
        language = "zh"
        if req and isinstance(req.get("language"), str):
            language = str(req["language"]).strip() or "zh"
        return _local_recording_stop_and_transcribe(language=language)

    @app.post("/api/connect-config")
    def connect_config(req: ConnectConfigRequest) -> dict[str, object]:
        host = req.host.strip()
        if not host:
            raise HTTPException(status_code=400, detail="host must not be empty")
        if req.port <= 0 or req.port > 65535:
            raise HTTPException(status_code=400, detail="port must be 1..65535")

        prev_cfg = load_runtime_config(config_file)
        cfg = prev_cfg.model_copy(update={"host": host, "port": req.port})
        saved = save_runtime_config(cfg, config_file)

        base_url = f"http://{host}:{req.port}"
        health_ok, health_error = _check_health(base_url)

        return {
            "success": True,
            "path": str(saved),
            "base_url": base_url,
            "health_ok": health_ok,
            "health_error": health_error,
        }

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
        model = req.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="model must not be empty")
        prev_cfg = load_runtime_config(config_file)
        cfg = prev_cfg.model_copy(
            update={
                "host": req.host,
                "port": req.port,
                "model": model,
                "device": req.device,
                "dtype": req.dtype,
                "hf_endpoint": req.hf_endpoint,
            }
        )
        saved = save_runtime_config(cfg, config_file)
        logger.info(
            "runtime config updated: model=%s device=%s dtype=%s port=%s",
            cfg.model,
            cfg.device,
            cfg.dtype,
            cfg.port,
        )
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

    @app.post("/api/postprocess/config")
    def postprocess_config(req: PostprocessConfigRequest) -> dict[str, object]:
        prev = load_runtime_config(config_file)
        base_url = req.base_url.strip()
        model = req.model.strip()
        system_prompt = req.system_prompt.strip() or POSTPROCESS_SYSTEM_PROMPT
        input_api_key = req.api_key.strip()
        api_key = input_api_key or prev.postprocess_api_key.strip()
        if req.enabled:
            if not base_url or not model or not api_key:
                raise HTTPException(status_code=400, detail="启用后处理前请先完整填写 Base URL / API Key / Model")
            try:
                _postprocess_connectivity_check(base_url, model, api_key)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="连接失败") from exc
        cfg = prev.model_copy(
            update={
                "postprocess_enabled": req.enabled,
                "postprocess_base_url": base_url,
                "postprocess_model": model,
                "postprocess_api_key": api_key,
                "postprocess_system_prompt": system_prompt,
            }
        )
        saved = save_runtime_config(cfg, config_file)
        return {
            "success": True,
            "path": str(saved),
            "api_key_set": bool(cfg.postprocess_api_key.strip()),
            "enabled": cfg.postprocess_enabled,
        }

    @app.post("/api/postprocess/test")
    def postprocess_test(req: PostprocessTestRequest) -> dict[str, object]:
        cfg = load_runtime_config(config_file)
        base_url = (req.base_url or "").strip() or cfg.postprocess_base_url.strip()
        model = (req.model or "").strip() or cfg.postprocess_model.strip()
        api_key = (req.api_key or "").strip() or cfg.postprocess_api_key.strip()
        raw = (req.text or "").strip() or "连接测试"
        if not base_url or not model or not api_key:
            raise HTTPException(status_code=400, detail="请先填写 Base URL / API Key / Model")
        try:
            reply = _postprocess_connectivity_check(base_url, model, api_key)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="连接失败") from exc
        return {"success": True, "message": "连接成功", "reply": reply}


    def _resolve_service(target: str | None = None) -> str:
        if target and target in MANAGED_ASR_SERVICES:
            return target
        cfg = load_runtime_config(config_file)
        if str(cfg.backend).strip().lower() == "openvino":
            return ASR_OPENVINO_SERVICE
        return ASR_TRANSFORMERS_SERVICE

    @app.post("/api/service/start")
    def start_service(service: str | None = None) -> dict[str, object]:
        resolved = _resolve_service(service)
        killed = _kill_unmanaged_asr_processes()
        code, output = _run_systemctl_user("start", resolved)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "start failed")
        logger.info("service started: %s", resolved)
        return {"success": True, "service": resolved, "killed_unmanaged": killed}

    @app.post("/api/service/restart")
    def restart_service(service: str | None = None) -> dict[str, object]:
        resolved = _resolve_service(service)
        killed = _kill_unmanaged_asr_processes()
        code, output = _run_systemctl_user("restart", resolved)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "restart failed")
        logger.info("service restarted: %s", resolved)
        return {"success": True, "service": resolved, "killed_unmanaged": killed}

    @app.post("/api/service/stop")
    def stop_service(service: str | None = None) -> dict[str, object]:
        resolved = _resolve_service(service)
        code, output = _run_systemctl_user("stop", resolved)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "stop failed")
        logger.info("service stopped: %s", resolved)
        return {"success": True, "service": resolved}

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
