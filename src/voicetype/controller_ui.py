from __future__ import annotations

import logging
import shutil
import subprocess
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
        cfg = AppConfig(
            host=req.host,
            port=req.port,
            model=req.model,
            device=req.device,
            hf_endpoint=req.hf_endpoint,
        )
        saved = save_runtime_config(cfg, config_file)
        logger.info("runtime config updated: model=%s device=%s port=%s", cfg.model, cfg.device, cfg.port)
        return {"success": True, "path": str(saved)}

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
