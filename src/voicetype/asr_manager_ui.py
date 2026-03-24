from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ASR_TRANSFORMERS_SERVICE = "asr-transformers.service"
ASR_OPENVINO_SERVICE = "asr-openvino.service"
ASR_MANAGER_UI_SERVICE = "asr-manager-ui.service"
ASR_SERVICES = [ASR_TRANSFORMERS_SERVICE, ASR_OPENVINO_SERVICE, ASR_MANAGER_UI_SERVICE]

CFG_DIR = Path.home() / ".config" / "asr-services"
TRANSFORMERS_ENV = CFG_DIR / "transformers.env"
OPENVINO_ENV = CFG_DIR / "openvino.env"
MANAGER_UI_ENV = CFG_DIR / "manager-ui.env"
STACK_ENV = CFG_DIR / "stack.env"


class TransformersConfigRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8789
    model: str = ""
    device: str = "cpu"
    dtype: str = "bfloat16"
    hf_endpoint: str = "https://hf-mirror.com"
    max_inference_batch_size: int = 1


class OpenVinoConfigRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8789
    model: str = ""
    device: str = "CPU"
    hf_endpoint: str = "https://hf-mirror.com"
    dtype: str = ""


class ManagerUiConfigRequest(BaseModel):
    ui_host: str = "127.0.0.1"
    ui_port: int = 8788


class ActivateBackendRequest(BaseModel):
    backend: str


class SimpleConfigRequest(BaseModel):
    backend: str = "transformers"
    host: str = "127.0.0.1"
    port: int = 8789
    model: str = ""
    device: str = "cpu"
    hf_endpoint: str = "https://hf-mirror.com"
    max_inference_batch_size: int = 1


def _run_systemctl_user(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def _run_journalctl(service: str, lines: int = 200) -> tuple[int, str]:
    proc = subprocess.run(
        ["journalctl", "--user", "-u", service, "-n", str(lines), "--no-pager"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def _ensure_service(name: str) -> str:
    if name not in ASR_SERVICES:
        raise HTTPException(status_code=400, detail=f"unsupported service: {name}")
    return name


def _service_status(name: str) -> dict[str, object]:
    active_code, active_text = _run_systemctl_user("is-active", name)
    enabled_code, enabled_text = _run_systemctl_user("is-enabled", name)
    return {
        "name": name,
        "active": active_code == 0,
        "active_state": active_text,
        "enabled": enabled_code == 0,
        "enabled_state": enabled_text,
    }


def _read_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_transformers_env() -> dict[str, str]:
    return {
        "HOST": "127.0.0.1",
        "PORT": "8789",
        "MODEL": "",
        "DEVICE": "cpu",
        "DTYPE": "bfloat16",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "MAX_INFERENCE_BATCH_SIZE": "1",
    }


def _default_openvino_env() -> dict[str, str]:
    return {
        "HOST": "127.0.0.1",
        "PORT": "8789",
        "MODEL": "",
        "DEVICE": "CPU",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "DTYPE": "",
    }


def _default_manager_ui_env() -> dict[str, str]:
    return {
        "UI_HOST": "127.0.0.1",
        "UI_PORT": "8788",
    }


def _default_stack_env() -> dict[str, str]:
    return {"BACKEND": "transformers"}


def _transformers_device_options() -> list[dict[str, str]]:
    options = [{"value": "cpu", "label": "CPU"}]
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                options.append({"value": f"cuda:{idx}", "label": f"GPU {idx} ({name})"})
    except Exception:
        pass
    return options


def _device_options(backend: str) -> list[dict[str, str]]:
    if backend == "openvino":
        return [{"value": "CPU", "label": "CPU"}]
    return _transformers_device_options()


def _normalize_device(backend: str, device: str) -> str:
    options = _device_options(backend)
    allowed = {x["value"] for x in options}
    if backend == "openvino":
        return "CPU"
    return device if device in allowed else "cpu"


def _current_configs() -> dict[str, dict[str, str]]:
    t = _default_transformers_env()
    t.update(_read_env(TRANSFORMERS_ENV))

    o = _default_openvino_env()
    o.update(_read_env(OPENVINO_ENV))

    m = _default_manager_ui_env()
    m.update(_read_env(MANAGER_UI_ENV))

    s = _default_stack_env()
    s.update(_read_env(STACK_ENV))

    return {
        "transformers": t,
        "openvino": o,
        "manager_ui": m,
        "stack": s,
    }


def render_ui() -> str:
    return """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>ASR 推理服务管理</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; background:#f5f7fb; color:#1f2937; font-family: \"Segoe UI\", \"PingFang SC\", \"Microsoft YaHei\", sans-serif; line-height:1.45; }
    .wrap { max-width: 1040px; margin: 24px auto; padding: 0 16px 28px; }
    h2 { margin: 0 0 14px; }
    h3 { margin: 0 0 12px; }
    .card { background:#fff; border:1px solid #dbe3ef; border-radius:12px; padding:18px; margin-bottom:16px; box-shadow:0 1px 2px rgba(15, 23, 42, 0.04); }
    .row { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-bottom:14px; }
    .row.one { grid-template-columns: minmax(0, 1fr); }
    .field { min-width: 0; }
    label { display:block; font-size:13px; color:#6b7280; margin-bottom:6px; }
    input, select { width:100%; border:1px solid #c7d2e5; border-radius:8px; padding:10px 12px; background:#fff; color:#111827; min-height:42px; min-width:0; }
    select { text-overflow: ellipsis; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; margin-bottom:10px; }
    button { border:1px solid #c7d2e5; border-radius:8px; padding:9px 14px; background:#fff; color:#111827; cursor:pointer; white-space:nowrap; min-height:40px; }
    .primary { background:#0f6fff; border-color:#0f6fff; color:#fff; }
    .danger { background:#dc2626; border-color:#dc2626; color:#fff; }
    .ok { color:#059669; font-weight:600; margin-top:6px; }
    .bad { color:#dc2626; font-weight:600; margin-top:6px; }
    .svc { padding:12px; border:1px solid #e5e7eb; border-radius:10px; margin-bottom:10px; background:#fafcff; }
    .svc-title { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px; }
    .badge { font-size:12px; padding:2px 8px; border-radius:999px; background:#e6f5ee; color:#047857; border:1px solid #a7e3c5; }
    .badge.off { background:#ffecec; color:#b42318; border-color:#f8b4b4; }
    pre { white-space:pre-wrap; background:#0c1220; color:#d7deed; border:1px solid #1f2a44; border-radius:8px; padding:10px; min-height:160px; max-height:420px; overflow:auto; }
    .muted { color:#6b7280; font-size:12px; margin-top:6px; }
    @media (max-width: 860px) {
      .row { grid-template-columns: minmax(0, 1fr); gap:10px; margin-bottom:12px; }
      .actions { gap:8px; }
      button { width:100%; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h2>ASR 推理服务管理</h2>

    <div class=\"card\">
      <div class=\"row one\"><div class=\"field\"><label>后端</label>
        <select id=\"backend\">
          <option value=\"transformers\">transformers</option>
          <option value=\"openvino\">openvino</option>
        </select>
      </div></div>
      <div class=\"row\">
        <div class=\"field\"><label>模型 ID / 路径</label><input id=\"model\" placeholder=\"Qwen/Qwen3-ASR-0.6B 或本地路径\"></div>
      </div>
      <div class=\"row\">
        <div class=\"field\"><label>Host</label><input id=\"host\" placeholder=\"127.0.0.1\"></div>
        <div class=\"field\"><label>Port</label><input id=\"port\" placeholder=\"8789\"></div>
      </div>
      <div class=\"row\">
        <div class=\"field\"><label>Device</label><select id=\"device\"></select></div>
        <div class=\"field\"><label>HF 镜像</label><input id=\"hf\" placeholder=\"https://hf-mirror.com\"></div>
      </div>
      <div class=\"row\" id=\"batchField\">
        <div class=\"field\"><label>最大推理批大小（仅 Transformers）</label><input id=\"batch\" type=\"number\" min=\"1\" value=\"1\" placeholder=\"1\"></div>
      </div>
      <div class=\"actions\">
        <button class=\"primary\" onclick=\"saveConfig()\">保存配置</button>
        <button onclick=\"startBackend()\">启动推理服务</button>
        <button onclick=\"restartBackend()\">重启推理服务</button>
        <button class=\"danger\" onclick=\"stopBackend()\">停止推理服务</button>
        <button onclick=\"refreshAll()\">刷新</button>
      </div>
      <div id=\"topStatus\"></div>
      <div class=\"muted\">首次使用：先保存配置再启动服务；已启动后修改参数：保存后需重载推理服务。</div>
    </div>

    <div class=\"card\">
      <h3>服务状态与操作</h3>
      <div id=\"serviceCards\"></div>
    </div>

    <div class=\"card\">
      <h3>日志输出</h3>
      <pre id=\"logs\"></pre>
    </div>
  </div>

  <script>
    const backend = document.getElementById('backend');
    const model = document.getElementById('model');
    const host = document.getElementById('host');
    const port = document.getElementById('port');
    const device = document.getElementById('device');
    const hf = document.getElementById('hf');
    const batch = document.getElementById('batch');
    const batchField = document.getElementById('batchField');

    async function req(url, opt) {
      const r = await fetch(url, opt);
      const t = await r.text();
      let j = {}; try { j = JSON.parse(t); } catch { j = { raw: t }; }
      if (!r.ok) throw new Error(j.detail || t || ('HTTP ' + r.status));
      return j;
    }
    function setTop(msg, ok=true) {
      const el = document.getElementById('topStatus');
      el.className = ok ? 'ok' : 'bad';
      el.textContent = msg;
    }
    function backendService(backend) {
      return backend === 'openvino' ? 'asr-openvino.service' : 'asr-transformers.service';
    }

    function toggleBatchField() {
      batchField.style.display = backend.value === 'transformers' ? '' : 'none';
    }

    async function loadDeviceOptions(selected = '') {
      const out = await req('/api/devices?backend=' + encodeURIComponent(backend.value));
      device.innerHTML = '';
      for (const item of out.devices || []) {
        const opt = document.createElement('option');
        opt.value = item.value;
        opt.textContent = item.label || item.value;
        device.appendChild(opt);
      }
      const values = Array.from(device.options).map(x => x.value);
      const prefer = selected && values.includes(selected) ? selected : (out.default_device || values[0] || '');
      if (prefer) device.value = prefer;
      device.disabled = backend.value === 'openvino';
    }

    async function reloadSystemd() {
      try {
        const out = await req('/api/systemd/reload', { method:'POST' });
        setTop('reload success: ' + out.timestamp, true);
      } catch (e) { setTop(String(e), false); }
    }

    async function saveConfig() {
      try {
        const payload = {
          backend: backend.value,
          host: host.value,
          port: Number(port.value),
          model: model.value,
          device: device.value,
          hf_endpoint: hf.value
        };
        if (backend.value === 'transformers') {
          payload.max_inference_batch_size = Number(batch.value || 1);
        }
        await req('/api/config/simple', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });
        setTop('配置已保存', true);
        await refreshAll();
      } catch (e) { setTop(String(e), false); }
    }

    async function startBackend() {
      try {
        await req('/api/services/' + backendService(backend.value) + '/start', { method:'POST' });
        setTop('推理服务已启动', true);
        await refreshAll();
      } catch (e) { setTop(String(e), false); }
    }

    async function restartBackend() {
      try {
        await req('/api/services/' + backendService(backend.value) + '/restart', { method:'POST' });
        setTop('推理服务已重启', true);
        await refreshAll();
      } catch (e) { setTop(String(e), false); }
    }

    async function stopBackend() {
      try {
        await req('/api/services/' + backendService(backend.value) + '/stop', { method:'POST' });
        setTop('推理服务已停止', true);
        await refreshAll();
      } catch (e) { setTop(String(e), false); }
    }

    async function action(name, op) {
      try {
        const out = await req('/api/services/' + encodeURIComponent(name) + '/' + op, { method:'POST' });
        setTop(name + ' ' + op + ' ok', true);
        document.getElementById('logs').textContent = out.output || '';
        await refreshAll();
      } catch (e) { setTop(String(e), false); }
    }

    async function viewLogs(name) {
      try {
        const out = await req('/api/services/' + encodeURIComponent(name) + '/logs?lines=200');
        document.getElementById('logs').textContent = out.output || '';
      } catch (e) { setTop(String(e), false); }
    }

    async function refreshAll() {
      try {
        const cfg = await req('/api/config');
        const t = cfg.transformers, o = cfg.openvino, s = cfg.stack;
        const b = (s.BACKEND || 'transformers');
        backend.value = b;
        const c = b === 'openvino' ? o : t;
        host.value = c.HOST || '';
        port.value = c.PORT || '';
        model.value = c.MODEL || '';
        hf.value = c.HF_ENDPOINT || '';
        batch.value = String(t.MAX_INFERENCE_BATCH_SIZE || '1');
        toggleBatchField();
        await loadDeviceOptions(c.DEVICE || '');

        const out = await req('/api/services');
        const box = document.getElementById('serviceCards');
        box.innerHTML = '';
        for (const svc of out.services) {
          const card = document.createElement('div');
          card.className = 'svc';
          const activeCls = svc.active ? 'badge' : 'badge off';
          const activeText = svc.active ? '运行中' : '未运行';
          card.innerHTML = `
            <div class=\"svc-title\"><b>${svc.name}</b><span class=\"${activeCls}\">${activeText}</span></div>
            <div>active: <b>${svc.active_state}</b> | enabled: <b>${svc.enabled_state}</b></div>
            <div style=\"margin-top:8px\">
              <button onclick=\"viewLogs('${svc.name}')\">查看日志</button>
            </div>`;
          box.appendChild(card);
        }
        setTop('', true);
      } catch (e) { setTop(String(e), false); }
    }

    backend.addEventListener('change', async () => {
      const cfg = await req('/api/config');
      const c = backend.value === 'openvino' ? cfg.openvino : cfg.transformers;
      host.value = c.HOST || '';
      port.value = c.PORT || '';
      model.value = c.MODEL || '';
      hf.value = c.HF_ENDPOINT || '';
      batch.value = String(cfg.transformers.MAX_INFERENCE_BATCH_SIZE || '1');
      toggleBatchField();
      await loadDeviceOptions(c.DEVICE || '');
    });

    refreshAll();
  </script>
</body>
</html>
"""


def create_asr_manager_app() -> FastAPI:
    app = FastAPI(title="ASR Manager UI", version="0.2.0")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(render_ui())

    @app.get("/ui", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        return HTMLResponse(render_ui())

    @app.get("/api/services")
    def services() -> dict[str, object]:
        return {"services": [_service_status(s) for s in ASR_SERVICES]}

    @app.get("/api/config")
    def config() -> dict[str, object]:
        return _current_configs()

    @app.get("/api/devices")
    def devices(backend: str = "transformers") -> dict[str, object]:
        b = backend.strip().lower()
        if b not in {"transformers", "openvino"}:
            raise HTTPException(status_code=400, detail=f"unsupported backend: {backend}")
        opts = _device_options(b)
        return {
            "backend": b,
            "devices": opts,
            "default_device": "CPU" if b == "openvino" else "cpu",
        }

    @app.post("/api/config/transformers")
    def config_transformers(req: TransformersConfigRequest) -> dict[str, object]:
        device = _normalize_device("transformers", req.device)
        _write_env(
            TRANSFORMERS_ENV,
            {
                "HOST": req.host,
                "PORT": str(req.port),
                "MODEL": req.model,
                "DEVICE": device,
                "DTYPE": req.dtype,
                "HF_ENDPOINT": req.hf_endpoint,
                "MAX_INFERENCE_BATCH_SIZE": str(req.max_inference_batch_size),
            },
        )
        return {"success": True, "path": str(TRANSFORMERS_ENV)}

    @app.post("/api/config/openvino")
    def config_openvino(req: OpenVinoConfigRequest) -> dict[str, object]:
        device = _normalize_device("openvino", req.device)
        _write_env(
            OPENVINO_ENV,
            {
                "HOST": req.host,
                "PORT": str(req.port),
                "MODEL": req.model,
                "DEVICE": device,
                "HF_ENDPOINT": req.hf_endpoint,
                "DTYPE": req.dtype,
            },
        )
        return {"success": True, "path": str(OPENVINO_ENV)}

    @app.post("/api/config/simple")
    def config_simple(req: SimpleConfigRequest) -> dict[str, object]:
        backend = req.backend.strip().lower()
        if backend not in {"transformers", "openvino"}:
            raise HTTPException(status_code=400, detail=f"unsupported backend: {req.backend}")
        if req.max_inference_batch_size < 1:
            raise HTTPException(status_code=400, detail="max_inference_batch_size must be >= 1")
        normalized_device = _normalize_device(backend, req.device)

        cfg = _current_configs()
        if backend == "transformers":
            base = _default_transformers_env()
            base.update(cfg["transformers"])
            base.update(
                {
                    "HOST": req.host,
                    "PORT": str(req.port),
                    "MODEL": req.model,
                    "DEVICE": normalized_device,
                    "HF_ENDPOINT": req.hf_endpoint,
                    "MAX_INFERENCE_BATCH_SIZE": str(req.max_inference_batch_size),
                }
            )
            _write_env(TRANSFORMERS_ENV, base)
        else:
            base = _default_openvino_env()
            base.update(cfg["openvino"])
            base.update(
                {
                    "HOST": req.host,
                    "PORT": str(req.port),
                    "MODEL": req.model,
                    "DEVICE": normalized_device,
                    "HF_ENDPOINT": req.hf_endpoint,
                }
            )
            _write_env(OPENVINO_ENV, base)

        _write_env(STACK_ENV, {"BACKEND": backend})
        return {"success": True, "backend": backend}

    @app.post("/api/config/manager-ui")
    def config_manager_ui(req: ManagerUiConfigRequest) -> dict[str, object]:
        _write_env(
            MANAGER_UI_ENV,
            {
                "UI_HOST": req.ui_host,
                "UI_PORT": str(req.ui_port),
            },
        )
        return {"success": True, "path": str(MANAGER_UI_ENV)}

    @app.post("/api/config/activate-backend")
    def activate_backend(req: ActivateBackendRequest) -> dict[str, object]:
        backend = req.backend.strip().lower()
        if backend not in {"transformers", "openvino"}:
            raise HTTPException(status_code=400, detail=f"unsupported backend: {req.backend}")
        _write_env(STACK_ENV, {"BACKEND": backend})
        return {"success": True, "backend": backend, "path": str(STACK_ENV)}

    @app.post("/api/systemd/reload")
    def systemd_reload() -> dict[str, object]:
        code, output = _run_systemctl_user("daemon-reload")
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "daemon-reload failed")
        return {"success": True, "timestamp": datetime.now().isoformat(timespec="seconds"), "output": output}

    @app.post("/api/services/{name}/start")
    def start_service(name: str) -> dict[str, object]:
        svc = _ensure_service(name)
        code, output = _run_systemctl_user("start", svc)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "start failed")
        return {"success": True, "service": svc, "output": output}

    @app.post("/api/services/{name}/stop")
    def stop_service(name: str) -> dict[str, object]:
        svc = _ensure_service(name)
        code, output = _run_systemctl_user("stop", svc)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "stop failed")
        return {"success": True, "service": svc, "output": output}

    @app.post("/api/services/{name}/restart")
    def restart_service(name: str) -> dict[str, object]:
        svc = _ensure_service(name)
        code, output = _run_systemctl_user("restart", svc)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "restart failed")
        return {"success": True, "service": svc, "output": output}

    @app.get("/api/services/{name}/status")
    def service_status(name: str) -> dict[str, object]:
        svc = _ensure_service(name)
        return _service_status(svc)

    @app.get("/api/services/{name}/logs")
    def service_logs(name: str, lines: int = 200) -> dict[str, object]:
        svc = _ensure_service(name)
        lines = min(max(lines, 10), 2000)
        code, output = _run_journalctl(svc, lines)
        if code != 0:
            raise HTTPException(status_code=500, detail=output or "journalctl failed")
        return {"success": True, "service": svc, "lines": lines, "output": output}

    return app
