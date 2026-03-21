from __future__ import annotations


def render_ui() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>VoiceType 控制台</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --line:#334155; --txt:#e2e8f0; --muted:#94a3b8; --ok:#16a34a; --warn:#f59e0b; --bad:#ef4444; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; background:linear-gradient(160deg,#020617,#111827 45%,#1f2937); color:var(--txt); }
    .wrap { max-width: 980px; margin: 28px auto; padding: 0 14px; }
    h1 { margin: 0 0 18px; font-size: 26px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .card { border:1px solid var(--line); background:rgba(17,24,39,.9); border-radius: 14px; padding: 14px; }
    .card h2 { margin:0 0 10px; font-size: 16px; color:#cbd5e1; }
    label { display:block; font-size:13px; color:var(--muted); margin:10px 0 6px; }
    input, select, textarea { width:100%; border:1px solid #475569; background:#0b1220; color:#e5e7eb; border-radius:10px; padding:10px; font-size:14px; }
    textarea { min-height: 132px; resize: vertical; }
    .row { display:flex; gap:10px; }
    .row > * { flex:1; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; margin-top: 12px; }
    button { border:1px solid #475569; background:#1e293b; color:#e2e8f0; border-radius:10px; padding:9px 12px; cursor:pointer; }
    button.primary { background:#2563eb; border-color:#2563eb; }
    button.warn { background:#d97706; border-color:#d97706; }
    .status { margin: 10px 0 0; font-size: 13px; color: var(--muted); white-space: pre-wrap; }
    .status.ok { color: #4ade80; }
    .status.bad { color: #f87171; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>VoiceType 本地配置面板</h1>
    <div class="grid">
      <section class="card">
        <h2>运行配置</h2>
        <label>模型（HF repo id 或本地绝对路径）</label>
        <input id="model" placeholder="Qwen/Qwen3-ASR-0.6B 或 /abs/path/to/model" />
        <div class="row">
          <div>
            <label>设备</label>
            <input id="device" placeholder="cpu 或 cuda:0" />
          </div>
          <div>
            <label>默认语言（留空=自动识别）</label>
            <input id="language" placeholder="例如：zh / en / yue（留空自动）" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>最大推理批大小</label>
            <input id="batch" type="number" min="1" />
          </div>
          <div>
            <label>HF 镜像（可空）</label>
            <input id="hf_endpoint" placeholder="https://hf-mirror.com" />
          </div>
        </div>
        <div class="btns">
          <button class="primary" id="saveRuntime">保存并热重载模型</button>
          <button class="warn" id="restartSvc">重载 systemd 服务</button>
        </div>
        <div id="cfgStatus" class="status"></div>
      </section>

      <section class="card">
        <h2>热词维护</h2>
        <label>热词输入（每行一个词，格式：词 或 词 权重）</label>
        <textarea id="hotwordsText" class="mono">pytorch 50
centos 20
VoiceType 40</textarea>
        <div class="row">
          <div>
            <label>合并模式</label>
            <select id="merge">
              <option value="false">false（覆盖）</option>
              <option value="true">true（合并）</option>
            </select>
          </div>
          <div>
            <label>分类名</label>
            <input id="category" value="custom" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>从本地文件加载（服务器文件路径）</label>
            <input id="hotwordsFile" placeholder="/path/to/hotwords.txt" />
          </div>
        </div>
        <div class="btns">
          <button class="primary" id="loadHotwordsText">加载输入框热词</button>
          <button id="loadHotwordsFile">从文件加载热词</button>
          <button id="clearHotwords">清空热词</button>
        </div>
        <div id="hotStatus" class="status"></div>
      </section>
    </div>

    <section class="card" style="margin-top:14px">
      <h2>当前状态</h2>
      <div id="state" class="status mono"></div>
    </section>
  </div>

  <script>
    async function req(url, opt) {
      const r = await fetch(url, opt);
      const text = await r.text();
      let json;
      try { json = JSON.parse(text); } catch { json = { raw: text }; }
      if (!r.ok) {
        throw new Error(json.detail || json.error || text || ("HTTP " + r.status));
      }
      return json;
    }
    function setStatus(id, msg, ok=true) {
      const el = document.getElementById(id);
      el.textContent = msg;
      el.className = "status " + (ok ? "ok" : "bad");
    }
    async function refreshState() {
      const s = await req("/v1/ui/state");
      document.getElementById("model").value = s.config.model || "";
      document.getElementById("device").value = s.config.device || "cpu";
      document.getElementById("language").value = s.config.default_language || "";
      document.getElementById("batch").value = s.config.max_inference_batch_size || 1;
      document.getElementById("hf_endpoint").value = s.config.hf_endpoint || "";
      document.getElementById("state").textContent = JSON.stringify(s, null, 2);
    }
    document.getElementById("saveRuntime").onclick = async () => {
      try {
        const body = {
          model: document.getElementById("model").value,
          device: document.getElementById("device").value,
          default_language: document.getElementById("language").value,
          max_inference_batch_size: parseInt(document.getElementById("batch").value || "1", 10),
          hf_endpoint: document.getElementById("hf_endpoint").value || null
        };
        const data = await req("/v1/ui/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        setStatus("cfgStatus", "已应用运行配置\\n" + JSON.stringify(data, null, 2), true);
        await refreshState();
      } catch (e) {
        setStatus("cfgStatus", String(e), false);
      }
    };
    document.getElementById("restartSvc").onclick = async () => {
      try {
        const data = await req("/v1/ui/service/restart", { method: "POST" });
        setStatus("cfgStatus", "服务已重载\\n" + JSON.stringify(data, null, 2), true);
      } catch (e) {
        setStatus("cfgStatus", String(e), false);
      }
    };
    document.getElementById("loadHotwordsText").onclick = async () => {
      try {
        const text = document.getElementById("hotwordsText").value;
        const category = document.getElementById("category").value.trim() || "custom";
        const merge = document.getElementById("merge").value === "true";
        const data = await req("/v1/ui/hotwords/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, merge, category })
        });
        setStatus("hotStatus", "热词已加载\\n" + JSON.stringify(data, null, 2), true);
        await refreshState();
      } catch (e) {
        setStatus("hotStatus", String(e), false);
      }
    };
    document.getElementById("loadHotwordsFile").onclick = async () => {
      try {
        const path = document.getElementById("hotwordsFile").value.trim();
        const merge = document.getElementById("merge").value === "true";
        const data = await req("/v1/ui/hotwords/file", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, merge })
        });
        setStatus("hotStatus", "热词文件已加载\\n" + JSON.stringify(data, null, 2), true);
        await refreshState();
      } catch (e) {
        setStatus("hotStatus", String(e), false);
      }
    };
    document.getElementById("clearHotwords").onclick = async () => {
      try {
        const data = await req("/v1/hotwords", { method: "DELETE" });
        setStatus("hotStatus", "热词已清空\\n" + JSON.stringify(data, null, 2), true);
        await refreshState();
      } catch (e) {
        setStatus("hotStatus", String(e), false);
      }
    };
    refreshState().catch(e => setStatus("cfgStatus", String(e), false));
  </script>
</body>
</html>
"""
