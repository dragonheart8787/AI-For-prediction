#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedPredictor HTTP 服務（僅標準函式庫：wsgiref + json）
適合本機／內網部署，無需 FastAPI/torch。

端點：
  GET  /                瀏覽器操作介面（選模型、填特徵、預測）
  GET  /health          健康檢查
  GET  /v1/models       列出 models 目錄內可載入的 task_*.pkl
  POST /v1/load_model   {"task_id": "stock_price_next"} 或 "__universal__"
  GET  /v1/model/info   模型摘要（含 feature_names）
  POST /v1/predict      {"X": [[...]], "domain": "..."} 或 {"rows": [{...},...]}（物件鍵＝特徵名，可含字串會編碼）
  POST /v1/predict_many {"X": [[...]], "domain": "...", "batch_size": 1024}
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_PREDICT_ROWS = int(os.environ.get("PREDICT_AI_HTTP_MAX_ROWS", "20000"))
MAX_BATCH_CAP = int(os.environ.get("PREDICT_AI_HTTP_MAX_BATCH", "8192"))


def _validate_X_matrix(X: Any) -> Optional[str]:
    """回傳錯誤碼字串；通過回傳 None。"""
    if not isinstance(X, list):
        return "X_must_be_list"
    if len(X) > MAX_PREDICT_ROWS:
        return "X_too_many_rows"
    for row in X:
        if not isinstance(row, (list, tuple)):
            return "X_row_must_be_list"
        for v in row:
            if isinstance(v, bool):
                return "X_values_must_be_numeric"
            if not isinstance(v, (int, float)):
                return "X_values_must_be_numeric"
    return None


def _validate_rows_objects(rows: Any) -> Optional[str]:
    """``rows`` 為物件陣列；值可為數字／字串／布林（由模型端編碼）。"""
    if not isinstance(rows, list):
        return "rows_must_be_list"
    if len(rows) > MAX_PREDICT_ROWS:
        return "rows_too_many_rows"
    for row in rows:
        if not isinstance(row, dict):
            return "row_must_be_object"
    return None


def _json_response(
    status: int,
    obj: Any,
    cors: bool = True,
    extra_headers: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    if cors:
        headers.append(("Access-Control-Allow-Origin", "*"))
    if extra_headers:
        headers.extend(extra_headers)
    return status, headers, body


def _parse_json_body(raw: bytes) -> Any:
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _html_response(
    status: int, html: str, cors: bool = True
) -> Tuple[int, List[Tuple[str, str]], bytes]:
    body = html.encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    if cors:
        headers.append(("Access-Control-Allow-Origin", "*"))
    return status, headers, body


def _predict_ui_page_html(model_path: str, model_loaded: bool, models_root: str) -> str:
    """單頁操作介面：選模型、動態特徵欄位、呼叫 /v1/predict。"""
    safe_path = html.escape(model_path or "（尚未載入）", quote=True)
    safe_root = html.escape(models_root, quote=True)
    status = (
        '<p class="ok">目前狀態：模型已載入，可直接預測。</p>'
        if model_loaded
        else '<p class="warn">目前狀態：尚未載入模型 — 請從下方選單選擇任務後按「載入模型」。</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>預測 AI — 操作介面</title>
<style>
:root {{ --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9cb3; --acc:#3d8bfd; --ok:#3fb950; --warn:#d4a72c; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", "Microsoft JhengHei", system-ui, sans-serif; background: var(--bg); color: var(--text);
  margin: 0; padding: 1rem 1.25rem 2rem; line-height: 1.5; max-width: 42rem; margin-left: auto; margin-right: auto; }}
h1 {{ font-size: 1.35rem; font-weight: 600; margin-bottom: 0.5rem; }}
p.ok {{ color: var(--ok); font-size: 0.9rem; }}
p.warn {{ color: var(--warn); font-size: 0.9rem; }}
.card {{ background: var(--card); border-radius: 12px; padding: 1rem 1.1rem; margin: 1rem 0; border: 1px solid #2d3a4d; }}
label {{ display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem; }}
select, input[type="text"] {{ width: 100%; padding: 0.5rem 0.6rem; border-radius: 8px; border: 1px solid #3d4f66;
  background: #0d1117; color: var(--text); font-size: 0.95rem; }}
.row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: flex-end; margin-top: 0.75rem; }}
.row select {{ flex: 1; min-width: 12rem; }}
button {{ cursor: pointer; border: none; border-radius: 8px; padding: 0.55rem 1rem; font-size: 0.9rem; font-weight: 600;
  background: var(--acc); color: #fff; }}
button.secondary {{ background: #30363d; color: var(--text); }}
button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
#featureInputs {{ display: grid; gap: 0.65rem; }}
.feat-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; align-items: center; }}
@media (max-width: 520px) {{ .feat-row {{ grid-template-columns: 1fr; }} }}
pre#out {{ background: #0d1117; border-radius: 8px; padding: 0.85rem; overflow: auto; font-size: 0.8rem; border: 1px solid #30363d; white-space: pre-wrap; word-break: break-all; }}
footer {{ font-size: 0.75rem; color: var(--muted); margin-top: 1.5rem; }}
footer code {{ font-size: 0.7rem; }}
</style>
</head>
<body>
<h1>預測 AI 操作介面</h1>
{status}
<div class="card">
  <label for="taskSel">模型／任務（來自 models 目錄）</label>
  <div class="row">
    <select id="taskSel" aria-label="任務"></select>
    <button type="button" id="btnLoad">載入模型</button>
  </div>
  <p id="loadMsg" style="font-size:0.85rem;color:var(--muted);margin:0.6rem 0 0;"></p>
</div>
<div class="card" id="featCard" hidden>
  <h2 style="font-size:1rem;margin:0 0 0.75rem;">輸入特徵</h2>
  <label for="domainIn">domain（領域）</label>
  <input type="text" id="domainIn" value="financial" placeholder="financial / weather / medical / custom">
  <div id="featureInputs" style="margin-top:0.75rem;"></div>
  <div class="row" style="margin-top:1rem;">
    <button type="button" id="btnDemo">填入示範值</button>
    <button type="button" id="btnPredict">預測（表單）</button>
  </div>
</div>
<div class="card" id="jsonCard" hidden>
  <h2 style="font-size:1rem;margin:0 0 0.35rem;">進階：JSON 批次（欄位名 + 可含文字）</h2>
  <p style="font-size:0.78rem;color:var(--muted);margin:0 0 0.5rem;line-height:1.45;">
    每列為一個物件，<strong>鍵名</strong>須與上方特徵名相同。數字可直接寫；<strong>文字</strong>會變成固定 0～1 編碼（與訓練時語意未對齊則預測僅供實驗）。也可貼多筆陣列一次預測。
  </p>
  <textarea id="jsonRows" rows="8" style="width:100%;font-family:ui-monospace,monospace;font-size:0.78rem;border-radius:8px;border:1px solid #3d4f66;background:#0d1117;color:var(--text);padding:0.5rem;resize:vertical;"></textarea>
  <div class="row" style="margin-top:0.65rem;">
    <button type="button" class="secondary" id="btnJsonExample">從表單填入 JSON</button>
    <button type="button" id="btnPredictJson">送出 JSON 預測</button>
  </div>
</div>
<div class="card" id="outCard" hidden>
  <h2 style="font-size:1rem;margin:0 0 0.5rem;">結果</h2>
  <pre id="out"></pre>
</div>
<footer>
  <p>模型目錄：<code>{safe_root}</code></p>
  <p>目前檔案：<code>{safe_path}</code></p>
  <p>API：<code>GET /v1/models</code> · <code>POST /v1/load_model</code> · <code>POST /v1/predict</code>（<code>X</code> 矩陣或 <code>rows</code> 物件陣列）</p>
</footer>
<script>
(function () {{
  const taskSel = document.getElementById('taskSel');
  const btnLoad = document.getElementById('btnLoad');
  const loadMsg = document.getElementById('loadMsg');
  const featCard = document.getElementById('featCard');
  const featureInputs = document.getElementById('featureInputs');
  const domainIn = document.getElementById('domainIn');
  const btnPredict = document.getElementById('btnPredict');
  const btnDemo = document.getElementById('btnDemo');
  const jsonCard = document.getElementById('jsonCard');
  const jsonRows = document.getElementById('jsonRows');
  const btnJsonExample = document.getElementById('btnJsonExample');
  const btnPredictJson = document.getElementById('btnPredictJson');
  const outCard = document.getElementById('outCard');
  const outPre = document.getElementById('out');

  const demoDefaults = {{
    open: 185, high: 187, low: 184, close: 186, volume: 8500000,
    new_cases: 350, new_deaths: 5, max_temp: 29, min_temp: 22, temp: 25, temperature: 25,
    sentiment: 0.15, demand: 12000, load: 11500
  }};

  function keyName(name) {{
    const parts = String(name).toLowerCase().split('__');
    return parts[parts.length - 1];
  }}

  async function refreshTaskList() {{
    loadMsg.textContent = '正在讀取模型清單…';
    try {{
      const r = await fetch('/v1/models');
      const j = await r.json();
      taskSel.innerHTML = '';
      const tasks = j.tasks || [];
      if (!tasks.length) {{
        loadMsg.textContent = '找不到 task_*.pkl，請先訓練模型。';
        return;
      }}
      tasks.forEach(function (t) {{
        const opt = document.createElement('option');
        opt.value = t.task_id;
        const lab = t.label || t.task_id;
        opt.textContent = lab + ' (' + t.filename + ')';
        taskSel.appendChild(opt);
      }});
      loadMsg.textContent = '請選擇任務後按「載入模型」，或若伺服器已載入模型將自動顯示特徵欄。';
    }} catch (e) {{
      loadMsg.textContent = '無法取得 /v1/models：' + e;
    }}
  }}

  async function bootstrapIfServerHasModel() {{
    try {{
      const ir = await fetch('/v1/model/info');
      if (!ir.ok) return;
      const info = await ir.json();
      buildFeatureForm(info);
      fillDemo();
      syncJsonFromForm();
      loadMsg.textContent = '伺服器已載入模型，可調整特徵後按「預測」。若要換模型請選擇任務後按「載入模型」。';
    }} catch (e) {{}}
  }}

  function buildFeatureForm(info) {{
    featureInputs.innerHTML = '';
    const names = info.feature_names || [];
    const dim = info.input_dim || names.length || 0;
    const useNames = names.length === dim && dim > 0;
    for (let i = 0; i < dim; i++) {{
      const row = document.createElement('div');
      row.className = 'feat-row';
      const nm = useNames ? names[i] : ('特徵_' + (i + 1));
      const label = document.createElement('label');
      label.textContent = nm;
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.dataset.feature = nm;
      inp.id = 'f_' + i;
      inp.placeholder = '數值';
      row.appendChild(label);
      row.appendChild(inp);
      featureInputs.appendChild(row);
    }}
    featCard.hidden = dim === 0;
    if (jsonCard) jsonCard.hidden = dim === 0;
  }}

  function syncJsonFromForm() {{
    if (!jsonRows) return;
    const inputs = featureInputs.querySelectorAll('input[type=text]');
    if (!inputs.length) return;
    const o = {{}};
    inputs.forEach(function (inp) {{
      const key = inp.dataset.feature;
      const raw = String(inp.value).replace(/,/g, '').trim();
      const v = parseFloat(raw);
      o[key] = Number.isNaN(v) ? (raw || '') : v;
    }});
    jsonRows.value = JSON.stringify([o], null, 2);
  }}

  function fillDemo() {{
    const inputs = featureInputs.querySelectorAll('input[type=text]');
    inputs.forEach(function (inp) {{
      const k = keyName(inp.dataset.feature || '');
      if (demoDefaults[k] !== undefined) inp.value = demoDefaults[k];
      else inp.value = (1 + inputs.length * 0.01).toFixed(3);
    }});
  }}

  async function loadModel() {{
    const tid = taskSel.value;
    if (!tid) return;
    loadMsg.textContent = '載入中…';
    btnLoad.disabled = true;
    try {{
      const r = await fetch('/v1/load_model', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ task_id: tid }})
      }});
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || j.detail || r.status);
      loadMsg.textContent = '已載入：' + (j.model_path || '');
      const ir = await fetch('/v1/model/info');
      const info = await ir.json();
      if (!ir.ok) throw new Error(info.error || 'info');
      buildFeatureForm(info);
      fillDemo();
      syncJsonFromForm();
      outCard.hidden = true;
    }} catch (e) {{
      loadMsg.textContent = '載入失敗：' + e;
      featCard.hidden = true;
      if (jsonCard) jsonCard.hidden = true;
    }}
    btnLoad.disabled = false;
  }}

  async function runPredict() {{
    const inputs = featureInputs.querySelectorAll('input[type=text]');
    const row = [];
    for (let i = 0; i < inputs.length; i++) {{
      const v = parseFloat(String(inputs[i].value).replace(/,/g, ''));
      if (Number.isNaN(v)) {{
        alert('請為所有特徵輸入有效數字：' + inputs[i].dataset.feature);
        return;
      }}
      row.push(v);
    }}
    outPre.textContent = '計算中…';
    outCard.hidden = false;
    btnPredict.disabled = true;
    try {{
      const r = await fetch('/v1/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ X: [row], domain: domainIn.value || 'custom' }})
      }});
      const j = await r.json();
      outPre.textContent = JSON.stringify(j, null, 2);
    }} catch (e) {{
      outPre.textContent = String(e);
    }}
    btnPredict.disabled = false;
  }}

  async function runPredictJson() {{
    if (!jsonRows) return;
    let rows;
    try {{
      rows = JSON.parse(jsonRows.value.trim());
    }} catch (e) {{
      alert('JSON 無法解析：' + e);
      return;
    }}
    if (!Array.isArray(rows)) rows = [rows];
    outPre.textContent = '計算中…';
    outCard.hidden = false;
    btnPredictJson.disabled = true;
    try {{
      const r = await fetch('/v1/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ rows: rows, domain: domainIn.value || 'custom' }})
      }});
      const j = await r.json();
      outPre.textContent = JSON.stringify(j, null, 2);
    }} catch (e) {{
      outPre.textContent = String(e);
    }}
    btnPredictJson.disabled = false;
  }}

  btnLoad.addEventListener('click', loadModel);
  btnPredict.addEventListener('click', runPredict);
  btnDemo.addEventListener('click', function () {{ fillDemo(); syncJsonFromForm(); }});
  if (btnJsonExample) btnJsonExample.addEventListener('click', syncJsonFromForm);
  if (btnPredictJson) btnPredictJson.addEventListener('click', runPredictJson);
  refreshTaskList().then(bootstrapIfServerHasModel);
}})();
</script>
</body>
</html>"""


class ModelRegistry:
    """管理 models 目錄與目前載入的 pickle；支援執行中切換任務。"""

    def __init__(self, models_root: str, initial_path: Optional[str] = None) -> None:
        self.models_root = os.path.abspath(models_root)
        self.initial_path = (
            os.path.abspath(initial_path) if initial_path and os.path.isfile(initial_path) else None
        )
        self.model_path = self.initial_path or ""
        self._predictor: Any = None

    def list_available_tasks(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        root = self.models_root
        if not os.path.isdir(root):
            return out
        for fn in sorted(os.listdir(root)):
            if fn.startswith("task_") and fn.endswith(".pkl"):
                tid = fn[5:-4]
                out.append({"task_id": tid, "filename": fn})
        uni = os.path.join(root, "universal_model.pkl")
        if os.path.isfile(uni):
            out.append(
                {
                    "task_id": "__universal__",
                    "filename": "universal_model.pkl",
                    "label": "universal（通用模型）",
                }
            )
        return out

    def path_for_task_id(self, task_id: str) -> str:
        if task_id == "__universal__":
            fn = "universal_model.pkl"
        else:
            fn = f"task_{task_id}.pkl"
        p = os.path.join(self.models_root, fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return p

    def load(self, path: Optional[str] = None) -> None:
        from unified_predict import UnifiedPredictor

        target = os.path.abspath(path or self.model_path or "")
        if not target or not os.path.isfile(target):
            raise FileNotFoundError(f"模型不存在: {target}")
        self._predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
        self._predictor.load_model(target)
        self.model_path = target
        logger.info("已載入模型: %s", target)

    def load_task_id(self, task_id: str) -> None:
        self.load(self.path_for_task_id(task_id))

    def predictor(self) -> Any:
        if self._predictor is None:
            raise RuntimeError("模型未載入")
        return self._predictor


class UnifiedPredictHTTPApp:
    """WSGI 應用：載入 pickle 模型並提供預測 JSON API 與瀏覽器 UI。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        models_root: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        mp: Optional[str] = None
        if model_path:
            mp = os.path.abspath(os.path.expanduser(str(model_path)))
            if not os.path.isfile(mp):
                mp = None
        if models_root is not None:
            mr = os.path.abspath(os.path.expanduser(str(models_root)))
        elif mp:
            mr = os.path.dirname(mp)
        else:
            try:
                from runtime_paths import resolve_models_dir

                mr = str(resolve_models_dir(None))
            except Exception:
                mr = os.path.abspath("models")
        self._registry = ModelRegistry(models_root=mr, initial_path=mp)
        self.model_path = self._registry.model_path
        if auto_load:
            reg = self._registry
            if reg.initial_path:
                reg.load(reg.initial_path)
            else:
                tasks = reg.list_available_tasks()
                pref = next((t for t in tasks if t["task_id"] == "stock_price_next"), None)
                pick = pref or (tasks[0] if tasks else None)
                if pick:
                    try:
                        reg.load(os.path.join(reg.models_root, pick["filename"]))
                    except Exception as ex:
                        logger.warning("自動載入模型略過: %s", ex)

    def load(self, path: Optional[str] = None) -> None:
        self._registry.load(path)

    @property
    def _predictor(self) -> Any:
        return self._registry._predictor

    @property
    def predictor(self) -> Any:
        return self._registry.predictor()

    def handle_request(
        self,
        method: str,
        path: str,
        raw_body: bytes,
        request_id: Optional[str] = None,
    ) -> Tuple[int, List[Tuple[str, str]], bytes]:
        path = path.split("?", 1)[0].rstrip("/") or "/"
        rid_h: List[Tuple[str, str]] = (
            [("X-Request-ID", str(request_id))] if request_id else []
        )

        if method == "GET" and path in ("/", "/index.html"):
            html = _predict_ui_page_html(
                self._registry.model_path,
                model_loaded=self._predictor is not None,
                models_root=self._registry.models_root,
            )
            return _html_response(200, html)

        if method == "GET" and path == "/favicon.ico":
            return (
                204,
                [
                    ("Access-Control-Allow-Origin", "*"),
                    ("Content-Length", "0"),
                ],
                b"",
            )

        if method == "GET" and path == "/health":
            ok = self._registry._predictor is not None
            return _json_response(
                200,
                {
                    "status": "ok" if ok else "no_model",
                    "model_path": self._registry.model_path,
                    "models_root": self._registry.models_root,
                },
                extra_headers=rid_h,
            )

        if method == "GET" and path == "/v1/models":
            tasks = self._registry.list_available_tasks()
            return _json_response(
                200,
                {
                    "models_dir": self._registry.models_root,
                    "tasks": tasks,
                },
                extra_headers=rid_h,
            )

        if method == "POST" and path == "/v1/load_model":
            try:
                data = _parse_json_body(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return _json_response(
                    400, {"error": "invalid_json", "detail": str(e)}, extra_headers=rid_h
                )
            if not isinstance(data, dict) or not data.get("task_id"):
                return _json_response(
                    400, {"error": "missing_task_id"}, extra_headers=rid_h
                )
            tid = str(data["task_id"])
            try:
                self._registry.load_task_id(tid)
            except FileNotFoundError:
                return _json_response(
                    404,
                    {"error": "model_file_not_found", "task_id": tid},
                    extra_headers=rid_h,
                )
            except Exception as e:
                logger.exception("load_model failed")
                return _json_response(
                    500, {"error": "load_failed", "detail": str(e)}, extra_headers=rid_h
                )
            return _json_response(
                200,
                {"ok": True, "model_path": self._registry.model_path, "task_id": tid},
                extra_headers=rid_h,
            )

        if self._registry._predictor is None:
            return _json_response(
                503,
                {
                    "error": "model_not_loaded",
                    "model_path": self._registry.model_path,
                    "hint": "POST /v1/load_model with task_id or open / to use UI",
                },
                extra_headers=rid_h,
            )

        if method == "GET" and path == "/v1/model/info":
            return _json_response(200, self.predictor.get_model_info(), extra_headers=rid_h)

        if method == "POST" and path == "/v1/predict":
            try:
                data = _parse_json_body(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return _json_response(
                    400, {"error": "invalid_json", "detail": str(e)}, extra_headers=rid_h
                )
            if not isinstance(data, dict):
                return _json_response(400, {"error": "body_must_be_object"}, extra_headers=rid_h)
            has_x = "X" in data
            has_rows = "rows" in data
            if (not has_x and not has_rows) or (has_x and has_rows):
                return _json_response(
                    400,
                    {"error": "provide_either_X_or_rows", "detail": "擇一提供：純數矩陣 X 或物件陣列 rows"},
                    extra_headers=rid_h,
                )
            domain = str(data.get("domain") or "financial")
            try:
                if has_x:
                    verr = _validate_X_matrix(data["X"])
                    if verr:
                        return _json_response(400, {"error": verr}, extra_headers=rid_h)
                    payload = data["X"]
                else:
                    rows = data["rows"]
                    if isinstance(rows, dict):
                        rows = [rows]
                    verr = _validate_rows_objects(rows)
                    if verr:
                        return _json_response(400, {"error": verr}, extra_headers=rid_h)
                    payload = rows
                out = self.predictor.predict(payload, domain=domain)
            except Exception as e:
                logger.exception("predict failed")
                return _json_response(
                    500, {"error": "predict_failed", "detail": str(e)}, extra_headers=rid_h
                )
            if request_id:
                out = dict(out)
                out["request_id"] = request_id
            return _json_response(200, out, extra_headers=rid_h)

        if method == "POST" and path == "/v1/predict_many":
            try:
                data = _parse_json_body(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return _json_response(
                    400, {"error": "invalid_json", "detail": str(e)}, extra_headers=rid_h
                )
            if not isinstance(data, dict) or "X" not in data:
                return _json_response(400, {"error": "missing_field_X"}, extra_headers=rid_h)
            verr = _validate_X_matrix(data["X"])
            if verr:
                return _json_response(400, {"error": verr}, extra_headers=rid_h)
            domain = str(data.get("domain") or "financial")
            batch_size = min(int(data.get("batch_size") or 1024), MAX_BATCH_CAP)
            try:
                out = self.predictor.predict_many(
                    data["X"], domain=domain, batch_size=batch_size
                )
            except Exception as e:
                logger.exception("predict_many failed")
                return _json_response(
                    500, {"error": "predict_many_failed", "detail": str(e)}, extra_headers=rid_h
                )
            if request_id:
                out = dict(out)
                out["request_id"] = request_id
            return _json_response(200, out, extra_headers=rid_h)

        if method == "OPTIONS":
            h = [
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type"),
            ]
            return 204, h, b""

        return _json_response(404, {"error": "not_found", "path": path})

    def __call__(
        self, environ: Dict[str, Any], start_response: Callable[..., Any]
    ) -> List[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")

        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            size = 0
        raw_body = environ["wsgi.input"].read(size) if size > 0 else b""
        request_id = environ.get("HTTP_X_REQUEST_ID") or environ.get("HTTP_X_TRACE_ID")

        status_code, headers, body = self.handle_request(
            method, path, raw_body, request_id=request_id
        )
        lines = {
            200: "200 OK",
            204: "204 No Content",
            400: "400 Bad Request",
            404: "404 Not Found",
            500: "500 Internal Server Error",
            503: "503 Service Unavailable",
        }
        start_response(lines.get(status_code, f"{status_code}"), headers)
        return [body]


def run_server() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="UnifiedPredictor HTTP 服務：瀏覽器 UI + JSON API（標準函式庫）"
    )
    env_model = os.environ.get("UNIFIED_MODEL_PATH", "").strip()
    parser.add_argument(
        "--model-path",
        type=str,
        default=env_model or "models/task_stock_price_next.pkl",
        help="啟動時載入的 pickle（檔案不存在則改為自動選 models 目錄內第一個 task_*.pkl）",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="模型目錄（預設：與 --model-path 同層，或 PREDICT_AI_MODELS_DIR / ./models）",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="啟動時不載入任何模型（僅 UI 手動載入）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mp_arg = str(Path(args.model_path).expanduser().resolve())
    if not os.path.isfile(mp_arg):
        mp_arg = None  # 交給 UnifiedPredictHTTPApp 從 models 目錄自動選
    mr_arg = args.models_dir
    if mr_arg:
        mr_arg = str(Path(mr_arg).expanduser().resolve())

    try:
        app = UnifiedPredictHTTPApp(
            model_path=mp_arg,
            models_root=mr_arg,
            auto_load=not args.no_auto_load,
        )
    except Exception as e:
        logger.error("無法建立服務: %s", e)
        sys.exit(1)

    from wsgiref.simple_server import make_server

    with make_server(args.host, args.port, app) as httpd:
        print(f"瀏覽器開啟 http://{args.host}:{args.port}/  即可操作預測")
        print(f"  模型目錄: {app._registry.models_root}")
        print(f"  目前模型: {app._registry.model_path or '（未載入）'}")
        print("  API: GET /v1/models  POST /v1/load_model  POST /v1/predict")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
