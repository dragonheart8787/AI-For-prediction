#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedPredictor HTTP 服務（僅標準函式庫：wsgiref + json）
適合本機／內網部署，無需 FastAPI/torch。

端點：
  GET  /                瀏覽器說明頁（HTML）
  GET  /health          健康檢查
  GET  /v1/model/info   模型摘要
  POST /v1/predict      {"X": [[...]], "domain": "financial"}
  POST /v1/predict_many {"X": [[...]], "domain": "...", "batch_size": 1024}
"""
from __future__ import annotations

import argparse
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


def _landing_page_html(model_path: str, model_loaded: bool, input_dim: Optional[int]) -> str:
    dim_hint = (
        f"<p>目前模型輸入維度（特徵數）：<strong>{input_dim}</strong> — <code>X</code> 每列需 {input_dim} 個數值。</p>"
        if input_dim is not None
        else "<p>載入模型後可從 <a href=\"/v1/model/info\">/v1/model/info</a> 查看 <code>input_dim</code>。</p>"
    )
    status_html = (
        '<p style="color:green">模型已載入，API 可用。</p>'
        if model_loaded
        else '<p style="color:#c60">模型未載入，請檢查 <code>--model-path</code>。</p>'
    )
    curl_example = "curl -s -X POST http://127.0.0.1:8765/v1/predict -H \"Content-Type: application/json\" -d \"{\\\"X\\\": [[1,2,3,4,5]], \\\"domain\\\": \\\"financial\\\"}\""
    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>預測服務</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;}}
code,pre{{background:#f4f4f4;padding:.2rem .4rem;border-radius:4px;font-size:90%;}}
pre{{overflow:auto;padding:1rem;}}</style></head><body>
<h1>UnifiedPredictor HTTP 服務</h1>
{status_html}
<p>模型檔：<code>{model_path}</code></p>
{dim_hint}
<h2>可用路徑</h2>
<ul>
<li><a href="/health"><code>GET /health</code></a> — JSON 健康狀態</li>
<li><a href="/v1/model/info"><code>GET /v1/model/info</code></a> — 模型資訊（JSON）</li>
<li><code>POST /v1/predict</code> — 單次／多筆矩陣預測</li>
<li><code>POST /v1/predict_many</code> — 大批次（可帶 batch_size）</li>
</ul>
<p>瀏覽器只適合開本頁與上述 GET；預測請用 <strong>curl、Postman、或前端 fetch</strong>。</p>
<h3>範例（請依實際特徵數改 X）</h3>
<pre>{curl_example}</pre>
<p><small>若出現 <code>No module named 'yfinance'</code> 僅影響<strong>訓練時爬蟲</strong>；服務載入的 .pkl 與 yfinance 無關。要即時股價可執行：<code>pip install yfinance</code></small></p>
</body></html>"""


class ModelRegistry:
    """啟動時載入一次模型；避免每個請求重讀檔。"""

    def __init__(self, model_path: str) -> None:
        self.model_path = os.path.abspath(model_path)
        self._predictor: Any = None

    def load(self) -> None:
        from unified_predict import UnifiedPredictor

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"模型不存在: {self.model_path}")
        self._predictor = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=True)
        self._predictor.load_model(self.model_path)
        logger.info("已載入模型: %s", self.model_path)

    def predictor(self) -> Any:
        if self._predictor is None:
            raise RuntimeError("模型未載入")
        return self._predictor


class UnifiedPredictHTTPApp:
    """WSGI 應用：載入 pickle 模型並提供預測 JSON API。"""

    def __init__(self, model_path: str, auto_load: bool = True) -> None:
        self.model_path = os.path.abspath(model_path)
        self._registry = ModelRegistry(self.model_path)
        if auto_load:
            self._registry.load()

    def load(self) -> None:
        self._registry.load()

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
            dim = None
            if self._predictor is not None:
                dim = getattr(self._predictor, "_input_dim", None)
            html = _landing_page_html(
                self.model_path,
                model_loaded=self._predictor is not None,
                input_dim=dim,
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
                {"status": "ok" if ok else "no_model", "model_path": self.model_path},
                extra_headers=rid_h,
            )

        if self._registry._predictor is None:
            return _json_response(
                503,
                {"error": "model_not_loaded", "model_path": self.model_path},
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
            if not isinstance(data, dict) or "X" not in data:
                return _json_response(400, {"error": "missing_field_X"}, extra_headers=rid_h)
            verr = _validate_X_matrix(data["X"])
            if verr:
                return _json_response(400, {"error": verr}, extra_headers=rid_h)
            domain = str(data.get("domain") or "financial")
            try:
                out = self.predictor.predict(data["X"], domain=domain)
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

    parser = argparse.ArgumentParser(description="UnifiedPredictor HTTP 服務（標準函式庫）")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("UNIFIED_MODEL_PATH", "models/task_stock_price_next.pkl"),
        help="pickle 模型路徑（相對路徑相對於**目前工作目錄**），或環境變數 UNIFIED_MODEL_PATH",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_path = str(Path(args.model_path).expanduser().resolve())
    try:
        app = UnifiedPredictHTTPApp(model_path)
    except Exception as e:
        logger.error("無法載入模型: %s", e)
        sys.exit(1)

    from wsgiref.simple_server import make_server

    with make_server(args.host, args.port, app) as httpd:
        print(f"預測服務 http://{args.host}:{args.port}  model={model_path}")
        print("  GET  /health  GET /v1/model/info  POST /v1/predict  POST /v1/predict_many")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
