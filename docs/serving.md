# HTTP 推論服務

實作於 [`model_serving/unified_http_service.py`](../model_serving/unified_http_service.py)，由 [`launch_predict_service.py`](../launch_predict_service.py) 載入訓練產生的 `.pkl` 後啟動（僅標準函式庫 **wsgiref**，無需 FastAPI）。

---

## 1. 啟動

```bash
python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765
```

可選環境變數：**`UNIFIED_MODEL_PATH`** — 若設定，可作為預設模型路徑（仍可用 CLI 覆寫，視實作為準；建議以 `--model-path` 為準）。

其他與上限相關：`PREDICT_AI_HTTP_MAX_ROWS`、`PREDICT_AI_HTTP_MAX_BATCH`（見程式碼頂部常數）。

---

## 2. HTTP 端點（正式清單）

以下為**唯一維護清單**（與 `README.md`、程式內註解一致）：

| 方法 | 路徑 | 說明 |
|------|------|------|
| **GET** | **`/`** | 瀏覽器說明頁（HTML），含連結至 `/health`、`/v1/model/info` |
| **GET** | **`/health`** | JSON 健康檢查 |
| **GET** | **`/v1/model/info`** | 模型摘要（JSON，含 `input_dim` 等） |
| **POST** | **`/v1/predict`** | 單次或多筆列預測；body 見 `examples/sample_predict_payload.json` |
| **POST** | **`/v1/predict_many`** | 大批次；可帶 `batch_size`（有上限） |

可選 HTTP header：**`X-Request-ID`** 或 **`X-Trace-ID`**（回應中會帶回追蹤用）。

---

## 3. 請求範例（跨平台）

**Bash / macOS / Linux**

```bash
export UNIFIED_MODEL_PATH="models/task_stock_price_next.pkl"

python launch_predict_service.py --model-path "$UNIFIED_MODEL_PATH" --port 8765 &
sleep 2
curl -s -X POST "http://127.0.0.1:8765/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" \
  -d '{"X": [[1.0, 2.0, 3.0, 4.0, 5.0]], "domain": "financial"}'
```

**PowerShell**

```powershell
$env:UNIFIED_MODEL_PATH = "models\task_stock_price_next.pkl"
Start-Process python -ArgumentList "launch_predict_service.py","--model-path",$env:UNIFIED_MODEL_PATH,"--port","8765" -NoNewWindow
Start-Sleep -Seconds 2
Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/predict" -Method Post `
  -ContentType "application/json" `
  -Headers @{"X-Request-ID"="demo-1"} `
  -Body '{"X": [[1.0, 2.0, 3.0, 4.0, 5.0]], "domain": "financial"}'
```

**Windows CMD**

```bat
set UNIFIED_MODEL_PATH=models\task_stock_price_next.pkl
start /B python launch_predict_service.py --model-path %UNIFIED_MODEL_PATH% --port 8765
timeout /t 2 >nul
curl -s -X POST http://127.0.0.1:8765/v1/predict -H "Content-Type: application/json" -H "X-Request-ID: demo-1" -d "{\"X\": [[1.0, 2.0, 3.0, 4.0, 5.0]], \"domain\": \"financial\"}"
```

特徵維度須與訓練一致；不確定時可先開 **`GET /v1/model/info`** 查看 `input_dim`。

---

## 4. 限制（生產前必讀）

- 無內建 TLS／認證／配額；請於反向代理或上層閘道補強。  
- `confidence`、區間 API 為啟發式，非共形／貝氏保證。  

更多能力與限制見 [capabilities.md](capabilities.md)。
