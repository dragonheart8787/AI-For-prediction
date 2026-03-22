# HTTP 推論服務

實作於 [`model_serving/unified_http_service.py`](../model_serving/unified_http_service.py)，由 [`launch_predict_service.py`](../launch_predict_service.py) 載入訓練產生的 `.pkl` 後啟動（僅標準函式庫 **wsgiref**，無需 FastAPI）。

---

## 1. 啟動

```bash
python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765
```

可選環境變數：**`UNIFIED_MODEL_PATH`** — 與 `--model-path` 對應；**路徑字串中間不可插入空白**（錯誤例：`models\t task_...`）。

其他與上限相關：`PREDICT_AI_HTTP_MAX_ROWS`、`PREDICT_AI_HTTP_MAX_BATCH`（見程式碼頂部常數）。

---

## 2. HTTP 端點（正式清單）

| 方法 | 路徑 | 說明 |
|------|------|------|
| **GET** | **`/`** | 瀏覽器說明頁（HTML），含連結至 `/health`、`/v1/model/info` |
| **GET** | **`/health`** | JSON 健康狀態 |
| **GET** | **`/v1/model/info`** | 模型摘要（JSON，含 `input_dim` 等，實際鍵依 `get_model_info()`） |
| **POST** | **`/v1/predict`** | 單次或多筆列預測 |
| **POST** | **`/v1/predict_many`** | 大批次；body 可帶 `batch_size`（有上限） |

可選 HTTP header：**`X-Request-ID`** 或 **`X-Trace-ID`**（成功時回應 JSON 可帶 `request_id`）。

---

## 3. JSON：請求與回應

### 3.1 `POST /v1/predict`

**請求 body（JSON）**

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| **`X`** | `number[][]` | 是 | 每列一筆樣本，每欄一特徵；維度須與訓練一致。 |
| **`domain`** | `string` | 否 | 預設 `"financial"`，影響內部 horizon 等邏輯。 |

範例：

```json
{
  "X": [[1.0, 2.0, 3.0, 4.0, 5.0]],
  "domain": "financial"
}
```

**成功 200（JSON）** — 由 `prediction_contracts.build_predict_result` 組裝，典型鍵包括：

| 鍵 | 說明 |
|----|------|
| **`predictions`** | 與 `X` 列數對應的預測值（巢狀 list） |
| **`model_type`** | 目前模型名稱／型別字串 |
| **`feature_names`** | 特徵名稱列表（若訓練時有） |
| **`metrics`** | 推論端通常為 `null` |
| **`artifacts`** | 含 `confidence`、`horizons`、`n_samples`、`domain` 等 |
| **`prediction`** | 與 `predictions` 相同（legacy 相容） |
| **`model`** | 與 `model_type` 相同（legacy 相容） |
| **`confidence`**、`horizons`、`n_samples`、`domain` | 由 `artifacts` 複寫到頂層（legacy） |
| **`request_id`** | 若有傳 `X-Request-ID`／`X-Trace-ID` |

範例（結構示意，數值為假資料）：

```json
{
  "predictions": [[0.0123]],
  "model_type": "linear",
  "feature_names": ["f0", "f1", "f2", "f3", "f4"],
  "metrics": null,
  "artifacts": {
    "confidence": [0.5],
    "horizons": [1],
    "n_samples": 1,
    "domain": "financial"
  },
  "prediction": [[0.0123]],
  "model": "linear",
  "confidence": [0.5],
  "horizons": [1],
  "n_samples": 1,
  "domain": "financial"
}
```

**錯誤（JSON）** 常見：

| HTTP | `error` 欄位 | 情境 |
|------|----------------|------|
| 400 | `invalid_json` | body 非合法 JSON |
| 400 | `missing_field_X` | 未提供 `X` |
| 400 | `X_must_be_list` / `X_row_must_be_list` / `X_values_must_be_numeric` / `X_too_many_rows` | 矩陣格式不符或超過列數上限 |
| 500 | `predict_failed` | 推論拋錯（`detail` 有訊息） |
| 503 | `model_not_loaded` | 模型未載入 |

### 3.2 `POST /v1/predict_many`

與 `/v1/predict` 相同 **`X`**／**`domain`**，另可選：

| 欄位 | 說明 |
|------|------|
| **`batch_size`** | 整數，預設 1024，實際會被伺服器上限截斷。 |

回應形狀與 `predict_many` 實作一致（通常同樣含 `predictions` 等契約鍵）；錯誤時 `error` 可能為 `predict_many_failed`。

### 3.3 `GET /health`

```json
{ "status": "ok", "model_path": "/abs/path/to/model.pkl" }
```

無模型時 `status` 可能為 `no_model`。

---

## 4. 請求範例（跨平台）

**Bash / macOS / Linux（建議）**

```bash
export UNIFIED_MODEL_PATH="models/task_stock_price_next.pkl"

python launch_predict_service.py --model-path "$UNIFIED_MODEL_PATH" --port 8765 &
sleep 2
curl -s -X POST "http://127.0.0.1:8765/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" \
  -d '{"X": [[1.0, 2.0, 3.0, 4.0, 5.0]], "domain": "financial"}'
```

專案亦提供 **`scripts/run_core_demo.sh`**（安裝／煙霧測試）；Windows 可用 **`scripts/run_core_demo.ps1`**。

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

**Windows CMD（注意：`set` 行內無引號時，路徑不可有空格）**

```bat
set UNIFIED_MODEL_PATH=models\task_stock_price_next.pkl
start /B python launch_predict_service.py --model-path %UNIFIED_MODEL_PATH% --port 8765
timeout /t 2 >nul
curl -s -X POST http://127.0.0.1:8765/v1/predict -H "Content-Type: application/json" -H "X-Request-ID: demo-1" -d "{\"X\": [[1.0, 2.0, 3.0, 4.0, 5.0]], \"domain\": \"financial\"}"
```

特徵維度須與訓練一致；請先 **`GET /v1/model/info`** 確認 `input_dim`。

---

## 5. 限制（生產前必讀）

- 無內建 TLS／認證／配額；請於反向代理或上層閘道補強。  
- `confidence`、區間 API 為啟發式，非共形／貝氏保證。  

更多能力與限制見 [capabilities.md](capabilities.md)。
