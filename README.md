# 多源時序預測管線（Predict-AI）

**一句話**：從 Yahoo / Open-Meteo / OWID / EIA / NewsAPI 等來源拉資料、時間對齊、訓練統一迴歸模型，並可選 HTTP 推論。

**版本**：`0.9.1`（見 [`VERSION`](VERSION)／[`CHANGELOG.md`](CHANGELOG.md)）  
**授權**：[MIT](LICENSE)  
**Repository**：[github.com/dragonheart8787/AI-For-prediction](https://github.com/dragonheart8787/AI-For-prediction)（若仍使用舊網址，GitHub 會自動轉址）

> **文件入口（請從此收斂）**：本檔為唯一首頁；其餘說明見 [`docs/architecture.md`](docs/architecture.md)、[`docs/training.md`](docs/training.md)、[`docs/serving.md`](docs/serving.md)、[`docs/capabilities.md`](docs/capabilities.md)。舊版長文與腳本在 [`archive/`](archive/README.md)。  
> **Repo 命名**：若可選，建議日後改為 `predict-ai` 等較短名稱；目前遠端網址仍以上方為準。

---

## 安裝（主線）

在**專案根目錄**執行：

```bash
pip install -r requirements-core.txt
```

選配（AutoML + PyTorch）：

```bash
pip install -r requirements-automl.txt
```

其餘歷史依賴清單見 [`archive/requirements_legacy/`](archive/requirements_legacy/)。環境變數範例：[`.env.example`](.env.example)（**勿 commit 含金鑰的 `.env`**）。

---

## 訓練

```bash
python crawler_train_pipeline.py stock_price_next --model linear
python crawler_train_pipeline.py stock_price_next --model linear --models-dir ./models
python crawler_train_pipeline.py stock_price_next --model linear --data-dir ./data
```

詳見 [`docs/training.md`](docs/training.md)。

**Push 前最低驗證**：

```bash
pip install -r requirements-core.txt
pytest -q tests/test_unified_predict.py
python crawler_train_pipeline.py --help
python launch_predict_service.py --help
```

或使用 `scripts/run_core_demo.sh`／`scripts/run_core_demo.ps1`。

---

## 啟動 HTTP 推論

預設模型路徑可透過環境變數 **`UNIFIED_MODEL_PATH`** 提供；亦建議明確傳 **`--model-path`**。

### Bash（macOS / Linux）

```bash
export UNIFIED_MODEL_PATH="models/task_stock_price_next.pkl"
python launch_predict_service.py --model-path "$UNIFIED_MODEL_PATH" --port 8765
```

### PowerShell

```powershell
$env:UNIFIED_MODEL_PATH = "models\task_stock_price_next.pkl"
python launch_predict_service.py --model-path $env:UNIFIED_MODEL_PATH --port 8765
```

### Windows CMD

```bat
set UNIFIED_MODEL_PATH=models\task_stock_price_next.pkl
python launch_predict_service.py --model-path %UNIFIED_MODEL_PATH% --port 8765
```

### curl 範例（Bash）

```bash
curl -s -X POST "http://127.0.0.1:8765/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" \
  -d '{"X": [[1.0, 2.0, 3.0, 4.0, 5.0]], "domain": "financial"}'
```

Windows CMD 若使用 `curl.exe`，JSON 需適當跳脫雙引號（見 [`docs/serving.md`](docs/serving.md)）。

JSON 範例檔：[examples/sample_predict_payload.json](examples/sample_predict_payload.json)

---

## HTTP 端點（正式清單）

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 瀏覽器說明頁（HTML） |
| GET | `/health` | 健康檢查（JSON） |
| GET | `/v1/model/info` | 模型摘要（JSON） |
| POST | `/v1/predict` | 預測 |
| POST | `/v1/predict_many` | 大批次（`batch_size` 有上限） |

說明與跨平台請求範例：[docs/serving.md](docs/serving.md)

---

## 支援能力（精簡）

| 類別 | 內容 |
|------|------|
| 模型 | linear、xgboost、lightgbm、ensemble；選配 **automl**、**mlp_torch / lstm / transformer** |
| 資料源 | Yahoo、Open-Meteo、OWID、EIA、NewsAPI |
| 流程 | collect → align → feature expand（可選）→ train → metrics → persist →（選）serve |

---

## 訓練後產物

| 位置 | 說明 |
|------|------|
| `models/`（或 `PREDICT_AI_MODELS_DIR`／`--models-dir`） | `task_<task_id>.pkl` 等 |
| `artifacts/<task>/<run_id>/` | run 摘要與 `model_path.txt` 等 |
| `data/experiment_runs.jsonl` | 實驗事件（可設定） |

---

## 主線目錄

```
config/           # 主設定（已合併原 configs/ 中與 enhanced 無關之用途說明見 archive）
data_connectors/
validation/
model_serving/    # 主 HTTP：unified_http_service.py（舊 shell 見 archive/legacy/serving）
crawler_train_pipeline.py
unified_predict.py
launch_predict_service.py
demo_backtest_all.py
automl/   nn_models/    # 選配
tests/  docs/  examples/  scripts/
archive/            # 非主線
reports/            # 長篇報告；零散舊報告見 archive/documentation
```

---

## 限制

- **Walk-forward**：`automl` 預設不於每折搜參；需 `automl_frozen`。深度模型預設跳過 wf，除非 `--wf-allow-deep`。  
- **ONNX**：主線以樹／線性為主；Torch 預設不轉 ONNX。  
- **外部 API**：無金鑰時部分 connector 為後備資料。  

---

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml)：多矩陣 pytest、`requirements-automl` 子集測試，以及 **compileall + import smoke + ruff（tests）**。

---

## GitHub Topics 建議

`machine-learning` `time-series` `forecasting` `automl` `pytorch` `xgboost` `lightgbm` `python`
