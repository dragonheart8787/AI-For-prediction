# 多源時序預測管線（Predict-AI）

**一句話**：從 Yahoo / Open-Meteo / OWID / EIA / NewsAPI 等來源拉資料、時間對齊、訓練統一迴歸模型，並可選 HTTP 推論。

**版本**：`0.9.0`（見 [`VERSION`](VERSION)／[`CHANGELOG.md`](CHANGELOG.md)）  
**授權**：[MIT](LICENSE)

---

## 安裝（主線，必裝）

在**專案根目錄**執行（路徑皆相對於目前工作目錄，勿硬編碼本機絕對路徑）：

```bash
pip install -r requirements-core.txt
```

選配（AutoML + PyTorch）：

```bash
pip install -r requirements-automl.txt
```

環境變數範例見 [`.env.example`](.env.example)（**勿將含 API Key 的 `.env`  commit**）。

---

## 訓練（主線，3 步內）

```bash
# 1) 訓練單一任務（任務 ID 定義於 config/prediction_schema.yaml）
python crawler_train_pipeline.py stock_price_next --model linear

# 2) 指定模型目錄（預設：./models 或環境變數 PREDICT_AI_MODELS_DIR）
python crawler_train_pipeline.py stock_price_next --model linear --models-dir ./models

# 3) 資料根目錄（僅在未自訂 --memory-path 且仍為預設記憶檔時，改寫為 <data-dir>/training_memory.jsonl）
python crawler_train_pipeline.py stock_price_next --model linear --data-dir ./data
```

**乾淨環境最低驗證**（push 前建議跑）：

```bash
pip install -r requirements-core.txt
pytest -q tests/test_unified_predict.py
python crawler_train_pipeline.py --help
python launch_predict_service.py --help
```

或使用 `scripts/run_core_demo.sh`／`scripts/run_core_demo.ps1`。

---

## 啟動 HTTP 推論

先完成訓練，確認 `models/task_<任務ID>.pkl` 存在（或 `--save` 路徑）。

```bash
# 環境變數（可選）
set UNIFIED_MODEL_PATH=models\task_stock_price_next.pkl

python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765
```

**curl 範例**（特徵維度須與訓練一致；下方僅示意 5 維）：

```bash
curl -s -X POST http://127.0.0.1:8765/v1/predict ^
  -H "Content-Type: application/json" ^
  -H "X-Request-ID: demo-1" ^
  -d "{\"X\": [[1.0, 2.0, 3.0, 4.0, 5.0]], \"domain\": \"financial\"}"
```

JSON 範例檔：[examples/sample_predict_payload.json](examples/sample_predict_payload.json)

---

## 支援能力（精簡）

| 類別 | 內容 |
|------|------|
| 模型 | linear、xgboost、lightgbm、ensemble；選配 **automl（Optuna）**、**mlp_torch / lstm / transformer** |
| 資料源 | Yahoo、Open-Meteo、OWID、EIA、NewsAPI（金鑰由設定／環境提供，見 `.env.example`） |
| HTTP | `GET /health`、`GET /v1/model/info`、`POST /v1/predict`、`POST /v1/predict_many` |
| 流程 | collect → align → feature expand（可選）→ train → metrics → persist →（選）serve |

**一條龍文件**：[`ADVANCED_SYSTEM_CAPABILITIES.md`](ADVANCED_SYSTEM_CAPABILITIES.md)  
**架構／訓練／服務索引**：[`docs/architecture.md`](docs/architecture.md)、[`docs/training.md`](docs/training.md)、[`docs/serving.md`](docs/serving.md)

---

## 訓練後產物

| 位置 | 說明 |
|------|------|
| `models/`（或 `PREDICT_AI_MODELS_DIR`／`--models-dir`） | `task_<task_id>.pkl`、`universal_model.pkl`（`--train-all`） |
| `artifacts/<task>/<run_id>/` | `config.json`、`metrics.json`、`summary.json`、`feature_manifest.json`、`model_path.txt`（指向實際 .pkl） |
| `data/experiment_runs.jsonl`（可於 schema 設定） | 實驗事件紀錄 |

---

## 主線目錄（精簡）

```
config/           # prediction_schema.yaml、tasks
data_connectors/  # 資料來源
crawler_train_pipeline.py
unified_predict.py
validation/       # walk-forward 等
model_serving/    # HTTP
automl/           # 選配 Optuna
nn_models/        # 選配 Torch
tests/
docs/
examples/
scripts/
```

歷史與長篇報告多在 `reports/`；實驗性說明見 `experimental/README.md`。

---

## 限制（務必閱讀）

- **Walk-forward**：`automl` 預設不跑每折搜參；需事先搜參並以 `automl_frozen` 傳入。深度模型預設跳過 wf，除非 `--wf-allow-deep`（且有 epoch cap）。
- **ONNX**：主線以樹／線性為主；Torch 預設不轉 ONNX。
- **外部 API**：無金鑰時部分 connector 走離線／合成後備，上限較低。
- **深度模型**：訓練時間與記憶體成本高，小樣本時樹模型常更穩。

---

## CI

見 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：`requirements-core.txt` 全測 + 選用 `requirements-automl.txt` 子集測試。

---

## 標籤建議（GitHub Topics）

`machine-learning` `time-series` `forecasting` `automl` `pytorch` `xgboost` `lightgbm` `onnx` `python`

---

*舊版行銷向「AGI」長文已由此 README 取代；細節請以 `docs/` 與 `ADVANCED_SYSTEM_CAPABILITIES.md` 為準。*
