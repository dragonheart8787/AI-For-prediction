# 進階系統能力說明（與程式碼對齊版）

> 本檔描述**主線已接線、可一條龍執行**的能力。重型子目錄（torch 為主的 NAS、分散式排程等）仍為實驗模組，需自行整合。

---

## 〇、架構重構（第一輪）

- 契約與設定：`prediction_contracts.py`、歸檔說明 [`archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md`](../archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md)
- 區間／可解釋性：`prediction_intervals.py`、`model_explainability.py`
- 實驗追溯：`pipeline_run_context.py`、`artifact_registry.py`、`experiment_log.log_training_event`
- 產物目錄：`artifacts/`（訓練 run 摘要與 `models/*.pkl` 路徑並存）

---

## 一、主線已落地（端到端）

| 能力 | 說明 | 位置 |
|------|------|------|
| 統一預測 | linear / xgboost / lightgbm / **ensemble**（線性+XGB）、**automl**（Optuna 搜模組+超參）、**mlp_torch / lstm / transformer**（PyTorch）、可選 ONNX（樹／線性；Torch 略過）、可設定限流 | `unified_predict.py`、`automl/optuna_runner.py`、`nn_models/torch_regressors.py` |
| **強化訓練** | 時序衍生特徵（`--rich-features`）、`--preset strong`、樹模型 **early stopping**（時間序驗證窗） | `feature_expansion.py`、`crawler_train_pipeline.py` |
| 資料連接器 | retry、timeout、backoff；OWID 離線後備 | `data_connectors/` |
| 爬蟲訓練管線 | 多源、`align_sources` 時間對齊、特徵重要性 | `crawler_train_pipeline.py` |
| **Walk-forward 驗證** | 時間序列交叉驗證（不含 replay 混合資料） | `--walk-forward`、`validation/walk_forward_eval.py` |
| **實驗紀錄** | JSON Lines，無需 MLflow | `experiment_log.py` → `data/experiment_runs.jsonl`（可於 `prediction_schema.yaml` 設定） |
| **HTTP 預測服務** | WSGI：`GET /` 說明頁、`GET /health`、`GET /v1/model/info`、`POST /v1/predict`、`POST /v1/predict_many` | `launch_predict_service.py`、`model_serving/unified_http_service.py` |
| 時間切分工具 | purge gap、walk-forward 索引 | `validation/time_series_split.py` |
| 多領域回測 CLI | 合成／爬蟲資料 + 指標 JSON 行 | `demo_backtest_all.py` |

---

## 二、設定檔

- **`config/prediction_schema.yaml`**  
  - `pipeline_defaults.walk_forward`：折數、測試窗、`purge_gap`、`min_train_size`  
  - `pipeline_defaults.experiment_log`：是否寫入、路徑  

---

## 三、建議指令（完整閉環）

```bash
pip install -r requirements-core.txt

# 選用 AutoML + 深度模型（Optuna + PyTorch）
# pip install -r requirements-automl.txt

# 1) 訓練 + walk-forward + 實驗紀錄 + 存模組
python crawler_train_pipeline.py stock_price_next --model linear --walk-forward

# 1b) 較強組合（建議：pip install xgboost yfinance pandas）
python crawler_train_pipeline.py stock_price_next --model ensemble --preset strong --rich-features --period 1y --walk-forward

# 1c) Optuna AutoML（可選納入 LSTM/Transformer：--automl-include-deep）
python crawler_train_pipeline.py stock_price_next --model automl --automl-trials 25 --preset strong

# 2) 啟動預測 API（另開終端）
# Windows CMD: set UNIFIED_MODEL_PATH=models\task_stock_price_next.pkl
# Bash: export UNIFIED_MODEL_PATH=models/task_stock_price_next.pkl
python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765

# 3) 測試 POST（需 curl 或自寫腳本）
# curl -X POST http://127.0.0.1:8765/v1/predict -H "Content-Type: application/json" -d "{\"X\":[[1,2,3,4,5]],\"domain\":\"financial\"}"

python -m pytest tests/ -q
python demo_backtest_all.py --model linear --batch 32
```

> 特徵維度須與訓練時一致；若不知維度，請以訓練日誌中的特徵數為準。  
> 如何把模型練得更強：見 **[`archive/documentation/強模型訓練要點.md`](../archive/documentation/強模型訓練要點.md)**（資料量與真實性優先於調參）；精簡版已併入 [training.md](training.md)。

---

## 四、實驗／擴充目錄（非主線預設）

**已掛主線（需另裝依賴）**：`automl/optuna_runner.py`（`--model automl`）、`nn_models/torch_regressors.py`（`--model lstm` 等）；請 `pip install -r requirements-automl.txt`。

以下仍多為實驗或重型模組，**未**預設用於上述 HTTP 一鍵流程：

`archive/legacy/packages/` 下舊版模組（如 `data_pipeline`、`gpu_acceleration`、`neural_architecture_search`、`reinforcement_learning`、`knowledge_distillation`）與 `model_serving/serving_engine.py`（舊版引擎）…

若需 **Kafka / K8s / Prometheus 全棧**，請在現有 `docker-compose.yml` 上自行擴充。

---

## 五、進階 API（已實作）

- **`UnifiedPredictor.predict_interval_naive(X, z_score=1.96)`**：以訓練集 `train_rmse` 當殘差標準差，產出 ±z·RMSE 對稱區間（粗估，無共形保證）。  
- **`UnifiedPredictor.export_torch_model_onnx(path)`**：嘗試匯出當前 `TorchRegressorWrapper`；失敗回傳 `False`。  
- **Torch 特徵解釋**：`fit` 後 `predictor._torch_feature_attr`（梯度近似，已正規化）；爬蟲管線特徵排序會優先使用。  
- **實驗紀錄**：`train_complete` 事件含 `model_requested`、`model_trained`、選用時 `automl` 摘要（已 JSON 安全化）。

---

## 六、已知限制

1. `confidence` 與 `predict_interval_naive` 均為啟發式／粗估，非貝氏／共形區間。  
2. 類別／文字欄位多為 hash 編碼。  
3. 無 API Key 時部分 connector 為合成後備資料。  
4. 生產環境請補 TLS、認證、配額與監控。

---

## 七、CI 選測

- 預設 workflow 僅 `requirements-core.txt`。  
- 可選 job：安裝 `requirements-automl.txt` 並跑 `tests/test_automl_torch.py`（見 `.github/workflows/ci.yml`）。

---

*與 `crawler_train_pipeline`、`launch_predict_service`、`experiment_log`、`validation` 同步維護。*
