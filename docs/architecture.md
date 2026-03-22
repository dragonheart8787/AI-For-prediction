# 架構說明

本專案主線是**多源時序資料 → 對齊與特徵 → 統一迴歸訓練 →（選用）HTTP 推論**。舊版 AGI／論文長文／實驗套件已收在 [`archive/`](../archive/README.md)。

---

## 0. 端到端流程（文字版）

主線一條龍可理解為下列階段（對應程式與設定）：

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     crawler_train_pipeline                 │
  [Yahoo / Meteo /   │  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────┴────┐
   OWID / EIA /      │  │ collect  │──▶│  align   │──▶│feature expand│──▶│   train    │
   NewsAPI …]        │  │(connectors)│  │schema_   │   │(optional)    │   │Unified     │
                    │  │          │  │infer     │   │feature_      │   │Predictor   │
                    │  └──────────┘   └──────────┘   │expansion     │   └──────┬─────┘
                    │                    │             └──────────────┘          │
                    │                    │                                       ▼
                    │                    │                              ┌───────────────┐
                    │                    │                              │ persist       │
                    │                    │                              │ models/*.pkl  │
                    │                    │                              │ artifacts/... │
                    │                    │                              │ experiment_log│
                    │                    │                              └───────┬───────┘
                    └────────────────────┴──────────────────────────────────────┘
                                                                                 │
                    ┌──────────────────────────────────────────────────────────────┘
                    ▼
            ┌───────────────┐         ┌─────────────────────────────┐
            │ (optional)    │         │ launch_predict_service +   │
            │ HTTP serve    │◀────────│ model_serving/ (WSGI)       │
            └───────────────┘         └─────────────────────────────┘
```

- **collect**：依 `config/prediction_schema.yaml` 任務挑 connector，拉取時間序列與外生欄位。  
- **align**：`schema_infer.align_sources`／`align_source_frames` 對齊時間索引；可 `prevent_leakage` 限制填補。  
- **feature expand**：可選 `--rich-features` 或 YAML 內 `feature_expansion`。  
- **train**：`UnifiedPredictor.fit`；可 `--walk-forward`；`automl` 在 wf 上需凍結參數。  
- **persist**：模型檔、`artifacts/<task>/<run_id>/`、`data/experiment_runs.jsonl`（路徑可設定）。  
- **serve**：載入 `.pkl`，對外 `GET/POST`（見 [serving.md](serving.md)）。

---

## 1. 元件與責任

| 區塊 | 說明 |
|------|------|
| **`crawler_train_pipeline.py`** | 任務驅動：依 `config/prediction_schema.yaml` 拉資料、`schema_infer.align_sources` 對齊、可選 `feature_expansion`、呼叫 `UnifiedPredictor`、walk-forward、實驗紀錄與 artifact。 |
| **`unified_predict.py`** | `UnifiedPredictor`：linear / xgboost / lightgbm / ensemble、選配 automl（Optuna）與 Torch 模型；lazy import 深度依賴。 |
| **`data_connectors/`** | Yahoo、Open-Meteo、OWID、EIA、NewsAPI 等；統一 timeout／retry 與例外型別。 |
| **`schema_infer.py`** | 多來源 DataFrame 對齊；`prevent_leakage` 時限制填補策略。 |
| **`validation/`** | 時間切分（purge gap）、`walk_forward_eval` 與 wf + automl 凍結參數約定。 |
| **`prediction_contracts.py`** | `fit`／`predict` 統一回傳契約與 legacy 鍵相容。 |
| **`prediction_intervals.py`**、**`model_explainability.py`** | 粗估區間、Torch 特徵归因（與核心推論分離）。 |
| **`pipeline_run_context.py`**、**`artifact_registry.py`**、**`experiment_log.py`** | `run_id`、產物目錄、`data/experiment_runs.jsonl` 事件。 |
| **`model_serving/unified_http_service.py`** | 僅標準函式庫 WSGI：`GET /` 說明頁、`/health`、`/v1/model/info`、POST 預測（見 [serving.md](serving.md)）。 |
| **`launch_predict_service.py`** | 一鍵載入 `.pkl` 並呼叫 `run_server()`。 |
| **`runtime_paths.py`** | `PREDICT_AI_MODELS_DIR`／`PREDICT_AI_DATA_DIR` 等路徑解析。 |
| **`automl/`**、**`nn_models/`** | 選配：Optuna 管線、Torch 迴歸包裝（需 `requirements-automl.txt`）。 |

---

## 2. 目錄（主線 — 與 repo 根目錄一致）

```
config/                 # 見 config/README.md；主線為 prediction_schema.yaml
data_connectors/
validation/
model_serving/
automl/                 # 選配
nn_models/              # 選配
crawler_train_pipeline.py
unified_predict.py
launch_predict_service.py
demo_backtest_all.py    # 多領域合成／指標 CLI（測試與示範）
tests/  docs/  examples/  scripts/
archive/                # 非主線
reports/legacy/         # 長篇／舊報告集中區
data/                   # 預設實驗 log、memory jsonl（.gitignore 視設定）
```

---

## 3. 延伸閱讀

- [訓練與指令](training.md)  
- [HTTP 服務與端點](serving.md)  
- [能力對照表（與程式碼同步）](capabilities.md)  
- 第一輪重構筆記（歸檔）：[`archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md`](../archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md)
