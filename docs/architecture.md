# 架構說明

本專案主線是**多源時序資料 → 對齊與特徵 → 統一迴歸訓練 →（選用）HTTP 推論**。舊版 AGI／論文長文／實驗套件已收在 [`archive/`](../archive/README.md)。

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

## 2. 目錄（主線）

```
config/                 # prediction_schema.yaml、tasks、schema
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
archive/                # 非主線：舊腳本、長文、範例資料、舊 docker 等
reports/                # 專案內長篇報告（若有的話）
data/                   # 預設實驗 log、memory jsonl（.gitignore 視設定）
```

---

## 3. 資料流（概念）

1. **Collect**：依任務從多 connector 取數（無金鑰時部分為離線／合成後備）。  
2. **Align**：時間索引對齊、缺失處理（防洩漏模式下禁用向後填補）。  
3. **Features**：可選 `--rich-features`／YAML 擴張。  
4. **Train**：`UnifiedPredictor.fit`；可 `--walk-forward`；`automl` 於 wf 需凍結參數。  
5. **Persist**：`models/task_<id>.pkl` 與 `artifacts/<task>/<run_id>/`。  
6. **Serve**（選）：`launch_predict_service.py` + WSGI。

---

## 4. 延伸閱讀

- [訓練與指令](training.md)  
- [HTTP 服務與端點](serving.md)  
- [能力對照表（與程式碼同步）](capabilities.md)  
- 第一輪重構筆記（歸檔）：[`archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md`](../archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md)
