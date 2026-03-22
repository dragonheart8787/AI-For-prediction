# 第一輪架構重構說明（2026）

## 新增模組（責任切分）

| 模組 | 用途 |
|------|------|
| `prediction_contracts.py` | `ModelConfig`、`RuntimeCapabilities`、`build_fit_result` / `build_predict_result` 統一回傳 |
| `prediction_intervals.py` | 近似預測區間（與 `UnifiedPredictor` 推論核心分離） |
| `model_explainability.py` | Torch 梯度近似特徵重要性 |
| `pipeline_run_context.py` | `run_id`、`RunMetadata`、`hash_config_blob`、`try_git_commit` |
| `artifact_registry.py` | `artifacts/{task}/{run_id}/` 目錄與 `write_run_bundle` |
| `artifacts/` | 預設產物根目錄（可用 `PREDICT_AI_ARTIFACTS_ROOT` 覆寫） |
| `experimental/` | 預留非主線程式搬移 |

## UnifiedPredictor

- 可選 `model_config: ModelConfig`；`fit`／`predict` 統一契約鍵 + **legacy** 頂層鍵（`prediction`、`confidence`、`train_*` 等）過渡相容。
- Torch／Optuna 維持 **lazy import**（僅在對應分支）。

## 爬蟲管線

- 訓練產生 `run_id`、寫入 `artifacts/.../config|metrics|summary|feature_manifest`。
- `experiment_log.log_training_event` 固定欄位 + 檔案輪替（`MAX_JSONL_BYTES`）。

## schema_infer

- `align_source_frames(DataFrame dict, ...)` + `AlignSummary`；`prevent_leakage` 時禁用 `bfill`。
- `align_sources(..., prevent_leakage=, log_summary=, return_summary=)`。

## walk_forward

- `model=automl` 預設仍跳過；若傳 `fit_kwargs["automl_frozen"] = {"model": "...", "fit_kw": {...}}` 則每折凍結重訓。
- 深度模型 + `--wf-allow-deep` 時 `torch_epochs` 會被 `wf_torch_epoch_cap`（預設 48）壓縮。

## HTTP

- `ModelRegistry` 啟動載入；`X` schema 驗證；列數上限；`batch_size` cap；`X-Request-ID`／`X-Trace-ID` 回顯。

## data_connectors

- 統一例外：`ConnectorTimeoutError`、`ConnectorRateLimitError` 等。

## 後續可選

- 將根目錄大量腳本移入 `experimental/`。
- 完全移除 legacy 頂層 predict/fit 鍵（需全專案搜尋遷移）。
