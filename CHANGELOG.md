# Changelog

## v0.9.0

- 主線：爬蟲 → 多源對齊 → 特徵展開 → `UnifiedPredictor` 訓練 → 產物寫入 `models/`、`artifacts/{task}/{run_id}/`
- 模型：linear / xgboost / lightgbm / ensemble；選配 **automl（Optuna）**、**mlp_torch / lstm / transformer（PyTorch）**
- 連接器：retry、timeout、backoff；統一例外型別（逾時、429、認證等）
- 驗證：walk-forward；automl 需 `automl_frozen` 才可對折重訓；深度模型預設跳過 wf（可 `--wf-allow-deep`）
- HTTP：`/health`、`/v1/predict`、`/v1/predict_many`；請求驗證、批次上限、`X-Request-ID`
- 依賴分層：`requirements-core.txt` / `requirements-automl.txt`
- 契約：`prediction_contracts` 統一 fit/predict 回傳；`prediction_intervals`、`model_explainability` 分離
- CI：core 全測 + 選用 automl/torch job

詳見 `docs/REFACTOR_ROUND1_ARCHITECTURE.md`、`ADVANCED_SYSTEM_CAPABILITIES.md`。
