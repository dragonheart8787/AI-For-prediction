# Changelog

## v0.9.1

- **Repo 佈局**：根目錄收斂為主線；大量舊腳本、長文、範例資料、舊 `configs/`／`serving/`／`paper/` 等移入 `archive/`（見 `archive/README.md`）。
- **文件**：`README.md` 為唯一首頁；補齊 `docs/architecture.md`、`docs/training.md`、`docs/serving.md`；原進階說明改為 `docs/capabilities.md`；HTTP 端點與 README／程式對齊（含 `GET /`、`GET /v1/model/info`）。
- **README**：補 Bash／PowerShell／CMD 與 `UNIFIED_MODEL_PATH` 範例；註記建議之 GitHub repo 命名。
- **CI**：新增 `quality` job（`compileall`、import smoke、`ruff check tests/`）。
- **其他**：`pyproject.toml` 增加 `tool.ruff`（排除 `archive` 等）；歷史 `requirements*.txt` 移至 `archive/requirements_legacy/`；舊 Docker 組態移至 `archive/legacy/deploy/`。

## v0.9.0

- 主線：爬蟲 → 多源對齊 → 特徵展開 → `UnifiedPredictor` 訓練 → 產物寫入 `models/`、`artifacts/{task}/{run_id}/`
- 模型：linear / xgboost / lightgbm / ensemble；選配 **automl（Optuna）**、**mlp_torch / lstm / transformer（PyTorch）**
- 連接器：retry、timeout、backoff；統一例外型別（逾時、429、認證等）
- 驗證：walk-forward；automl 需 `automl_frozen` 才可對折重訓；深度模型預設跳過 wf（可 `--wf-allow-deep`）
- HTTP：`/health`、`/v1/predict`、`/v1/predict_many`；請求驗證、批次上限、`X-Request-ID`
- 依賴分層：`requirements-core.txt` / `requirements-automl.txt`
- 契約：`prediction_contracts` 統一 fit/predict 回傳；`prediction_intervals`、`model_explainability` 分離
- CI：core 全測 + 選用 automl/torch job

詳見歸檔：`archive/documentation/REFACTOR_ROUND1_ARCHITECTURE.md`；主線能力表：`docs/capabilities.md`。
