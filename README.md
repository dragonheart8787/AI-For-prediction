# 多源時序預測管線（Predict-AI）

**一句話**：從 Yahoo / Open-Meteo / OWID / EIA / NewsAPI 等來源拉資料、時間對齊、訓練統一迴歸模型，並可選 HTTP 推論。

**版號（唯一來源）**：請以根目錄 [`VERSION`](VERSION) 為準；目前為 **`0.9.2`**，並與 [`pyproject.toml`](pyproject.toml)、[`CHANGELOG.md`](CHANGELOG.md) 最新條目一致。  
**授權**：[MIT](LICENSE)  
**Repository**：[github.com/dragonheart8787/AI-For-prediction](https://github.com/dragonheart8787/AI-For-prediction)

> **文件入口**：本檔為唯一首頁 → [`docs/architecture.md`](docs/architecture.md)、[`docs/training.md`](docs/training.md)、[`docs/serving.md`](docs/serving.md)、[`docs/capabilities.md`](docs/capabilities.md)。舊版腳本與長文 → [`archive/`](archive/README.md)；專案內報告 → [`reports/legacy/`](reports/legacy/README.md)。  
> **GitHub About**：首頁右側 Description／Topics 請依 [`.github/GITHUB_ABOUT.md`](.github/GITHUB_ABOUT.md) 手動貼上（網頁不會自動讀取該檔）。

---

## 安裝（主線）

**Bash / macOS / Linux / WSL**

```bash
pip install -r requirements-core.txt
```

選配（AutoML + PyTorch）：

```bash
pip install -r requirements-automl.txt
```

歷史依賴清單：[archive/requirements_legacy/](archive/requirements_legacy/)。環境變數範例：[`.env.example`](.env.example)（**勿 commit 含金鑰的 `.env`**）。

---

## 訓練（首選 Bash）

```bash
python crawler_train_pipeline.py stock_price_next --model linear
python crawler_train_pipeline.py stock_price_next --model linear --models-dir ./models
python crawler_train_pipeline.py stock_price_next --model linear --data-dir ./data
```

詳見 [`docs/training.md`](docs/training.md)（含 **core vs automl**、常見產物）。

**Push 前最低驗證**

```bash
pip install -r requirements-core.txt
pytest -q tests/test_unified_predict.py
python crawler_train_pipeline.py --help
python launch_predict_service.py --help
```

或使用 **`scripts/run_core_demo.sh`**（macOS/Linux）／**`scripts/run_core_demo.ps1`**（Windows）。

---

## 啟動 HTTP 推論

環境變數 **`UNIFIED_MODEL_PATH`** 與 **`--model-path`** 皆指向同一個 `.pkl`。**請注意**：Windows CMD 的 `set` 行內**不可**在路徑中插入多餘空白（錯誤例：`models\t ask_stock_price_next.pkl` — 會變成兩個 token）。

### Bash（macOS / Linux）— 建議

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

### curl（Bash）

```bash
curl -s -X POST "http://127.0.0.1:8765/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-1" \
  -d '{"X": [[1.0, 2.0, 3.0, 4.0, 5.0]], "domain": "financial"}'
```

Windows CMD 的 `curl.exe` 跳脫規則見 [`docs/serving.md`](docs/serving.md)。JSON 範例檔：[examples/sample_predict_payload.json](examples/sample_predict_payload.json)。

---

## HTTP 端點（正式清單）

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 瀏覽器說明頁（HTML） |
| GET | `/health` | 健康檢查（JSON） |
| GET | `/v1/model/info` | 模型摘要（JSON） |
| POST | `/v1/predict` | 預測 |
| POST | `/v1/predict_many` | 大批次（`batch_size` 有上限） |

請求／回應 JSON 形狀見 [`docs/serving.md`](docs/serving.md)。

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

## 主線目錄（與 repo 根目錄一致）

```
config/            # 見 config/README.md（含 train_enhanced 歸檔 YAML）
data_connectors/
validation/
model_serving/
crawler_train_pipeline.py
unified_predict.py
launch_predict_service.py
demo_backtest_all.py
automl/   nn_models/       # 選配
tests/  docs/  examples/  scripts/
archive/                  # 非主線
reports/legacy/           # 長篇／舊報告
```

---

## 限制

- **Walk-forward**：`automl` 預設不於每折搜參；需 `automl_frozen`。深度模型預設跳過 wf，除非 `--wf-allow-deep`。  
- **ONNX**：主線以樹／線性為主；Torch 預設不轉 ONNX。  
- **外部 API**：無金鑰時部分 connector 為後備資料。  

---

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml)：多矩陣 pytest、`requirements-automl` 子集測試，以及 **quality**（**全部已追蹤 `.py` 之 compileall（排除 `archive/`）**、import smoke、`ruff check tests/`）。

---

## Release

請在 GitHub 建立 **Release**（或至少本機 `git tag`）：目前展示版號建議 **`v0.9.2`**（與 `VERSION` 一致）。若日後封板可再考慮 `v1.0.0-beta`。

---

## GitHub Topics 建議

`machine-learning` `time-series` `forecasting` `automl` `pytorch` `onnx` `xgboost` `lightgbm` `python`

（請於倉庫 About 手動加入，見上節連結。）
