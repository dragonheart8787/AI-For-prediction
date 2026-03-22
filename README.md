# 多源時序預測管線（Predict-AI）

**一句話**：從 Yahoo / Open-Meteo / OWID / EIA / NewsAPI 等來源拉資料、時間對齊、訓練統一迴歸模型，並可選 HTTP 推論。

**版號（唯一來源）**：請以根目錄 [`VERSION`](VERSION) 為準；目前為 **`0.9.3`**，並與 [`pyproject.toml`](pyproject.toml)、[`CHANGELOG.md`](CHANGELOG.md) **首條 `## v…`**、本 README 一致（CI 會跑 [`scripts/check_repo_policy.py`](scripts/check_repo_policy.py) 驗證）。  
**授權**：[MIT](LICENSE)  
**Repository**：[github.com/dragonheart8787/AI-For-prediction](https://github.com/dragonheart8787/AI-For-prediction)

### 你看到的畫面與這裡不符？

若 GitHub 首頁根目錄仍出現 **`accelerators/`、`configs/`、`serving/`、`paper/`、`AGI_*.md`** 等，幾乎可判定是 **錯倉庫**、**舊預設分支**，或 **未 `git pull`**：

| 檢查 | 預期（本倉庫 `main` 最新） |
|------|---------------------------|
| 遠端 URL | `https://github.com/dragonheart8787/AI-For-prediction.git`（舊名 `-AI-For-prediction` 可能仍轉址，但內容應已同步） |
| `git log --oneline -5` | 應見 **多個** commit（非單一 Initial commit） |
| 根層資料夾 | **不應**有上表舊目錄；它們應只在 **`archive/`** 內 |
| [`VERSION`](VERSION) 與 [`CHANGELOG.md`](CHANGELOG.md) 第一條 | 版號相同（目前 **`0.9.3`**） |
| [Actions](https://github.com/dragonheart8787/AI-For-prediction/actions) | 應有 **3 個 job**：`test`、`test-automl-torch`、`quality` |

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

Workflow 檔：[`.github/workflows/ci.yml`](https://github.com/dragonheart8787/AI-For-prediction/blob/main/.github/workflows/ci.yml)（檔案頂部註解亦列出 job）。實際跑 **3 個 job**：

| Job | 內容 |
|-----|------|
| **`test`** | `ubuntu-latest` + `windows-latest` × Python 3.11／3.12；`pip install -r requirements-core.txt`；`pytest tests/` |
| **`test-automl-torch`** | Ubuntu；core + automl；`pytest tests/test_automl_torch.py` |
| **`quality`** | `scripts/check_repo_policy.py`；`compileall` 所有已追蹤 `*.py`（**排除 `archive/`**）；import smoke；`ruff check tests/` |

---

## Release

步驟見 [`.github/RELEASE_INSTRUCTIONS.md`](.github/RELEASE_INSTRUCTIONS.md)。目前 tag 建議與 [`VERSION`](VERSION) 一致：**`v0.9.3`**。

---

## GitHub Topics 建議

Topics **須逐個加入**（每個以小寫／數字開頭、僅含連字號，見 [`.github/GITHUB_ABOUT.md`](.github/GITHUB_ABOUT.md)）。建議九個：

`machine-learning`、`time-series`、`forecasting`、`automl`、`pytorch`、`onnx`、`xgboost`、`lightgbm`、`python`
