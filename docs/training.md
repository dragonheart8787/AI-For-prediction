# 訓練說明

> **本文為訓練正文**（安裝分層、CLI、產物、強化手段），不是僅列連結的 stub。HTTP 見 [`serving.md`](serving.md)。

---

## 1. 安裝：core 與 automl 差異

| 面向 | **`requirements-core.txt`** | **`requirements-automl.txt`**（疊加安裝） |
|------|-----------------------------|------------------------------------------|
| **用途** | 主線爬蟲管線、樹／線性、`ensemble`、測試與 HTTP 載入推論 | Optuna **AutoML**、`mlp_torch` / **LSTM** / **Transformer** 等 **PyTorch** 路徑 |
| **典型指令** | `python crawler_train_pipeline.py TASK --model linear` | 先 `pip install -r requirements-automl.txt`，再 `--model automl` 或 `--model lstm` 等 |
| **CI** | 全矩陣 `pytest tests/` | 另 job：`tests/test_automl_torch.py` |

```bash
pip install -r requirements-core.txt
# 需要 AutoML / 深度模型時再加：
# pip install -r requirements-automl.txt
```

環境變數範例：專案根目錄 [`.env.example`](../.env.example)。

---

## 2. 常用 CLI（`crawler_train_pipeline.py`）

任務 ID 定義於 **`config/prediction_schema.yaml`**。

```bash
# 最小：單一任務 + 線性模型
python crawler_train_pipeline.py stock_price_next --model linear

# 模型與產物目錄
python crawler_train_pipeline.py stock_price_next --model linear --models-dir ./models

# 資料根（影響預設 memory／實驗路徑等，見 runtime_paths）
python crawler_train_pipeline.py stock_price_next --model linear --data-dir ./data

# 強化：衍生特徵 + 樹模型強預設 + 區間 1y + walk-forward
python crawler_train_pipeline.py stock_price_next \
  --model ensemble \
  --preset strong \
  --rich-features \
  --period 1y \
  --walk-forward
```

**完整旗標**請以官方說明為準：

```bash
python crawler_train_pipeline.py --help
```

---

## 3. 模型與強化手段（精簡）

演算法只是上限的一部分；**資料量、特徵、標籤**更關鍵。

| 手段 | 說明 |
|------|------|
| **衍生時序特徵** | `--rich-features` 或 YAML `feature_expansion` |
| **強預設（樹）** | `--preset strong` |
| **Early stopping** | 時間序後段作驗證（樹模型） |
| **Ensemble** | `--model ensemble`（建議已裝 `xgboost`） |
| **AutoML** | `--model automl`（需 Optuna；見 `requirements-automl.txt`） |
| **深度模型** | `lstm` / `transformer` 等（需 torch）；walk-forward 預設跳過，需 `--wf-allow-deep` |
| **Walk-forward** | `--walk-forward`；`automl` 每折需事先搜參並以 `automl_frozen` 傳入（防洩漏與成本） |

較完整表格與範例（歸檔原文目錄）：[`legacy-archive/archive/documentation`](https://github.com/dragonheart8787/AI-For-prediction/tree/legacy-archive/archive/documentation)（內含 `強模型訓練要點.md` 等）。

---

## 4. 常見產物（訓練後）

| 路徑 | 說明 |
|------|------|
| **`models/task_<task_id>.pkl`**（或 `PREDICT_AI_MODELS_DIR`／`--models-dir`） | 主線儲存的預測器快照，供 `launch_predict_service` 載入。 |
| **`artifacts/<task>/<run_id>/`** | `config.json`、`metrics.json`、`summary.json`、`feature_manifest.json`、`model_path.txt` 等 run 摘要。 |
| **`data/experiment_runs.jsonl`** | 實驗事件（路徑可於 schema 設定）。 |
| **`data/training_memory.jsonl`** | 部分管線記憶／重放設定（依 CLI／schema）。 |

---

## 5. 相關文件

- [架構與流程圖](architecture.md)  
- [HTTP 推論](serving.md)  
- [能力對照](capabilities.md)  
- [`config/README.md`](../config/README.md)  
