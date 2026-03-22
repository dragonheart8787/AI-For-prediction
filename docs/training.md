# 訓練說明

---

## 1. 安裝

```bash
pip install -r requirements-core.txt
# 選配 AutoML + PyTorch
# pip install -r requirements-automl.txt
```

環境變數範例：專案根目錄 [`.env.example`](../.env.example)。

---

## 2. 基本指令

任務 ID 定義於 `config/prediction_schema.yaml`。

```bash
python crawler_train_pipeline.py stock_price_next --model linear
python crawler_train_pipeline.py stock_price_next --model linear --models-dir ./models
python crawler_train_pipeline.py stock_price_next --model linear --data-dir ./data
```

完整參數：

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

較完整表格與範例（歸檔原文）：[`archive/documentation/強模型訓練要點.md`](../archive/documentation/強模型訓練要點.md)。

**盡力壓榨管線的範例**（需 `xgboost`、`yfinance` 等）：

```bash
pip install xgboost yfinance pandas

python crawler_train_pipeline.py stock_price_next \
  --model ensemble \
  --preset strong \
  --rich-features \
  --period 1y \
  --walk-forward
```

Windows CMD 將 `\` 換成 `^` 斷行即可。

---

## 4. 產物位置

| 路徑 | 說明 |
|------|------|
| `models/` 或 `PREDICT_AI_MODELS_DIR`／`--models-dir` | `task_<task_id>.pkl` 等 |
| `artifacts/<task>/<run_id>/` | `config.json`、`metrics.json`、`summary.json`、`feature_manifest.json`、`model_path.txt` |
| `data/experiment_runs.jsonl` | 實驗事件（路徑可於 schema 設定） |

---

## 5. 相關文件

- [架構總覽](architecture.md)  
- [HTTP 推論](serving.md)  
- [能力對照](capabilities.md)  
