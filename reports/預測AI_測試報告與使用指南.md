# 預測 AI 測試報告與使用指南

> 本報告包含測試結果、使用方式與訓練教學，讓您能快速上手並訓練自己的預測模型。

---

## 一、測試報告

### 1.1 測試執行摘要

| 項目 | 結果 |
|------|------|
| **測試套件** | `tests/test_unified_predict.py` |
| **總測試數** | 42 |
| **通過** | 42 ✓ |
| **失敗** | 0 |
| **成功率** | 100% |
| **執行時間** | ~2 秒 |

### 1.2 測試項目清單

| 類別 | 測試項目 | 狀態 |
|------|----------|------|
| **基本 fit/predict** | test_linear_fit_returns_metrics | ✓ |
| | test_predict_returns_dict | ✓ |
| | test_predict_shape | ✓ |
| | test_predict_before_fit_raises | ✓ |
| | test_fit_dimension_mismatch_raises | ✓ |
| | test_fit_empty_raises | ✓ |
| **多地平線** | test_multi_horizon_fit_predict | ✓ |
| **批次預測** | test_predict_many_basic | ✓ |
| | test_predict_many_single_batch | ✓ |
| | test_predict_many_before_fit_raises | ✓ |
| **輸入格式** | test_list_of_lists | ✓ |
| | test_dict_input | ✓ |
| | test_1d_array_reshaped | ✓ |
| | test_pandas_dataframe | ✓ |
| **資料驗證** | test_nan_replaced | ✓ |
| | test_predict_with_nan | ✓ |
| **正規化** | test_normalize_improves_or_maintains | ✓ |
| | test_scaler_applied_on_predict | ✓ |
| **評估** | test_evaluate_returns_metrics | ✓ |
| | test_evaluate_before_fit_raises | ✓ |
| **模型儲存/載入** | test_save_and_load | ✓ |
| | test_save_before_fit_raises | ✓ |
| | test_load_missing_file_raises | ✓ |
| **快取** | test_cache_returns_same_result | ✓ |
| | test_cache_cleared_on_fit | ✓ |
| **LRU Cache** | test_get_set, test_eviction, test_clear | ✓ |
| **Token Bucket** | test_allow_within_capacity, test_refill | ✓ |
| **置信度** | test_confidence_between_0_and_1 | ✓ |
| | test_confidence_constant_prediction | ✓ |
| | test_confidence_high_variance | ✓ |
| **ONNX** | test_auto_onnx_linear | ✓ |
| | test_auto_onnx_disabled | ✓ |
| **模型資訊** | test_info_before_fit, test_info_after_fit | ✓ |
| **領域地平線** | test_financial_horizons | ✓ |
| | test_unknown_domain_fallback | ✓ |
| **回退模型** | test_fallback_with_unknown_model | ✓ |
| **端到端** | test_full_pipeline | ✓ |
| | test_predict_many_consistency | ✓ |

### 1.3 執行測試指令

```bash
cd <專案根目錄>
python -m pytest tests/test_unified_predict.py -v --tb=short
```

---

## 二、如何使用這個預測 AI

### 2.1 快速開始（3 步驟）

```python
from unified_predict import UnifiedPredictor
import numpy as np

# 1. 建立預測器
predictor = UnifiedPredictor(auto_onnx=False)

# 2. 準備資料並訓練
X = np.random.randn(100, 5)   # 100 筆樣本，5 個特徵
y = np.random.randn(100)       # 100 個目標值
metrics = predictor.fit(X, y, model="linear")

# 3. 預測
result = predictor.predict(X[:10])
print(result["prediction"])   # 預測值
print(result["confidence"])   # 置信度 0~1
```

### 2.2 支援的輸入格式

| 格式 | 範例 |
|------|------|
| **numpy 陣列** | `np.array([[1,2,3], [4,5,6]])` |
| **二維列表** | `[[1,2,3], [4,5,6]]` |
| **字典列表** | `[{"a":1, "b":2}, {"a":3, "b":4}]` |
| **pandas DataFrame** | `df[["f1", "f2", "f3"]]` |

### 2.3 支援的模型

| 模型 | 說明 | 安裝需求 |
|------|------|----------|
| `linear` | 線性回歸（預設） | scikit-learn |
| `xgboost` | 梯度提升樹 | `pip install xgboost` |
| `lightgbm` | 輕量級梯度提升 | `pip install lightgbm` |

### 2.4 預測結果格式

```python
result = predictor.predict(X)

# result 結構：
{
    "domain": "financial",           # 領域
    "horizons": [1, 5, 10, 20],     # 預測地平線
    "model": "linear",               # 使用的模型
    "prediction": [[0.5], [0.3]],   # 預測值（每筆樣本）
    "confidence": 0.85,              # 整體置信度 0~1
    "n_samples": 2                  # 預測樣本數
}
```

---

## 三、如何訓練這個 AI

### 3.1 基本訓練流程

```python
from unified_predict import UnifiedPredictor
import numpy as np

# 建立預測器（可選：normalize=True 啟用資料正規化）
predictor = UnifiedPredictor(normalize=True, auto_onnx=False)

# 準備訓練資料
X_train = np.random.randn(500, 8)   # 500 筆，8 個特徵
y_train = np.random.randn(500)      # 目標值

# 訓練並取得指標
metrics = predictor.fit(X_train, y_train, model="linear")

print("訓練指標：", metrics)
# 輸出範例：
# {
#   "train_rmse": 0.12,
#   "train_mae": 0.09,
#   "train_r2": 0.95,
#   "samples": 500,
#   "features": 8,
#   "model": "linear"
# }
```

### 3.2 使用自己的資料訓練

```python
import pandas as pd
from unified_predict import UnifiedPredictor

# 從 CSV 載入
df = pd.read_csv("your_data.csv")

# 假設：前 5 欄是特徵，最後 1 欄是目標
X = df.iloc[:, :5].values
y = df.iloc[:, 5].values

predictor = UnifiedPredictor(normalize=True)
predictor.fit(X, y, model="linear")

# 驗證集評估
X_val = df_val.iloc[:, :5].values
y_val = df_val.iloc[:, 5].values
eval_metrics = predictor.evaluate(X_val, y_val)
print("驗證 RMSE:", eval_metrics["rmse"])
print("驗證 R²:", eval_metrics["r2"])
```

### 3.3 多地平線預測（多目標）

```python
# y 可以是 [N, H] 維度，H 為多個預測目標
X = np.random.randn(200, 4)
y = np.random.randn(200, 3)   # 3 個地平線/目標

predictor = UnifiedPredictor()
predictor.fit(X, y, model="linear")

result = predictor.predict(X[:5])
# result["prediction"] 形狀為 5x3
```

### 3.4 進階訓練參數（XGBoost / LightGBM）

```python
predictor.fit(
    X, y,
    model="xgboost",
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
)
```

### 3.5 儲存與載入訓練好的模型

```python
# 訓練後儲存
predictor.fit(X_train, y_train, model="linear")
predictor.save_model("my_model.pkl")

# 之後載入使用
predictor2 = UnifiedPredictor()
predictor2.load_model("my_model.pkl")
result = predictor2.predict(X_new)
```

---

## 四、完整範例腳本

### 4.1 從頭到尾的範例

```python
"""
預測 AI 完整使用範例
"""
import numpy as np
from unified_predict import UnifiedPredictor

# === 1. 產生或載入資料 ===
rng = np.random.default_rng(42)
X = rng.normal(size=(500, 6))
w = rng.normal(size=(6,))
y = X @ w + rng.normal(scale=0.2, size=(500,))

# 分割訓練/驗證
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

# === 2. 建立並訓練 ===
predictor = UnifiedPredictor(normalize=True)
train_metrics = predictor.fit(X_train, y_train, model="linear")
print("訓練 R²:", train_metrics["train_r2"])

# === 3. 驗證 ===
eval_metrics = predictor.evaluate(X_val, y_val)
print("驗證 RMSE:", eval_metrics["rmse"])
print("驗證 R²:", eval_metrics["r2"])

# === 4. 預測 ===
result = predictor.predict(X_val[:10])
print("前 10 筆預測:", result["prediction"])
print("置信度:", result["confidence"])

# === 5. 儲存模型 ===
predictor.save_model("trained_model.pkl")
print("模型已儲存")
```

### 4.2 執行內建示範

```bash
cd <專案根目錄>
python unified_predict.py
```

會輸出訓練指標、驗證指標、預測結果與模型資訊。

---

## 五、依賴安裝

### 5.1 最低需求（線性回歸）

```bash
pip install numpy scikit-learn pyyaml
```

### 5.2 完整功能（含 XGBoost、LightGBM、pandas）

```bash
pip install numpy scikit-learn pyyaml pandas xgboost lightgbm
```

---

## 六、常見問題

| 問題 | 解法 |
|------|------|
| `Model is not fitted` | 先呼叫 `fit()` 再 `predict()` |
| `Rate limited` | 預測請求過於頻繁，稍後再試 |
| X 與 y 樣本數不一致 | 檢查 `X.shape[0] == len(y)` |
| 想用 DataFrame | 直接傳入 `df`，會自動提取數值欄位 |
| 資料有 NaN | 會自動替換為 0，並記錄警告 |

---

## 七、爬蟲訓練管線（爬取資料 → 訓練預測 AI）

### 7.1 功能說明

- **爬蟲取得資料**：依預測任務自動選擇 Yahoo、EIA、NewsAPI、Open-Meteo、OWID 等來源
- **預測相關訓練**：將爬取資料轉成 X/y，訓練 UnifiedPredictor 做實際預測
- **AI 學習所需資料**：訓練後分析特徵重要性，告訴你「預測這個目標需要哪些資料」

### 7.2 可用預測任務（共 17 種）

| 任務 ID | 說明 | 資料來源 |
|---------|------|----------|
| `stock_price_next` | 股價下一日預測 | Yahoo |
| `stock_return` | 股票報酬率預測 | Yahoo |
| `crypto_btc_next` | 比特幣價格預測 | Yahoo |
| `crypto_eth_next` | 以太坊價格預測 | Yahoo |
| `forex_usdjpy_next` | 美元日圓匯率預測 | Yahoo |
| `forex_eurusd_next` | 歐元美元匯率預測 | Yahoo |
| `gold_price_next` | 黃金價格預測 | Yahoo |
| `oil_price_next` | 原油價格預測 | Yahoo |
| `sp500_next` | 標普500指數預測 | Yahoo |
| `twii_next` | 台灣加權指數預測 | Yahoo |
| `nasdaq_next` | 那斯達克指數預測 | Yahoo |
| `temperature_next` | 氣溫下一時段預測 | Open-Meteo |
| `energy_demand_next` | 能源需求預測 | EIA |
| `news_sentiment_next` | 新聞情感趨勢預測 | NewsAPI |
| `covid_cases_next` | 新增病例預測 | OWID |
| `multi_source_price` | 多源股價預測（金融+新聞） | Yahoo + NewsAPI |

### 7.3 預測目標顧問（AI 知道要找什麼資料）

當使用者提出預測目標時，AI 會自動判斷需要爬取哪些資料：

```bash
# 查詢：預測股價需要什麼資料？
python prediction_target_advisor.py "我想預測股價"

# 輸出範例：
# 要預測「股價下一日預測」，需要爬取以下資料：
#   - Yahoo Finance 股價資料：open、high、low、close、volume
# 執行訓練指令: python crawler_train_pipeline.py stock_price_next
```

支援的自然語言範例：`股價`、`台北氣溫`、`能源需求`、`疫情`、`新聞情感` 等。

### 7.4 使用方式

```bash
# 列出所有任務與所需資料
python crawler_train_pipeline.py --list-tasks

# 用自然語言指定預測目標（AI 自動匹配任務）
python crawler_train_pipeline.py "我想預測股價"
python crawler_train_pipeline.py "台北氣溫"
python crawler_train_pipeline.py "能源需求"

# 或用任務 ID
python crawler_train_pipeline.py stock_price_next --model linear

# 指定股票與期間
python crawler_train_pipeline.py stock_price_next --symbol ^GSPC --period 6mo

# 儲存訓練好的模型
python crawler_train_pipeline.py stock_price_next --save models/stock_predictor.pkl

# 使用 XGBoost 訓練
python crawler_train_pipeline.py stock_price_next --model xgboost

# 啟用經驗回放防遺忘（增量訓練時混合舊任務樣本）
python crawler_train_pipeline.py "黃金價格" --replay

# 訓練所有任務並建立通用模型（可預測未訓練過的目標）
python crawler_train_pipeline.py --train-all --model linear
```

### 7.5 預測所需資料學習

訓練完成後，管線會輸出**特徵重要性**，例如：

```
特徵重要性排序（預測此目標最需要的資料）:
  1. high     #################### 1.54
  2. low      ################## 1.30
  3. close    ############### 0.96
  4. open     ########## 0.52
  5. volume   - 0.05

-> 預測 [股價下一日預測] 建議優先收集的資料: ['high', 'low', 'close', 'open', 'volume']
```

代表 AI 已學習到：預測股價時，`high`、`low`、`close` 最重要，`volume` 影響較小。

### 7.6 防遺忘機制（經驗回放）

增量訓練新任務時，使用 `--replay` 可混合舊任務樣本，避免遺忘已學到的知識：

```bash
python crawler_train_pipeline.py "黃金價格" --replay
```

### 7.7 預測未訓練過的目標

先執行全任務訓練建立通用模型：

```bash
python crawler_train_pipeline.py --train-all
```

之後可用 `universal_predictor.py` 預測**從未訓練過的任務**：

```bash
python universal_predictor.py unknown_task --features open high low close volume --values 100 102 98 101 1000000
```

預測策略：專用模型 > 通用模型 > 最相似任務 > 零樣本均值。

### 7.8 自訂預測任務與關鍵字

- **新增任務**：編輯 `config/prediction_schema.yaml`，指定 `target_source`、`data_sources` 等。
- **新增關鍵字**：編輯 `config/prediction_target_keywords.yaml`，在對應任務下加入關鍵字，使用者說出該詞時即會匹配該任務。

編輯 `config/prediction_schema.yaml`，可新增任務並指定：

- `target_source`：目標欄位（如 close、temperature_2m）
- `target_horizon`：預測步數（1 = 下一步）
- `data_sources`：要爬取的 connector 與特徵欄位

---

## 八、報告產生資訊

- **產生日期**：2025-03-15
- **測試環境**：Windows 10, Python 3.14
- **核心模組**：`unified_predict.py`
