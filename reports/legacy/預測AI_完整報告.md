# 預測 AI 系統 - 完整報告

**文件版本**：1.0  
**產生日期**：2025 年 3 月  
**專案路徑**：請以本機 clone 後之**專案根目錄**為準（勿硬編碼絕對路徑）。

---

## 目錄

1. [摘要](#一摘要)
2. [專案概述](#二專案概述)
3. [系統架構](#三系統架構)
4. [能力說明](#四能力說明)
5. [測試報告](#五測試報告)
6. [效能基準](#六效能基準)
7. [使用指南](#七使用指南)
8. [配置說明](#八配置說明)
9. [已知限制與風險](#九已知限制與風險)
10. [建議改進](#十建議改進)
11. [附錄](#十一附錄)

---

## 一、摘要

本專案為一整合式**預測 AI 系統**，具備以下核心特色：

- **統一預測介面**：支援線性回歸、XGBoost、LightGBM、ONNX 等多種模型
- **爬蟲訓練管線**：從 5 類資料來源（Yahoo、EIA、NewsAPI、Open-Meteo、OWID）爬取資料並訓練
- **17 種預測任務**：涵蓋股價、加密貨幣、外匯、商品、指數、天氣、能源、新聞、疫情等
- **自然語言任務匹配**：使用者輸入「我想預測股價」等描述，自動對應任務與資料來源
- **防遺忘機制**：經驗回放避免增量訓練時遺忘舊知識
- **通用預測**：可預測未訓練過的目標（通用模型、最近任務、零樣本回退）

**測試摘要**：核心測試 42/42 通過，效能基準 6/6 通過。

---

## 二、專案概述

### 2.1 專案目標

打造一個可擴充的預測 AI 平台，能夠：

1. 從多種外部來源爬取資料  
2. 根據預測目標自動選擇資料來源  
3. 訓練並部署預測模型  
4. 支援未訓練過的預測目標泛化推論  

### 2.2 主要模組

| 模組 | 檔案 | 用途 |
|------|------|------|
| 統一預測器 | `unified_predict.py` | 訓練、預測、儲存、評估 |
| 爬蟲訓練管線 | `crawler_train_pipeline.py` | 爬取→轉換→訓練→儲存 |
| 預測目標顧問 | `prediction_target_advisor.py` | 自然語言→任務→資料需求 |
| 訓練資料儲存 | `training_data_store.py` | 經驗回放、防遺忘 |
| 通用預測器 | `universal_predictor.py` | 未訓練目標預測 |
| 資料連接器 | `data_connectors/*.py` | Yahoo、EIA、NewsAPI、Open-Meteo、OWID、REST |
| Schema 推斷 | `schema_infer.py` | 時間戳推斷、rows→features |
| 特徵儲存 | `feature_store.py` | JSONL 特徵儲存 |

### 2.3 關鍵配置檔

| 檔案 | 用途 |
|------|------|
| `config/tasks.yaml` | 領域、地平線、模型路由、ONNX |
| `config/prediction_schema.yaml` | 預測任務、目標、資料來源、canonical 特徵 |
| `config/prediction_target_keywords.yaml` | 自然語言關鍵字→任務對應 |
| `config/schema.json` | 資料 schema |

---

## 三、系統架構

```
                         ┌─────────────────────────────┐
                         │    預測目標顧問              │
                         │  自然語言 → 任務 ID          │
                         │  任務 → 建議資料來源         │
                         └──────────────┬──────────────┘
                                        │
┌───────────────────────────────────────┼───────────────────────────────────────┐
│                                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  爬蟲訓練管線 (crawler_train_pipeline.py)                               │  │
│  │  • 單任務：crawl_and_build_xy → fit → 特徵重要性                        │  │
│  │  • 全任務：--train-all → 合併 canonical → 通用模型                      │  │
│  │  • 防遺忘：--replay → merge_with_replay                                 │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                           │
│  ┌───────────────────────────────┼───────────────────────────────────────┐  │
│  │  資料連接器 (data_connectors)  │  TrainingDataStore                      │  │
│  │  Yahoo / EIA / NewsAPI /       │  經驗回放、全任務合併                   │  │
│  │  OpenMeteo / OWID / REST       │  data/training_memory.jsonl             │  │
│  └───────────────────────────────┼───────────────────────────────────────┘  │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  UnifiedPredictor (unified_predict.py)                                   │  │
│  │  linear / xgboost / lightgbm / ensemble / automl / torch(LSTM等) / ONNX   │  │
│  │  LRU 快取、限流、正規化、儲存                                             │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   ▼
              ┌────────────────────────────────────────────────────────────────┐
              │  UniversalPredictor (universal_predictor.py)                    │
              │  策略：專用模型 → 通用模型 → 最近任務 → 零樣本均值              │
              │  支援未訓練過的預測目標                                          │
              └────────────────────────────────────────────────────────────────┘
```

---

## 四、能力說明

### 4.1 UnifiedPredictor（統一預測器）

| 能力項目 | 狀態 | 說明 |
|----------|------|------|
| 線性回歸 | ✓ | sklearn LinearRegression |
| XGBoost | ✓ | 需安裝 xgboost |
| LightGBM | ✓ | 需安裝 lightgbm |
| ONNX 包裝 | ✓ | 自動轉 linear/xgb/lgb 為快速推論 |
| 多地平線輸出 | ✓ | 依 domain horizons |
| 輸入格式 | ✓ | numpy, list, dict, pandas DataFrame |
| 資料正規化 | ✓ | StandardScaler 可選 |
| 模型儲存/載入 | ✓ | pickle |
| 評估指標 | ✓ | fit 回傳 train_rmse/mae/r2，evaluate() |
| LRU 快取 | ✓ | 64 條目 |
| 限流 | ✓ | Token Bucket 60 容量 |
| NaN/Inf 處理 | ✓ | 自動替換為 0 |

### 4.2 爬蟲訓練管線

| 能力項目 | 狀態 | 說明 |
|----------|------|------|
| 單任務訓練 | ✓ | 爬取 → X/y → 訓練 → 儲存 |
| 自然語言匹配 | ✓ | 透過 prediction_target_advisor |
| 全任務訓練 | ✓ | `--train-all` |
| 經驗回放防遺忘 | ✓ | `--replay` |
| 特徵重要性分析 | ✓ | 線性係數、XGB/LGB feature_importances_ |
| 17 種預測任務 | ✓ | 見 4.6 節 |

### 4.3 預測目標顧問

| 能力項目 | 狀態 | 說明 |
|----------|------|------|
| 自然語言→任務 ID | ✓ | 關鍵字匹配 |
| 資料需求建議 | ✓ | connector、features |
| 未知目標回退 | ✓ | default_task |

### 4.4 訓練資料儲存與通用預測器

| 模組 | 能力 |
|------|------|
| TrainingDataStore | 儲存訓練樣本、經驗回放、全任務合併 |
| UniversalPredictor | 專用模型、通用模型、最近任務、零樣本 |

### 4.5 資料連接器

| Connector | 資料類型 | 備註 |
|-----------|----------|------|
| YahooFinanceConnector | 股價、加密貨幣、外匯、商品、指數 | symbol: AAPL, BTC-USD, GC=F 等 |
| EIAConnector | 能源 | 需 API key，無時離線後備 |
| NewsAPIConnector | 新聞情感 | 需 API key，離線後備 |
| OpenMeteoConnector | 天氣 | 免費 API |
| OWIDConnector | 疫情 | 需網路 |
| RESTConnector | 通用 REST | 自訂 url、root_path、fields |

### 4.6 預測任務一覽（17 種）

| 類別 | 任務 ID | 說明 |
|------|---------|------|
| 股票 | stock_price_next, stock_return | 股價、報酬率 |
| 加密貨幣 | crypto_btc_next, crypto_eth_next | 比特幣、以太坊 |
| 外匯 | forex_usdjpy_next, forex_eurusd_next | 美元日圓、歐元美元 |
| 商品 | gold_price_next, oil_price_next | 黃金、原油 |
| 指數 | sp500_next, twii_next, nasdaq_next | 標普、台股、那斯達克 |
| 天氣 | temperature_next | 氣溫 |
| 能源 | energy_demand_next | 能源需求 |
| 新聞 | news_sentiment_next | 新聞情感 |
| 疫情 | covid_cases_next | 新增病例 |
| 綜合 | multi_source_price | 股價+新聞 |

---

## 五、測試報告

### 5.1 測試結果總表

| 測試套件 | 通過 | 失敗 | 總數 | 備註 |
|----------|------|------|------|------|
| test_unified_predict.py | 42 | 0 | 42 | 核心功能 |
| benchmark_all.py | 6 | 0 | 6 | 全模組基準 |
| test_data_connectors.py | - | - | - | 需修正匯入 |
| test_performance.py | - | - | - | 使用舊 API |

### 5.2 測試項目清單（test_unified_predict）

| 類別 | 項目數 | 狀態 |
|------|--------|------|
| 基本 fit/predict | 6 | 全通過 |
| 多地平線 | 1 | 通過 |
| 批次預測 | 3 | 通過 |
| 輸入格式 (list/dict/array/DataFrame) | 4 | 通過 |
| 資料驗證 (NaN/Inf) | 2 | 通過 |
| 正規化 | 2 | 通過 |
| 評估 | 2 | 通過 |
| 模型儲存/載入 | 3 | 通過 |
| 快取 | 2 | 通過 |
| LRU Cache | 3 | 通過 |
| Token Bucket | 2 | 通過 |
| 置信度 | 3 | 通過 |
| ONNX | 2 | 通過 |
| 模型資訊 | 2 | 通過 |
| 地平線 | 2 | 通過 |
| 回退模型 | 1 | 通過 |
| 端到端 | 2 | 通過 |

### 5.3 執行指令

```bash
cd <專案根目錄>
python -m pytest tests/test_unified_predict.py -v --tb=short
```

---

## 六、效能基準

### 6.1 基準測試結果（benchmark_all.py）

| 模組 | 指標 | 數值 |
|------|------|------|
| **UnifiedPredictor** | 訓練時間 | 2.29 ms |
| | 訓練 R² | 0.9932 |
| | 評估 R² | 0.9932 |
| | 50 次 predict | 3.03 ms |
| | predict_many 吞吐 | ~615,820 samples/s |
| **PredictionAdvisor** | 5 次查詢 | 71.93 ms |
| | 平均單次 | 14.39 ms |
| **TrainingDataStore** | add | 0.76 ms |
| | replay | 0.17 ms |
| **CrawlerPipeline** | 單任務爬取 | 16.71 ms |
| | 樣本數 | 29（stock_price_next）|
| **DataConnectors** | yahoo | 30 筆, 0.36 ms |
| | eia | 52 筆, 0.33 ms |
| | newsapi | 32 筆, 0.36 ms |
| | open_meteo | 48 筆, ~1070 ms |

### 6.2 執行指令

```bash
python benchmark_all.py
```

結果輸出：`reports/benchmark_report.json`

---

## 七、使用指南

### 7.1 快速開始（3 步驟）

```python
from unified_predict import UnifiedPredictor
import numpy as np

predictor = UnifiedPredictor(auto_onnx=False)
X = np.random.randn(100, 5)
y = np.random.randn(100)
predictor.fit(X, y, model="linear")
result = predictor.predict(X[:10])
print(result["prediction"], result["confidence"])
```

### 7.2 爬蟲訓練管線

```bash
# 列出所有任務
python crawler_train_pipeline.py --list-tasks

# 自然語言指定預測目標
python crawler_train_pipeline.py "我想預測股價"
python crawler_train_pipeline.py "黃金價格"

# 單任務訓練
python crawler_train_pipeline.py stock_price_next --model linear

# 經驗回放防遺忘
python crawler_train_pipeline.py "能源需求" --replay

# 全任務訓練建立通用模型
python crawler_train_pipeline.py --train-all
```

### 7.3 預測目標顧問

```bash
python prediction_target_advisor.py "我想預測股價"
# 輸出：需爬取 Yahoo Finance 的 open, high, low, close, volume
```

### 7.4 預測未訓練過的目標

```bash
# 先執行 --train-all 建立通用模型
python crawler_train_pipeline.py --train-all

# 預測未知任務
python universal_predictor.py unknown_task --features open high low close volume --values 100 102 98 101 1000000
```

### 7.5 依賴安裝

```bash
# 最低需求
pip install numpy scikit-learn pyyaml

# 完整功能
pip install numpy scikit-learn pyyaml pandas xgboost lightgbm yfinance
```

---

## 八、配置說明

### 8.1 config/prediction_schema.yaml

- `canonical_features`：跨任務通用特徵順序  
- `prediction_tasks`：任務 ID、display_name、domain、target_source、target_horizon、data_sources  

### 8.2 config/prediction_target_keywords.yaml

- `target_keywords`：任務 ID → 關鍵字列表  
- `default_task`：未匹配時預設任務  

### 8.3 config/tasks.yaml

- `domains`：領域與 horizons  
- `routing`：default_model、baselines  

---

## 九、已知限制與風險

### 9.1 技術限制

| 項目 | 說明 |
|------|------|
| 限流 | predict 約 60 次/分鐘，超過會 RuntimeError |
| 多資料源時間對齊 | 僅按 timestamp 排序，無交叉對齊 |
| canonical 對齊 | 缺失特徵填 0 |
| JSONL 儲存 | 大資料量 I/O 較慢 |
| OWID | 需網路，離線無法爬取 |

### 9.2 依賴風險

- yfinance：Yahoo 變更時可能失效  
- NewsAPI、EIA：需 API key 才有真實資料  
- scikit-learn：必要，無時退回 dict 模型  

### 9.3 測試缺口

- test_data_connectors：需更新匯入  
- test_performance：需改寫為現有 API  

---

## 十、建議改進

1. 限流可調：建構時可傳入 capacity、refill 或關閉  
2. 爬蟲重試：connector 加入重試與退避  
3. 更新 test_performance、test_data_connectors 以符合現有 API  
4. 將 benchmark 納入 CI，定期記錄效能  

---

## 十一、附錄

### A. 檔案清單（核心）

```
unified_predict.py           # 統一預測器
crawler_train_pipeline.py    # 爬蟲訓練管線
prediction_target_advisor.py # 預測目標顧問
training_data_store.py       # 訓練資料儲存
universal_predictor.py       # 通用預測器
benchmark_all.py             # 效能基準
data_connectors/
  yahoo.py, eia.py, newsapi.py, open_meteo.py, owid.py, rest_generic.py
config/
  tasks.yaml, prediction_schema.yaml, prediction_target_keywords.yaml
tests/
  test_unified_predict.py
reports/
  預測AI_完整報告.md
  benchmark_report.json
```

### B. 常見問題

| 問題 | 解法 |
|------|------|
| Model is not fitted | 先呼叫 fit() |
| Rate limited | 稍後再試或調整限流 |
| X 與 y 樣本數不一致 | 檢查 shape/len |
| 資料有 NaN | 會自動替換為 0 |

### C. 報告產生資訊

- **測試環境**：Windows 10, Python 3.14  
- **報告產生**：2025 年 3 月  

---

*本報告整合測試結果、效能基準、使用指南與系統分析，作為專案完整文件。*
