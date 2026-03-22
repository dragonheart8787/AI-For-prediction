# 🚀 AGI智能預測系統使用指南

## 📋 系統概述

這是一個完整的AGI（人工通用智能）預測系統，具備以下功能：

- 🕷️ **數據爬取**: 自動收集金融、天氣、醫療、能源等領域數據
- 🧠 **模型訓練**: 使用LSTM和Transformer架構訓練AI模型
- 🎯 **智能預測**: 多領域預測和模型融合
- 📊 **性能監控**: 實時監控系統性能和模型表現
- 🔄 **自動恢復**: 故障檢測和自動修復

## 🛠️ 快速開始

### 1. 環境準備

確保您已安裝以下Python套件：

```bash
pip install numpy pandas scikit-learn joblib aiohttp beautifulsoup4
```

### 2. 運行完整演示

```bash
python agi_complete_demo.py
```

這將執行完整的AGI流程：
1. 爬取多領域數據
2. 訓練AI模型
3. 進行智能預測
4. 展示系統狀態

### 3. 單獨運行各組件

#### 數據爬取
```bash
python data_crawler.py
```

#### 模型訓練
```bash
python data_trainer.py
```

#### 預測演示
```bash
python predict_demo.py
```

## 📊 系統功能詳解

### 🕷️ 數據爬取系統

**支持的數據類型：**
- **金融數據**: 股價、交易量、技術指標
- **天氣數據**: 溫度、濕度、氣壓、風速
- **醫療數據**: 疾病統計、病例數、康復率
- **能源數據**: 用電量、發電量、可再生能源比例
- **新聞數據**: 情感分析、分類

**使用方法：**
```python
from data_crawler import DataCrawler

# 創建爬蟲實例
crawler = DataCrawler()

# 爬取所有數據
await crawler.crawl_all_data()

# 查看數據摘要
summary = crawler.get_data_summary()
print(summary)
```

### 🧠 模型訓練系統

**支持的模型類型：**
- **LSTM**: 適合時間序列預測
- **Transformer**: 適合複雜模式識別

**訓練的領域：**
- 金融預測 (股價、趨勢)
- 天氣預測 (溫度、天氣變化)
- 醫療預測 (疾病傳播、風險評估)
- 能源預測 (用電需求、價格)

**使用方法：**
```python
from data_trainer import AGITrainingSystem

# 創建訓練系統
training_system = AGITrainingSystem()

# 訓練所有模型
results = await training_system.train_all_models()

# 查看訓練摘要
summary = training_system.model_trainer.get_training_summary()
print(summary)
```

### 🎯 智能預測系統

**預測功能：**
- 智能模型選擇
- 多模型融合
- 置信度評估
- 實時性能監控

**使用方法：**
```python
from agi_new_features import EnhancedAPI
import numpy as np

# 創建API實例
api = EnhancedAPI()

# 進行預測
data = np.random.randn(1, 20)  # 輸入數據
result = await api.smart_predict('financial', data, use_fusion=True)

print(f"預測結果: {result}")
print(f"置信度: {result.get('confidence', 0):.3f}")
```

## 🌍 真實世界應用場景

### 1. 股票投資決策
```python
# 預測股價走勢
financial_data = np.random.randn(1, 20)
result = await api.smart_predict('financial', financial_data)
```

### 2. 天氣預報
```python
# 預測天氣變化
weather_data = np.random.randn(1, 15)
result = await api.smart_predict('weather', weather_data)
```

### 3. 疾病傳播預測
```python
# 預測疾病傳播趨勢
medical_data = np.random.randn(1, 12)
result = await api.smart_predict('medical', medical_data)
```

### 4. 能源需求預測
```python
# 預測用電需求
energy_data = np.random.randn(1, 18)
result = await api.smart_predict('energy', energy_data)
```

## 📈 系統監控

### 性能指標
- **準確率**: 模型預測準確程度
- **置信度**: 預測結果的可信度
- **處理時間**: 預測響應速度
- **融合效果**: 多模型融合的改善程度

### 查看系統狀態
```python
# 獲取系統狀態
status = api.get_system_status()

# 查看性能摘要
performance = status.get('performance_summary', {})
for metric, data in performance.items():
    print(f"{metric}: {data}")
```

## 🔧 高級功能

### 1. 自定義數據爬取
```python
# 爬取特定股票數據
await crawler.crawl_financial_data(['AAPL', 'GOOGL'], days=60)

# 爬取特定地區天氣數據
await crawler.crawl_weather_data(['Taipei', 'Tokyo'], days=30)
```

### 2. 模型性能調優
```python
# 查看模型訓練歷史
history = training_system.model_trainer.training_history
for model_name, data in history.items():
    print(f"{model_name}: {data['final_metrics']}")
```

### 3. 預測結果分析
```python
# 分析融合效果
if 'fusion' in result:
    fusion = result['fusion']
    print(f"融合模型數: {fusion.get('model_count', 0)}")
    print(f"融合置信度: {fusion.get('confidence', 0):.3f}")
    print(f"權重分配: {fusion.get('weights', [])}")
```

## 📁 文件結構

```
預測ai/
├── data_crawler.py          # 數據爬蟲系統
├── data_trainer.py          # 模型訓練系統
├── agi_new_features.py      # 智能預測API
├── agi_complete_demo.py     # 完整演示
├── predict_demo.py          # 預測演示
├── agi_storage/            # 數據和模型儲存
│   ├── crawled_data.db     # 爬取數據庫
│   ├── models/            # 訓練好的模型
│   └── training_summary.json # 訓練摘要
└── *.json                 # 各種報告文件
```

## 🚨 故障排除

### 常見問題

1. **數據爬取失敗**
   - 檢查網絡連接
   - 確認數據庫權限
   - 查看日誌文件

2. **模型訓練失敗**
   - 確認數據量充足
   - 檢查記憶體使用
   - 驗證輸入數據格式

3. **預測結果異常**
   - 檢查輸入數據維度
   - 確認模型已正確載入
   - 查看置信度指標

### 日誌查看
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🎯 最佳實踐

1. **定期更新數據**: 建議每天運行數據爬取
2. **模型重訓練**: 根據性能指標定期重訓練模型
3. **監控系統**: 定期檢查系統狀態和性能
4. **備份重要數據**: 定期備份訓練好的模型和數據

## 📞 技術支持

如果遇到問題，請檢查：
1. Python版本 (建議 3.8+)
2. 依賴套件版本
3. 系統日誌
4. 數據庫連接

---

**🎉 恭喜！您現在已經掌握了AGI智能預測系統的使用方法。開始您的AI預測之旅吧！** 