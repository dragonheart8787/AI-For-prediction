# 🚀 真實AI模型時間序列預測系統

## 📋 系統概述

這是一個完整的時間序列預測系統，包含真實的AI模型實現、數據收集、模型訓練和預測功能。系統支持多種時間序列模型，能夠從真實數據源收集數據，訓練模型，並進行高精度預測。

## 🌟 主要特性

### 🔍 真實數據收集
- **股票數據**: 使用yfinance API收集實時股票數據
- **加密貨幣數據**: 支持多種加密貨幣的價格和交易數據
- **經濟指標**: 通過quandl API獲取經濟數據
- **技術指標**: 自動計算RSI、MACD、移動平均等技術指標

### 🧠 真實AI模型
- **LSTM模型**: 使用PyTorch實現的長短期記憶網絡
- **Transformer模型**: 基於注意力機制的深度學習模型
- **模型訓練**: 真實的梯度下降訓練過程
- **模型評估**: 完整的性能指標評估（MSE、MAE、RMSE、R²）

### 🚀 高級功能
- **多模型集成**: 支持多個模型同時預測和訓練
- **連續學習**: 系統能夠持續學習和改進
- **實時預測**: 支持實時數據輸入和預測
- **模型管理**: 自動保存、加載和版本控制

## 📁 文件結構

```
├── real_lstm_model.py          # 真實LSTM模型實現
├── ultimate_time_series_agi.py # 核心AGI系統
├── demo_real_ai_models.py      # 演示腳本
├── requirements.txt            # 依賴包列表
└── REAL_AI_MODELS_README.md   # 本文檔
```

## 🛠️ 安裝和配置

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置API密鑰（可選）

如果需要使用真實的數據源，請配置以下API密鑰：

```python
# 在ultimate_time_series_agi.py中配置
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
QUANDL_API_KEY = "your_api_key_here"
```

## 🚀 使用方法

### 1. 快速開始

運行演示腳本：

```bash
python demo_real_ai_models.py
```

### 2. 基本使用

```python
from ultimate_time_series_agi import UltimateTimeSeriesConfig, UltimateTimeSeriesAGI
import asyncio

async def main():
    # 創建配置
    config = UltimateTimeSeriesConfig()
    
    # 創建AGI系統
    agi_system = UltimateTimeSeriesAGI(config)
    
    # 啟動系統
    await agi_system.start_system()
    
    # 收集數據
    result = await agi_system.collect_training_data('stocks')
    
    # 訓練模型
    training_result = await agi_system.train_all_models('AAPL')
    
    # 進行預測
    prediction = await agi_system.make_prediction(test_data, 'ensemble')
    
    # 清理資源
    agi_system.cleanup()

asyncio.run(main())
```

### 3. 使用LSTM模型

```python
from real_lstm_model import LSTMTimeSeriesPredictor
import pandas as pd

# 創建預測器
predictor = LSTMTimeSeriesPredictor(
    input_size=5,
    hidden_size=64,
    num_layers=2
)

# 準備數據
data = pd.read_csv('your_data.csv')
X_train, X_test, y_train, y_test = predictor.prepare_data(
    data, 
    target_column='Close',
    sequence_length=60
)

# 訓練模型
training_result = predictor.train(X_train, y_train, epochs=100)

# 進行預測
prediction = predictor.predict(X_test)

# 評估模型
evaluation = predictor.evaluate(X_test, y_test)
```

## 📊 支持的模型類型

### 1. 零訓練模型（Zero-Shot）
- **TimesFM**: 無需訓練的時間序列預測
- **Chronos-Bolt**: 基於Transformer的零訓練模型
- **TimeGPT**: 大型語言模型用於時間序列

### 2. 深度學習模型
- **LSTM**: 長短期記憶網絡
- **Transformer**: 基於注意力的模型
- **TFT**: 時間融合Transformer
- **N-BEATS**: 神經基於擴展的自適應時間序列

### 3. 經典統計模型
- **ARIMA**: 自回歸積分移動平均
- **Prophet**: Facebook的時間序列預測工具
- **ETS**: 指數平滑
- **VAR**: 向量自回歸

## 🔧 配置選項

### 數據收集配置

```python
data_collection = {
    'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
    'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD'],
    'economic': ['GDP', 'INFLATION', 'UNEMPLOYMENT'],
    'update_interval': 3600  # 1小時更新一次
}
```

### 模型配置

```python
model_config = {
    'sequence_length': 60,      # 輸入序列長度
    'forecast_horizon': 30,     # 預測時間範圍
    'training_epochs': 100,     # 訓練輪數
    'batch_size': 32,          # 批次大小
    'learning_rate': 0.001     # 學習率
}
```

## 📈 性能指標

系統提供完整的模型評估指標：

- **MSE (均方誤差)**: 預測誤差的平方平均值
- **MAE (平均絕對誤差)**: 預測誤差的絕對值平均值
- **RMSE (均方根誤差)**: MSE的平方根
- **R² (決定係數)**: 模型解釋變異的比例

## 🔍 故障排除

### 常見問題

1. **CUDA內存不足**
   - 減少batch_size或sequence_length
   - 使用CPU訓練：`device='cpu'`

2. **數據格式錯誤**
   - 確保數據包含必要的列
   - 檢查數據類型是否正確

3. **模型訓練失敗**
   - 檢查學習率設置
   - 增加early_stopping_patience

### 日誌和調試

系統提供詳細的日誌記錄：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 高級功能

### 1. 自定義模型

```python
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

### 2. 數據預處理管道

```python
def custom_preprocessing(data):
    # 自定義數據清理邏輯
    data = data.dropna()
    data = data.fillna(method='ffill')
    return data
```

### 3. 模型集成策略

```python
def custom_ensemble(predictions):
    # 自定義集成方法
    weights = [0.4, 0.3, 0.3]  # 加權平均
    return np.average(predictions, weights=weights, axis=0)
```

## 📚 技術文檔

### 架構設計

系統採用模塊化設計，主要組件包括：

1. **數據收集器**: 負責從各種數據源獲取數據
2. **模型管理器**: 管理多個AI模型的訓練和預測
3. **預測引擎**: 協調多模型預測和集成
4. **存儲系統**: 管理模型、數據和配置的持久化

### 異步處理

系統使用Python的asyncio進行異步處理，支持：

- 並發數據收集
- 並行模型訓練
- 實時預測響應

### 內存管理

- 自動垃圾回收
- 模型權重優化
- 數據批次處理

## 🤝 貢獻指南

歡迎貢獻代碼和改進建議！

### 開發環境設置

1. Fork項目
2. 創建功能分支
3. 提交更改
4. 發起Pull Request

### 代碼規範

- 使用Python 3.8+
- 遵循PEP 8規範
- 添加適當的註釋和文檔
- 包含單元測試

## 📄 許可證

本項目採用MIT許可證。

## 📞 聯繫方式

如有問題或建議，請通過以下方式聯繫：

- 創建GitHub Issue
- 發送郵件至開發團隊
- 參與社區討論

## 🎯 未來計劃

- [ ] 支持更多深度學習架構
- [ ] 集成雲端訓練服務
- [ ] 添加可視化儀表板
- [ ] 支持實時數據流
- [ ] 多語言支持

---

**🌟 感謝使用真實AI模型時間序列預測系統！**
