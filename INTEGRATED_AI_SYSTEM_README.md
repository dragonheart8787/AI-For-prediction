# 🚀 整合AI系統 - 智能金融預測平台

## 🌟 系統概述

整合AI系統是一個完整的端到端金融預測解決方案，結合了：

- **📊 大規模數據爬取** - 自動爬取股票、加密貨幣、外匯、商品等金融數據
- **🤖 智能AI模型訓練** - 統計模型、機器學習、深度學習等多種模型
- **🔗 模型融合與集成** - 智能權重分配和集成預測
- **🎯 任務導向優化** - 根據不同預測任務自動調整模型權重
- **🚀 GPU加速支持** - 充分利用GPU資源加速訓練和預測
- **📈 自動評估與選擇** - 智能選擇最佳模型和集成策略

## 🏗️ 系統架構

```
整合AI系統
├── 📊 數據爬取層 (EnhancedDataCrawler)
│   ├── 股票數據 (40+ 主要股票)
│   ├── 加密貨幣 (24+ 主要幣種)
│   ├── 外匯數據 (24+ 貨幣對)
│   ├── 商品數據 (16+ 大宗商品)
│   └── 指數數據 (16+ 主要指數)
├── 🤖 AI訓練層 (EnhancedAITrainer)
│   ├── 經典統計模型 (ARIMA, ETS)
│   ├── 機器學習模型 (Random Forest, Gradient Boosting)
│   ├── 深度學習模型 (LSTM, GRU, Transformer)
│   └── 特徵工程 (技術指標、統計特徵、季節性特徵)
├── 🔗 模型融合層
│   ├── 任務導向權重分配
│   ├── 集成學習策略
│   └── 性能評估與選擇
└── 📈 預測與評估層
    ├── 多模型集成預測
    ├── 不確定性估計
    └── 預測性能評估
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install pandas numpy scikit-learn statsmodels yfinance aiohttp torch
```

### 2. 運行系統

```bash
python start_integrated_system.py
```

### 3. 選擇功能

系統提供友好的菜單界面：

- **選項1**: 運行完整工作流程 (推薦)
- **選項2**: 只爬取數據
- **選項3**: 只訓練AI模型
- **選項4**: 模型評估與選擇
- **選項5**: 查看系統狀態
- **選項6**: 查看可用模型
- **選項7**: 系統配置
- **選項8**: 退出

## 📊 數據爬取功能

### 支持的數據源

- **股票**: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX 等40+股票
- **加密貨幣**: BTC-USD, ETH-USD, ADA-USD, DOT-USD, SOL-USD 等24+幣種
- **外匯**: EURUSD, GBPUSD, JPYUSD, CADUSD, AUDUSD 等24+貨幣對
- **商品**: 黃金(GC), 原油(CL), 白銀(SI), 天然氣(NG) 等16+商品
- **指數**: S&P500, 道瓊斯, 納斯達克, 富時100, 日經225 等16+指數

### 數據特徵

- 自動爬取2年歷史數據
- 日頻數據 (可配置為分鐘級)
- 包含開盤價、最高價、最低價、收盤價、成交量
- 自動數據清洗和驗證
- SQLite數據庫存儲，支持快速查詢

## 🤖 AI模型訓練

### 支持的模型類型

#### 1. 經典統計模型
- **ARIMA/SARIMA**: 自回歸積分移動平均模型
- **ETS**: 指數平滑模型，支持趨勢和季節性
- **GARCH**: 廣義自回歸條件異方差模型

#### 2. 機器學習模型
- **Random Forest**: 隨機森林回歸
- **Gradient Boosting**: 梯度提升樹
- **SVR**: 支持向量回歸
- **MLP**: 多層感知器

#### 3. 深度學習模型
- **LSTM**: 長短期記憶網絡
- **GRU**: 門控循環單元
- **Transformer**: 注意力機制模型

### 特徵工程

- **技術指標**: SMA, EMA, RSI, MACD, 布林帶
- **統計特徵**: 收益率、波動率、偏度、峰度
- **時間特徵**: 日、週、月、季度、季節性
- **滯後特徵**: 1-10期的價格和收益率滯後
- **滾動特徵**: 5、10、20、50期的移動統計

## 🔗 模型融合策略

### 任務導向權重分配

系統根據不同預測任務自動調整模型權重：

- **短期預測** (1-7天): 重視快速響應和趨勢捕捉
- **中期預測** (8-30天): 平衡準確性和穩定性
- **長期預測** (31-90天): 重視結構性和泛化能力
- **高頻交易**: 重視波動率和低延遲
- **風險管理**: 重視波動率預測和風險度量

### 集成方法

- **加權平均**: 基於模型性能的動態權重
- **Stacking**: 元學習器組合多個基礎模型
- **Voting**: 多模型投票決策
- **貝葉斯融合**: 概率加權集成

## 🚀 GPU加速支持

### 支持的GPU後端

- **PyTorch**: 主要GPU加速框架
- **TensorFlow**: 可選GPU後端
- **CuPy**: 數值計算GPU加速
- **Numba**: CUDA JIT編譯

### GPU優化特性

- 自動GPU檢測和選擇
- 動態CPU/GPU切換
- 並行模型訓練
- 記憶體優化管理

## 📈 性能評估

### 評估指標

- **預測準確性**: MSE, MAE, RMSE, R²
- **模型穩定性**: AIC, BIC, 交叉驗證
- **金融指標**: Sharpe比率, 最大回撤, Hit比率
- **不確定性**: 預測區間, 置信度

### 自動模型選擇

- 性能排名和篩選
- 過擬合檢測
- 模型複雜度評估
- 集成策略推薦

## ⚙️ 系統配置

### 配置文件

創建 `integrated_config.json` 來自定義配置：

```json
{
  "workflow": {
    "auto_crawling": true,
    "auto_training": true,
    "crawling_interval_hours": 24,
    "training_interval_hours": 48,
    "max_symbols_per_training": 10
  },
  "data_management": {
    "min_data_points": 200,
    "data_retention_days": 365,
    "backup_enabled": true
  },
  "model_management": {
    "model_evaluation": true,
    "model_selection": "best_performance",
    "ensemble_methods": ["weighted_average", "stacking"]
  },
  "performance": {
    "enable_gpu": true,
    "parallel_processing": true,
    "max_workers": 4
  }
}
```

## 📁 文件結構

```
預測ai/
├── 🚀 啟動腳本
│   ├── start_integrated_system.py      # 主啟動腳本
│   └── start_web_server.py             # Web服務器啟動
├── 📊 數據爬取
│   ├── enhanced_data_crawler.py        # 增強版數據爬取器
│   └── crawler_config.json             # 爬取器配置
├── 🤖 AI訓練
│   ├── enhanced_ai_trainer.py          # 增強版AI訓練器
│   └── real_lstm_model.py              # 真實LSTM模型
├── 🔗 模型融合
│   ├── super_enhanced_ts_system.py     # 超級增強版TS系統
│   ├── advanced_model_fusion.py        # 高級模型融合
│   └── gpu_cpu_selector.py             # GPU/CPU選擇器
├── 🌐 Web界面
│   ├── web_server.py                   # Flask Web服務器
│   ├── templates/index.html            # 前端界面
│   └── requirements_web.txt            # Web依賴
├── 📊 數據庫
│   ├── enhanced_financial_data.db      # 金融數據庫
│   └── super_fusion_agi.db            # AGI系統數據庫
├── 🤖 訓練模型
│   └── trained_models/                 # 訓練好的模型
├── 📋 結果報告
│   ├── integrated_results/              # 整合系統結果
│   └── super_enhanced_ts_results/      # TS系統結果
└── 📚 文檔
    ├── INTEGRATED_AI_SYSTEM_README.md  # 本文件
    └── GPU_CPU_SELECTION_README.md     # GPU選擇說明
```

## 🔧 高級功能

### 1. 自定義數據源

修改 `crawler_config.json` 添加新的數據源：

```json
{
  "data_sources": {
    "custom_stocks": {
      "symbols": ["YOUR_SYMBOL1", "YOUR_SYMBOL2"],
      "period": "2y",
      "interval": "1d"
    }
  }
}
```

### 2. 自定義模型

在 `enhanced_ai_trainer.py` 中添加新的模型類型：

```python
def train_custom_model(self, data, symbol):
    # 實現你的自定義模型
    pass
```

### 3. 自定義評估指標

添加新的評估指標：

```python
def custom_evaluation_metric(self, y_true, y_pred):
    # 實現你的評估指標
    return score
```

## 📊 使用示例

### 示例1: 完整工作流程

```python
from integrated_ai_system import IntegratedAISystem

# 創建系統實例
system = IntegratedAISystem()

# 運行完整工作流程
results = await system.run_complete_workflow()

# 查看結果
print(f"爬取成功率: {results['summary']['data_crawling']['success_rate']:.2%}")
print(f"訓練成功率: {results['summary']['model_training']['success_rate']:.2%}")
```

### 示例2: 單獨數據爬取

```python
from enhanced_data_crawler import EnhancedDataCrawler

# 創建爬取器
crawler = EnhancedDataCrawler()

# 開始爬取
results = await crawler.start_crawling()

# 獲取訓練數據
training_data = crawler.get_data_for_training(
    data_type="stocks",
    min_data_points=200
)
```

### 示例3: 單獨模型訓練

```python
from enhanced_ai_trainer import EnhancedAITrainer

# 創建訓練器
trainer = EnhancedAITrainer()

# 獲取數據
training_data = trainer.get_training_data(
    data_type="stocks",
    min_data_points=200
)

# 訓練模型
results = trainer.train_models_for_multiple_symbols(training_data)
```

## 🚨 注意事項

### 1. 數據爬取限制

- 遵守yfinance的API使用條款
- 避免過於頻繁的請求
- 建議在非交易時間進行大規模爬取

### 2. 模型訓練資源

- 深度學習模型需要較多計算資源
- 建議使用GPU加速訓練
- 大規模訓練前檢查可用記憶體

### 3. 數據質量

- 自動數據清洗可能無法處理所有異常
- 建議定期檢查數據質量
- 重要預測前進行數據驗證

## 🔮 未來發展

### 計劃功能

- **實時數據流**: WebSocket實時數據更新
- **更多數據源**: 新聞、社交媒體、經濟指標
- **高級模型**: 圖神經網絡、強化學習
- **雲端部署**: Docker容器化、Kubernetes編排
- **API服務**: RESTful API接口
- **移動應用**: 手機APP支持

### 貢獻指南

歡迎貢獻代碼和改進建議：

1. Fork項目
2. 創建功能分支
3. 提交Pull Request
4. 參與討論和改進

## 📞 技術支持

### 常見問題

**Q: 系統無法啟動怎麼辦？**
A: 檢查依賴項安裝，確保所有必要的包都已安裝

**Q: GPU加速不工作怎麼辦？**
A: 檢查CUDA安裝，使用 `gpu_cpu_selector.py` 診斷

**Q: 數據爬取失敗怎麼辦？**
A: 檢查網絡連接，確認yfinance服務可用

**Q: 模型訓練太慢怎麼辦？**
A: 啟用GPU加速，減少訓練符號數量，調整模型參數

### 聯繫方式

- 項目Issues: GitHub Issues頁面
- 技術討論: GitHub Discussions
- 郵件支持: [your-email@example.com]

## 📄 許可證

本項目採用 MIT 許可證，詳見 LICENSE 文件。

---

**🎉 感謝使用整合AI系統！**

這個系統結合了最先進的AI技術和金融預測方法，為你提供專業級的金融預測解決方案。如果你覺得有用，請給我們一個⭐！
