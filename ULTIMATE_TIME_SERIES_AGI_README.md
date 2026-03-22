# 🚀 終極時間序列預測AGI系統

## 🌟 系統概述

終極時間序列預測AGI系統整合了所有頂級時間序列預測模型，提供零樣本、深度學習、經典統計等多種預測方法，支持智能集成預測。

## 🎯 核心特性

### ✨ 零樣本基礎模型
- **TimesFM (Google)**: 時間序列基礎模型，支持零樣本/少樣本預測
- **Chronos-Bolt (Amazon)**: 序列token化，類語言模型機率式零樣本預測
- **TimeGPT (Nixtla)**: 商用API的TS基礎模型，開箱可用
- **Lag-Llama**: 開源TS基礎模型，零樣本＋可微調
- **Moirai (Salesforce)**: 多頻率/多變量通用預測

### 🧠 強力深度學習模型
- **TFT (Temporal Fusion Transformer)**: 多地平線、可解釋注意力
- **N-BEATS / N-HiTS**: 結構簡潔、效果強勁的MLP堆疊家族
- **PatchTST**: 專攻長序列與多變量，近年競賽表現亮眼
- **iTransformer**: 維度反轉的Transformer，SOTA代表
- **Informer**: 長序列Transformer，ProbSparse注意力
- **DLinear**: 線性基線，出奇有效
- **DeepAR**: RNN機率式，天然產生預測區間

### 📈 經典統計模型
- **ARIMA / SARIMA / SARIMAX**: 季節性/外生變數支持
- **ETS (指數平滑)**: 趨勢與季節性分解
- **Prophet**: 趨勢＋季節＋假日可調，開箱即用
- **Theta**: M3/M4競賽中很強的簡潔模型
- **TBATS**: 多重/非整數季節性處理
- **VAR / SVAR**: 多變量同時建模

### 🔄 智能集成預測
- **多模型類型集成**: 零樣本(30%) + 深度學習(50%) + 經典統計(20%)
- **動態權重調整**: 根據模型性能自動調整權重
- **置信區間計算**: 提供預測不確定性評估
- **並行預測執行**: 異步並行處理，提升效率

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements_ultimate_time_series.txt
```

### 2. 運行系統

#### 演示模式（推薦新手）
```bash
python run_ultimate_time_series.py demo
```

#### 測試模式
```bash
python run_ultimate_time_series.py test
```

#### 主運行模式
```bash
python run_ultimate_time_series.py main
```

### 3. 使用示例

```python
import asyncio
import numpy as np
from ultimate_time_series_agi import UltimateTimeSeriesConfig, UltimateTimeSeriesAGI

async def main():
    # 創建配置
    config = UltimateTimeSeriesConfig()
    
    # 初始化系統
    agi_system = UltimateTimeSeriesAGI(config)
    await agi_system.start_system()
    
    # 準備輸入數據
    input_sequence = np.random.randn(1, 100)  # 100個時間步
    
    # 進行預測
    result = await agi_system.make_prediction(input_sequence, 'ensemble_all')
    
    if 'error' not in result:
        print(f"預測結果: {result['prediction']}")
        print(f"置信區間: {result['confidence_interval']}")
        print(f"使用模型: {result['models_used']}")
    
    # 清理資源
    agi_system.cleanup()

# 運行
asyncio.run(main())
```

## 🎯 預測方法選擇指南

### 🌟 零樣本預測（推薦場景）
- **什麼時候選**: 想「先不訓練」，多序列快速跑出還不錯的預測/區間
- **適用場景**: 
  - 新業務快速驗證
  - 數據量不足
  - 需要快速原型
- **優勢**: 無需訓練，即插即用
- **模型**: TimesFM, Chronos-Bolt, TimeGPT

### 🧠 深度學習預測（推薦場景）
- **什麼時候選**: 有訓練資源，要比經典法更高準度，或要吃很多外生特徵
- **適用場景**:
  - 大量歷史數據
  - 複雜時序模式
  - 多變量預測
- **優勢**: 高精度，處理複雜模式
- **模型**: TFT, N-BEATS, PatchTST, iTransformer

### 📈 經典統計預測（推薦場景）
- **什麼時候選**: 資料量不大、季節性明顯、要快速可解釋baseline
- **適用場景**:
  - 小數據集
  - 季節性明顯
  - 需要可解釋性
- **優勢**: 穩健、可解釋、計算效率高
- **模型**: ARIMA, Prophet, ETS, VAR

### 🚀 終極集成預測（推薦場景）
- **什麼時候選**: 追求最高預測精度，有足夠計算資源
- **適用場景**:
  - 生產環境
  - 高精度要求
  - 多模型比較
- **優勢**: 最高精度，魯棒性強
- **方法**: 智能權重集成所有模型類型

## 🔧 系統配置

### 預測配置
```python
config.prediction_config = {
    'forecast_horizon': 30,        # 預測步數
    'confidence_level': 0.95,      # 置信水平
    'ensemble_method': 'weighted_average',  # 集成方法
    'auto_retrain_interval': 24    # 自動重訓練間隔（小時）
}
```

### 模型配置
```python
# 零樣本模型
config.zero_shot_models = {
    'timesfm': {'enabled': True, 'max_sequence_length': 1000},
    'chronos_bolt': {'enabled': True, 'max_sequence_length': 2000},
    # ... 更多模型
}

# 深度學習模型
config.deep_learning_models = {
    'tft': {'enabled': True, 'max_sequence_length': 1000, 'hidden_size': 64},
    'n_beats': {'enabled': True, 'max_sequence_length': 1000, 'num_stacks': 3},
    # ... 更多模型
}
```

## 📊 性能指標

### 預測精度
- **MSE (均方誤差)**: 預測值與真實值的均方誤差
- **MAE (平均絕對誤差)**: 預測值與真實值的平均絕對誤差
- **MAPE (平均絕對百分比誤差)**: 相對誤差評估

### 系統性能
- **執行時間**: 單次預測的執行時間
- **內存使用**: 模型載入和預測過程的內存佔用
- **並發能力**: 同時處理多個預測請求的能力

### 模型穩定性
- **預測方差**: 集成預測的方差，評估穩定性
- **置信區間**: 預測結果的置信區間寬度
- **異常檢測**: 異常預測結果的識別能力

## 🔮 未來擴展

### 模型增強
- [ ] 集成更多SOTA時間序列模型
- [ ] 支持自定義模型添加
- [ ] 自動超參數優化

### 功能增強
- [ ] 實時數據流預測
- [ ] 多頻率時間序列支持
- [ ] 異常檢測和預警
- [ ] 預測結果可視化

### 部署優化
- [ ] Docker容器化
- [ ] Kubernetes部署
- [ ] 微服務架構
- [ ] 負載均衡

## 🤝 貢獻指南

歡迎貢獻代碼、報告問題或提出建議！

### 貢獻方式
1. Fork 本項目
2. 創建特性分支
3. 提交更改
4. 發起 Pull Request

### 開發環境設置
```bash
git clone <repository-url>
cd ultimate-time-series-agi
pip install -r requirements_ultimate_time_series.txt
python run_ultimate_time_series.py test
```

## 📄 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

感謝所有開源時間序列預測模型的開發者和研究團隊，特別是：
- Google Research (TimesFM)
- Amazon Web Services (Chronos-Bolt)
- Nixtla (TimeGPT)
- Salesforce Research (Moirai)
- 以及所有貢獻者

---

**�� 讓時間序列預測變得簡單而強大！**
