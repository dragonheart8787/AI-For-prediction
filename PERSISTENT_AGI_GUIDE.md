# 持久化AGI預測系統使用指南

## 🚀 系統概述

持久化AGI預測系統是一個具備以下功能的完整AI預測平台：

- ✅ **持續運行**: 系統可以24/7持續運行
- ✅ **本地儲存**: 所有模型和資料都儲存在本地SQLite資料庫
- ✅ **雲端上傳**: 支援將模型上傳到雲端儲存
- ✅ **真實訓練**: 具備真實的深度學習訓練功能
- ✅ **即時預測**: 提供即時的AI預測服務
- ✅ **API接口**: 提供RESTful API供外部調用

## 📁 檔案結構

```
├── agi_persistent.py          # 核心持久化AGI系統
├── run_persistent_agi.py      # 運行腳本
├── agi_api.py                 # FastAPI接口
├── requirements_persistent.txt # 依賴文件
├── PERSISTENT_AGI_GUIDE.md   # 使用指南
└── agi_storage/              # 本地儲存目錄
    ├── models/               # 模型檔案
    ├── data/                 # 資料檔案
    ├── state/                # 狀態檔案
    └── agi_database.db      # SQLite資料庫
```

## 🛠️ 安裝與設置

### 1. 安裝依賴

```bash
pip install -r requirements_persistent.txt
```

### 2. 安裝額外依賴（可選）

如果需要完整的深度學習功能：

```bash
pip install torch tensorflow scikit-learn
```

如果需要真實的雲端儲存：

```bash
pip install boto3 google-cloud-storage azure-storage-blob
```

## 🚀 快速開始

### 方法1: 使用運行腳本

```bash
# 運行完整演示
python run_persistent_agi.py --demo

# 運行預測演示
python run_persistent_agi.py --prediction

# 運行訓練演示
python run_persistent_agi.py --training

# 運行雲端儲存演示
python run_persistent_agi.py --cloud

# 運行狀態檢查
python run_persistent_agi.py --status

# 運行所有演示
python run_persistent_agi.py --all
```

### 方法2: 使用API接口

```bash
# 啟動API服務器
python agi_api.py
```

然後訪問 `http://localhost:8000` 查看API文檔。

## 📊 系統功能

### 1. 模型訓練

系統支援以下模型的真實訓練：

- **LSTM模型**: 用於時間序列預測
- **Transformer模型**: 用於序列到序列預測

訓練過程包括：
- 批次訓練
- 損失計算
- 準確率監控
- 訓練歷史記錄

### 2. 預測功能

提供即時的AI預測服務：

```python
# 使用API進行預測
import requests
import numpy as np

# 準備輸入資料
input_data = np.random.randn(5, 10).tolist()

# 發送預測請求
response = requests.post("http://localhost:8000/predict", json={
    "model_name": "financial_lstm",
    "input_data": input_data
})

result = response.json()
print(f"預測結果: {result['prediction']}")
print(f"置信度: {result['confidence']}")
```

### 3. 持續學習

系統具備持續學習功能：

- **自動重新訓練**: 當模型性能下降時自動重新訓練
- **性能監控**: 實時監控模型性能
- **背景運行**: 在背景持續運行，不影響主要功能

### 4. 本地儲存

所有資料都儲存在本地：

- **模型檔案**: 儲存在 `agi_storage/models/`
- **資料庫**: SQLite資料庫 `agi_storage/agi_database.db`
- **狀態檔案**: 系統狀態儲存在 `agi_storage/state/`

### 5. 雲端儲存

支援雲端儲存功能：

```python
# 上傳模型到雲端
response = requests.post("http://localhost:8000/cloud", json={
    "model_name": "financial_lstm",
    "operation": "upload"
})

# 從雲端下載模型
response = requests.post("http://localhost:8000/cloud", json={
    "model_name": "financial_lstm",
    "operation": "download"
})
```

## 🔧 API接口

### 主要端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 系統根路徑 |
| `/predict` | POST | 進行預測 |
| `/train` | POST | 訓練模型 |
| `/status` | GET | 獲取系統狀態 |
| `/cloud` | POST | 雲端操作 |
| `/continuous/start` | POST | 啟動持續運行 |
| `/continuous/stop` | POST | 停止持續運行 |
| `/models` | GET | 列出所有模型 |
| `/predictions/recent` | GET | 獲取最近預測 |
| `/health` | GET | 健康檢查 |

### 使用示例

#### 1. 進行預測

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "financial_lstm",
    "input_data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
  }'
```

#### 2. 訓練模型

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["financial_lstm", "weather_transformer"],
    "training_epochs": 100,
    "batch_size": 32
  }'
```

#### 3. 獲取系統狀態

```bash
curl "http://localhost:8000/status"
```

## 📈 監控與管理

### 1. 系統狀態監控

```python
import requests

# 獲取系統狀態
status = requests.get("http://localhost:8000/status").json()
print(f"總模型數: {status['total_models']}")
print(f"總預測數: {status['total_predictions']}")
print(f"系統運行: {status['system_running']}")
```

### 2. 模型性能監控

```python
# 獲取模型列表
models = requests.get("http://localhost:8000/models").json()
for model in models['models']:
    print(f"模型: {model['name']}")
    print(f"類型: {model['type']}")
    print(f"準確率: {model['accuracy']}")
```

### 3. 預測歷史

```python
# 獲取最近預測
predictions = requests.get("http://localhost:8000/predictions/recent?limit=10").json()
for pred in predictions['predictions']:
    print(f"模型: {pred['model_name']}")
    print(f"置信度: {pred['confidence']}")
    print(f"時間: {pred['timestamp']}")
```

## 🔧 配置選項

### PersistentConfig 配置

```python
from agi_persistent import PersistentConfig

config = PersistentConfig(
    # 本地儲存路徑
    local_storage_path="./agi_storage",
    model_storage_path="./agi_storage/models",
    
    # 訓練配置
    training_epochs=100,
    training_batch_size=32,
    learning_rate=0.001,
    
    # 持續學習配置
    continuous_learning_enabled=True,
    retrain_interval_hours=24,
    performance_threshold=0.8,
    
    # 雲端配置
    cloud_enabled=True,
    cloud_storage_url="https://api.example.com/agi-storage"
)
```

## 🚨 故障排除

### 常見問題

1. **模型訓練失敗**
   - 檢查是否有足夠的記憶體
   - 確認依賴已正確安裝
   - 檢查日誌檔案 `agi_persistent.log`

2. **API服務器無法啟動**
   - 確認端口8000未被佔用
   - 檢查FastAPI依賴是否安裝
   - 查看錯誤訊息

3. **雲端上傳失敗**
   - 確認網路連接
   - 檢查雲端API配置
   - 確認API金鑰是否正確

4. **資料庫錯誤**
   - 檢查SQLite權限
   - 確認儲存目錄存在
   - 重新初始化資料庫

### 日誌檔案

系統會生成詳細的日誌檔案：

- `agi_persistent.log`: 主要系統日誌
- API服務器日誌: 在控制台輸出

## 🔮 未來發展

### 計劃中的功能

1. **更多模型類型**
   - CNN模型
   - GNN模型
   - 強化學習模型

2. **高級功能**
   - 模型版本管理
   - A/B測試
   - 自動超參數調優

3. **雲端整合**
   - AWS S3整合
   - Google Cloud Storage整合
   - Azure Blob Storage整合

4. **監控儀表板**
   - 即時性能監控
   - 預測準確率追蹤
   - 系統資源使用情況

## 📞 支援

如果您遇到任何問題或有建議，請：

1. 檢查日誌檔案
2. 查看API文檔 (`http://localhost:8000/docs`)
3. 運行健康檢查 (`http://localhost:8000/health`)

---

**🎉 恭喜！您現在擁有一個完整的持久化AGI預測系統！** 