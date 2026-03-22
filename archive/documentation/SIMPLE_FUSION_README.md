# 簡化模型融合系統

## 概述

這是一個簡化版的模型融合系統，專為解決Windows環境下的編碼問題和複雜異步操作而設計。系統可以創建本地模型並進行融合，無需下載外部預訓練模型。

## 主要特性

- **本地模型創建**: 自動創建本地時間序列預測模型
- **模型融合**: 支持多種融合策略
- **Windows兼容**: 解決了編碼問題
- **簡化操作**: 避免複雜的異步操作
- **交互式界面**: 提供友好的命令行菜單

## 系統架構

### 支持的模型類型

1. **TimesFM** - 零訓練時間序列預測模型 (Transformer)
2. **Chronos-Bolt** - 基於Transformer的時間序列模型 (T5)
3. **TFT** - 時間融合Transformer模型
4. **N-BEATS** - 神經基於擴展的自適應時間序列
5. **LSTM** - 預訓練的LSTM時間序列模型

### 融合策略

1. **加權平均 (Weighted Average)**: 簡單的模型權重組合
2. **堆疊 (Stacking)**: 使用元學習器組合多個模型
3. **投票 (Voting)**: 多數投票決策
4. **神經網絡融合 (Neural Fusion)**: 使用神經網絡學習最佳組合

## 安裝和運行

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 運行系統

```bash
python start_simple_fusion_system.py
```

## 使用方法

### 主菜單選項

1. **創建所有本地模型**: 為所有預定義模型創建本地配置
2. **查看可用模型**: 顯示已創建的模型列表
3. **創建融合模型**: 選擇模型和融合策略創建融合模型
4. **查看融合模型**: 顯示已創建的融合模型
5. **系統狀態**: 查看系統整體狀態
6. **退出**: 退出系統

### 操作流程

#### 步驟1: 創建本地模型
```
選擇操作 (1-6): 1
```
系統會自動為所有預定義模型創建本地配置文件和狀態文件。

#### 步驟2: 查看可用模型
```
選擇操作 (1-6): 2
```
顯示所有已創建的模型及其狀態信息。

#### 步驟3: 創建融合模型
```
選擇操作 (1-6): 3
```
1. 輸入要融合的模型名稱（用逗號分隔）
2. 選擇融合類型（1-4）
3. 系統自動創建融合模型配置

#### 步驟4: 查看融合模型
```
選擇操作 (1-6): 4
```
顯示所有已創建的融合模型及其配置信息。

## 文件結構

```
project/
├── start_simple_fusion_system.py    # 主啟動腳本
├── model_downloader.py              # 模型下載器（可選）
├── advanced_model_fusion.py         # 高級融合功能（可選）
├── pretrained_models/               # 模型存儲目錄
│   ├── timesfm/                     # TimesFM模型
│   ├── chronos/                     # Chronos模型
│   ├── tft/                         # TFT模型
│   ├── nbeats/                      # N-BEATS模型
│   ├── lstm_pretrained/             # LSTM模型
│   └── fusion_models/               # 融合模型配置
├── simple_fusion_system.log         # 系統日誌
└── SIMPLE_FUSION_README.md          # 本說明文件
```

## 模型配置

每個本地模型都會創建以下文件：

- **`local_config.json`**: 模型架構配置
- **`status.json`**: 模型狀態信息

### 配置示例

```json
{
  "model_type": "transformer",
  "input_size": 512,
  "output_size": 1,
  "hidden_size": 1024,
  "num_layers": 2,
  "dropout": 0.1,
  "is_local": true,
  "created_time": "2024-01-01 12:00:00"
}
```

## 融合模型配置

融合模型會創建包含以下信息的配置文件：

```json
{
  "fusion_type": "weighted_average",
  "base_models": ["timesfm", "chronos", "tft"],
  "created_time": "2024-01-01 12:00:00",
  "status": "created"
}
```

## 故障排除

### 常見問題

1. **編碼錯誤**: 系統已自動處理Windows編碼問題
2. **模型創建失敗**: 檢查目錄權限和磁盤空間
3. **融合模型創建失敗**: 確保基礎模型已存在

### 日誌文件

系統會創建 `simple_fusion_system.log` 文件記錄所有操作，可用於調試問題。

## 擴展功能

### 添加新模型

在 `SimpleModelDownloader` 類的 `pretrained_models` 字典中添加新模型配置：

```python
'new_model': {
    'name': 'New Model',
    'description': '新模型描述',
    'type': 'deep_learning',
    'architecture': 'transformer',
    'input_size': 256,
    'output_size': 1
}
```

### 自定義融合策略

在 `SimpleModelFusion` 類中添加新的融合方法。

## 技術特點

- **同步操作**: 避免異步操作的複雜性
- **錯誤處理**: 完善的異常處理機制
- **狀態管理**: 詳細的模型狀態追蹤
- **配置持久化**: 所有配置自動保存到文件
- **跨平台兼容**: 支持Windows、Linux、macOS

## 性能考慮

- 本地模型創建速度很快（毫秒級）
- 融合模型配置創建即時完成
- 系統啟動時間短
- 內存佔用低

## 未來發展

1. **模型訓練**: 添加實際的模型訓練功能
2. **預測功能**: 實現時間序列預測
3. **性能評估**: 添加模型性能評估指標
4. **Web界面**: 開發基於Web的管理界面
5. **數據集成**: 支持外部數據源

## 聯繫和支持

如有問題或建議，請查看日誌文件或聯繫開發團隊。

---

**注意**: 這是一個簡化版本，專注於解決編碼和異步操作問題。如需完整功能，請參考其他相關文件。
