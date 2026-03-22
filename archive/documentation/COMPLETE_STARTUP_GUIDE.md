# 🚀 SuperFusionAGI 完整功能啟動指南

## 📋 概述

本指南將幫助您快速啟動 SuperFusionAGI 系統的所有功能，包括插件系統、數據爬取、預測系統、模型融合等。

## 🎯 啟動選項

### 1. ⚡ 快速啟動（推薦新手）

```bash
python quick_start.py
```

**適用於：**
- 初次使用系統
- 只想體驗核心功能
- 系統依賴不完整

**功能包括：**
- 🔌 插件系統
- 📊 數據爬取系統
- 🎬 演示系統
- 📝 示例插件

### 2. 🚀 完整啟動（推薦進階用戶）

```bash
python start_all_systems.py
```

**適用於：**
- 想要使用所有功能
- 系統依賴完整
- 有足夠的系統資源

**功能包括：**
- 🔌 插件系統
- 📊 數據爬取系統
- 🔮 預測系統
- 🔄 模型融合系統
- 🌐 Web服務器
- 📈 監控系統
- 🧠 持續學習系統

## 🔧 系統依賴

### 必需依賴
```bash
pip install asyncio aiohttp pandas numpy
```

### 完整依賴
```bash
pip install -r requirements.txt
```

## 📖 詳細使用指南

### 1. 插件系統使用

#### 創建新插件
```bash
python demo_plugin_creation.py
```

#### 測試插件功能
```bash
python demo_plugin_integration.py
```

#### 手動創建插件
```python
from plugin_manager import PluginManager

pm = PluginManager()
template = pm.create_plugin_template("my_plugin", "custom_type")

# 保存模板
with open("my_plugin.py", "w") as f:
    f.write(template)
```

### 2. 數據爬取系統使用

#### 啟動爬蟲
```bash
python start_enhanced_comprehensive_crawler.py
```

#### 菜單選項
1. 🚀 開始全面數據爬取
2. 📊 查看數據類型
3. 🔧 插件管理
4. 📈 數據摘要
5. 🧪 測試插件
6. 📋 爬取歷史

### 3. 預測系統使用

#### 啟動預測系統
```bash
python run_agi.py
```

#### 可用預測類型
- 金融預測
- 天氣預測
- 醫療預測
- 能源預測
- 語言處理

### 4. Web界面使用

#### 訪問Web界面
啟動完整系統後，訪問：
```
http://localhost:8000
```

#### 可用API端點
- `/` - 主頁
- `/predict` - 預測接口
- `/train` - 訓練接口
- `/status` - 系統狀態
- `/models` - 模型列表

## 🎬 演示功能

### 1. 插件創建演示
```bash
python demo_plugin_creation.py
```

**演示內容：**
- 自動生成插件模板
- 創建多種類型插件
- 展示插件管理器功能

### 2. 插件集成演示
```bash
python demo_plugin_integration.py
```

**演示內容：**
- 動態加載插件
- 測試插件功能
- 自動生成配置和文檔

### 3. 完整系統演示
```bash
python agi_complete_demo.py
```

**演示內容：**
- 數據爬取
- 模型訓練
- 智能預測
- 系統狀態

## 📊 監控和報告

### 啟動報告
系統啟動後會自動生成報告：
- `startup_report.json` - 完整啟動報告
- `quick_startup_report.json` - 快速啟動報告

### 系統狀態檢查
```bash
# 查看系統狀態
python -c "from agi_predictor import AGIEngine; print(AGIEngine().get_system_status())"
```

## 🔧 故障排除

### 常見問題

#### 1. 模組導入錯誤
**解決方案：**
```bash
pip install -r requirements.txt
```

#### 2. 數據庫錯誤
**解決方案：**
```bash
# 重新初始化數據庫
python -c "from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler; crawler = EnhancedComprehensiveDataCrawler()"
```

#### 3. 插件加載失敗
**解決方案：**
```bash
# 檢查插件目錄
ls plugins/
# 重新創建插件
python demo_plugin_creation.py
```

#### 4. Web服務器啟動失敗
**解決方案：**
```bash
# 檢查端口是否被佔用
netstat -an | grep 8000
# 使用不同端口
python web_server.py --port 8001
```

### 日誌文件
- `enhanced_crawler.log` - 爬蟲系統日誌
- `agi_predictor.log` - 預測系統日誌
- `agi_persistent.log` - 持久化系統日誌

## 🚀 進階功能

### 1. 自定義配置
編輯配置文件：
- `config.json` - 主配置
- `enhanced_crawler_config.json` - 爬蟲配置
- `monitoring_config.json` - 監控配置
- `learning_config.json` - 學習配置

### 2. 插件開發
參考文檔：
- `SYSTEM_SELF_EXTENDING_DEMO.md` - 系統自擴展演示
- `PLUGIN_README.md` - 插件開發指南

### 3. API集成
```python
import requests

# 預測API
response = requests.post("http://localhost:8000/predict", json={
    "model_name": "financial_lstm",
    "input_data": [[1.0, 2.0, 3.0]],
    "description": "股票預測"
})

# 訓練API
response = requests.post("http://localhost:8000/train", json={
    "model_names": ["financial_lstm"],
    "training_epochs": 100
})
```

## 📞 支持和幫助

### 文檔資源
- `README.md` - 主文檔
- `AGI_USER_GUIDE.md` - 用戶指南
- `SYSTEM_SELF_EXTENDING_DEMO.md` - 系統演示
- `ULTIMATE_AGI_SYSTEM_GUIDE.md` - 終極系統指南

### 示例代碼
- `demo_plugin_creation.py` - 插件創建示例
- `demo_plugin_integration.py` - 插件集成示例
- `sample_plugin.py` - 示例插件

### 測試腳本
- `test_all_features.py` - 功能測試
- `test_fixes.py` - 修復測試

## 🎉 開始使用

1. **選擇啟動方式**
   ```bash
   # 快速啟動（推薦）
   python quick_start.py
   
   # 完整啟動
   python start_all_systems.py
   ```

2. **運行演示**
   ```bash
   python demo_plugin_creation.py
   ```

3. **開始爬取數據**
   ```bash
   python start_enhanced_comprehensive_crawler.py
   ```

4. **訪問Web界面**
   打開瀏覽器訪問：`http://localhost:8000`

---

**🎯 現在您已經準備好使用 SuperFusionAGI 系統的所有功能了！**

**🚀 祝您使用愉快！**
