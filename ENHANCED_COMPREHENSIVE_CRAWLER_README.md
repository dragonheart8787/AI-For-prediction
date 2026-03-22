# 🚀 增強版綜合數據爬取系統

## 📖 概述

增強版綜合數據爬取系統是一個功能強大的數據收集平台，能夠自動爬取各種類型的數據和信息。系統採用插件式架構，支持無限擴展新的數據類型，並具備自動發現和加載新插件的功能。

## ✨ 核心特性

### 🔌 插件式架構
- **無限擴展**: 支持添加任意數量的新數據類型
- **自動發現**: 系統自動掃描和加載新插件
- **熱重載**: 支持插件的動態加載和卸載
- **標準化接口**: 統一的插件開發規範

### 📊 多樣化數據支持
- **金融數據**: 股票、加密貨幣、外匯、商品、指數
- **經濟指標**: GDP、通脹率、失業率等
- **新聞情感**: 新聞文章和情感分析
- **社交媒體**: 趨勢話題和用戶行為
- **天氣數據**: 實時天氣和氣候信息
- **地緣政治**: 國際事件和影響分析
- **供應鏈**: 物流和供應鏈信息
- **能源市場**: 石油、天然氣等能源數據

### 🚀 高性能爬取
- **異步處理**: 使用 `asyncio` 實現高並發
- **智能限流**: 自動控制爬取頻率
- **錯誤處理**: 完善的異常處理和重試機制
- **狀態監控**: 實時監控爬取進度和狀態

### 💾 數據管理
- **SQLite數據庫**: 輕量級、高性能的數據存儲
- **JSON格式**: 靈活的數據結構支持
- **自動備份**: 定期數據備份和版本管理
- **數據摘要**: 快速查看數據統計信息

## 🏗️ 系統架構

```
增強版綜合數據爬取系統
├── 插件管理器 (PluginManager)
│   ├── 插件發現器 (PluginDiscovery)
│   ├── 插件加載器 (PluginLoader)
│   └── 插件註冊表 (PluginRegistry)
├── 數據爬取器 (EnhancedComprehensiveDataCrawler)
│   ├── 內置插件
│   ├── 自定義插件
│   └── 數據存儲
└── 用戶界面
    ├── 命令行界面
    └── 配置管理
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements_comprehensive_crawler.txt
```

### 2. 運行系統

```bash
python start_enhanced_comprehensive_crawler.py
```

### 3. 開始爬取

在系統菜單中選擇 "1. 🚀 開始全面數據爬取"

## 📁 文件結構

```
enhanced_comprehensive_crawler/
├── enhanced_comprehensive_crawler.py    # 主系統文件
├── plugin_manager.py                    # 插件管理器
├── start_enhanced_comprehensive_crawler.py  # 啟動腳本
├── demo_dynamic_plugin_loading.py      # 動態插件加載演示
├── enhanced_crawler_config.json        # 配置文件
├── requirements_comprehensive_crawler.txt  # 依賴文件
└── README.md                           # 說明文檔
```

## 🔌 插件開發

### 插件基類

所有插件都必須繼承 `DataCrawlerPlugin` 基類：

```python
from plugin_manager import DataCrawlerPlugin

class MyCustomPlugin(DataCrawlerPlugin):
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        # 實現您的爬取邏輯
        pass
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['my_data_type']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {'requests': '>=2.25.0'}
```

### 插件接口規範

#### `crawl(config: Dict) -> Dict[str, Any]`
- **功能**: 執行爬取操作
- **參數**: `config` - 爬取配置
- **返回**: 包含爬取結果的字典

#### `get_supported_types() -> List[str]`
- **功能**: 返回插件支持的數據類型列表
- **返回**: 字符串列表

#### `get_requirements() -> Dict[str, str]`
- **功能**: 返回插件的依賴要求
- **返回**: 依賴包和版本要求

### 插件返回格式

```python
{
    'success': True,                    # 是否成功
    'data_type': 'my_data_type',       # 數據類型
    'data': [...],                     # 爬取的數據
    'metadata': {                      # 元數據
        'total_records': 100,
        'crawled_at': '2024-01-15T10:00:00Z'
    }
}
```

## 🛠️ 使用方法

### 1. 查看可用數據類型

```bash
# 在主菜單中選擇 "2. 🔍 查看可用數據類型"
```

### 2. 管理插件

```bash
# 在主菜單中選擇 "3. 🔌 管理插件"
# 可以查看、測試、創建插件
```

### 3. 創建新插件

```bash
# 在插件管理菜單中選擇 "4. 📝 創建插件模板"
# 輸入插件名稱和類型
# 編輯生成的模板文件
```

### 4. 手動添加數據源

```python
from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler

crawler = EnhancedComprehensiveDataCrawler()

# 創建自定義插件
custom_plugin = MyCustomPlugin()

# 添加到系統
crawler.add_custom_data_source(
    name="my_data",
    config={"enabled": True, "priority": 10},
    plugin=custom_plugin
)
```

## ⚙️ 配置說明

### 主要配置項

```json
{
    "data_sources": {
        "stocks": {"enabled": true, "priority": 1},
        "crypto": {"enabled": true, "priority": 2}
    },
    "crawling": {
        "max_concurrent": 10,
        "rate_limit": 0.1,
        "retry_count": 3,
        "timeout": 30
    },
    "storage": {
        "database": "enhanced_comprehensive_financial_data.db",
        "backup_interval": 3600
    },
    "plugins": {
        "auto_discovery": true,
        "discovery_interval": 300,
        "hot_reload": true
    }
}
```

### 配置參數說明

- **max_concurrent**: 最大並發爬取數
- **rate_limit**: 爬取頻率限制（秒）
- **retry_count**: 失敗重試次數
- **timeout**: 請求超時時間
- **discovery_interval**: 插件發現間隔（秒）

## 📊 數據庫結構

### comprehensive_data 表
- `id`: 主鍵
- `data_type`: 數據類型
- `source_name`: 數據源名稱
- `symbol`: 符號/標識
- `timestamp`: 時間戳
- `data_json`: 數據內容（JSON格式）
- `metadata_json`: 元數據（JSON格式）

### data_source_status 表
- `source_name`: 數據源名稱
- `success_rate`: 成功率
- `error_count`: 錯誤次數
- `total_attempts`: 總嘗試次數
- `last_updated`: 最後更新時間

### crawling_history 表
- `session_id`: 爬取會話ID
- `start_time`: 開始時間
- `end_time`: 結束時間
- `total_sources`: 總數據源數
- `successful_sources`: 成功數據源數
- `failed_sources`: 失敗數據源數
- `total_records`: 總記錄數

## 🔧 高級功能

### 自動插件發現

系統會自動掃描 `plugins/` 目錄，發現新的插件文件：

```bash
# 創建插件目錄
mkdir plugins

# 將插件文件放入目錄
# 系統會自動發現和加載
```

### 插件熱重載

支持插件的動態重新加載，無需重啟系統：

```python
# 修改插件文件後
# 系統會自動檢測變化並重新加載
```

### 數據源優先級

可以設置數據源的爬取優先級：

```json
{
    "data_sources": {
        "stocks": {"enabled": true, "priority": 1},      # 最高優先級
        "crypto": {"enabled": true, "priority": 2},      # 次高優先級
        "forex": {"enabled": true, "priority": 3}        # 較低優先級
    }
}
```

## 🚨 注意事項

### 1. 依賴管理
- 確保安裝所有必要的Python包
- 某些插件可能需要額外的API密鑰
- 注意包版本兼容性

### 2. 資源使用
- 控制並發數量，避免過度消耗資源
- 設置適當的爬取頻率，避免被目標網站封鎖
- 定期清理舊數據，避免數據庫過大

### 3. 錯誤處理
- 實現完善的異常處理邏輯
- 設置合理的重試機制
- 記錄詳細的錯誤日誌

### 4. 法律合規
- 遵守目標網站的robots.txt規則
- 尊重API使用限制和條款
- 注意數據使用和隱私保護

## 🔍 故障排除

### 常見問題

#### 1. 插件加載失敗
- 檢查插件文件語法是否正確
- 確認插件類繼承了正確的基類
- 查看錯誤日誌獲取詳細信息

#### 2. 數據爬取失敗
- 檢查網絡連接
- 確認API密鑰是否有效
- 查看目標網站是否更改了結構

#### 3. 數據庫錯誤
- 檢查數據庫文件權限
- 確認SQLite版本兼容性
- 查看磁盤空間是否充足

### 日誌文件

系統會生成詳細的日誌文件：
- `enhanced_comprehensive_crawler.log`: 主系統日誌
- `comprehensive_crawler.log`: 原始爬取器日誌

## 🚀 未來發展

### 計劃功能
- [ ] Web界面支持
- [ ] 分布式爬取
- [ ] 更多數據源支持
- [ ] 機器學習數據分析
- [ ] 實時數據流處理

### 貢獻指南

歡迎貢獻代碼和改進建議：
1. Fork 項目
2. 創建功能分支
3. 提交更改
4. 發起 Pull Request

## 📞 支持與聯繫

如果您在使用過程中遇到問題或有改進建議，請：
- 查看日誌文件獲取錯誤信息
- 檢查配置文件設置
- 參考本文檔的故障排除部分
- 提交 Issue 或 Pull Request

## 📄 許可證

本項目採用 MIT 許可證，詳見 LICENSE 文件。

---

**🎉 感謝使用增強版綜合數據爬取系統！**

讓您的數據收集工作更加高效和智能！
