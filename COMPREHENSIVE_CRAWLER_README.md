# 🚀 全面數據爬取器 - Comprehensive Data Crawler

## 📖 概述

全面數據爬取器是一個強大的、可擴展的數據收集系統，能夠爬取各種類型的金融、經濟、社會和環境數據。系統採用插件化架構，支持動態擴展新的數據源和爬取類型。

## 🌟 核心特性

### 🔌 插件化架構
- **25+ 內置數據類型**：股票、加密貨幣、外匯、商品、指數等
- **可擴展設計**：支持自定義插件開發
- **動態加載**：無需重啟系統即可添加新功能

### 📊 支持的數據類型

#### 1. 金融數據
- **股票數據** (Stocks)：價格、成交量、技術指標
- **加密貨幣** (Cryptocurrency)：BTC、ETH等主流幣種
- **外匯數據** (Forex)：主要貨幣對匯率
- **商品數據** (Commodities)：黃金、原油、農產品等
- **市場指數** (Market Indices)：S&P 500、道瓊斯等

#### 2. 經濟指標
- **宏觀經濟**：GDP、CPI、失業率、利率
- **貿易數據**：進出口、貿易順差
- **貨幣政策**：央行公告、利率決策

#### 3. 社會數據
- **新聞情感**：財經新聞、情感分析
- **社交媒體**：Twitter、Reddit等平台數據
- **天氣數據**：溫度、濕度、風速等氣象信息

#### 4. 專業數據
- **財報日曆**：公司財報發布時間
- **內幕交易**：高管持股變動
- **期權流動**：期權交易數據
- **機構持股**：基金、保險公司持倉

### 🚀 高性能特性
- **異步爬取**：使用 `asyncio` 實現並行處理
- **智能重試**：自動處理網絡錯誤和API限制
- **速率控制**：避免觸發反爬蟲機制
- **數據持久化**：SQLite數據庫存儲

## 🛠️ 安裝與配置

### 1. 環境要求
```bash
Python 3.8+
pip install -r requirements.txt
```

### 2. 依賴包
```bash
pip install aiohttp pandas numpy yfinance requests
```

### 3. 配置文件
系統使用 `enhanced_crawler_config.json` 作為配置文件，包含：
- 數據源配置
- 爬取參數設置
- 存儲選項
- 擴展性配置

## 🚀 快速開始

### 1. 基本使用
```python
from comprehensive_data_crawler import ComprehensiveDataCrawler

# 創建爬取器實例
crawler = ComprehensiveDataCrawler()

# 開始全面爬取
results = await crawler.start_comprehensive_crawling()
```

### 2. 啟動用戶界面
```bash
python start_comprehensive_crawler.py
```

### 3. 直接運行爬取器
```bash
python comprehensive_data_crawler.py
```

## 🔌 插件開發

### 1. 創建自定義插件
```python
from comprehensive_data_crawler import DataCrawlerPlugin

class MyCustomPlugin(DataCrawlerPlugin):
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        # 實現爬取邏輯
        pass
    
    def get_supported_types(self) -> List[str]:
        return ['my_data_type']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}
```

### 2. 註冊插件
```python
# 添加到爬取器
crawler.add_custom_data_source("my_source", config, MyCustomPlugin())
```

### 3. 示例插件
- `example_custom_plugin.py`：包含三個示例插件
  - 公司財報數據插件
  - 真實天氣數據插件
  - 新聞情感分析插件

## 📊 數據存儲

### 1. 數據庫結構
- **comprehensive_data**：主數據表
- **data_source_status**：數據源狀態
- **plugins**：插件信息

### 2. 數據格式
```json
{
    "data_type": "stocks",
    "source_name": "yfinance",
    "symbol": "AAPL",
    "timestamp": "2024-01-01T00:00:00",
    "data_json": "{...}",
    "metadata_json": "{...}"
}
```

### 3. 數據查詢
```python
# 獲取數據摘要
summary = crawler.get_data_summary()

# 獲取訓練數據
training_data = crawler.get_data_for_training(
    symbols=["AAPL", "MSFT"],
    data_type="stocks",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## ⚙️ 配置選項

### 1. 數據源配置
```json
{
    "stocks": {
        "enabled": true,
        "priority": 1,
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "period": "2y",
        "interval": "1d"
    }
}
```

### 2. 爬取參數
```json
{
    "crawling": {
        "max_concurrent": 15,
        "rate_limit": 0.05,
        "retry_attempts": 5,
        "timeout": 45
    }
}
```

### 3. 存儲選項
```json
{
    "storage": {
        "database": "comprehensive_financial_data.db",
        "backup_interval": "6h",
        "max_data_age": "10y",
        "compression": true
    }
}
```

## 🔍 監控與管理

### 1. 系統狀態
- 數據源運行狀態
- 插件健康狀況
- 爬取成功率統計

### 2. 日誌記錄
- 詳細的操作日誌
- 錯誤追蹤和調試
- 性能監控

### 3. 數據備份
- 自動數據庫備份
- 配置版本控制
- 災難恢復

## 🚀 高級功能

### 1. 智能重試
- 自動檢測失敗原因
- 指數退避重試策略
- 網絡異常處理

### 2. 速率控制
- 動態調整請求頻率
- API限制感知
- 反爬蟲對策

### 3. 數據驗證
- 數據完整性檢查
- 異常值檢測
- 格式驗證

## 📈 性能優化

### 1. 並行處理
- 多數據源並行爬取
- 異步I/O操作
- 線程池管理

### 2. 內存管理
- 數據流式處理
- 垃圾回收優化
- 緩存策略

### 3. 網絡優化
- 連接池管理
- 請求合併
- 響應壓縮

## 🔧 故障排除

### 1. 常見問題
- **API限制**：調整速率限制和重試策略
- **網絡超時**：增加超時時間和重試次數
- **數據格式錯誤**：檢查插件實現和數據驗證

### 2. 調試技巧
- 啟用詳細日誌
- 使用示例插件測試
- 檢查配置文件語法

### 3. 性能調優
- 調整並發數量
- 優化數據庫查詢
- 監控系統資源

## 🌐 擴展性

### 1. 新數據源
- 實現 `DataCrawlerPlugin` 接口
- 配置數據源參數
- 註冊到爬取器

### 2. 自定義存儲
- 支持多種數據庫
- 文件系統存儲
- 雲存儲集成

### 3. API集成
- RESTful API
- GraphQL支持
- WebSocket實時數據

## 📚 示例和教程

### 1. 基本爬取
```python
# 爬取股票數據
stocks_config = {
    "symbols": ["AAPL", "MSFT"],
    "period": "1y",
    "interval": "1d"
}

results = await crawler._crawl_data_source(stocks_config)
```

### 2. 自定義插件
```python
# 創建天氣數據插件
weather_plugin = WeatherDataRealPlugin(api_key="your_key")
crawler.add_custom_data_source("weather", config, weather_plugin)
```

### 3. 數據分析
```python
# 獲取數據摘要
summary = crawler.get_data_summary()
for data_type, stats in summary.items():
    print(f"{data_type}: {stats['record_count']} 條記錄")
```

## 🤝 貢獻指南

### 1. 開發環境
- 克隆代碼庫
- 安裝依賴
- 運行測試

### 2. 代碼規範
- 遵循PEP 8
- 添加類型註解
- 編寫文檔字符串

### 3. 測試要求
- 單元測試覆蓋
- 集成測試
- 性能測試

## 📄 許可證

本項目採用 MIT 許可證，詳見 LICENSE 文件。

## 📞 支持與聯繫

- **問題報告**：GitHub Issues
- **功能請求**：GitHub Discussions
- **技術支持**：文檔和示例

## 🔮 未來規劃

### 1. 短期目標
- 更多數據源支持
- 改進錯誤處理
- 性能優化

### 2. 中期目標
- 機器學習集成
- 實時數據流
- 分佈式爬取

### 3. 長期願景
- AI驅動的數據發現
- 自動化數據質量評估
- 企業級部署支持

---

**🎯 開始使用全面數據爬取器，讓數據收集變得簡單高效！**
