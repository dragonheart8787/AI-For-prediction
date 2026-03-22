#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動態插件加載演示
展示如何向綜合數據爬取系統添加新的爬取信息類型
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from comprehensive_data_crawler import ComprehensiveDataCrawler, DataCrawlerPlugin
from example_custom_plugin import WeatherDataRealPlugin, NewsSentimentRealPlugin
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomNewsPlugin(DataCrawlerPlugin):
    """自定義新聞爬取插件示例"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行新聞爬取"""
        logger.info("📰 開始爬取自定義新聞數據...")
        
        # 模擬新聞數據
        news_data = [
            {
                'title': 'AI技術在金融領域的應用',
                'content': '人工智能技術正在改變金融行業的預測和分析方式...',
                'source': 'TechNews',
                'published_at': '2024-01-15T10:00:00Z',
                'sentiment': 'positive',
                'category': 'technology'
            },
            {
                'title': '全球經濟復甦趨勢分析',
                'content': '根據最新數據顯示，全球經濟正在逐步復甦...',
                'source': 'EconReport',
                'published_at': '2024-01-15T09:30:00Z',
                'sentiment': 'neutral',
                'category': 'economics'
            }
        ]
        
        return {
            'success': True,
            'data_type': 'custom_news',
            'data': news_data,
            'metadata': {
                'total_articles': len(news_data),
                'crawled_at': '2024-01-15T10:00:00Z'
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['custom_news']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class SocialMediaTrendsPlugin(DataCrawlerPlugin):
    """社交媒體趨勢插件示例"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行社交媒體趨勢爬取"""
        logger.info("📱 開始爬取社交媒體趨勢數據...")
        
        # 模擬社交媒體趨勢數據
        trends_data = [
            {
                'platform': 'Twitter',
                'trending_topic': '#AI預測',
                'tweet_count': 15420,
                'sentiment': 'positive',
                'trending_since': '2024-01-15T08:00:00Z'
            },
            {
                'platform': 'Reddit',
                'trending_topic': 'r/MachineLearning',
                'post_count': 892,
                'sentiment': 'neutral',
                'trending_since': '2024-01-15T07:30:00Z'
            }
        ]
        
        return {
            'success': True,
            'data_type': 'social_trends',
            'data': trends_data,
            'metadata': {
                'total_platforms': len(trends_data),
                'crawled_at': '2024-01-15T10:00:00Z'
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['social_trends']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class MarketSentimentPlugin(DataCrawlerPlugin):
    """市場情緒插件示例"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行市場情緒爬取"""
        logger.info("📊 開始爬取市場情緒數據...")
        
        # 模擬市場情緒數據
        sentiment_data = [
            {
                'market': 'S&P 500',
                'fear_greed_index': 65,
                'vix_level': 18.5,
                'bullish_percentage': 58.2,
                'timestamp': '2024-01-15T10:00:00Z'
            },
            {
                'market': 'NASDAQ',
                'fear_greed_index': 72,
                'vix_level': 16.8,
                'bullish_percentage': 62.1,
                'timestamp': '2024-01-15T10:00:00Z'
            }
        ]
        
        return {
            'success': True,
            'data_type': 'market_sentiment',
            'data': sentiment_data,
            'metadata': {
                'total_markets': len(sentiment_data),
                'crawled_at': '2024-01-15T10:00:00Z'
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['market_sentiment']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

async def demonstrate_dynamic_plugin_loading():
    """演示動態插件加載功能"""
    print("🚀 綜合數據爬取系統 - 動態插件加載演示")
    print("=" * 60)
    
    # 初始化爬取器
    crawler = ComprehensiveDataCrawler()
    
    # 顯示初始可用的數據類型
    print("\n📋 初始可用的數據類型:")
    initial_types = crawler.get_available_data_types()
    for i, data_type in enumerate(initial_types, 1):
        print(f"  {i}. {data_type}")
    
    print(f"\n總共 {len(initial_types)} 種數據類型")
    
    # 動態添加新的插件
    print("\n🔧 開始動態添加新的爬取信息類型...")
    
    # 1. 添加自定義新聞插件
    custom_news_plugin = CustomNewsPlugin()
    crawler.add_custom_data_source(
        name="custom_news",
        config={"enabled": True, "priority": 5},
        plugin=custom_news_plugin
    )
    print("✅ 已添加: 自定義新聞爬取插件")
    
    # 2. 添加社交媒體趨勢插件
    social_trends_plugin = SocialMediaTrendsPlugin()
    crawler.add_custom_data_source(
        name="social_trends",
        config={"enabled": True, "priority": 6},
        plugin=social_trends_plugin
    )
    print("✅ 已添加: 社交媒體趨勢爬取插件")
    
    # 3. 添加市場情緒插件
    market_sentiment_plugin = MarketSentimentPlugin()
    crawler.add_custom_data_source(
        name="market_sentiment",
        config={"enabled": True, "priority": 7},
        plugin=market_sentiment_plugin
    )
    print("✅ 已添加: 市場情緒爬取插件")
    
    # 4. 嘗試加載真實的天氣數據插件（如果API密鑰可用）
    try:
        weather_plugin = WeatherDataRealPlugin()
        crawler.add_custom_data_source(
            name="real_weather",
            config={"enabled": True, "priority": 8},
            plugin=weather_plugin
        )
        print("✅ 已添加: 真實天氣數據爬取插件")
    except Exception as e:
        print(f"⚠️ 天氣插件加載失敗: {e}")
    
    # 顯示更新後的數據類型
    print("\n📋 更新後可用的數據類型:")
    updated_types = crawler.get_available_data_types()
    for i, data_type in enumerate(updated_types, 1):
        print(f"  {i}. {data_type}")
    
    print(f"\n總共 {len(updated_types)} 種數據類型")
    print(f"新增了 {len(updated_types) - len(initial_types)} 種數據類型")
    
    # 演示新插件的爬取功能
    print("\n🧪 測試新插件的爬取功能...")
    
    # 測試自定義新聞插件
    print("\n📰 測試自定義新聞插件:")
    try:
        news_result = await custom_news_plugin.crawl({})
        if news_result['success']:
            print(f"  ✅ 成功爬取 {len(news_result['data'])} 條新聞")
            for news in news_result['data'][:2]:  # 只顯示前2條
                print(f"    - {news['title']} ({news['sentiment']})")
        else:
            print("  ❌ 新聞爬取失敗")
    except Exception as e:
        print(f"  ❌ 新聞插件測試失敗: {e}")
    
    # 測試社交媒體趨勢插件
    print("\n📱 測試社交媒體趨勢插件:")
    try:
        trends_result = await social_trends_plugin.crawl({})
        if trends_result['success']:
            print(f"  ✅ 成功爬取 {len(trends_result['data'])} 個趨勢")
            for trend in trends_result['data']:
                print(f"    - {trend['platform']}: {trend['trending_topic']}")
        else:
            print("  ❌ 社交媒體趨勢爬取失敗")
    except Exception as e:
        print(f"  ❌ 社交媒體趨勢插件測試失敗: {e}")
    
    # 測試市場情緒插件
    print("\n📊 測試市場情緒插件:")
    try:
        sentiment_result = await market_sentiment_plugin.crawl({})
        if sentiment_result['success']:
            print(f"  ✅ 成功爬取 {len(sentiment_result['data'])} 個市場情緒")
            for sentiment in sentiment_result['data']:
                print(f"    - {sentiment['market']}: 恐懼貪婪指數 {sentiment['fear_greed_index']}")
        else:
            print("  ❌ 市場情緒爬取失敗")
    except Exception as e:
        print(f"  ❌ 市場情緒插件測試失敗: {e}")
    
    # 演示如何創建和添加更多自定義插件
    print("\n🔧 插件創建指南:")
    print("1. 繼承 DataCrawlerPlugin 基類")
    print("2. 實現 crawl() 方法")
    print("3. 實現 get_supported_types() 方法")
    print("4. 實現 get_requirements() 方法")
    print("5. 使用 add_custom_data_source() 添加到系統")
    
    print("\n📝 示例代碼:")
    print("""
class MyCustomPlugin(DataCrawlerPlugin):
    async def crawl(self, config):
        # 實現爬取邏輯
        pass
    
    def get_supported_types(self):
        return ['my_data_type']
    
    def get_requirements(self):
        return {'requests': '>=2.25.0'}

# 添加到系統
crawler.add_custom_data_source(
    name="my_data",
    config={"enabled": True, "priority": 10},
    plugin=MyCustomPlugin()
)
""")
    
    print("\n🎯 系統特點:")
    print("✅ 支持無限擴展新的數據類型")
    print("✅ 插件式架構，易於維護")
    print("✅ 自動數據庫管理")
    print("✅ 異步爬取，高效並行")
    print("✅ 配置驅動，靈活控制")
    
    return crawler

async def main():
    """主函數"""
    try:
        crawler = await demonstrate_dynamic_plugin_loading()
        
        print("\n" + "=" * 60)
        print("🎉 動態插件加載演示完成！")
        print("\n💡 現在您可以:")
        print("1. 運行 start_comprehensive_crawler.py 來使用完整的爬取系統")
        print("2. 創建自己的自定義插件")
        print("3. 將新插件添加到系統中")
        print("4. 讓系統自動爬取所有類型的數據")
        
    except Exception as e:
        logger.error(f"演示過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 修復Windows編碼問題
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    asyncio.run(main())
