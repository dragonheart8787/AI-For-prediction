#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例自定義插件
展示如何擴展全面數據爬取器來爬取新的數據類型
"""

import asyncio
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from comprehensive_data_crawler import DataCrawlerPlugin

logger = logging.getLogger(__name__)

class ExampleCustomPlugin(DataCrawlerPlugin):
    """示例自定義插件 - 爬取公司財報數據"""
    
    def __init__(self):
        self.name = "company_earnings"
        self.description = "爬取公司財報數據，包括營收、利潤、每股收益等"
        
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("📊 開始爬取公司財報數據...")
        
        # 從配置中獲取參數
        companies = config.get("companies", ["AAPL", "MSFT", "GOOGL"])
        periods = config.get("periods", ["annual", "quarterly"])
        
        results = []
        
        for company in companies:
            try:
                # 這裡可以實現實際的財報數據爬取邏輯
                # 例如使用 Alpha Vantage API、Yahoo Finance 等
                
                # 模擬數據（實際使用時替換為真實API調用）
                company_data = await self._fetch_company_earnings(company, periods)
                results.extend(company_data)
                
                logger.info(f"✅ 成功爬取 {company} 的財報數據")
                
                # 避免過於頻繁的請求
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ 爬取 {company} 財報數據失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'company_earnings',
            'data': results,
            'metadata': {
                'total_companies': len(companies),
                'periods': periods,
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    async def _fetch_company_earnings(self, company: str, periods: List[str]) -> List[Dict]:
        """獲取公司財報數據"""
        # 這裡是模擬數據，實際使用時應該調用真實的API
        mock_data = []
        
        for period in periods:
            if period == "annual":
                # 模擬年度數據
                for year in range(2020, 2024):
                    mock_data.append({
                        'company': company,
                        'period': 'annual',
                        'year': year,
                        'revenue': 1000000000 + (year - 2020) * 100000000,  # 模擬增長
                        'net_income': 100000000 + (year - 2020) * 20000000,
                        'eps': 2.0 + (year - 2020) * 0.5,
                        'timestamp': datetime(year, 12, 31),
                        'data_source': 'example_plugin'
                    })
            elif period == "quarterly":
                # 模擬季度數據
                for year in range(2022, 2024):
                    for quarter in range(1, 5):
                        mock_data.append({
                            'company': company,
                            'period': 'quarterly',
                            'year': year,
                            'quarter': quarter,
                            'revenue': 250000000 + (year - 2022) * 25000000,
                            'net_income': 25000000 + (year - 2022) * 5000000,
                            'eps': 0.5 + (year - 2022) * 0.1,
                            'timestamp': datetime(year, quarter * 3, 31),
                            'data_source': 'example_plugin'
                        })
        
        return mock_data
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['company_earnings', 'financial_statements', 'corporate_finance']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {
            'requests': '>=2.25.0',
            'pandas': '>=1.3.0'
        }

class WeatherDataRealPlugin(DataCrawlerPlugin):
    """真實天氣數據插件 - 使用OpenWeatherMap API"""
    
    def __init__(self, api_key: str = None):
        self.name = "weather_data_real"
        self.description = "使用OpenWeatherMap API爬取真實天氣數據"
        self.api_key = api_key or "your_api_key_here"  # 需要設置真實的API密鑰
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("🌤️ 開始爬取真實天氣數據...")
        
        cities = config.get("cities", ["London", "New York", "Tokyo"])
        metrics = config.get("metrics", ["temperature", "humidity", "pressure", "wind"])
        
        results = []
        
        for city in cities:
            try:
                # 獲取當前天氣
                current_weather = await self._fetch_current_weather(city)
                if current_weather:
                    results.append(current_weather)
                
                # 獲取5天預報
                forecast = await self._fetch_forecast(city)
                if forecast:
                    results.extend(forecast)
                
                logger.info(f"✅ 成功爬取 {city} 的天氣數據")
                
                # 避免API限制
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 爬取 {city} 天氣數據失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'weather_data_real',
            'data': results,
            'metadata': {
                'total_cities': len(cities),
                'metrics': metrics,
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat(),
                'api_source': 'OpenWeatherMap'
            }
        }
    
    async def _fetch_current_weather(self, city: str) -> Dict:
        """獲取當前天氣"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'  # 使用攝氏度
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'city': city,
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg'),
                'description': data['weather'][0]['description'],
                'data_type': 'current_weather'
            }
            
        except Exception as e:
            logger.error(f"獲取 {city} 當前天氣失敗: {e}")
            return None
    
    async def _fetch_forecast(self, city: str) -> List[Dict]:
        """獲取5天預報"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecasts = []
            
            for item in data['list']:
                forecasts.append({
                    'city': city,
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'wind_direction': item['wind'].get('deg'),
                    'description': item['weather'][0]['description'],
                    'data_type': 'weather_forecast'
                })
            
            return forecasts
            
        except Exception as e:
            logger.error(f"獲取 {city} 預報失敗: {e}")
            return []
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['weather_data', 'climate_data', 'meteorological_data']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {
            'requests': '>=2.25.0',
            'openweathermap': '>=0.3.0'  # 可選的專用庫
        }

class NewsSentimentRealPlugin(DataCrawlerPlugin):
    """真實新聞情感插件 - 使用NewsAPI"""
    
    def __init__(self, api_key: str = None):
        self.name = "news_sentiment_real"
        self.description = "使用NewsAPI爬取真實新聞數據並進行情感分析"
        self.api_key = api_key or "your_news_api_key_here"  # 需要設置真實的API密鑰
        self.base_url = "https://newsapi.org/v2"
        
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("📰 開始爬取真實新聞數據...")
        
        keywords = config.get("keywords", ["AI", "technology", "finance"])
        sources = config.get("sources", ["techcrunch", "reuters", "bloomberg"])
        
        results = []
        
        for keyword in keywords:
            try:
                # 搜索相關新聞
                news_articles = await self._search_news(keyword, sources)
                if news_articles:
                    results.extend(news_articles)
                
                logger.info(f"✅ 成功爬取關鍵詞 '{keyword}' 的新聞數據")
                
                # 避免API限制
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 爬取關鍵詞 '{keyword}' 新聞失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'news_sentiment_real',
            'data': results,
            'metadata': {
                'total_keywords': len(keywords),
                'sources': sources,
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat(),
                'api_source': 'NewsAPI'
            }
        }
    
    async def _search_news(self, keyword: str, sources: List[str]) -> List[Dict]:
        """搜索新聞"""
        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': keyword,
                'sources': ','.join(sources),
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                # 簡單的情感分析（實際使用時可以使用更複雜的NLP庫）
                sentiment = self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                
                articles.append({
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'content': article.get('content'),
                    'url': article.get('url'),
                    'source': article.get('source', {}).get('name'),
                    'published_at': article.get('publishedAt'),
                    'keyword': keyword,
                    'sentiment_score': sentiment['score'],
                    'sentiment_label': sentiment['label'],
                    'data_type': 'news_article'
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"搜索新聞失敗: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """簡單的情感分析"""
        if not text:
            return {'score': 0, 'label': 'neutral'}
        
        # 簡單的關鍵詞情感分析
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'success', 'up', 'rise']
        negative_words = ['bad', 'terrible', 'negative', 'loss', 'decline', 'down', 'fall', 'crisis', 'risk']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'score': 0.5, 'label': 'positive'}
        elif negative_count > positive_count:
            return {'score': -0.5, 'label': 'negative'}
        else:
            return {'score': 0, 'label': 'neutral'}
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['news_data', 'sentiment_analysis', 'media_content']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {
            'requests': '>=2.25.0',
            'textblob': '>=0.15.3',  # 用於更準確的情感分析
            'nltk': '>=3.6.0'  # 自然語言處理
        }

# 使用示例
async def demo_custom_plugins():
    """演示自定義插件的使用"""
    print("🚀 演示自定義插件...")
    
    # 創建插件實例
    earnings_plugin = ExampleCustomPlugin()
    weather_plugin = WeatherDataRealPlugin()
    news_plugin = NewsSentimentRealPlugin()
    
    # 配置
    earnings_config = {
        "companies": ["AAPL", "MSFT"],
        "periods": ["annual", "quarterly"]
    }
    
    weather_config = {
        "cities": ["London", "Tokyo"],
        "metrics": ["temperature", "humidity"]
    }
    
    news_config = {
        "keywords": ["AI", "technology"],
        "sources": ["techcrunch", "reuters"]
    }
    
    # 執行爬取
    print("\n📊 爬取財報數據...")
    earnings_result = await earnings_plugin.crawl(earnings_config)
    print(f"   結果: {earnings_result['successful_records']} 條記錄")
    
    print("\n🌤️ 爬取天氣數據...")
    weather_result = await weather_plugin.crawl(weather_config)
    print(f"   結果: {weather_result['successful_records']} 條記錄")
    
    print("\n📰 爬取新聞數據...")
    news_result = await news_plugin.crawl(news_config)
    print(f"   結果: {news_result['successful_records']} 條記錄")
    
    print("\n✅ 演示完成！")

if __name__ == "__main__":
    # 運行演示
    asyncio.run(demo_custom_plugins())
