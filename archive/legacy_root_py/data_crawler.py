#!/usr/bin/env python3
"""
AGI數據爬蟲系統
收集多領域數據用於AGI模型訓練

支持的數據類型:
- 金融數據 (股價、匯率、加密貨幣)
- 天氣數據 (溫度、濕度、氣壓)
- 醫療數據 (疾病統計、健康指標)
- 能源數據 (用電量、發電量)
- 新聞數據 (情感分析)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import random

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCrawler:
    """多功能數據爬蟲"""
    
    def __init__(self, db_path: str = "./agi_storage/crawled_data.db"):
        self.db_path = db_path
        self.session = None
        self._init_database()
    
    def _init_database(self):
        """初始化數據庫"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 創建數據表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS financial_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    date TEXT NOT NULL,
                    temperature REAL,
                    humidity REAL,
                    pressure REAL,
                    wind_speed REAL,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS medical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease TEXT NOT NULL,
                    date TEXT NOT NULL,
                    cases INTEGER,
                    deaths INTEGER,
                    recovered INTEGER,
                    region TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS energy_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    date TEXT NOT NULL,
                    consumption REAL,
                    generation REAL,
                    renewable_percentage REAL,
                    price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    date TEXT NOT NULL,
                    sentiment REAL,
                    category TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ 數據庫初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 數據庫初始化失敗: {e}")
    
    async def start_session(self):
        """啟動HTTP會話"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
    
    async def close_session(self):
        """關閉HTTP會話"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def crawl_financial_data(self, symbols: List[str] = None, days: int = 30):
        """爬取金融數據"""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        await self.start_session()
        
        try:
            logger.info(f"💰 開始爬取金融數據: {symbols}")
            
            for symbol in symbols:
                # 模擬金融數據（實際應用中會使用真實API）
                data = await self._simulate_financial_data(symbol, days)
                
                # 保存到數據庫
                await self._save_financial_data(symbol, data)
                
                logger.info(f"✅ 完成 {symbol} 數據爬取")
                await asyncio.sleep(1)  # 避免過於頻繁的請求
            
            logger.info("✅ 金融數據爬取完成")
            
        except Exception as e:
            logger.error(f"❌ 金融數據爬取失敗: {e}")
    
    async def _simulate_financial_data(self, symbol: str, days: int) -> List[Dict]:
        """模擬金融數據"""
        data = []
        base_price = random.uniform(100, 500)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # 模擬價格波動
            change = random.uniform(-0.05, 0.05)
            base_price *= (1 + change)
            
            open_price = base_price
            high_price = base_price * random.uniform(1.0, 1.03)
            low_price = base_price * random.uniform(0.97, 1.0)
            close_price = base_price * random.uniform(0.98, 1.02)
            volume = random.randint(1000000, 10000000)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open_price': round(open_price, 2),
                'high_price': round(high_price, 2),
                'low_price': round(low_price, 2),
                'close_price': round(close_price, 2),
                'volume': volume
            })
        
        return data
    
    async def _save_financial_data(self, symbol: str, data: List[Dict]):
        """保存金融數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute('''
                    INSERT INTO financial_data 
                    (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, item['date'], item['open_price'], item['high_price'],
                     item['low_price'], item['close_price'], item['volume']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存金融數據失敗: {e}")
    
    async def crawl_weather_data(self, locations: List[str] = None, days: int = 30):
        """爬取天氣數據"""
        if locations is None:
            locations = ['Taipei', 'Tokyo', 'New York', 'London', 'Sydney']
        
        await self.start_session()
        
        try:
            logger.info(f"🌤️ 開始爬取天氣數據: {locations}")
            
            for location in locations:
                # 模擬天氣數據
                data = await self._simulate_weather_data(location, days)
                
                # 保存到數據庫
                await self._save_weather_data(location, data)
                
                logger.info(f"✅ 完成 {location} 天氣數據爬取")
                await asyncio.sleep(1)
            
            logger.info("✅ 天氣數據爬取完成")
            
        except Exception as e:
            logger.error(f"❌ 天氣數據爬取失敗: {e}")
    
    async def _simulate_weather_data(self, location: str, days: int) -> List[Dict]:
        """模擬天氣數據"""
        data = []
        
        # 根據地點設置基礎溫度
        base_temps = {
            'Taipei': 25, 'Tokyo': 20, 'New York': 15, 
            'London': 12, 'Sydney': 22
        }
        base_temp = base_temps.get(location, 20)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # 模擬溫度變化
            temp_change = random.uniform(-5, 5)
            temperature = base_temp + temp_change
            
            humidity = random.uniform(40, 90)
            pressure = random.uniform(1000, 1020)
            wind_speed = random.uniform(0, 20)
            
            descriptions = ['Sunny', 'Cloudy', 'Rainy', 'Windy', 'Clear']
            description = random.choice(descriptions)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'description': description
            })
        
        return data
    
    async def _save_weather_data(self, location: str, data: List[Dict]):
        """保存天氣數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute('''
                    INSERT INTO weather_data 
                    (location, date, temperature, humidity, pressure, wind_speed, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (location, item['date'], item['temperature'], item['humidity'],
                     item['pressure'], item['wind_speed'], item['description']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存天氣數據失敗: {e}")
    
    async def crawl_medical_data(self, diseases: List[str] = None, days: int = 30):
        """爬取醫療數據"""
        if diseases is None:
            diseases = ['COVID-19', 'Influenza', 'Diabetes', 'Heart Disease', 'Cancer']
        
        try:
            logger.info(f"🏥 開始爬取醫療數據: {diseases}")
            
            for disease in diseases:
                # 模擬醫療數據
                data = await self._simulate_medical_data(disease, days)
                
                # 保存到數據庫
                await self._save_medical_data(disease, data)
                
                logger.info(f"✅ 完成 {disease} 醫療數據爬取")
                await asyncio.sleep(1)
            
            logger.info("✅ 醫療數據爬取完成")
            
        except Exception as e:
            logger.error(f"❌ 醫療數據爬取失敗: {e}")
    
    async def _simulate_medical_data(self, disease: str, days: int) -> List[Dict]:
        """模擬醫療數據"""
        data = []
        regions = ['Taiwan', 'Japan', 'USA', 'UK', 'Australia']
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            for region in regions:
                # 模擬病例數據
                base_cases = random.randint(10, 1000)
                cases = base_cases + random.randint(-50, 50)
                deaths = random.randint(0, cases // 10)
                recovered = random.randint(0, cases - deaths)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'cases': max(0, cases),
                    'deaths': max(0, deaths),
                    'recovered': max(0, recovered),
                    'region': region
                })
        
        return data
    
    async def _save_medical_data(self, disease: str, data: List[Dict]):
        """保存醫療數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute('''
                    INSERT INTO medical_data 
                    (disease, date, cases, deaths, recovered, region)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (disease, item['date'], item['cases'], item['deaths'],
                     item['recovered'], item['region']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存醫療數據失敗: {e}")
    
    async def crawl_energy_data(self, regions: List[str] = None, days: int = 30):
        """爬取能源數據"""
        if regions is None:
            regions = ['Taiwan', 'Japan', 'USA', 'Germany', 'Australia']
        
        try:
            logger.info(f"⚡ 開始爬取能源數據: {regions}")
            
            for region in regions:
                # 模擬能源數據
                data = await self._simulate_energy_data(region, days)
                
                # 保存到數據庫
                await self._save_energy_data(region, data)
                
                logger.info(f"✅ 完成 {region} 能源數據爬取")
                await asyncio.sleep(1)
            
            logger.info("✅ 能源數據爬取完成")
            
        except Exception as e:
            logger.error(f"❌ 能源數據爬取失敗: {e}")
    
    async def _simulate_energy_data(self, region: str, days: int) -> List[Dict]:
        """模擬能源數據"""
        data = []
        
        # 根據地區設置基礎用電量
        base_consumption = random.uniform(1000, 5000)
        base_generation = base_consumption * random.uniform(0.8, 1.2)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # 模擬能源數據變化
            consumption = base_consumption + random.uniform(-200, 200)
            generation = base_generation + random.uniform(-300, 300)
            renewable_percentage = random.uniform(10, 40)
            price = random.uniform(0.1, 0.3)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'consumption': round(consumption, 2),
                'generation': round(generation, 2),
                'renewable_percentage': round(renewable_percentage, 1),
                'price': round(price, 3)
            })
        
        return data
    
    async def _save_energy_data(self, region: str, data: List[Dict]):
        """保存能源數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute('''
                    INSERT INTO energy_data 
                    (region, date, consumption, generation, renewable_percentage, price)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (region, item['date'], item['consumption'], item['generation'],
                     item['renewable_percentage'], item['price']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存能源數據失敗: {e}")
    
    async def crawl_news_data(self, categories: List[str] = None, count: int = 100):
        """爬取新聞數據"""
        if categories is None:
            categories = ['Technology', 'Finance', 'Health', 'Politics', 'Sports']
        
        try:
            logger.info(f"📰 開始爬取新聞數據: {categories}")
            
            for category in categories:
                # 模擬新聞數據
                data = await self._simulate_news_data(category, count)
                
                # 保存到數據庫
                await self._save_news_data(category, data)
                
                logger.info(f"✅ 完成 {category} 新聞數據爬取")
                await asyncio.sleep(1)
            
            logger.info("✅ 新聞數據爬取完成")
            
        except Exception as e:
            logger.error(f"❌ 新聞數據爬取失敗: {e}")
    
    async def _simulate_news_data(self, category: str, count: int) -> List[Dict]:
        """模擬新聞數據"""
        data = []
        sources = ['Reuters', 'Bloomberg', 'CNN', 'BBC', 'TechCrunch']
        
        titles = {
            'Technology': [
                'AI Breakthrough in Machine Learning',
                'New Smartphone Features Announced',
                'Cybersecurity Threats on the Rise',
                'Quantum Computing Advances',
                '5G Network Expansion'
            ],
            'Finance': [
                'Stock Market Reaches New Highs',
                'Central Bank Policy Changes',
                'Cryptocurrency Market Volatility',
                'Economic Recovery Indicators',
                'Investment Trends Analysis'
            ],
            'Health': [
                'New Medical Treatment Discovered',
                'Public Health Guidelines Updated',
                'Vaccine Development Progress',
                'Mental Health Awareness Campaign',
                'Healthcare System Reforms'
            ]
        }
        
        category_titles = titles.get(category, [f'{category} News'])
        
        for i in range(count):
            date = datetime.now() - timedelta(days=random.randint(0, 30))
            title = random.choice(category_titles) + f" #{i+1}"
            content = f"This is a simulated news article about {category.lower()} topics. Article number {i+1}."
            source = random.choice(sources)
            sentiment = random.uniform(-1, 1)  # -1到1之間的情感分數
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'title': title,
                'content': content,
                'source': source,
                'sentiment': round(sentiment, 3),
                'category': category
            })
        
        return data
    
    async def _save_news_data(self, category: str, data: List[Dict]):
        """保存新聞數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in data:
                cursor.execute('''
                    INSERT INTO news_data 
                    (title, content, source, date, sentiment, category)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (item['title'], item['content'], item['source'],
                     item['date'], item['sentiment'], item['category']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存新聞數據失敗: {e}")
    
    async def crawl_all_data(self):
        """爬取所有類型的數據"""
        logger.info("🚀 開始全面數據爬取...")
        
        try:
            # 並行爬取所有數據
            tasks = [
                self.crawl_financial_data(),
                self.crawl_weather_data(),
                self.crawl_medical_data(),
                self.crawl_energy_data(),
                self.crawl_news_data()
            ]
            
            await asyncio.gather(*tasks)
            
            logger.info("🎉 所有數據爬取完成！")
            
        except Exception as e:
            logger.error(f"❌ 全面數據爬取失敗: {e}")
        finally:
            await self.close_session()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """獲取數據摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            summary = {}
            
            # 統計各表數據量
            tables = ['financial_data', 'weather_data', 'medical_data', 'energy_data', 'news_data']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary[table] = count
            
            conn.close()
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 獲取數據摘要失敗: {e}")
            return {}

async def main():
    """主函數"""
    crawler = DataCrawler()
    
    print("🕷️ AGI數據爬蟲系統")
    print("=" * 50)
    
    # 爬取所有數據
    await crawler.crawl_all_data()
    
    # 顯示數據摘要
    summary = crawler.get_data_summary()
    print("\n📊 數據爬取摘要:")
    for table, count in summary.items():
        print(f"  {table}: {count} 條記錄")
    
    print("\n✅ 數據爬取完成！")

if __name__ == "__main__":
    asyncio.run(main()) 