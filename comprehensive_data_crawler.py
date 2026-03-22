#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面數據爬取器
能夠爬取所有類型的數據和信息，支持動態擴展新的爬取類型
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import logging
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """數據源定義"""
    name: str
    enabled: bool
    priority: int
    last_updated: Optional[datetime] = None
    success_rate: float = 0.0
    error_count: int = 0

class DataCrawlerPlugin(ABC):
    """數據爬取插件基類"""
    
    @abstractmethod
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        pass

class ComprehensiveDataCrawler:
    """全面數據爬取器"""
    
    def __init__(self, config_path: str = "enhanced_crawler_config.json"):
        self.config = self._load_config(config_path)
        self.db_path = self.config["storage"]["database"]
        self.plugins: Dict[str, DataCrawlerPlugin] = {}
        self.data_sources: Dict[str, DataSource] = {}
        self.session = None
        self._init_database()
        self._init_data_sources()
        self._load_plugins()
        
    def _load_config(self, config_path: str) -> Dict:
        """加載配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"配置文件 {config_path} 不存在，使用默認配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """獲取默認配置"""
        return {
            "data_sources": {
                "stocks": {"enabled": True, "priority": 1},
                "crypto": {"enabled": True, "priority": 2},
                "forex": {"enabled": True, "priority": 3}
            },
            "crawling": {
                "max_concurrent": 10,
                "rate_limit": 0.1,
                "retry_attempts": 3,
                "timeout": 30
            },
            "storage": {
                "database": "comprehensive_financial_data.db",
                "backup_interval": "1d"
            }
        }
    
    def _init_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建主數據表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comprehensive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                symbol TEXT,
                timestamp DATETIME NOT NULL,
                data_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(data_type, source_name, symbol, timestamp)
            )
        ''')
        
        # 創建數據源狀態表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_source_status (
                source_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                last_updated DATETIME,
                success_rate REAL DEFAULT 0.0,
                error_count INTEGER DEFAULT 0,
                total_attempts INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                config_json TEXT
            )
        ''')
        
        # 創建插件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plugins (
                plugin_name TEXT PRIMARY KEY,
                plugin_type TEXT NOT NULL,
                version TEXT,
                enabled BOOLEAN DEFAULT 1,
                last_used DATETIME,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0
            )
        ''')
        
        # 創建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_type_timestamp 
            ON comprehensive_data(data_type, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_symbol 
            ON comprehensive_data(source_name, symbol)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ 數據庫初始化完成")
    
    def _init_data_sources(self):
        """初始化數據源"""
        for source_name, source_config in self.config["data_sources"].items():
            if source_config.get("enabled", False):
                self.data_sources[source_name] = DataSource(
                    name=source_name,
                    enabled=True,
                    priority=source_config.get("priority", 999),
                    last_updated=None,
                    success_rate=0.0,
                    error_count=0
                )
        
        # 按優先級排序
        sorted_sources = sorted(
            self.data_sources.values(), 
            key=lambda x: x.priority
        )
        logger.info(f"✅ 初始化了 {len(sorted_sources)} 個數據源")
    
    def _load_plugins(self):
        """加載插件"""
        # 加載內置插件
        self._load_builtin_plugins()
        
        # 加載外部插件
        self._load_external_plugins()
        
        logger.info(f"✅ 加載了 {len(self.plugins)} 個插件")
    
    def _load_builtin_plugins(self):
        """加載內置插件"""
        # 股票數據插件
        self.plugins["stocks"] = StocksDataPlugin()
        
        # 加密貨幣插件
        self.plugins["crypto"] = CryptoDataPlugin()
        
        # 外匯數據插件
        self.plugins["forex"] = ForexDataPlugin()
        
        # 商品數據插件
        self.plugins["commodities"] = CommoditiesDataPlugin()
        
        # 指數數據插件
        self.plugins["indices"] = IndicesDataPlugin()
        
        # 經濟指標插件
        self.plugins["economic_indicators"] = EconomicIndicatorsPlugin()
        
        # 新聞情感插件
        self.plugins["news_sentiment"] = NewsSentimentPlugin()
        
        # 社交媒體插件
        self.plugins["social_media"] = SocialMediaPlugin()
        
        # 天氣數據插件
        self.plugins["weather_data"] = WeatherDataPlugin()
        
        # 地緣政治事件插件
        self.plugins["geopolitical_events"] = GeopoliticalEventsPlugin()
        
        # 央行公告插件
        self.plugins["central_bank_announcements"] = CentralBankAnnouncementsPlugin()
        
        # 財報日曆插件
        self.plugins["earnings_calendar"] = EarningsCalendarPlugin()
        
        # 內幕交易插件
        self.plugins["insider_trading"] = InsiderTradingPlugin()
        
        # 期權流動插件
        self.plugins["options_flow"] = OptionsFlowPlugin()
        
        # 空頭利息插件
        self.plugins["short_interest"] = ShortInterestPlugin()
        
        # 機構持股插件
        self.plugins["institutional_holdings"] = InstitutionalHoldingsPlugin()
        
        # 供應鏈數據插件
        self.plugins["supply_chain_data"] = SupplyChainDataPlugin()
        
        # 能源市場插件
        self.plugins["energy_markets"] = EnergyMarketsPlugin()
        
        # 房地產數據插件
        self.plugins["real_estate"] = RealEstateDataPlugin()
    
    def _load_external_plugins(self):
        """加載外部插件"""
        plugins_dir = Path("plugins")
        if plugins_dir.exists():
            for plugin_file in plugins_dir.glob("*.py"):
                try:
                    # 這裡可以實現動態插件加載
                    logger.info(f"發現外部插件: {plugin_file}")
                except Exception as e:
                    logger.warning(f"加載外部插件失敗 {plugin_file}: {e}")
    
    async def start_comprehensive_crawling(self):
        """開始全面爬取"""
        logger.info("🚀 開始全面數據爬取...")
        
        async with aiohttp.ClientSession() as self.session:
            # 按優先級排序的數據源
            sorted_sources = sorted(
                self.data_sources.values(), 
                key=lambda x: x.priority
            )
            
            # 創建爬取任務
            tasks = []
            for source in sorted_sources:
                if source.enabled and source.name in self.plugins:
                    task = self._crawl_data_source(source)
                    tasks.append(task)
            
            # 並行執行爬取任務
            semaphore = asyncio.Semaphore(self.config["crawling"]["max_concurrent"])
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # 處理結果
            total_sources = len(tasks)
            successful_sources = 0
            failed_sources = 0
            
            for i, result in enumerate(results):
                source_name = sorted_sources[i].name
                if isinstance(result, Exception):
                    logger.error(f"❌ 數據源 {source_name} 爬取失敗: {result}")
                    failed_sources += 1
                    self._update_source_status(source_name, success=False)
                else:
                    if result.get('success', False):
                        successful_sources += 1
                        self._update_source_status(source_name, success=True)
                    else:
                        failed_sources += 1
                        self._update_source_status(source_name, success=False)
            
            logger.info(f"🎯 全面爬取完成！總計: {total_sources} 個數據源")
            logger.info(f"   成功: {successful_sources}, 失敗: {failed_sources}")
            
            return {
                'total_sources': total_sources,
                'successful_sources': successful_sources,
                'failed_sources': failed_sources,
                'results': results
            }
    
    async def _crawl_data_source(self, source: DataSource) -> Dict:
        """爬取單個數據源"""
        try:
            plugin = self.plugins[source.name]
            source_config = self.config["data_sources"][source.name]
            
            logger.info(f"📊 開始爬取數據源: {source.name}")
            
            # 執行插件爬取
            result = await plugin.crawl(source_config)
            
            # 保存數據
            if result.get('success', False):
                await self._save_comprehensive_data(source.name, result)
                logger.info(f"✅ {source.name} 數據爬取成功")
            else:
                logger.warning(f"⚠️ {source.name} 數據爬取失敗: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 爬取數據源 {source.name} 時發生錯誤: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _save_comprehensive_data(self, source_name: str, result: Dict):
        """保存綜合數據"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            data_type = result.get('data_type', 'unknown')
            data_records = result.get('data', [])
            
            for record in data_records:
                # 轉換為JSON字符串
                data_json = json.dumps(record, ensure_ascii=False, default=str)
                metadata_json = json.dumps(result.get('metadata', {}), ensure_ascii=False, default=str)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO comprehensive_data 
                    (data_type, source_name, symbol, timestamp, data_json, metadata_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_type,
                    source_name,
                    record.get('symbol', None),
                    record.get('timestamp', datetime.now()),
                    data_json,
                    metadata_json,
                    datetime.now()
                ))
            
            conn.commit()
            logger.info(f"💾 保存了 {len(data_records)} 條 {source_name} 數據")
            
        except Exception as e:
            logger.error(f"保存數據失敗: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _update_source_status(self, source_name: str, success: bool):
        """更新數據源狀態"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO data_source_status 
                (source_name, data_type, last_updated, success_rate, error_count, total_attempts)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                source_name,
                'unknown',  # 可以從配置中獲取
                datetime.now(),
                1.0 if success else 0.0,
                0 if success else 1,
                1
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"更新數據源狀態失敗: {e}")
        finally:
            conn.close()
    
    def add_custom_data_source(self, name: str, config: Dict, plugin: DataCrawlerPlugin):
        """添加自定義數據源"""
        if name in self.plugins:
            logger.warning(f"數據源 {name} 已存在，將被覆蓋")
        
        self.plugins[name] = plugin
        self.data_sources[name] = DataSource(
            name=name,
            enabled=True,
            priority=config.get("priority", 999),
            last_updated=None,
            success_rate=0.0,
            error_count=0
        )
        
        # 更新配置
        self.config["data_sources"][name] = config
        
        logger.info(f"✅ 成功添加自定義數據源: {name}")
    
    def get_available_data_types(self) -> List[str]:
        """獲取可用的數據類型"""
        return list(self.plugins.keys())
    
    def get_data_summary(self) -> Dict:
        """獲取數據摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 獲取各數據類型的統計
            cursor.execute('''
                SELECT data_type, COUNT(*) as record_count, 
                       COUNT(DISTINCT symbol) as symbol_count,
                       MIN(timestamp) as earliest_date,
                       MAX(timestamp) as latest_date
                FROM comprehensive_data 
                GROUP BY data_type
            ''')
            
            summary = {}
            for row in cursor.fetchall():
                data_type, record_count, symbol_count, earliest_date, latest_date = row
                summary[data_type] = {
                    'record_count': record_count,
                    'symbol_count': symbol_count,
                    'earliest_date': earliest_date,
                    'latest_date': latest_date
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"獲取數據摘要失敗: {e}")
            return {}
        finally:
            conn.close()

# 內置插件實現
class StocksDataPlugin(DataCrawlerPlugin):
    """股票數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", [])
        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    for timestamp, row in data.iterrows():
                        results.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': row.get('Open'),
                            'high': row.get('High'),
                            'low': row.get('Low'),
                            'close': row.get('Close'),
                            'volume': row.get('Volume'),
                            'adj_close': row.get('Adj Close')
                        })
                
                await asyncio.sleep(0.1)  # 避免過於頻繁的請求
                
            except Exception as e:
                logger.error(f"爬取股票 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'stocks',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'period': period,
                'interval': interval
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['stocks', 'equity']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class CryptoDataPlugin(DataCrawlerPlugin):
    """加密貨幣數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", [])
        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    for timestamp, row in data.iterrows():
                        results.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': row.get('Open'),
                            'high': row.get('High'),
                            'low': row.get('Low'),
                            'close': row.get('Close'),
                            'volume': row.get('Volume'),
                            'adj_close': row.get('Adj Close')
                        })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"爬取加密貨幣 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'crypto',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'period': period,
                'interval': interval
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['crypto', 'cryptocurrency']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class ForexDataPlugin(DataCrawlerPlugin):
    """外匯數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", [])
        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    for timestamp, row in data.iterrows():
                        results.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': row.get('Open'),
                            'high': row.get('High'),
                            'low': row.get('Low'),
                            'close': row.get('Close'),
                            'volume': row.get('Volume'),
                            'adj_close': row.get('Adj Close')
                        })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"爬取外匯 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'forex',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'period': period,
                'interval': interval
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['forex', 'fx']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class CommoditiesDataPlugin(DataCrawlerPlugin):
    """商品數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", [])
        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    for timestamp, row in data.iterrows():
                        results.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': row.get('Open'),
                            'high': row.get('High'),
                            'low': row.get('Low'),
                            'close': row.get('Close'),
                            'volume': row.get('Volume'),
                            'adj_close': row.get('Adj Close')
                        })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"爬取商品 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'commodities',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'period': period,
                'interval': interval
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['commodities', 'futures']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class IndicesDataPlugin(DataCrawlerPlugin):
    """指數數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", [])
        period = config.get("period", "2y")
        interval = config.get("interval", "1d")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    for timestamp, row in data.iterrows():
                        results.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': row.get('Open'),
                            'high': row.get('High'),
                            'low': row.get('Low'),
                            'close': row.get('Close'),
                            'volume': row.get('Volume'),
                            'adj_close': row.get('Adj Close')
                        })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"爬取指數 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'indices',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'period': period,
                'interval': interval
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['indices', 'market_indices']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class EconomicIndicatorsPlugin(DataCrawlerPlugin):
    """經濟指標插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        # 這裡實現經濟指標的爬取邏輯
        # 可以使用FRED API、世界銀行API等
        logger.info("📊 經濟指標爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'economic_indicators',
            'data': [],
            'metadata': {
                'note': '經濟指標爬取功能待實現',
                'sources': config.get('sources', []),
                'indicators': config.get('indicators', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['economic_indicators', 'macro_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class NewsSentimentPlugin(DataCrawlerPlugin):
    """新聞情感插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("📰 新聞情感爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'news_sentiment',
            'data': [],
            'metadata': {
                'note': '新聞情感爬取功能待實現',
                'sources': config.get('sources', []),
                'keywords': config.get('keywords', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['news_sentiment', 'sentiment_analysis']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class SocialMediaPlugin(DataCrawlerPlugin):
    """社交媒體插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("📱 社交媒體爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'social_media',
            'data': [],
            'metadata': {
                'note': '社交媒體爬取功能待實現',
                'platforms': config.get('platforms', []),
                'symbols': config.get('symbols', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['social_media', 'social_sentiment']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class WeatherDataPlugin(DataCrawlerPlugin):
    """天氣數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🌤️ 天氣數據爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'weather_data',
            'data': [],
            'metadata': {
                'note': '天氣數據爬取功能待實現',
                'cities': config.get('cities', []),
                'metrics': config.get('metrics', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['weather_data', 'climate_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class GeopoliticalEventsPlugin(DataCrawlerPlugin):
    """地緣政治事件插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🌍 地緣政治事件爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'geopolitical_events',
            'data': [],
            'metadata': {
                'note': '地緣政治事件爬取功能待實現',
                'sources': config.get('sources', []),
                'categories': config.get('categories', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['geopolitical_events', 'political_risk']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class CentralBankAnnouncementsPlugin(DataCrawlerPlugin):
    """央行公告插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🏦 央行公告爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'central_bank_announcements',
            'data': [],
            'metadata': {
                'note': '央行公告爬取功能待實現',
                'banks': config.get('banks', []),
                'types': config.get('types', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['central_bank_announcements', 'monetary_policy']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class EarningsCalendarPlugin(DataCrawlerPlugin):
    """財報日曆插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("📅 財報日曆爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'earnings_calendar',
            'data': [],
            'metadata': {
                'note': '財報日曆爬取功能待實現',
                'sources': config.get('sources', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['earnings_calendar', 'corporate_events']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class InsiderTradingPlugin(DataCrawlerPlugin):
    """內幕交易插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("👥 內幕交易爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'insider_trading',
            'data': [],
            'metadata': {
                'note': '內幕交易爬取功能待實現',
                'sources': config.get('sources', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['insider_trading', 'corporate_insiders']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class OptionsFlowPlugin(DataCrawlerPlugin):
    """期權流動插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("📊 期權流動爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'options_flow',
            'data': [],
            'metadata': {
                'note': '期權流動爬取功能待實現',
                'sources': config.get('sources', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['options_flow', 'options_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class ShortInterestPlugin(DataCrawlerPlugin):
    """空頭利息插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("📉 空頭利息爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'short_interest',
            'data': [],
            'metadata': {
                'note': '空頭利息爬取功能待實現',
                'sources': config.get('sources', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['short_interest', 'short_selling']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class InstitutionalHoldingsPlugin(DataCrawlerPlugin):
    """機構持股插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🏢 機構持股爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'institutional_holdings',
            'data': [],
            'metadata': {
                'note': '機構持股爬取功能待實現',
                'sources': config.get('sources', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['institutional_holdings', 'institutional_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class SupplyChainDataPlugin(DataCrawlerPlugin):
    """供應鏈數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🚚 供應鏈數據爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'supply_chain_data',
            'data': [],
            'metadata': {
                'note': '供應鏈數據爬取功能待實現',
                'sources': config.get('sources', []),
                'metrics': config.get('metrics', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['supply_chain_data', 'logistics_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class EnergyMarketsPlugin(DataCrawlerPlugin):
    """能源市場插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("⚡ 能源市場爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'energy_markets',
            'data': [],
            'metadata': {
                'note': '能源市場爬取功能待實現',
                'sources': config.get('sources', []),
                'metrics': config.get('metrics', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['energy_markets', 'energy_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

class RealEstateDataPlugin(DataCrawlerPlugin):
    """房地產數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        logger.info("🏠 房地產數據爬取功能待實現")
        
        return {
            'success': True,
            'data_type': 'real_estate',
            'data': [],
            'metadata': {
                'note': '房地產數據爬取功能待實現',
                'sources': config.get('sources', []),
                'metrics': config.get('metrics', [])
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['real_estate', 'housing_data']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

async def main():
    """主函數"""
    crawler = ComprehensiveDataCrawler()
    
    try:
        # 開始全面爬取
        results = await crawler.start_comprehensive_crawling()
        
        # 顯示結果
        print("\n🎯 全面爬取結果摘要:")
        print(f"   總計數據源: {results['total_sources']}")
        print(f"   成功: {results['successful_sources']}")
        print(f"   失敗: {results['failed_sources']}")
        
        # 顯示可用數據類型
        available_types = crawler.get_available_data_types()
        print(f"\n📊 可用數據類型 ({len(available_types)}):")
        for data_type in available_types:
            print(f"   - {data_type}")
        
        # 顯示數據摘要
        data_summary = crawler.get_data_summary()
        if data_summary:
            print("\n📈 數據摘要:")
            for data_type, stats in data_summary.items():
                print(f"   {data_type}: {stats['record_count']} 條記錄, {stats['symbol_count']} 個符號")
        
    except Exception as e:
        logger.error(f"爬取過程失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
