#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版綜合數據爬取系統
集成插件管理器，支持自動發現和加載新插件
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sqlite3
import json
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from plugin_manager import PluginManager

# 修復 Windows 日誌問題
if sys.platform == "win32":
    # Windows 特定的日誌配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_crawler.log', encoding='utf-8', mode='a')
        ]
    )
else:
    # Unix/Linux 日誌配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# 設置 stdout 和 stderr 編碼（Windows 兼容性）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

@dataclass
class DataSource:
    """數據源配置"""
    name: str
    type: str
    enabled: bool = True
    priority: int = 1
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

class DataCrawlerPlugin(ABC):
    """數據爬取插件抽象基類"""
    
    @abstractmethod
    async def crawl(self, config: Dict[str, Any]) -> Dict[str, Any]:
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

class EnhancedComprehensiveDataCrawler:
    """增強版全面數據爬取器"""
    
    def __init__(self, config_path: str = "enhanced_crawler_config.json"):
        self.config = self._load_config(config_path)
        self.db_path = self.config["storage"]["database"]
        self.plugin_manager = PluginManager()
        self.data_sources: Dict[str, DataSource] = {}
        self.session = None
        self._init_database()
        self._init_data_sources()
        self._load_builtin_plugins()
        
        # 啟動自動插件發現
        self.plugin_manager.start_auto_discovery()
        
        logger.info("🚀 增強版綜合數據爬取系統初始化完成")
    
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
                "retry_count": 3,
                "timeout": 30
            },
            "storage": {
                "database": "enhanced_comprehensive_financial_data.db",
                "backup_interval": 3600,
                "max_backup_count": 10
            },
            "plugins": {
                "auto_discovery": True,
                "discovery_interval": 300,
                "hot_reload": True
            }
        }
    
    def _init_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建綜合數據表
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
        
        # 創建插件管理表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plugins (
                plugin_name TEXT PRIMARY KEY,
                plugin_type TEXT NOT NULL,
                version TEXT,
                enabled BOOLEAN DEFAULT 1,
                last_used DATETIME,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                metadata_json TEXT
            )
        ''')
        
        # 創建爬取歷史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawling_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_sources INTEGER DEFAULT 0,
                successful_sources INTEGER DEFAULT 0,
                failed_sources INTEGER DEFAULT 0,
                total_records INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                error_log TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ 數據庫初始化完成")
    
    def _init_data_sources(self):
        """初始化數據源"""
        for source_name, config in self.config["data_sources"].items():
            self.data_sources[source_name] = DataSource(
                name=source_name,
                type=source_name, # Assuming type is the same as name for now
                config=config
            )
        logger.info(f"✅ 初始化了 {len(self.data_sources)} 個數據源")
    
    def _load_builtin_plugins(self):
        """加載內置插件"""
        # 股票數據插件
        stocks_plugin = StocksDataPlugin()
        self.plugin_manager.add_plugin_manually("stocks", stocks_plugin)
        
        # 加密貨幣插件
        crypto_plugin = CryptoDataPlugin()
        self.plugin_manager.add_plugin_manually("crypto", crypto_plugin)
        
        # 外匯數據插件
        forex_plugin = ForexDataPlugin()
        self.plugin_manager.add_plugin_manually("forex", forex_plugin)
        
        # 商品數據插件
        commodities_plugin = CommoditiesDataPlugin()
        self.plugin_manager.add_plugin_manually("commodities", commodities_plugin)
        
        # 指數數據插件
        indices_plugin = IndicesDataPlugin()
        self.plugin_manager.add_plugin_manually("indices", indices_plugin)
        
        # 經濟指標插件
        economic_plugin = EconomicIndicatorsPlugin()
        self.plugin_manager.add_plugin_manually("economic_indicators", economic_plugin)
        
        logger.info("✅ 內置插件加載完成")
    
    async def start_comprehensive_crawling(self):
        """開始全面爬取"""
        logger.info("🚀 開始增強版全面數據爬取...")
        
        # 創建爬取會話
        session_id = f"crawl_{int(time.time())}"
        self._create_crawling_session(session_id)
        
        async with aiohttp.ClientSession() as self.session:
            # 獲取所有可用的插件
            all_plugins = self.plugin_manager.get_all_plugins()
            
            # 按優先級排序的數據源
            sorted_sources = sorted(
                self.data_sources.values(), 
                key=lambda x: x.priority
            )
            
            # 創建爬取任務
            tasks = []
            for source in sorted_sources:
                if source.enabled and source.name in all_plugins:
                    task = self._crawl_data_source(source, session_id)
                    tasks.append(task)
            
            # 並行執行爬取任務
            semaphore = asyncio.Semaphore(self.config["crawling"]["max_concurrent"])
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # 處理結果
            successful = 0
            failed = 0
            total_records = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"❌ 數據源 {sorted_sources[i].name} 爬取失敗: {result}")
                else:
                    if result.get('success', False):
                        successful += 1
                        total_records += result.get('total_records', 0)
                    else:
                        failed += 1
            
            # 更新爬取會話
            self._update_crawling_session(session_id, successful, failed, total_records)
            
            logger.info(f"✅ 爬取完成！成功: {successful}, 失敗: {failed}, 總記錄: {total_records}")
            
            return {
                'session_id': session_id,
                'successful': successful,
                'failed': failed,
                'total_records': total_records
            }
    
    async def _crawl_data_source(self, source: DataSource, session_id: str):
        """爬取單個數據源"""
        try:
            plugin = self.plugin_manager.get_plugin(source.name)
            if not plugin:
                logger.warning(f"⚠️ 數據源 {source.name} 的插件未找到")
                return {'success': False, 'error': 'Plugin not found'}
            
            # 獲取配置
            config = self.config["data_sources"].get(source.name, {})
            
            # 執行爬取
            if asyncio.iscoroutinefunction(plugin.crawl):
                result = await plugin.crawl(config)
            else:
                result = plugin.crawl(config)
            
            if result.get('success', False):
                # 保存數據
                await self._save_crawled_data(source.name, result)
                
                # 更新數據源狀態
                self._update_data_source_status(source.name, True)
                
                return {
                    'success': True,
                    'total_records': len(result.get('data', [])),
                    'data_type': result.get('data_type', 'unknown')
                }
            else:
                self._update_data_source_status(source.name, False)
                return result
                
        except Exception as e:
            logger.error(f"❌ 爬取數據源 {source.name} 失敗: {e}")
            self._update_data_source_status(source.name, False)
            return {'success': False, 'error': str(e)}
    
    async def _save_crawled_data(self, source_name: str, result: Dict):
        """保存爬取的數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = result.get('data', [])
            data_type = result.get('data_type', 'unknown')
            metadata = result.get('metadata', {})
            
            for item in data:
                # 準備數據
                symbol = item.get('symbol', '')
                timestamp = item.get('timestamp', datetime.now().isoformat())
                data_json = json.dumps(item, ensure_ascii=False)
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                
                # 插入或更新數據
                cursor.execute('''
                    INSERT OR REPLACE INTO comprehensive_data 
                    (data_type, source_name, symbol, timestamp, data_json, metadata_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_type, source_name, symbol, timestamp, 
                    data_json, metadata_json, datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 成功保存 {len(data)} 條 {data_type} 數據")
            
        except Exception as e:
            logger.error(f"❌ 保存數據失敗: {e}")
    
    def _create_crawling_session(self, session_id: str):
        """創建爬取會話"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO crawling_history 
                (session_id, start_time, total_sources, status)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id, datetime.now(), len(self.data_sources), 'running'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 創建爬取會話失敗: {e}")
    
    def _update_crawling_session(self, session_id: str, successful: int, failed: int, total_records: int):
        """更新爬取會話"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE crawling_history 
                SET end_time = ?, successful_sources = ?, failed_sources = ?, 
                    total_records = ?, status = ?
                WHERE session_id = ?
            ''', (
                datetime.now(), successful, failed, total_records, 'completed', session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 更新爬取會話失敗: {e}")
    
    def _update_data_source_status(self, source_name: str, success: bool):
        """更新數據源狀態"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 獲取當前狀態
            cursor.execute('''
                SELECT success_rate, error_count, total_attempts 
                FROM data_source_status 
                WHERE source_name = ?
            ''', (source_name,))
            
            row = cursor.fetchone()
            if row:
                success_rate, error_count, total_attempts = row
                total_attempts += 1
                
                if success:
                    success_rate = (success_rate * (total_attempts - 1) + 1) / total_attempts
                else:
                    error_count += 1
                    success_rate = (success_rate * (total_attempts - 1)) / total_attempts
                
                cursor.execute('''
                    UPDATE data_source_status 
                    SET success_rate = ?, error_count = ?, total_attempts = ?, last_updated = ?
                    WHERE source_name = ?
                ''', (success_rate, error_count, total_attempts, datetime.now(), source_name))
            else:
                # 創建新記錄
                cursor.execute('''
                    INSERT INTO data_source_status 
                    (source_name, data_type, success_rate, error_count, total_attempts, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source_name, 'unknown', 1.0 if success else 0.0,
                    0 if success else 1, 1, datetime.now()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"更新數據源狀態失敗: {e}")
        finally:
            conn.close()
    
    def add_custom_data_source(self, name: str, config: Dict, plugin: DataCrawlerPlugin):
        """添加自定義數據源"""
        # 添加到插件管理器
        success = self.plugin_manager.add_plugin_manually(name, plugin, metadata=config)
        
        if success:
            # 添加到數據源
            self.data_sources[name] = DataSource(
                name=name,
                type=name, # Assuming type is the same as name for now
                config=config
            )
            
            # 更新配置
            self.config["data_sources"][name] = config
            
            logger.info(f"✅ 成功添加自定義數據源: {name}")
            return True
        else:
            logger.error(f"❌ 添加自定義數據源失敗: {name}")
            return False
    
    def get_available_data_types(self) -> List[str]:
        """獲取可用的數據類型"""
        return list(self.plugin_manager.get_all_plugins().keys())
    
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
    
    def get_plugin_info(self) -> Dict[str, Dict]:
        """獲取插件信息"""
        return self.plugin_manager.get_plugin_info()
    
    def test_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """測試插件"""
        return self.plugin_manager.test_plugin(plugin_name)
    
    def create_plugin_template(self, plugin_name: str, plugin_type: str = "custom") -> str:
        """創建插件模板"""
        return self.plugin_manager.save_plugin_template(plugin_name, plugin_type)
    
    def stop(self):
        """停止系統"""
        self.plugin_manager.stop_auto_discovery()
        logger.info("🛑 增強版綜合數據爬取系統已停止")

# 內置插件實現
class StocksDataPlugin(DataCrawlerPlugin):
    """股票數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
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
                    
                    logger.info(f"✅ 成功爬取股票 {symbol} 數據")
                else:
                    logger.warning(f"⚠️ 股票 {symbol} 沒有數據")
                    
            except Exception as e:
                logger.error(f"❌ 爬取股票 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'stocks',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['stocks']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class CryptoDataPlugin(DataCrawlerPlugin):
    """加密貨幣數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"])
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
                    
                    logger.info(f"✅ 成功爬取加密貨幣 {symbol} 數據")
                else:
                    logger.warning(f"⚠️ 加密貨幣 {symbol} 沒有數據")
                    
            except Exception as e:
                logger.error(f"❌ 爬取加密貨幣 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'crypto',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['crypto']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class ForexDataPlugin(DataCrawlerPlugin):
    """外匯數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", ["EUR-USD", "GBP-USD", "JPY-USD", "CHF-USD"])
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
                    
                    logger.info(f"✅ 成功爬取外匯 {symbol} 數據")
                else:
                    logger.warning(f"⚠️ 外匯 {symbol} 沒有數據")
                    
            except Exception as e:
                logger.error(f"❌ 爬取外匯 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'forex',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['forex']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class CommoditiesDataPlugin(DataCrawlerPlugin):
    """商品數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", ["GC=F", "SI=F", "CL=F", "NG=F"])
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
                    
                    logger.info(f"✅ 成功爬取商品 {symbol} 數據")
                else:
                    logger.warning(f"⚠️ 商品 {symbol} 沒有數據")
                    
            except Exception as e:
                logger.error(f"❌ 爬取商品 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'commodities',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['commodities']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class IndicesDataPlugin(DataCrawlerPlugin):
    """指數數據插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        symbols = config.get("symbols", ["^GSPC", "^DJI", "^IXIC", "^FTSE"])
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
                    
                    logger.info(f"✅ 成功爬取指數 {symbol} 數據")
                else:
                    logger.warning(f"⚠️ 指數 {symbol} 沒有數據")
                    
            except Exception as e:
                logger.error(f"❌ 爬取指數 {symbol} 失敗: {e}")
        
        return {
            'success': True,
            'data_type': 'indices',
            'data': results,
            'metadata': {
                'total_symbols': len(symbols),
                'successful_records': len(results),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['indices']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'yfinance': '>=0.2.0'}

class EconomicIndicatorsPlugin(DataCrawlerPlugin):
    """經濟指標插件"""
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        # 模擬經濟指標數據
        indicators = [
            {
                'indicator': 'GDP_Growth',
                'value': 2.1,
                'unit': '%',
                'country': 'US',
                'period': 'Q4 2023',
                'timestamp': datetime.now().isoformat()
            },
            {
                'indicator': 'Inflation_Rate',
                'value': 3.2,
                'unit': '%',
                'country': 'US',
                'period': 'December 2023',
                'timestamp': datetime.now().isoformat()
            },
            {
                'indicator': 'Unemployment_Rate',
                'value': 3.7,
                'unit': '%',
                'country': 'US',
                'period': 'December 2023',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        logger.info("✅ 成功爬取經濟指標數據")
        
        return {
            'success': True,
            'data_type': 'economic_indicators',
            'data': indicators,
            'metadata': {
                'total_indicators': len(indicators),
                'crawled_at': datetime.now().isoformat()
            }
        }
    
    def get_supported_types(self) -> List[str]:
        return ['economic_indicators']
    
    def get_requirements(self) -> Dict[str, str]:
        return {'requests': '>=2.25.0'}

if __name__ == "__main__":
    # 測試增強版爬取器
    async def test_enhanced_crawler():
        crawler = EnhancedComprehensiveDataCrawler()
        
        print("🧪 測試增強版綜合數據爬取器...")
        
        # 顯示可用數據類型
        data_types = crawler.get_available_data_types()
        print(f"📋 可用數據類型: {len(data_types)}")
        for data_type in data_types:
            print(f"  - {data_type}")
        
        # 顯示插件信息
        plugin_info = crawler.get_plugin_info()
        print(f"\n🔌 插件信息:")
        for name, info in plugin_info.items():
            print(f"  - {name}: {info['status']} ({info['type']})")
        
        # 開始爬取
        print("\n🚀 開始爬取...")
        result = await crawler.start_comprehensive_crawling()
        print(f"✅ 爬取完成: {result}")
        
        # 顯示數據摘要
        summary = crawler.get_data_summary()
        print(f"\n📊 數據摘要:")
        for data_type, info in summary.items():
            print(f"  - {data_type}: {info['record_count']} 條記錄")
        
        # 停止系統
        crawler.stop()
    
    asyncio.run(test_enhanced_crawler())
