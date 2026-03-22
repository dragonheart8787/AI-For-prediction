#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版數據爬取器
支持爬取大量真實金融數據，包括股票、加密貨幣、經濟指標等
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import yfinance as yf
import quandl
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDataCrawler:
    """增強版數據爬取器"""
    
    def __init__(self, config_path: str = "crawler_config.json"):
        self.config = self._load_config(config_path)
        self.session = None
        self.db_path = "enhanced_financial_data.db"
        self._init_database()
        
    def _load_config(self, config_path: str) -> Dict:
        """加載爬取器配置"""
        default_config = {
            "data_sources": {
                "stocks": {
                    "symbols": [
                        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX",
                        "TSM", "ASML", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC",
                        "QCOM", "TXN", "MU", "KLAC", "LRCX", "AMAT", "ADI", "MCHP",
                        "NXPI", "MRVL", "WDC", "STX", "HPQ", "DELL", "CSCO", "JPM",
                        "BAC", "WFC", "GS", "MS", "JNJ", "PFE", "UNH", "HD", "PG"
                    ],
                    "period": "2y",
                    "interval": "1d"
                },
                "crypto": {
                    "symbols": [
                        "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD", "MATIC-USD",
                        "AVAX-USD", "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
                        "XRP-USD", "DOGE-USD", "SHIB-USD", "TRX-USD", "ETC-USD", "FIL-USD",
                        "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD", "THETA-USD", "FTM-USD"
                    ],
                    "period": "2y",
                    "interval": "1d"
                },
                "forex": {
                    "symbols": [
                        "EURUSD=X", "GBPUSD=X", "JPYUSD=X", "CADUSD=X", "AUDUSD=X", "CHFUSD=X",
                        "NZDUSD=X", "SEKUSD=X", "NOKUSD=X", "DKKUSD=X", "PLNUSD=X", "CZKUSD=X",
                        "HUFUSD=X", "RUBUSD=X", "TRYUSD=X", "ZARUSD=X", "BRLUSD=X", "MXNUSD=X",
                        "INRUSD=X", "KRWUSD=X", "CNYUSD=X", "HKDUSD=X", "SGDUSD=X", "THBUSD=X"
                    ],
                    "period": "2y",
                    "interval": "1d"
                },
                "commodities": {
                    "symbols": [
                        "GC=F", "CL=F", "SI=F", "NG=F", "PL=F", "PA=F", "HG=F", "ZC=F",
                        "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F", "SB=F", "CC=F", "CT=F"
                    ],
                    "period": "2y",
                    "interval": "1d"
                },
                "indices": {
                    "symbols": [
                        "^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225", "^HSI", "^GDAXI", "^FCHI",
                        "^STOXX50E", "^VIX", "^TNX", "^TYX", "^DXY", "^XAU", "^XAG", "^CRB"
                    ],
                    "period": "2y",
                    "interval": "1d"
                },
                "economic_indicators": {
                    "sources": ["FRED", "WORLD_BANK", "IMF"],
                    "indicators": [
                        "GDP", "CPI", "UNEMPLOYMENT", "INTEREST_RATE", "INFLATION",
                        "TRADE_BALANCE", "FOREIGN_EXCHANGE", "MONEY_SUPPLY"
                    ]
                }
            },
            "crawling": {
                "max_concurrent": 10,
                "rate_limit": 0.1,  # 秒
                "retry_attempts": 3,
                "timeout": 30
            },
            "storage": {
                "database": "enhanced_financial_data.db",
                "backup_interval": "1d",
                "max_data_age": "5y"
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合併配置
                    for key, value in user_config.items():
                        if key in default_config:
                            if isinstance(value, dict) and isinstance(default_config[key], dict):
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
            except Exception as e:
                logger.warning(f"無法加載用戶配置: {e}")
        
        return default_config
    
    def _init_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建數據表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                adj_close REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, data_type, timestamp)
            )
        ''')
        
        # 創建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_type_timestamp 
            ON financial_data(symbol, data_type, timestamp)
        ''')
        
        # 創建元數據表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                last_updated DATETIME,
                data_count INTEGER DEFAULT 0,
                min_date DATETIME,
                max_date DATETIME,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ 數據庫初始化完成")
    
    async def start_crawling(self):
        """開始爬取數據"""
        logger.info("🚀 開始增強版數據爬取...")
        
        # 創建aiohttp會話
        async with aiohttp.ClientSession() as self.session:
            tasks = []
            
            # 爬取股票數據
            if self.config["data_sources"]["stocks"]["symbols"]:
                tasks.append(self._crawl_stocks())
            
            # 爬取加密貨幣數據
            if self.config["data_sources"]["crypto"]["symbols"]:
                tasks.append(self._crawl_crypto())
            
            # 爬取外匯數據
            if self.config["data_sources"]["forex"]["symbols"]:
                tasks.append(self._crawl_forex())
            
            # 爬取商品數據
            if self.config["data_sources"]["commodities"]["symbols"]:
                tasks.append(self._crawl_commodities())
            
            # 爬取指數數據
            if self.config["data_sources"]["indices"]["symbols"]:
                tasks.append(self._crawl_indices())
            
            # 爬取經濟指標
            if self.config["data_sources"]["economic_indicators"]["indicators"]:
                tasks.append(self._crawl_economic_indicators())
            
            # 並行執行所有爬取任務
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 處理結果
            total_symbols = 0
            successful_symbols = 0
            
            for result in results:
                if isinstance(result, dict):
                    total_symbols += result.get('total', 0)
                    successful_symbols += result.get('successful', 0)
                elif isinstance(result, Exception):
                    logger.error(f"爬取任務失敗: {result}")
            
            logger.info(f"🎯 爬取完成！總計: {total_symbols} 個符號，成功: {successful_symbols} 個")
            return {
                'total': total_symbols,
                'successful': successful_symbols,
                'failed': total_symbols - successful_symbols
            }
    
    async def _crawl_stocks(self) -> Dict:
        """爬取股票數據"""
        logger.info("📈 開始爬取股票數據...")
        symbols = self.config["data_sources"]["stocks"]["symbols"]
        period = self.config["data_sources"]["stocks"]["period"]
        interval = self.config["data_sources"]["stocks"]["interval"]
        
        results = await self._crawl_yfinance_data(symbols, "stocks", period, interval)
        logger.info(f"✅ 股票數據爬取完成: {results['successful']}/{results['total']} 成功")
        return results
    
    async def _crawl_crypto(self) -> Dict:
        """爬取加密貨幣數據"""
        logger.info("🪙 開始爬取加密貨幣數據...")
        symbols = self.config["data_sources"]["crypto"]["symbols"]
        period = self.config["data_sources"]["crypto"]["period"]
        interval = self.config["data_sources"]["crypto"]["interval"]
        
        results = await self._crawl_yfinance_data(symbols, "crypto", period, interval)
        logger.info(f"✅ 加密貨幣數據爬取完成: {results['successful']}/{results['total']} 成功")
        return results
    
    async def _crawl_forex(self) -> Dict:
        """爬取外匯數據"""
        logger.info("💱 開始爬取外匯數據...")
        symbols = self.config["data_sources"]["forex"]["symbols"]
        period = self.config["data_sources"]["forex"]["period"]
        interval = self.config["data_sources"]["forex"]["interval"]
        
        results = await self._crawl_yfinance_data(symbols, "forex", period, interval)
        logger.info(f"✅ 外匯數據爬取完成: {results['successful']}/{results['total']} 成功")
        return results
    
    async def _crawl_commodities(self) -> Dict:
        """爬取商品數據"""
        logger.info("🛢️ 開始爬取商品數據...")
        symbols = self.config["data_sources"]["commodities"]["symbols"]
        period = self.config["data_sources"]["commodities"]["period"]
        interval = self.config["data_sources"]["commodities"]["interval"]
        
        results = await self._crawl_yfinance_data(symbols, "commodities", period, interval)
        logger.info(f"✅ 商品數據爬取完成: {results['successful']}/{results['total']} 成功")
        return results
    
    async def _crawl_indices(self) -> Dict:
        """爬取指數數據"""
        logger.info("📊 開始爬取指數數據...")
        symbols = self.config["data_sources"]["indices"]["symbols"]
        period = self.config["data_sources"]["indices"]["period"]
        interval = self.config["data_sources"]["indices"]["interval"]
        
        results = await self._crawl_yfinance_data(symbols, "indices", period, interval)
        logger.info(f"✅ 指數數據爬取完成: {results['successful']}/{results['total']} 成功")
        return results
    
    async def _crawl_economic_indicators(self) -> Dict:
        """爬取經濟指標數據"""
        logger.info("📊 開始爬取經濟指標數據...")
        # 這裡可以集成FRED、世界銀行等API
        # 暫時返回空結果
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    async def _crawl_yfinance_data(self, symbols: List[str], data_type: str, 
                                  period: str, interval: str) -> Dict:
        """使用yfinance爬取數據"""
        total = len(symbols)
        successful = 0
        failed = 0
        
        # 創建任務列表
        tasks = []
        for symbol in symbols:
            task = self._crawl_single_symbol(symbol, data_type, period, interval)
            tasks.append(task)
        
        # 並行執行，但限制並發數
        semaphore = asyncio.Semaphore(self.config["crawling"]["max_concurrent"])
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # 統計結果
        for result in results:
            if isinstance(result, dict) and result.get('success'):
                successful += 1
            else:
                failed += 1
                if isinstance(result, Exception):
                    logger.error(f"爬取失敗: {result}")
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed
        }
    
    async def _crawl_single_symbol(self, symbol: str, data_type: str, 
                                  period: str, interval: str) -> Dict:
        """爬取單個符號的數據"""
        try:
            # 使用yfinance獲取數據
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"⚠️ 符號 {symbol} 沒有數據")
                return {'success': False, 'symbol': symbol, 'error': 'No data'}
            
            # 保存到數據庫
            await self._save_data_to_db(symbol, data_type, data)
            
            # 更新元數據
            await self._update_metadata(symbol, data_type, data)
            
            logger.info(f"✅ {symbol} 數據爬取成功: {len(data)} 條記錄")
            return {'success': True, 'symbol': symbol, 'records': len(data)}
            
        except Exception as e:
            logger.error(f"❌ 爬取 {symbol} 失敗: {e}")
            return {'success': False, 'symbol': symbol, 'error': str(e)}
    
    async def _save_data_to_db(self, symbol: str, data_type: str, data: pd.DataFrame):
        """保存數據到數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO financial_data 
                    (symbol, data_type, timestamp, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, data_type, timestamp,
                    row.get('Open', None),
                    row.get('High', None),
                    row.get('Low', None),
                    row.get('Close', None),
                    row.get('Volume', None),
                    row.get('Adj Close', None)
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"保存數據失敗: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def _update_metadata(self, symbol: str, data_type: str, data: pd.DataFrame):
        """更新元數據"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            min_date = data.index.min()
            max_date = data.index.max()
            count = len(data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO data_metadata 
                (symbol, data_type, last_updated, data_count, min_date, max_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, data_type, datetime.now(), count, min_date, max_date))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"更新元數據失敗: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_available_data(self) -> Dict:
        """獲取可用的數據信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 獲取所有數據類型統計
            cursor.execute('''
                SELECT data_type, COUNT(DISTINCT symbol) as symbol_count, 
                       SUM(data_count) as total_records
                FROM data_metadata 
                GROUP BY data_type
            ''')
            
            stats = {}
            for row in cursor.fetchall():
                data_type, symbol_count, total_records = row
                stats[data_type] = {
                    'symbols': symbol_count,
                    'total_records': total_records or 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"獲取數據統計失敗: {e}")
            return {}
        finally:
            conn.close()
    
    def get_data_for_training(self, symbols: List[str] = None, 
                             data_type: str = None, 
                             start_date: str = None, 
                             end_date: str = None) -> Dict[str, pd.DataFrame]:
        """獲取用於訓練的數據"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = "SELECT * FROM financial_data WHERE 1=1"
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY symbol, timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return {}
            
            # 按符號分組
            grouped_data = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data.set_index('timestamp', inplace=True)
                symbol_data.sort_index(inplace=True)
                grouped_data[symbol] = symbol_data
            
            return grouped_data
            
        except Exception as e:
            logger.error(f"獲取訓練數據失敗: {e}")
            return {}
        finally:
            conn.close()

async def main():
    """主函數"""
    crawler = EnhancedDataCrawler()
    
    try:
        # 開始爬取
        results = await crawler.start_crawling()
        
        # 顯示結果
        print("\n🎯 爬取結果摘要:")
        print(f"   總計符號: {results['total']}")
        print(f"   成功: {results['successful']}")
        print(f"   失敗: {results['failed']}")
        
        # 顯示可用數據
        available_data = crawler.get_available_data()
        print("\n📊 可用數據統計:")
        for data_type, stats in available_data.items():
            print(f"   {data_type}: {stats['symbols']} 個符號, {stats['total_records']} 條記錄")
        
    except Exception as e:
        logger.error(f"爬取過程失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
