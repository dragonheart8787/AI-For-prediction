"""
強化版AGI預測系統 V2.0 - Enhanced AGI Prediction System
修復了資料庫游標錯誤、雲端上傳問題，並增強了系統穩定性
"""

import os
import json
import pickle
import sqlite3
import asyncio
import logging
import datetime
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import partial
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
import zipfile
import shutil
import traceback

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_enhanced_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedConfig:
    """強化配置"""
    # 本地儲存路徑
    local_storage_path: str = "./agi_storage"
    model_storage_path: str = "./agi_storage/models"
    data_storage_path: str = "./agi_storage/data"
    state_storage_path: str = "./agi_storage/state"
    
    # 雲端配置
    cloud_enabled: bool = True
    cloud_storage_url: str = "https://api.example.com/agi-storage"
    cloud_api_key: str = ""
    cloud_retry_attempts: int = 3
    cloud_timeout: int = 30
    
    # 資料庫配置
    db_path: str = "./agi_storage/agi_enhanced_v2.db"
    
    # 訓練配置
    training_batch_size: int = 32
    training_epochs: int = 100
    learning_rate: float = 0.001
    
    # 持續學習配置
    continuous_learning_enabled: bool = True
    retrain_interval_hours: int = 24
    performance_threshold: float = 0.8
    auto_optimization_enabled: bool = True
    
    # 預測配置
    prediction_cache_size: int = 1000
    confidence_threshold: float = 0.7
    
    # 系統穩定性配置
    health_check_interval: int = 300  # 5分鐘
    max_retry_attempts: int = 5
    error_recovery_enabled: bool = True

class EnhancedStorage:
    """強化儲存管理器"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.conn = None
        self.cursor = None
        self._ensure_directories()
        self._init_database()
        
    def _ensure_directories(self):
        """確保目錄存在"""
        directories = [
            self.config.local_storage_path,
            self.config.model_storage_path,
            self.config.data_storage_path,
            self.config.state_storage_path
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"[SUCCESS] 目錄已確保: {directory}")
    
    def _init_database(self):
        """初始化資料庫"""
        try:
            self.conn = sqlite3.connect(self.config.db_path)
            self.cursor = self.conn.cursor()
            
            # 創建表格
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_data BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    input_data TEXT,
                    prediction_result TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 資料庫初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 資料庫初始化失敗: {e}")
            raise
    
    def get_cursor(self):
        """獲取資料庫游標，確保連接有效"""
        try:
            if self.conn is None or self.cursor is None:
                self._init_database()
            return self.cursor
        except Exception as e:
            logger.error(f"❌ 獲取游標失敗: {e}")
            return None
    
    def save_model(self, name: str, model_type: str, model_data: Any, 
                   metadata: Dict[str, Any] = None):
        """儲存模型"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return False
                
            model_bytes = pickle.dumps(model_data)
            metadata_str = json.dumps(metadata) if metadata else "{}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO models 
                (name, type, version, model_data, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (name, model_type, '1.0', model_bytes, metadata_str))
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 模型已儲存: {name} v1.0")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型儲存失敗: {e}")
            return False
    
    def load_model(self, name: str) -> Optional[Any]:
        """載入模型"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return None
                
            cursor.execute('SELECT model_data FROM models WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                model_data = pickle.loads(result[0])
                logger.info(f"[SUCCESS] 模型已載入: {name}")
                return model_data
            else:
                logger.warning(f"[WARNING] 模型未找到: {name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}")
            return None
    
    def save_prediction(self, model_name: str, input_data: Any, 
                       prediction_result: Any, confidence: float):
        """儲存預測結果"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return False
                
            cursor.execute('''
                INSERT INTO predictions 
                (model_name, input_data, prediction_result, confidence)
                VALUES (?, ?, ?, ?)
            ''', (model_name, json.dumps(input_data), 
                  json.dumps(prediction_result), confidence))
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 預測結果已儲存: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 預測結果儲存失敗: {e}")
            return False
    
    def save_training_history(self, model_name: str, epoch: int, 
                            loss: float, accuracy: float):
        """儲存訓練歷史"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return False
                
            cursor.execute('''
                INSERT INTO training_history 
                (model_name, epoch, loss, accuracy)
                VALUES (?, ?, ?, ?)
            ''', (model_name, epoch, loss, accuracy))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 訓練歷史儲存失敗: {e}")
            return False
    
    def save_system_state(self, key: str, value: Any):
        """儲存系統狀態"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return False
                
            cursor.execute('''
                INSERT OR REPLACE INTO system_state 
                (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, json.dumps(value)))
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 系統狀態已儲存: {key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系統狀態儲存失敗: {e}")
            return False
    
    def load_system_state(self, key: str):
        """載入系統狀態"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return None
                
            cursor.execute('''
                SELECT value FROM system_state WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
            
        except Exception as e:
            logger.error(f"❌ 系統狀態載入失敗: {e}")
            return None
    
    def save_performance_metric(self, model_name: str, metric_name: str, 
                               metric_value: float):
        """儲存性能指標"""
        try:
            cursor = self.get_cursor()
            if not cursor:
                return False
                
            cursor.execute('''
                INSERT INTO performance_metrics 
                (model_name, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (model_name, metric_name, metric_value))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能指標儲存失敗: {e}")
            return False
    
    def close(self):
        """關閉資料庫連接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("[SUCCESS] 資料庫連接已關閉")
        except Exception as e:
            logger.error(f"❌ 關閉資料庫連接失敗: {e}")

class EnhancedCloudStorage:
    """強化雲端儲存管理器"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.session = requests.Session()
        if config.cloud_api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.cloud_api_key}'})
    
    async def upload_model(self, model_name: str, model_data: Any, 
                          metadata: Dict[str, Any]) -> bool:
        """上傳模型到雲端"""
        for attempt in range(self.config.cloud_retry_attempts):
            try:
                if not self.config.cloud_enabled:
                    logger.info("[WARNING] 雲端儲存已停用")
                    return False
                
                # 序列化模型
                model_bytes = pickle.dumps(model_data)
                
                # 準備上傳資料
                files = {'model': (f'{model_name}.pkl', model_bytes)}
                data = {'metadata': json.dumps(metadata)}
                
                # 上傳到雲端
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.session.post(
                        f"{self.config.cloud_storage_url}/upload",
                        files=files, 
                        data=data,
                        timeout=self.config.cloud_timeout
                    )
                )
                
                if response.status_code == 200:
                    logger.info(f"[SUCCESS] 模型已上傳到雲端: {model_name}")
                    return True
                else:
                    logger.warning(f"⚠️ 雲端上傳失敗 (嘗試 {attempt + 1}/{self.config.cloud_retry_attempts}): {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ 雲端上傳異常 (嘗試 {attempt + 1}/{self.config.cloud_retry_attempts}): {e}")
                
                if attempt < self.config.cloud_retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 指數退避
                
        logger.error(f"❌ 雲端上傳失敗，已重試 {self.config.cloud_retry_attempts} 次")
        return False
    
    async def download_model(self, model_name: str) -> Optional[Any]:
        """從雲端下載模型"""
        for attempt in range(self.config.cloud_retry_attempts):
            try:
                if not self.config.cloud_enabled:
                    return None
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(
                        f"{self.config.cloud_storage_url}/download/{model_name}",
                        timeout=self.config.cloud_timeout
                    )
                )
                
                if response.status_code == 200:
                    model_data = pickle.loads(response.content)
                    logger.info(f"[SUCCESS] 模型已從雲端下載: {model_name}")
                    return model_data
                else:
                    logger.warning(f"⚠️ 雲端下載失敗 (嘗試 {attempt + 1}/{self.config.cloud_retry_attempts}): {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ 雲端下載異常 (嘗試 {attempt + 1}/{self.config.cloud_retry_attempts}): {e}")
                
                if attempt < self.config.cloud_retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                
        logger.error(f"❌ 雲端下載失敗，已重試 {self.config.cloud_retry_attempts} 次")
        return None
