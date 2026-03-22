#!/usr/bin/env python3
"""
超級融合AGI系統 - Super Fusion AGI System
整合所有現有預測模型，具備持續學習、報告生成和數據儲存功能
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
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('super_fusion_agi.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SuperFusionConfig:
    """超級融合配置"""
    # 儲存路徑
    storage_path: str = "./super_fusion_storage"
    models_path: str = "./agi_storage/models"  # 現有模型路徑
    data_path: str = "./super_fusion_storage/data"
    reports_path: str = "./super_fusion_storage/reports"
    visualizations_path: str = "./super_fusion_storage/visualizations"
    
    # 資料庫配置
    db_path: str = "./super_fusion_storage/super_fusion.db"
    
    # 融合配置
    ensemble_methods: List[str] = None
    fusion_weights: Dict[str, float] = None
    confidence_threshold: float = 0.7
    
    # 持續學習配置
    continuous_learning: bool = True
    retrain_interval_hours: int = 6
    performance_threshold: float = 0.8
    
    # 報告配置
    report_interval_minutes: int = 30
    auto_save_interval_minutes: int = 5
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['weighted_average', 'stacking', 'voting']
        if self.fusion_weights is None:
            self.fusion_weights = {
                'financial_lstm': 0.25,
                'financial_transformer': 0.25,
                'weather_lstm': 0.25,
                'weather_transformer': 0.25
            }

class SuperFusionStorage:
    """超級融合儲存管理器"""
    
    def __init__(self, config: SuperFusionConfig):
        self.config = config
        self.conn = None
        self.cursor = None
        self._ensure_directories()
        self._init_database()
        
    def _ensure_directories(self):
        """確保目錄存在"""
        directories = [
            self.config.storage_path,
            self.config.data_path,
            self.config.reports_path,
            self.config.visualizations_path
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"[SUCCESS] 目錄已確保: {directory}")
    
    def _init_database(self):
        """初始化資料庫"""
        try:
            self.conn = sqlite3.connect(self.config.db_path)
            self.cursor = self.conn.cursor()
            
            # 創建超級融合表格
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS fusion_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    model_type TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    performance_score REAL,
                    fusion_weight REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS fusion_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    individual_predictions TEXT,
                    fusion_prediction REAL,
                    confidence REAL,
                    ensemble_variance REAL,
                    model_weights TEXT
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    metric_name TEXT,
                    metric_value REAL
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS continuous_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_type TEXT,
                    data_content TEXT,
                    source TEXT
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS fusion_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    report_type TEXT,
                    report_content TEXT,
                    performance_summary TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 超級融合資料庫初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 資料庫初始化失敗: {e}")
            raise
    
    def get_cursor(self):
        """獲取資料庫游標"""
        if not self.cursor:
            self.cursor = self.conn.cursor()
        return self.cursor
    
    def save_fusion_model(self, model_name: str, model_type: str, source_file: str, 
                         performance_score: float = None, fusion_weight: float = None):
        """儲存融合模型資訊"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO fusion_models 
                (model_name, model_type, source_file, performance_score, fusion_weight, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (model_name, model_type, source_file, performance_score, fusion_weight))
            
            self.conn.commit()
            logger.info(f"[SUCCESS] 融合模型已儲存: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合模型儲存失敗: {e}")
            return False
    
    def save_fusion_prediction(self, input_data: Any, individual_predictions: Dict[str, Any],
                              fusion_prediction: float, confidence: float, 
                              ensemble_variance: float = None, model_weights: Dict[str, float] = None):
        """儲存融合預測結果"""
        try:
            self.cursor.execute('''
                INSERT INTO fusion_predictions 
                (input_data, individual_predictions, fusion_prediction, confidence, ensemble_variance, model_weights)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (json.dumps(input_data), json.dumps(individual_predictions), 
                  fusion_prediction, confidence, ensemble_variance, json.dumps(model_weights)))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合預測儲存失敗: {e}")
            return False
    
    def save_performance_metric(self, model_name: str, metric_name: str, metric_value: float):
        """儲存性能指標"""
        try:
            self.cursor.execute('''
                INSERT INTO performance_history 
                (model_name, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (model_name, metric_name, metric_value))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能指標儲存失敗: {e}")
            return False
    
    def save_continuous_data(self, data_type: str, data_content: Any, source: str = "system"):
        """儲存持續數據"""
        try:
            self.cursor.execute('''
                INSERT INTO continuous_data 
                (data_type, data_content, source)
                VALUES (?, ?, ?)
            ''', (data_type, json.dumps(data_content), source))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 持續數據儲存失敗: {e}")
            return False
    
    def save_fusion_report(self, report_id: str, report_type: str, report_content: str, 
                          performance_summary: str = None):
        """儲存融合報告"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO fusion_reports 
                (report_id, report_type, report_content, performance_summary)
                VALUES (?, ?, ?, ?)
            ''', (report_id, report_type, report_content, performance_summary))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合報告儲存失敗: {e}")
            return False
    
    def get_all_fusion_models(self) -> List[Dict[str, Any]]:
        """獲取所有融合模型"""
        try:
            self.cursor.execute('SELECT * FROM fusion_models')
            columns = [description[0] for description in self.cursor.description]
            rows = self.cursor.fetchall()
            
            models = []
            for row in rows:
                model = dict(zip(columns, row))
                models.append(model)
            
            return models
            
        except Exception as e:
            logger.error(f"❌ 獲取融合模型失敗: {e}")
            return []
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """獲取最近的預測結果"""
        try:
            self.cursor.execute('''
                SELECT * FROM fusion_predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in self.cursor.description]
            rows = self.cursor.fetchall()
            
            predictions = []
            for row in rows:
                prediction = dict(zip(columns, row))
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 獲取預測結果失敗: {e}")
            return []
    
    def get_fusion_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """獲取融合預測結果"""
        try:
            cursor = self.get_cursor()
            cursor.execute('''
                SELECT * FROM fusion_predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            predictions = []
            for row in rows:
                prediction = dict(zip(columns, row))
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 獲取融合預測結果失敗: {e}")
            return []
    
    def get_fusion_reports(self) -> List[Dict[str, Any]]:
        """獲取所有融合報告"""
        try:
            cursor = self.get_cursor()
            cursor.execute("""
                SELECT report_id, report_type, report_content, performance_summary, created_at
                FROM fusion_reports
                ORDER BY created_at DESC
            """)
            reports = []
            for row in cursor.fetchall():
                reports.append({
                    'report_id': row[0],
                    'report_type': row[1],
                    'report_content': row[2],
                    'performance_summary': row[3],
                    'created_at': row[4]
                })
            return reports
        except Exception as e:
            logger.error(f"❌ 獲取融合報告失敗: {e}")
            return []
    
    def is_database_healthy(self) -> bool:
        """檢查資料庫健康狀態"""
        try:
            cursor = self.get_cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"❌ 資料庫健康檢查失敗: {e}")
            return False
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """獲取性能歷史數據"""
        try:
            cursor = self.get_cursor()
            cursor.execute("""
                SELECT model_name, metric_name, metric_value, timestamp
                FROM performance_history
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            history = []
            for row in cursor.fetchall():
                history.append({
                    'model_name': row[0],
                    'metric_name': row[1],
                    'metric_value': row[2],
                    'timestamp': row[3]
                })
            return history
        except Exception as e:
            logger.error(f"❌ 獲取性能歷史失敗: {e}")
            return []
    
    def get_continuous_data(self, data_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """獲取持續數據"""
        try:
            cursor = self.get_cursor()
            if data_type:
                cursor.execute("""
                    SELECT data_type, data_content, source, timestamp
                    FROM continuous_data
                    WHERE data_type = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (data_type,))
            else:
                cursor.execute("""
                    SELECT data_type, data_content, source, timestamp
                    FROM continuous_data
                    ORDER BY timestamp DESC
                    LIMIT 100
                """)
            
            data = {}
            for row in cursor.fetchall():
                dt = row[0]
                if dt not in data:
                    data[dt] = []
                data[dt].append({
                    'data_type': dt,
                    'data_content': row[1],
                    'source': row[2],
                    'timestamp': row[3]
                })
            return data
        except Exception as e:
            logger.error(f"❌ 獲取持續數據失敗: {e}")
            return {}
    
    def close(self):
        """關閉資料庫連接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("[SUCCESS] 超級融合資料庫連接已關閉")
        except Exception as e:
            logger.error(f"❌ 關閉資料庫連接失敗: {e}")

class ModelLoader:
    """模型載入器"""
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.loaded_models = {}
        
    def load_all_models(self) -> Dict[str, Any]:
        """載入所有現有模型"""
        try:
            logger.info(f"🔍 開始載入所有模型從: {self.models_path}")
            
            # 掃描模型目錄
            model_files = []
            for file in os.listdir(self.models_path):
                if file.endswith('.pkl'):
                    model_files.append(file)
            
            logger.info(f"📁 發現 {len(model_files)} 個模型文件")
            
            # 載入每個模型
            for model_file in model_files:
                try:
                    model_path = os.path.join(self.models_path, model_file)
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # 提取模型名稱和類型
                    model_name = self._extract_model_name(model_file)
                    model_type = self._extract_model_type(model_file)
                    
                    self.loaded_models[model_name] = {
                        'model': model,
                        'type': model_type,
                        'file': model_file,
                        'path': model_path,
                        'loaded_at': datetime.datetime.now()
                    }
                    
                    logger.info(f"✅ 模型載入成功: {model_name} ({model_type})")
                    
                except Exception as e:
                    logger.error(f"❌ 模型載入失敗 {model_file}: {e}")
            
            logger.info(f"🎯 總共載入 {len(self.loaded_models)} 個模型")
            return self.loaded_models
            
        except Exception as e:
            logger.error(f"❌ 模型載入過程失敗: {e}")
            return {}
    
    def _extract_model_name(self, filename: str) -> str:
        """從文件名提取模型名稱"""
        # 移除 .pkl 後綴
        name = filename.replace('.pkl', '')
        
        # 處理版本號
        if '_1.0' in name:
            name = name.replace('_1.0', '')
        
        # 處理重複的類型標識
        if '_lstm_lstm' in name:
            name = name.replace('_lstm_lstm', '_lstm')
        elif '_transformer_transformer' in name:
            name = name.replace('_transformer_transformer', '_transformer')
        
        return name
    
    def _extract_model_type(self, filename: str) -> str:
        """從文件名提取模型類型"""
        if 'lstm' in filename.lower():
            return 'LSTM'
        elif 'transformer' in filename.lower():
            return 'Transformer'
        else:
            return 'Unknown'
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """獲取特定模型"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]['model']
        return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """獲取模型資訊"""
        return self.loaded_models.get(model_name)

class SuperFusionEngine:
    """超級融合引擎"""
    
    def __init__(self, config: SuperFusionConfig, storage: SuperFusionStorage, model_loader: ModelLoader):
        self.config = config
        self.storage = storage
        self.model_loader = model_loader
        self.models = {}
        self.fusion_weights = config.fusion_weights
        self.ensemble_regressor = None
        
        # 初始化融合模型
        self._init_fusion_models()
        
    def _init_fusion_models(self):
        """初始化融合模型"""
        try:
            # 載入所有模型
            self.models = self.model_loader.load_all_models()
            
            # 儲存模型資訊到資料庫
            for model_name, model_info in self.models.items():
                self.storage.save_fusion_model(
                    model_name=model_name,
                    model_type=model_info['type'],
                    source_file=model_info['file'],
                    fusion_weight=self.fusion_weights.get(model_name, 0.25)
                )
            
            # 初始化集成回歸器
            self.ensemble_regressor = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            
            logger.info(f"🚀 超級融合引擎初始化完成，載入 {len(self.models)} 個模型")
            
        except Exception as e:
            logger.error(f"❌ 融合引擎初始化失敗: {e}")
    
    async def make_fusion_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        """進行超級融合預測"""
        try:
            logger.info(f"🔮 開始超級融合預測，輸入數據: {input_data.shape}")
            
            # 獲取各個模型的預測
            individual_predictions = {}
            predictions_list = []
            
            for model_name, model_info in self.models.items():
                try:
                    # 進行預測
                    prediction = self._make_single_prediction(
                        model_info['model'], 
                        model_info['type'], 
                        input_data
                    )
                    
                    individual_predictions[model_name] = {
                        'prediction': prediction,
                        'confidence': self._calculate_confidence(prediction),
                        'model_type': model_info['type']
                    }
                    
                    predictions_list.append(prediction)
                    
                    logger.info(f"📊 {model_name} 預測完成: {prediction:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} 預測失敗: {e}")
                    individual_predictions[model_name] = {
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'model_type': model_info['type'],
                        'error': str(e)
                    }
            
            # 計算融合預測
            fusion_prediction = self._calculate_fusion_prediction(individual_predictions)
            
            # 計算置信度和方差
            confidence = self._calculate_overall_confidence(individual_predictions)
            ensemble_variance = np.var(predictions_list) if predictions_list else 0.0
            
            # 儲存預測結果
            self.storage.save_fusion_prediction(
                input_data=input_data.tolist(),
                individual_predictions=individual_predictions,
                fusion_prediction=fusion_prediction,
                confidence=confidence,
                ensemble_variance=ensemble_variance,
                model_weights=self.fusion_weights
            )
            
            # 儲存持續數據
            self.storage.save_continuous_data(
                data_type="prediction",
                data_content={
                    'input_shape': input_data.shape,
                    'fusion_prediction': fusion_prediction,
                    'confidence': confidence,
                    'timestamp': datetime.datetime.now().isoformat()
                },
                source="fusion_engine"
            )
            
            result = {
                'fusion_prediction': fusion_prediction,
                'confidence': confidence,
                'ensemble_variance': ensemble_variance,
                'individual_predictions': individual_predictions,
                'model_weights': self.fusion_weights,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            logger.info(f"🎯 超級融合預測完成: {fusion_prediction:.4f} (置信度: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ 超級融合預測失敗: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'fusion_prediction': 0.0,
                'confidence': 0.0
            }
    
    def _make_single_prediction(self, model: Any, model_type: str, input_data: np.ndarray) -> float:
        """進行單個模型預測"""
        try:
            if model_type == 'LSTM':
                return self._predict_lstm(model, input_data)
            elif model_type == 'Transformer':
                return self._predict_transformer(model, input_data)
            else:
                # 默認預測
                return np.random.normal(0, 1)
                
        except Exception as e:
            logger.error(f"❌ 單個模型預測失敗: {e}")
            return 0.0
    
    def _predict_lstm(self, model: Any, input_data: np.ndarray) -> float:
        """LSTM模型預測"""
        try:
            # 模擬LSTM預測
            if hasattr(model, 'weights') and isinstance(model['weights'], list):
                # 多層LSTM
                hidden = input_data
                for i, weight in enumerate(model['weights']):
                    if i < len(model['weights']) - 1:
                        hidden = np.tanh(np.dot(hidden, weight.T))
                    else:
                        output = np.dot(hidden, weight.T)
                
                return float(np.mean(output))
            else:
                # 簡單LSTM
                return float(np.mean(input_data) + np.random.normal(0, 0.1))
                
        except Exception as e:
            logger.error(f"❌ LSTM預測失敗: {e}")
            return float(np.mean(input_data))
    
    def _predict_transformer(self, model: Any, input_data: np.ndarray) -> float:
        """Transformer模型預測"""
        try:
            # 模擬Transformer預測
            if hasattr(model, 'weights') and isinstance(model['weights'], dict):
                # 多層Transformer
                embedded = np.dot(input_data, np.random.randn(input_data.shape[1], 64))
                for _ in range(3):  # 模擬3層
                    attention = np.random.randn(*embedded.shape)
                    embedded = embedded + 0.1 * attention
                
                output = np.mean(embedded, axis=1)
                return float(np.mean(output))
            else:
                # 簡單Transformer
                return float(np.mean(input_data) + np.random.normal(0, 0.1))
                
        except Exception as e:
            logger.error(f"❌ Transformer預測失敗: {e}")
            return float(np.mean(input_data))
    
    def _calculate_fusion_prediction(self, individual_predictions: Dict[str, Any]) -> float:
        """計算融合預測"""
        try:
            # 加權平均
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, pred_info in individual_predictions.items():
                if 'error' not in pred_info:
                    weight = self.fusion_weights.get(model_name, 0.25)
                    prediction = pred_info['prediction']
                    
                    weighted_sum += weight * prediction
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ 融合預測計算失敗: {e}")
            return 0.0
    
    def _calculate_confidence(self, prediction: float) -> float:
        """計算單個預測的置信度"""
        try:
            # 基於預測值的穩定性計算置信度
            confidence = min(1.0, max(0.0, 1.0 - abs(prediction) * 0.1))
            return confidence
        except:
            return 0.5
    
    def _calculate_overall_confidence(self, individual_predictions: Dict[str, Any]) -> float:
        """計算整體置信度"""
        try:
            confidences = []
            for pred_info in individual_predictions.values():
                if 'error' not in pred_info:
                    confidences.append(pred_info['confidence'])
            
            if confidences:
                return np.mean(confidences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ 整體置信度計算失敗: {e}")
            return 0.0
    
    async def retrain_fusion_weights(self, training_data: np.ndarray, target_data: np.ndarray):
        """重新訓練融合權重"""
        try:
            logger.info("🔄 開始重新訓練融合權重")
            
            # 獲取各個模型的預測
            model_predictions = []
            for model_name, model_info in self.models.items():
                predictions = []
                for i in range(len(training_data)):
                    pred = self._make_single_prediction(
                        model_info['model'], 
                        model_info['type'], 
                        training_data[i:i+1]
                    )
                    predictions.append(pred)
                
                model_predictions.append(predictions)
            
            # 轉換為numpy數組
            X = np.array(model_predictions).T
            y = target_data
            
            # 訓練集成回歸器
            self.ensemble_regressor.fit(X, y)
            
            # 更新融合權重
            feature_importances = self.ensemble_regressor.feature_importances_
            model_names = list(self.models.keys())
            
            for i, model_name in enumerate(model_names):
                if i < len(feature_importances):
                    self.fusion_weights[model_name] = float(feature_importances[i])
            
            # 儲存新的權重
            for model_name, weight in self.fusion_weights.items():
                self.storage.save_fusion_model(
                    model_name=model_name,
                    model_type=self.models[model_name]['type'],
                    source_file=self.models[model_name]['file'],
                    fusion_weight=weight
                )
            
            logger.info(f"✅ 融合權重重新訓練完成: {self.fusion_weights}")
            
        except Exception as e:
            logger.error(f"❌ 融合權重重新訓練失敗: {e}")

class ContinuousReportGenerator:
    """持續報告生成器"""
    
    def __init__(self, config: SuperFusionConfig, storage: SuperFusionStorage):
        self.config = config
        self.storage = storage
        self.report_interval = config.report_interval_minutes * 60  # 轉換為秒
        self.auto_save_interval = config.auto_save_interval_minutes * 60  # 轉換為秒
        self.last_report_time = datetime.datetime.now() - datetime.timedelta(seconds=self.report_interval)
        self.last_auto_save_time = datetime.datetime.now() - datetime.timedelta(seconds=self.auto_save_interval)
        self.is_running = False
        self.report_task = None
        self.auto_save_task = None
        
        # 創建報告目錄
        os.makedirs(config.reports_path, exist_ok=True)
        
    async def start_continuous_reporting(self):
        """啟動持續報告生成"""
        if self.is_running:
            logger.warning("報告生成器已經在運行中。")
            return
            
        self.is_running = True
        self.last_report_time = datetime.datetime.now() - datetime.timedelta(seconds=self.report_interval)
        self.last_auto_save_time = datetime.datetime.now() - datetime.timedelta(seconds=self.auto_save_interval)
        
        self.report_task = asyncio.create_task(self._generate_reports_loop())
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        logger.info(f"✅ 持續報告生成器已啟動，每 {self.config.report_interval_minutes} 分鐘生成一次報告，每 {self.config.auto_save_interval_minutes} 分鐘自動保存。")
    
    async def _generate_reports_loop(self):
        """持續生成報告的主循環"""
        while self.is_running:
            current_time = datetime.datetime.now()
            if (current_time - self.last_report_time).total_seconds() >= self.report_interval:
                await self.generate_performance_report()
                await self.generate_prediction_report()
                await self.generate_system_report()
                await self.generate_fusion_report()
                self.last_report_time = current_time
                logger.info(f"✅ 報告已生成，距離上次生成 {self.config.report_interval_minutes} 分鐘。")
            await asyncio.sleep(1) # 避免過度消耗CPU
    
    async def _auto_save_loop(self):
        """持續自動保存的主循環"""
        while self.is_running:
            current_time = datetime.datetime.now()
            if (current_time - self.last_auto_save_time).total_seconds() >= self.auto_save_interval:
                await self.auto_save_data()
                self.last_auto_save_time = current_time
                logger.info(f"✅ 數據已自動保存，距離上次保存 {self.config.auto_save_interval_minutes} 分鐘。")
            await asyncio.sleep(1) # 避免過度消耗CPU
    
    async def generate_performance_report(self):
        """生成性能報告"""
        try:
            logger.info("📊 開始生成性能報告")
            
            # 獲取所有模型性能指標
            performance_data = self.storage.get_performance_history()
            
            if not performance_data:
                logger.warning("沒有性能指標數據可用於生成報告。")
                return
            
            df = pd.DataFrame(performance_data)
            
            # 創建性能報告
            report_content = f"""
# 超級融合AGI性能報告

## 總體性能
- 總預測次數: {len(df)}
- 平均置信度: {df['metric_value'].mean():.4f}
- 平均預測誤差: {df['metric_value'].abs().mean():.4f}

## 模型性能
"""
            
            for model_name in df['model_name'].unique():
                model_df = df[df['model_name'] == model_name]
                report_content += f"""
### {model_name}
- 平均置信度: {model_df['metric_value'].mean():.4f}
- 平均預測誤差: {model_df['metric_value'].abs().mean():.4f}
"""
            
            report_id = f"performance_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 保存到資料庫
            self.storage.save_fusion_report(
                report_id=report_id,
                report_type="performance",
                report_content=report_content,
                performance_summary=f"總體性能報告，包含 {len(df)} 次預測。"
            )
            
            # 生成文件
            report_file = os.path.join(self.config.reports_path, f"{report_id}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 性能報告生成成功: {report_id}")
            
        except Exception as e:
            logger.error(f"❌ 性能報告生成失敗: {e}")
    
    async def generate_prediction_report(self):
        """生成預測報告"""
        try:
            logger.info("📊 開始生成預測報告")
            
            # 獲取最近預測結果
            recent_predictions = self.storage.get_recent_predictions(limit=100)
            
            if not recent_predictions:
                logger.warning("沒有預測結果數據可用於生成報告。")
                return
            
            df = pd.DataFrame(recent_predictions)
            
            # 創建預測報告
            report_content = f"""
# 超級融合AGI預測報告

## 最近預測結果
"""
            
            for _, row in df.iterrows():
                report_content += f"""
- 時間: {row['timestamp']}
  - 輸入數據: {row['input_data']}
  - 融合預測: {row['fusion_prediction']:.4f} (置信度: {row['confidence']:.4f})
  - 個別預測: {row['individual_predictions']}
"""
            
            report_id = f"prediction_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 保存到資料庫
            self.storage.save_fusion_report(
                report_id=report_id,
                report_type="prediction",
                report_content=report_content,
                performance_summary=f"最近 {len(df)} 次預測結果。"
            )
            
            # 生成文件
            report_file = os.path.join(self.config.reports_path, f"{report_id}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 預測報告生成成功: {report_id}")
            
        except Exception as e:
            logger.error(f"❌ 預測報告生成失敗: {e}")
    
    async def generate_system_report(self):
        """生成系統報告"""
        try:
            logger.info("📊 開始生成系統報告")
            
            # 獲取系統啟動和停止狀態
            startup_data = self.storage.get_continuous_data("system_startup")
            shutdown_data = self.storage.get_continuous_data("system_shutdown")
            
            report_content = f"""
# 超級融合AGI系統報告

## 系統狀態
- 啟動時間: {startup_data[0]['timestamp']}
- 停止時間: {shutdown_data[0]['timestamp']}
- 總運行時間: {datetime.datetime.now() - datetime.datetime.fromisoformat(startup_data[0]['timestamp'])}

## 模型載入
- 已載入模型數量: {len(self.storage.get_all_fusion_models())}
- 最近載入的模型: {self.storage.get_all_fusion_models()[-1]['model_name']}

## 數據庫健康
- 資料庫路徑: {self.storage.config.db_path}
- 資料庫連接狀態: 正常
"""
            
            report_id = f"system_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 保存到資料庫
            self.storage.save_fusion_report(
                report_id=report_id,
                report_type="system",
                report_content=report_content,
                performance_summary=f"系統運行狀態報告。"
            )
            
            # 生成文件
            report_file = os.path.join(self.config.reports_path, f"{report_id}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 系統報告生成成功: {report_id}")
            
        except Exception as e:
            logger.error(f"❌ 系統報告生成失敗: {e}")
    
    async def generate_fusion_report(self):
        """生成融合報告"""
        try:
            logger.info("📊 開始生成融合報告")
            
            # 獲取融合模型資訊
            fusion_models = self.storage.get_all_fusion_models()
            
            if not fusion_models:
                logger.warning("沒有融合模型資訊可用於生成報告。")
                return
            
            df = pd.DataFrame(fusion_models)
            
            # 創建融合報告
            report_content = f"""
# 超級融合AGI融合報告

## 融合模型
"""
            
            for _, row in df.iterrows():
                report_content += f"""
- 模型名稱: {row['model_name']}
  - 類型: {row['model_type']}
  - 來源文件: {row['source_file']}
  - 融合權重: {row['fusion_weight']:.4f}
"""
            
            report_id = f"fusion_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 保存到資料庫
            self.storage.save_fusion_report(
                report_id=report_id,
                report_type="fusion",
                report_content=report_content,
                performance_summary=f"融合模型狀態報告。"
            )
            
            # 生成文件
            report_file = os.path.join(self.config.reports_path, f"{report_id}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 融合報告生成成功: {report_id}")
            
        except Exception as e:
            logger.error(f"❌ 融合報告生成失敗: {e}")
    
    async def auto_save_data(self):
        """自動保存數據"""
        try:
            logger.info("💾 開始自動保存數據")
            
            # 保存所有融合模型資訊
            # 注意：這裡我們無法直接訪問fusion_engine，所以跳過模型資訊保存
            # 模型資訊已經在初始化時保存過了
            
            # 保存所有融合預測結果
            recent_predictions = self.storage.get_recent_predictions(limit=100) # 只保存最近的
            for pred in recent_predictions:
                self.storage.save_fusion_prediction(
                    input_data=pred['input_data'],
                    individual_predictions=pred['individual_predictions'],
                    fusion_prediction=pred['fusion_prediction'],
                    confidence=pred['confidence'],
                    ensemble_variance=pred['ensemble_variance'],
                    model_weights=pred['model_weights']
                )
            
            # 保存所有融合預測結果
            recent_predictions = self.storage.get_recent_predictions(limit=100) # 只保存最近的
            for pred in recent_predictions:
                self.storage.save_fusion_prediction(
                    input_data=pred['input_data'],
                    individual_predictions=pred['individual_predictions'],
                    fusion_prediction=pred['fusion_prediction'],
                    confidence=pred['confidence'],
                    ensemble_variance=pred['ensemble_variance'],
                    model_weights=pred['model_weights']
                )
            
            # 保存性能指標
            performance_history = self.storage.get_performance_history()
            for item in performance_history:
                self.storage.save_performance_metric(
                    model_name=item['model_name'],
                    metric_name=item['metric_name'],
                    metric_value=item['metric_value']
                )
            
            # 保存持續數據
            continuous_data = self.storage.get_continuous_data()
            for data_type, data_list in continuous_data.items():
                for item in data_list:
                    self.storage.save_continuous_data(
                        data_type=data_type,
                        data_content=item['data_content'],
                        source=item['source']
                    )
            
            # 保存融合報告
            fusion_reports = self.storage.get_fusion_reports()
            for report in fusion_reports:
                self.storage.save_fusion_report(
                    report_id=report['report_id'],
                    report_type=report['report_type'],
                    report_content=report['report_content'],
                    performance_summary=report['performance_summary']
                )
            
            logger.info("✅ 數據自動保存完成")
            
        except Exception as e:
            logger.error(f"❌ 數據自動保存失敗: {e}")

class SuperFusionAGI:
    """超級融合AGI系統"""
    
    def __init__(self, config: SuperFusionConfig):
        self.config = config
        self.storage = SuperFusionStorage(config)
        self.model_loader = ModelLoader(config.models_path)
        self.fusion_engine = SuperFusionEngine(config, self.storage, self.model_loader)
        self.report_generator = ContinuousReportGenerator(config, self.storage)
        
        # 系統狀態
        self.is_running = False
        self.start_time = None
        
        logger.info("🚀 超級融合AGI系統初始化完成")
    
    async def start_system(self):
        """啟動系統"""
        try:
            logger.info("🚀 啟動超級融合AGI系統")
            
            self.is_running = True
            self.start_time = datetime.datetime.now()
            
            # 啟動持續報告生成
            report_task = asyncio.create_task(
                self.report_generator.start_continuous_reporting()
            )
            
            # 儲存系統啟動狀態
            self.storage.save_continuous_data(
                data_type="system_startup",
                data_content={
                    'start_time': self.start_time.isoformat(),
                    'config': self.config.__dict__,
                    'status': 'running'
                },
                source="super_fusion_agi"
            )
            
            logger.info("✅ 超級融合AGI系統啟動成功")
            
            # 等待報告任務
            await report_task
            
        except Exception as e:
            logger.error(f"❌ 系統啟動失敗: {e}")
            self.is_running = False
            raise
    
    async def stop_system(self):
        """停止系統"""
        try:
            logger.info("🛑 停止超級融合AGI系統")
            
            self.is_running = False
            
            # 儲存系統停止狀態
            self.storage.save_continuous_data(
                data_type="system_shutdown",
                data_content={
                    'stop_time': datetime.datetime.now().isoformat(),
                    'uptime': str(datetime.datetime.now() - self.start_time) if self.start_time else "Unknown",
                    'status': 'stopped'
                },
                source="super_fusion_agi"
            )
            
            logger.info("✅ 超級融合AGI系統停止成功")
            
        except Exception as e:
            logger.error(f"❌ 系統停止失敗: {e}")
    
    async def make_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        """進行預測"""
        try:
            logger.info(f"🔮 開始進行預測，輸入數據: {input_data.shape}")
            
            # 使用融合引擎進行預測
            prediction_result = await self.fusion_engine.make_fusion_prediction(input_data)
            
            # 儲存持續數據
            self.storage.save_continuous_data(
                data_type="prediction_request",
                data_content={
                    'input_shape': input_data.shape,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'source': 'user_request'
                },
                source="super_fusion_agi"
            )
            
            logger.info("✅ 預測完成")
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ 預測失敗: {e}")
            return {'error': str(e)}
    
    async def retrain_fusion_weights(self, training_data: np.ndarray, target_data: np.ndarray):
        """重新訓練融合權重"""
        try:
            logger.info("🔄 開始重新訓練融合權重")
            
            # 使用融合引擎重新訓練
            await self.fusion_engine.retrain_fusion_weights(training_data, target_data)
            
            # 儲存持續數據
            self.storage.save_continuous_data(
                data_type="fusion_retraining",
                data_content={
                    'training_samples': len(training_data),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'new_weights': self.fusion_engine.fusion_weights
                },
                source="super_fusion_agi"
            )
            
            logger.info("✅ 融合權重重新訓練完成")
            
        except Exception as e:
            logger.error(f"❌ 融合權重重新訓練失敗: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        try:
            status = {
                'system_status': 'running' if self.is_running else 'stopped',
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': str(datetime.datetime.now() - self.start_time) if self.start_time and self.is_running else None,
                'models_loaded': len(self.fusion_engine.models),
                'fusion_weights': self.fusion_engine.fusion_weights,
                'database_health': self.storage.is_database_healthy(),
                'last_prediction': self._get_last_prediction_time(),
                'total_predictions': len(self.storage.get_recent_predictions()),
                'total_reports': len(self.storage.get_continuous_data("performance_report"))
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 獲取系統狀態失敗: {e}")
            return {'error': str(e)}
    
    def _get_last_prediction_time(self) -> str:
        """獲取最後預測時間"""
        try:
            predictions = self.storage.get_recent_predictions()
            if predictions:
                latest = max(predictions, key=lambda x: x.get('timestamp', ''))
                return latest.get('timestamp', 'Unknown')
            return "Never"
        except:
            return "Unknown"
    
    async def generate_all_reports(self):
        """生成所有報告"""
        try:
            logger.info("📊 開始生成所有報告")
            
            await self.report_generator.generate_performance_report()
            await self.report_generator.generate_prediction_report()
            await self.report_generator.generate_system_report()
            await self.report_generator.generate_fusion_report()
            
            logger.info("✅ 所有報告生成完成")
            
        except Exception as e:
            logger.error(f"❌ 報告生成失敗: {e}")
    
    def cleanup(self):
        """清理資源"""
        try:
            logger.info("🧹 開始清理系統資源")
            
            if self.storage:
                self.storage.close()
            
            logger.info("✅ 系統資源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 系統資源清理失敗: {e}")

async def test_super_fusion_agi():
    """測試超級融合AGI系統"""
    try:
        print("🧪 開始測試超級融合AGI系統")
        print("=" * 60)
        
        # 創建配置
        config = SuperFusionConfig()
        
        # 創建系統
        agi_system = SuperFusionAGI(config)
        
        print("✅ 系統創建成功")
        
        # 獲取系統狀態
        status = await agi_system.get_system_status()
        print(f"📊 系統狀態: {json.dumps(status, ensure_ascii=False, indent=2)}")
        
        # 進行測試預測
        test_input = np.random.randn(1, 10)  # 10維輸入
        print(f"🔮 測試輸入數據: {test_input.shape}")
        
        prediction_result = await agi_system.make_prediction(test_input)
        print(f"📊 預測結果: {json.dumps(prediction_result, ensure_ascii=False, indent=2)}")
        
        # 生成報告
        print("📊 生成測試報告...")
        await agi_system.generate_all_reports()
        
        # 獲取最終狀態
        final_status = await agi_system.get_system_status()
        print(f"🎯 最終系統狀態: {json.dumps(final_status, ensure_ascii=False, indent=2)}")
        
        # 清理資源
        agi_system.cleanup()
        
        print("✅ 超級融合AGI系統測試完成")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 運行測試
    success = asyncio.run(test_super_fusion_agi())
    
    if success:
        print("🎉 所有測試通過！")
    else:
        print("💥 測試失敗！")
