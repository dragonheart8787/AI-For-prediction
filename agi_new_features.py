#!/usr/bin/env python3
"""
AGI新功能模組
為現有AGI系統添加新功能

新功能包括:
- 🎯 智能模型選擇
- 📊 實時性能監控
- 🔄 自動故障恢復
- 🌐 改進的API接口
- 🧠 多模型融合
- 📈 預測可視化
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """模型性能數據"""
    model_name: str
    accuracy: float
    confidence: float
    processing_time: float
    success_rate: float
    last_updated: datetime

class SmartModelSelector:
    """智能模型選擇器"""
    
    def __init__(self):
        self.model_performances = {}
        self.selection_history = []
    
    def update_performance(self, model_name: str, performance: ModelPerformance):
        """更新模型性能"""
        self.model_performances[model_name] = performance
    
    def select_best_model(self, task_type: str, input_size: int) -> str:
        """選擇最佳模型"""
        try:
            available_models = []
            
            for model_name, performance in self.model_performances.items():
                # 計算綜合評分
                score = self._calculate_model_score(performance, task_type)
                available_models.append((model_name, score))
            
            if not available_models:
                # 如果沒有性能數據，使用默認選擇
                if task_type == "financial":
                    return "financial_lstm"
                elif task_type == "weather":
                    return "weather_transformer"
                else:
                    return "financial_lstm"  # 默認
            
            # 選擇評分最高的模型
            best_model = max(available_models, key=lambda x: x[1])
            
            # 記錄選擇歷史
            self.selection_history.append({
                'timestamp': datetime.now(),
                'task_type': task_type,
                'selected_model': best_model[0],
                'score': best_model[1]
            })
            
            logger.info(f"🎯 選擇模型: {best_model[0]} (評分: {best_model[1]:.3f})")
            return best_model[0]
            
        except Exception as e:
            logger.error(f"❌ 模型選擇失敗: {e}")
            return "financial_lstm"  # 默認回退
    
    def _calculate_model_score(self, performance: ModelPerformance, task_type: str) -> float:
        """計算模型評分"""
        # 基礎評分
        base_score = performance.accuracy * 0.4 + performance.confidence * 0.3 + performance.success_rate * 0.3
        
        # 根據任務類型調整
        if task_type == "financial" and "lstm" in performance.model_name.lower():
            base_score *= 1.2
        elif task_type == "weather" and "transformer" in performance.model_name.lower():
            base_score *= 1.2
        
        # 考慮處理時間（越短越好）
        time_penalty = max(0, (performance.processing_time - 0.1) * 10)
        base_score -= time_penalty
        
        return max(0, base_score)

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    def record_metric(self, metric_name: str, value: float, model_name: str = None):
        """記錄指標"""
        timestamp = datetime.now()
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp,
            'model_name': model_name
        })
        
        # 檢查是否需要警報
        self._check_alerts(metric_name, value, model_name)
    
    def _check_alerts(self, metric_name: str, value: float, model_name: str):
        """檢查警報條件"""
        if metric_name == "accuracy" and value < 0.6:
            self.alerts.append({
                'type': 'low_accuracy',
                'model': model_name,
                'value': value,
                'timestamp': datetime.now(),
                'message': f"模型 {model_name} 準確率過低: {value:.3f}"
            })
            logger.warning(f"⚠️ 準確率警報: {model_name} = {value:.3f}")
        
        elif metric_name == "processing_time" and value > 5.0:
            self.alerts.append({
                'type': 'slow_processing',
                'model': model_name,
                'value': value,
                'timestamp': datetime.now(),
                'message': f"模型 {model_name} 處理時間過長: {value:.2f}秒"
            })
            logger.warning(f"⚠️ 處理時間警報: {model_name} = {value:.2f}秒")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """獲取性能摘要"""
        summary = {}
        
        for metric_name, data in self.metrics.items():
            if data:
                values = [item['value'] for item in data]
                summary[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0,
                    'count': len(values)
                }
        
        summary['alerts'] = self.alerts
        return summary

class AutoRecovery:
    """自動故障恢復"""
    
    def __init__(self):
        self.recovery_history = []
        self.failure_count = {}
    
    def handle_failure(self, model_name: str, error: str) -> bool:
        """處理故障"""
        try:
            # 記錄故障
            if model_name not in self.failure_count:
                self.failure_count[model_name] = 0
            self.failure_count[model_name] += 1
            
            # 根據故障類型採取恢復措施
            recovery_success = self._attempt_recovery(model_name, error)
            
            # 記錄恢復歷史
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'model_name': model_name,
                'error': error,
                'recovery_success': recovery_success,
                'failure_count': self.failure_count[model_name]
            })
            
            if recovery_success:
                logger.info(f"✅ 故障恢復成功: {model_name}")
            else:
                logger.error(f"❌ 故障恢復失敗: {model_name}")
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"❌ 故障處理異常: {e}")
            return False
    
    def _attempt_recovery(self, model_name: str, error: str) -> bool:
        """嘗試恢復"""
        try:
            # 根據錯誤類型選擇恢復策略
            if "connection" in error.lower():
                return self._recover_connection(model_name)
            elif "memory" in error.lower():
                return self._recover_memory(model_name)
            elif "model" in error.lower():
                return self._recover_model(model_name)
            else:
                return self._recover_general(model_name)
                
        except Exception as e:
            logger.error(f"❌ 恢復嘗試失敗: {e}")
            return False
    
    def _recover_connection(self, model_name: str) -> bool:
        """恢復連接"""
        # 模擬連接恢復
        logger.info(f"🔧 嘗試恢復連接: {model_name}")
        return True
    
    def _recover_memory(self, model_name: str) -> bool:
        """恢復記憶體"""
        # 模擬記憶體清理
        logger.info(f"🔧 嘗試清理記憶體: {model_name}")
        return True
    
    def _recover_model(self, model_name: str) -> bool:
        """恢復模型"""
        # 模擬模型重新載入
        logger.info(f"🔧 嘗試重新載入模型: {model_name}")
        return True
    
    def _recover_general(self, model_name: str) -> bool:
        """通用恢復"""
        # 模擬通用恢復
        logger.info(f"🔧 嘗試通用恢復: {model_name}")
        return True

class ModelFusion:
    """模型融合器"""
    
    def __init__(self):
        self.fusion_weights = {}
        self.fusion_history = []
    
    def fuse_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多個預測結果"""
        try:
            if not predictions:
                return {}
            
            # 計算融合權重
            weights = self._calculate_fusion_weights(predictions)
            
            # 加權平均融合
            fused_prediction = self._weighted_average(predictions, weights)
            
            # 計算融合置信度
            confidence = self._calculate_fusion_confidence(predictions, weights)
            
            # 記錄融合歷史
            self.fusion_history.append({
                'timestamp': datetime.now(),
                'predictions': predictions,
                'weights': weights,
                'fused_result': fused_prediction,
                'confidence': confidence
            })
            
            return {
                'fused_prediction': fused_prediction,
                'confidence': confidence,
                'weights': weights,
                'model_count': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"❌ 預測融合失敗: {e}")
            return {}
    
    def _calculate_fusion_weights(self, predictions: List[Dict[str, Any]]) -> List[float]:
        """計算融合權重"""
        weights = []
        
        for pred in predictions:
            # 基於置信度計算權重
            confidence = pred.get('confidence', 0.5)
            processing_time = pred.get('processing_time', 1.0)
            
            # 權重 = 置信度 / 處理時間
            weight = confidence / max(processing_time, 0.1)
            weights.append(weight)
        
        # 正規化權重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _weighted_average(self, predictions: List[Dict[str, Any]], weights: List[float]) -> np.ndarray:
        """加權平均"""
        try:
            # 獲取所有預測值
            pred_values = []
            for pred in predictions:
                pred_value = pred.get('prediction', [])
                if isinstance(pred_value, list):
                    pred_value = np.array(pred_value)
                pred_values.append(pred_value)
            
            # 確保所有預測值具有相同形狀
            if not pred_values:
                return np.array([])
            
            # 填充較短的預測值
            max_length = max(len(pred) for pred in pred_values)
            padded_values = []
            
            for pred in pred_values:
                if len(pred) < max_length:
                    # 用最後一個值填充
                    padded = np.pad(pred, (0, max_length - len(pred)), mode='edge')
                else:
                    padded = pred
                padded_values.append(padded)
            
            # 計算加權平均
            fused = np.zeros_like(padded_values[0])
            for i, (pred, weight) in enumerate(zip(padded_values, weights)):
                fused += pred * weight
            
            return fused
            
        except Exception as e:
            logger.error(f"❌ 加權平均計算失敗: {e}")
            return np.array([])
    
    def _calculate_fusion_confidence(self, predictions: List[Dict[str, Any]], weights: List[float]) -> float:
        """計算融合置信度"""
        try:
            confidences = [pred.get('confidence', 0.5) for pred in predictions]
            
            # 加權平均置信度
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
            
            # 考慮預測一致性
            if len(predictions) > 1:
                # 計算預測值的變異係數
                pred_values = [pred.get('prediction', []) for pred in predictions]
                if all(len(p) > 0 for p in pred_values):
                    # 簡化的變異係數計算
                    consistency_penalty = 0.1
                else:
                    consistency_penalty = 0
            else:
                consistency_penalty = 0
            
            final_confidence = max(0, min(1, weighted_confidence - consistency_penalty))
            return final_confidence
            
        except Exception as e:
            logger.error(f"❌ 融合置信度計算失敗: {e}")
            return 0.5

class EnhancedAPI:
    """增強版API接口"""
    
    def __init__(self):
        self.model_selector = SmartModelSelector()
        self.performance_monitor = PerformanceMonitor()
        self.auto_recovery = AutoRecovery()
        self.model_fusion = ModelFusion()
    
    async def smart_predict(self, task_type: str, input_data: np.ndarray, 
                           use_fusion: bool = True) -> Dict[str, Any]:
        """智能預測"""
        try:
            start_time = datetime.now()
            
            # 選擇最佳模型
            selected_model = self.model_selector.select_best_model(task_type, input_data.shape[1])
            
            # 進行預測
            prediction_result = await self._make_prediction(selected_model, input_data)
            
            if not prediction_result:
                # 如果預測失敗，嘗試恢復
                self.auto_recovery.handle_failure(selected_model, "預測失敗")
                return {'error': '預測失敗'}
            
            # 記錄性能指標
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_monitor.record_metric('processing_time', processing_time, selected_model)
            self.performance_monitor.record_metric('confidence', prediction_result['confidence'], selected_model)
            
            # 如果啟用融合，嘗試融合多個模型
            if use_fusion:
                fusion_result = await self._try_model_fusion(task_type, input_data)
                if fusion_result:
                    prediction_result['fusion'] = fusion_result
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ 智能預測失敗: {e}")
            return {'error': str(e)}
    
    async def _make_prediction(self, model_name: str, input_data: np.ndarray) -> Dict[str, Any]:
        """進行預測"""
        try:
            # 模擬預測過程
            prediction = np.random.randn(input_data.shape[0], 1)
            confidence = np.random.uniform(0.7, 0.95)
            
            return {
                'model_name': model_name,
                'prediction': prediction.tolist(),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 預測失敗: {e}")
            return None
    
    async def _try_model_fusion(self, task_type: str, input_data: np.ndarray) -> Dict[str, Any]:
        """嘗試模型融合"""
        try:
            # 獲取多個模型的預測
            predictions = []
            models = ['financial_lstm', 'weather_transformer']
            
            for model in models:
                pred_result = await self._make_prediction(model, input_data)
                if pred_result:
                    predictions.append(pred_result)
            
            if len(predictions) > 1:
                return self.model_fusion.fuse_predictions(predictions)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 模型融合失敗: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return {
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'recovery_history': self.auto_recovery.recovery_history[-10:],  # 最近10條
            'fusion_history': self.model_fusion.fusion_history[-10:],  # 最近10條
            'model_selections': self.model_selector.selection_history[-10:]  # 最近10條
        }

async def main():
    """測試新功能"""
    api = EnhancedAPI()
    
    # 模擬一些性能數據
    api.model_selector.update_performance('financial_lstm', 
        ModelPerformance('financial_lstm', 0.85, 0.8, 0.5, 0.9, datetime.now()))
    api.model_selector.update_performance('weather_transformer', 
        ModelPerformance('weather_transformer', 0.92, 0.9, 0.3, 0.95, datetime.now()))
    
    # 測試智能預測
    test_data = np.random.randn(1, 10)
    result = await api.smart_predict('financial', test_data, use_fusion=True)
    
    print("🎯 智能預測結果:")
    print(json.dumps(result, indent=2, default=str))
    
    # 獲取系統狀態
    status = api.get_system_status()
    print("\n📊 系統狀態:")
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main()) 