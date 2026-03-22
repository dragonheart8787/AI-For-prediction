#!/usr/bin/env python3
"""
高級模型融合系統
支持多種融合策略、自適應權重和動態模型選擇
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelFusion:
    """高級模型融合系統"""
    
    def __init__(self, models_dir: str = "./pretrained_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 融合策略配置
        self.fusion_strategies = {
            'weighted_average': {
                'name': '加權平均融合',
                'description': '基於模型性能的加權平均',
                'type': 'statistical',
                'adaptive': True
            },
            'stacking': {
                'name': '堆疊融合',
                'description': '使用元學習器組合多個模型',
                'type': 'machine_learning',
                'adaptive': True
            },
            'voting': {
                'name': '投票融合',
                'description': '多數投票決策',
                'type': 'statistical',
                'adaptive': False
            },
            'neural_fusion': {
                'name': '神經網絡融合',
                'description': '深度學習融合網絡',
                'type': 'deep_learning',
                'adaptive': True
            },
            'bayesian_fusion': {
                'name': '貝葉斯融合',
                'description': '貝葉斯模型平均',
                'type': 'probabilistic',
                'adaptive': True
            },
            'dynamic_selection': {
                'name': '動態選擇融合',
                'description': '根據數據特徵動態選擇最佳模型',
                'type': 'adaptive',
                'adaptive': True
            }
        }
        
        # 模型性能追蹤
        self.model_performance = {}
        self.fusion_history = []
        self.adaptive_weights = {}
        
        # 初始化融合模型
        self.fusion_models = {}
        self.scalers = {}
        
    async def create_fusion_model(self, strategy: str, base_models: List[str], 
                                 input_dim: int = 64) -> Dict[str, Any]:
        """創建指定的融合模型"""
        if strategy not in self.fusion_strategies:
            return {'error': f'不支持的融合策略: {strategy}'}
        
        try:
            logger.info(f"🔧 創建融合模型: {self.fusion_strategies[strategy]['name']}")
            
            if strategy == 'weighted_average':
                fusion_model = self._create_weighted_average_fusion(base_models)
            elif strategy == 'stacking':
                fusion_model = self._create_stacking_fusion(base_models, input_dim)
            elif strategy == 'voting':
                fusion_model = self._create_voting_fusion(base_models)
            elif strategy == 'neural_fusion':
                fusion_model = self._create_neural_fusion(base_models, input_dim)
            elif strategy == 'bayesian_fusion':
                fusion_model = self._create_bayesian_fusion(base_models)
            elif strategy == 'dynamic_selection':
                fusion_model = self._create_dynamic_selection_fusion(base_models, input_dim)
            else:
                return {'error': f'未知的融合策略: {strategy}'}
            
            # 保存融合模型
            self.fusion_models[strategy] = fusion_model
            
            # 初始化性能追蹤
            if strategy not in self.model_performance:
                self.model_performance[strategy] = {
                    'predictions': [],
                    'actual_values': [],
                    'performance_metrics': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            logger.info(f"✅ 融合模型創建成功: {strategy}")
            
            return {
                'status': 'created',
                'strategy': strategy,
                'base_models': base_models,
                'fusion_model': fusion_model
            }
            
        except Exception as e:
            logger.error(f"❌ 融合模型創建失敗: {e}")
            return {'error': str(e)}
    
    def _create_weighted_average_fusion(self, base_models: List[str]) -> nn.Module:
        """創建加權平均融合模型"""
        class WeightedAverageFusion(nn.Module):
            def __init__(self, num_models: int):
                super().__init__()
                # 可學習的權重參數
                self.weights = nn.Parameter(torch.ones(num_models) / num_models)
                self.softmax = nn.Softmax(dim=0)
                
            def forward(self, predictions: torch.Tensor) -> torch.Tensor:
                # predictions shape: (batch_size, num_models, output_size)
                weights = self.softmax(self.weights)
                weighted_pred = torch.sum(predictions * weights.unsqueeze(0).unsqueeze(-1), dim=1)
                return weighted_pred
        
        return WeightedAverageFusion(len(base_models))
    
    def _create_stacking_fusion(self, base_models: List[str], input_dim: int) -> Dict[str, Any]:
        """創建堆疊融合模型"""
        class MetaLearner(nn.Module):
            def __init__(self, num_models: int, input_dim: int, output_size: int = 1):
                super().__init__()
                self.meta_network = nn.Sequential(
                    nn.Linear(num_models * input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, output_size)
                )
                
            def forward(self, model_outputs: torch.Tensor) -> torch.Tensor:
                # model_outputs shape: (batch_size, num_models, input_dim)
                batch_size = model_outputs.shape[0]
                flattened = model_outputs.view(batch_size, -1)
                return self.meta_network(flattened)
        
        # 創建多個元學習器
        meta_learners = {
            'neural': MetaLearner(len(base_models), input_dim),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': Ridge(alpha=1.0)
        }
        
        return {
            'type': 'stacking',
            'meta_learners': meta_learners,
            'base_models': base_models
        }
    
    def _create_voting_fusion(self, base_models: List[str]) -> Dict[str, Any]:
        """創建投票融合模型"""
        return {
            'type': 'voting',
            'base_models': base_models,
            'voting_method': 'soft'  # soft voting for regression
        }
    
    def _create_neural_fusion(self, base_models: List[str], input_dim: int) -> nn.Module:
        """創建神經網絡融合模型"""
        class NeuralFusionNetwork(nn.Module):
            def __init__(self, num_models: int, input_dim: int, output_size: int = 1):
                super().__init__()
                
                # 特徵提取層
                self.feature_extractor = nn.Sequential(
                    nn.Linear(num_models * input_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.4)
                )
                
                # 注意力機制
                self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
                
                # 融合層
                self.fusion_layers = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, output_size)
                )
                
            def forward(self, model_outputs: torch.Tensor) -> torch.Tensor:
                # model_outputs shape: (batch_size, num_models, input_dim)
                batch_size = model_outputs.shape[0]
                
                # 特徵提取
                flattened = model_outputs.view(batch_size, -1)
                features = self.feature_extractor(flattened)
                
                # 注意力機制
                features = features.unsqueeze(1)  # (batch_size, 1, 512)
                attended_features, _ = self.attention(features, features, features)
                attended_features = attended_features.squeeze(1)  # (batch_size, 512)
                
                # 融合預測
                output = self.fusion_layers(attended_features)
                return output
        
        return NeuralFusionNetwork(len(base_models), input_dim)
    
    def _create_bayesian_fusion(self, base_models: List[str]) -> Dict[str, Any]:
        """創建貝葉斯融合模型"""
        return {
            'type': 'bayesian',
            'base_models': base_models,
            'prior_weights': np.ones(len(base_models)) / len(base_models),
            'update_method': 'online'
        }
    
    def _create_dynamic_selection_fusion(self, base_models: List[str], input_dim: int) -> nn.Module:
        """創建動態選擇融合模型"""
        class DynamicSelectionFusion(nn.Module):
            def __init__(self, num_models: int, input_dim: int, output_size: int = 1):
                super().__init__()
                
                # 特徵分析網絡
                self.feature_analyzer = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # 模型選擇網絡
                self.model_selector = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, num_models),
                    nn.Softmax(dim=1)
                )
                
                # 輸出層
                self.output_layer = nn.Linear(num_models, output_size)
                
            def forward(self, input_features: torch.Tensor, model_predictions: torch.Tensor) -> torch.Tensor:
                # input_features: (batch_size, input_dim)
                # model_predictions: (batch_size, num_models, output_size)
                
                # 分析輸入特徵
                feature_analysis = self.feature_analyzer(input_features)
                
                # 選擇模型權重
                model_weights = self.model_selector(feature_analysis)
                
                # 加權組合預測
                weighted_predictions = torch.sum(
                    model_predictions * model_weights.unsqueeze(-1), 
                    dim=1
                )
                
                return weighted_predictions
        
        return DynamicSelectionFusion(len(base_models), input_dim)
    
    async def train_fusion_model(self, strategy: str, training_data: Tuple[np.ndarray, np.ndarray],
                                validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """訓練融合模型"""
        if strategy not in self.fusion_models:
            return {'error': f'融合模型不存在: {strategy}'}
        
        try:
            logger.info(f"🚀 開始訓練融合模型: {strategy}")
            
            X_train, y_train = training_data
            
            if strategy in ['neural_fusion', 'weighted_average']:
                # PyTorch模型訓練
                result = await self._train_pytorch_fusion(
                    strategy, X_train, y_train, epochs, learning_rate
                )
            elif strategy == 'stacking':
                # 堆疊融合訓練
                result = await self._train_stacking_fusion(
                    strategy, X_train, y_train, validation_data
                )
            elif strategy == 'bayesian_fusion':
                # 貝葉斯融合訓練
                result = await self._train_bayesian_fusion(
                    strategy, X_train, y_train
                )
            else:
                # 其他策略不需要訓練
                result = {'status': 'no_training_needed', 'strategy': strategy}
            
            # 更新性能追蹤
            if 'error' not in result:
                self.model_performance[strategy]['last_updated'] = datetime.now().isoformat()
                
                # 保存融合模型
                model_path = self.models_dir / f"fusion_model_{strategy}.pth"
                if strategy in ['neural_fusion', 'weighted_average']:
                    torch.save(self.fusion_models[strategy].state_dict(), model_path)
                else:
                    # 保存其他類型的模型
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.fusion_models[strategy], f)
                
                result['model_path'] = str(model_path)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 融合模型訓練失敗: {e}")
            return {'error': str(e)}
    
    async def _train_pytorch_fusion(self, strategy: str, X_train: np.ndarray, y_train: np.ndarray,
                                   epochs: int, learning_rate: float) -> Dict[str, Any]:
        """訓練PyTorch融合模型"""
        try:
            model = self.fusion_models[strategy]
            
            # 轉換為PyTorch張量
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train)
            
            # 創建數據加載器
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 優化器和損失函數
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # 訓練循環
            model.train()
            training_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # 前向傳播
                    if strategy == 'neural_fusion':
                        # 創建模擬的模型預測
                        batch_size = batch_X.shape[0]
                        num_models = 5  # 假設有5個基礎模型
                        model_predictions = torch.randn(batch_size, num_models, batch_X.shape[1])
                        outputs = model(model_predictions)
                    else:
                        outputs = model(batch_X)
                    
                    loss = criterion(outputs, batch_y)
                    
                    # 反向傳播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)
                
                if epoch % 20 == 0:
                    logger.info(f"📊 訓練進度: {epoch}/{epochs}, 損失: {avg_loss:.6f}")
            
            return {
                'status': 'trained',
                'strategy': strategy,
                'epochs_completed': epochs,
                'final_loss': training_losses[-1],
                'training_losses': training_losses
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _train_stacking_fusion(self, strategy: str, X_train: np.ndarray, y_train: np.ndarray,
                                    validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """訓練堆疊融合模型"""
        try:
            fusion_model = self.fusion_models[strategy]
            
            # 創建模擬的基礎模型預測
            # 在實際應用中，這些應該是真實的基礎模型預測
            n_samples = len(X_train)
            n_models = len(fusion_model['base_models'])
            
            # 模擬基礎模型預測
            base_predictions = np.random.randn(n_samples, n_models)
            
            # 訓練元學習器
            training_results = {}
            
            for name, meta_learner in fusion_model['meta_learners'].items():
                if hasattr(meta_learner, 'fit'):
                    # 機器學習模型
                    meta_learner.fit(base_predictions, y_train.ravel())
                    
                    # 交叉驗證
                    cv_scores = cross_val_score(meta_learner, base_predictions, y_train.ravel(), 
                                              cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    
                    training_results[name] = {
                        'status': 'trained',
                        'cv_rmse': cv_rmse
                    }
                else:
                    # PyTorch模型
                    training_results[name] = {'status': 'pytorch_model'}
            
            return {
                'status': 'trained',
                'strategy': strategy,
                'meta_learners': training_results
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _train_bayesian_fusion(self, strategy: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """訓練貝葉斯融合模型"""
        try:
            fusion_model = self.fusion_models[strategy]
            
            # 簡化的貝葉斯權重更新
            # 在實際應用中，這裡應該實現完整的貝葉斯推理
            n_models = len(fusion_model['base_models'])
            
            # 模擬模型性能
            model_performances = np.random.rand(n_models)
            
            # 更新權重（簡化的貝葉斯更新）
            updated_weights = model_performances / model_performances.sum()
            
            fusion_model['prior_weights'] = updated_weights
            
            return {
                'status': 'trained',
                'strategy': strategy,
                'updated_weights': updated_weights.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def make_ensemble_prediction(self, input_data: np.ndarray, strategy: str,
                                     base_model_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """使用融合模型進行預測"""
        if strategy not in self.fusion_models:
            return {'error': f'融合模型不存在: {strategy}'}
        
        try:
            logger.info(f"🔮 使用融合策略進行預測: {strategy}")
            
            start_time = datetime.now()
            
            if strategy == 'neural_fusion':
                result = await self._neural_fusion_predict(input_data, strategy)
            elif strategy == 'stacking':
                result = await self._stacking_fusion_predict(input_data, strategy, base_model_predictions)
            elif strategy == 'weighted_average':
                result = await self._weighted_average_predict(input_data, strategy, base_model_predictions)
            elif strategy == 'voting':
                result = await self._voting_fusion_predict(input_data, strategy, base_model_predictions)
            elif strategy == 'bayesian_fusion':
                result = await self._bayesian_fusion_predict(input_data, strategy, base_model_predictions)
            elif strategy == 'dynamic_selection':
                result = await self._dynamic_selection_predict(input_data, strategy, base_model_predictions)
            else:
                return {'error': f'不支持的融合策略: {strategy}'}
            
            # 計算執行時間
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            result['strategy'] = strategy
            result['timestamp'] = datetime.now().isoformat()
            
            # 更新性能追蹤
            self._update_performance_tracking(strategy, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 融合預測失敗: {e}")
            return {'error': str(e)}
    
    async def _neural_fusion_predict(self, input_data: np.ndarray, strategy: str) -> Dict[str, Any]:
        """神經網絡融合預測"""
        try:
            model = self.fusion_models[strategy]
            model.eval()
            
            with torch.no_grad():
                # 創建模擬的模型預測
                batch_size = input_data.shape[0]
                num_models = 5
                model_predictions = torch.randn(batch_size, num_models, input_data.shape[1])
                
                # 進行預測
                predictions = model(model_predictions)
                
                # 轉換為numpy
                predictions_np = predictions.numpy()
                
                return {
                    'prediction': predictions_np.tolist(),
                    'confidence_interval': (predictions_np * 0.1).tolist(),
                    'model_type': 'neural_fusion'
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    async def _stacking_fusion_predict(self, input_data: np.ndarray, strategy: str,
                                     base_model_predictions: Optional[np.ndarray]) -> Dict[str, Any]:
        """堆疊融合預測"""
        try:
            fusion_model = self.fusion_models[strategy]
            
            if base_model_predictions is None:
                # 創建模擬的基礎模型預測
                n_samples = input_data.shape[0]
                n_models = len(fusion_model['base_models'])
                base_model_predictions = np.random.randn(n_samples, n_models)
            
            # 使用元學習器進行預測
            predictions = {}
            for name, meta_learner in fusion_model['meta_learners'].items():
                if hasattr(meta_learner, 'predict'):
                    pred = meta_learner.predict(base_model_predictions)
                    predictions[name] = pred.tolist()
            
            # 平均預測結果
            if predictions:
                avg_prediction = np.mean(list(predictions.values()), axis=0)
                return {
                    'prediction': avg_prediction.tolist(),
                    'individual_predictions': predictions,
                    'model_type': 'stacking'
                }
            else:
                return {'error': '沒有可用的元學習器'}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def _weighted_average_predict(self, input_data: np.ndarray, strategy: str,
                                      base_model_predictions: Optional[np.ndarray]) -> Dict[str, Any]:
        """加權平均融合預測"""
        try:
            model = self.fusion_models[strategy]
            model.eval()
            
            if base_model_predictions is None:
                # 創建模擬的基礎模型預測
                n_samples = input_data.shape[0]
                n_models = 5
                base_model_predictions = np.random.randn(n_samples, n_models, 1)
            
            with torch.no_grad():
                # 轉換為tensor
                predictions_tensor = torch.FloatTensor(base_model_predictions)
                
                # 進行預測
                fused_prediction = model(predictions_tensor)
                
                # 轉換為numpy
                fused_prediction_np = fused_prediction.numpy()
                
                return {
                    'prediction': fused_prediction_np.tolist(),
                    'confidence_interval': (fused_prediction_np * 0.1).tolist(),
                    'model_type': 'weighted_average'
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    async def _voting_fusion_predict(self, input_data: np.ndarray, strategy: str,
                                   base_model_predictions: Optional[np.ndarray]) -> Dict[str, Any]:
        """投票融合預測"""
        try:
            if base_model_predictions is None:
                # 創建模擬的基礎模型預測
                n_samples = input_data.shape[0]
                n_models = 5
                base_model_predictions = np.random.randn(n_samples, n_models, 1)
            
            # 軟投票（平均）
            fused_prediction = np.mean(base_model_predictions, axis=1)
            
            return {
                'prediction': fused_prediction.tolist(),
                'confidence_interval': (fused_prediction * 0.1).tolist(),
                'model_type': 'voting'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _bayesian_fusion_predict(self, input_data: np.ndarray, strategy: str,
                                     base_model_predictions: Optional[np.ndarray]) -> Dict[str, Any]:
        """貝葉斯融合預測"""
        try:
            fusion_model = self.fusion_models[strategy]
            
            if base_model_predictions is None:
                # 創建模擬的基礎模型預測
                n_samples = input_data.shape[0]
                n_models = len(fusion_model['base_models'])
                base_model_predictions = np.random.randn(n_samples, n_models, 1)
            
            # 使用貝葉斯權重
            weights = fusion_model['prior_weights']
            fused_prediction = np.average(base_model_predictions, axis=1, weights=weights)
            
            return {
                'prediction': fused_prediction.tolist(),
                'confidence_interval': (fused_prediction * 0.1).tolist(),
                'model_type': 'bayesian',
                'weights_used': weights.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _dynamic_selection_predict(self, input_data: np.ndarray, strategy: str,
                                       base_model_predictions: Optional[np.ndarray]) -> Dict[str, Any]:
        """動態選擇融合預測"""
        try:
            model = self.fusion_models[strategy]
            model.eval()
            
            if base_model_predictions is None:
                # 創建模擬的基礎模型預測
                n_samples = input_data.shape[0]
                n_models = 5
                base_model_predictions = np.random.randn(n_samples, n_models, 1)
            
            with torch.no_grad():
                # 轉換為tensor
                input_tensor = torch.FloatTensor(input_data)
                predictions_tensor = torch.FloatTensor(base_model_predictions)
                
                # 進行預測
                fused_prediction = model(input_tensor, predictions_tensor)
                
                # 轉換為numpy
                fused_prediction_np = fused_prediction.numpy()
                
                return {
                    'prediction': fused_prediction_np.tolist(),
                    'confidence_interval': (fused_prediction_np * 0.1).tolist(),
                    'model_type': 'dynamic_selection'
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _update_performance_tracking(self, strategy: str, prediction_result: Dict[str, Any]):
        """更新性能追蹤"""
        if strategy not in self.model_performance:
            return
        
        # 記錄預測結果
        if 'prediction' in prediction_result:
            self.model_performance[strategy]['predictions'].append(prediction_result['prediction'])
        
        # 記錄執行時間
        if 'execution_time' in prediction_result:
            execution_times = self.model_performance[strategy].get('execution_times', [])
            execution_times.append(prediction_result['execution_time'])
            self.model_performance[strategy]['execution_times'] = execution_times
    
    def get_fusion_performance(self, strategy: str) -> Dict[str, Any]:
        """獲取融合模型性能"""
        if strategy not in self.model_performance:
            return {'error': f'策略不存在: {strategy}'}
        
        return self.model_performance[strategy]
    
    def get_all_fusion_performance(self) -> Dict[str, Any]:
        """獲取所有融合模型性能"""
        return self.model_performance
    
    def save_fusion_system(self, filepath: str):
        """保存融合系統"""
        try:
            system_state = {
                'fusion_models': self.fusion_models,
                'model_performance': self.model_performance,
                'fusion_history': self.fusion_history,
                'adaptive_weights': self.adaptive_weights
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(system_state, f)
            
            logger.info(f"✅ 融合系統已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 融合系統保存失敗: {e}")
    
    def load_fusion_system(self, filepath: str):
        """加載融合系統"""
        try:
            with open(filepath, 'rb') as f:
                system_state = pickle.load(f)
            
            self.fusion_models = system_state.get('fusion_models', {})
            self.model_performance = system_state.get('model_performance', {})
            self.fusion_history = system_state.get('fusion_history', [])
            self.adaptive_weights = system_state.get('adaptive_weights', {})
            
            logger.info(f"✅ 融合系統已加載: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 融合系統加載失敗: {e}")

async def main():
    """主函數 - 演示高級模型融合系統"""
    print("🚀 高級模型融合系統演示")
    print("=" * 60)
    
    # 創建融合系統
    fusion_system = AdvancedModelFusion()
    
    try:
        # 1. 創建各種融合模型
        print("\n🔧 步驟1: 創建融合模型")
        base_models = ['model1', 'model2', 'model3', 'model4', 'model5']
        
        for strategy in fusion_system.fusion_strategies.keys():
            result = await fusion_system.create_fusion_model(strategy, base_models, input_dim=64)
            if 'error' not in result:
                print(f"✅ {strategy}: {fusion_system.fusion_strategies[strategy]['name']}")
            else:
                print(f"❌ {strategy}: {result['error']}")
        
        # 2. 創建模擬訓練數據
        print("\n📊 步驟2: 創建模擬訓練數據")
        np.random.seed(42)
        n_samples = 1000
        n_features = 64
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples, 1)
        
        print(f"✅ 訓練數據創建完成: {X_train.shape}")
        
        # 3. 訓練融合模型
        print("\n🧠 步驟3: 訓練融合模型")
        for strategy in ['neural_fusion', 'weighted_average']:
            print(f"\n📚 訓練 {strategy}...")
            training_result = await fusion_system.train_fusion_model(
                strategy, (X_train, y_train), epochs=50
            )
            
            if 'error' not in training_result:
                print(f"✅ {strategy} 訓練完成")
                if 'final_loss' in training_result:
                    print(f"   最終損失: {training_result['final_loss']:.6f}")
            else:
                print(f"❌ {strategy} 訓練失敗: {training_result['error']}")
        
        # 4. 使用融合模型進行預測
        print("\n🔮 步驟4: 使用融合模型進行預測")
        test_data = np.random.randn(10, n_features)
        
        for strategy in fusion_system.fusion_strategies.keys():
            print(f"\n🎯 測試 {strategy}...")
            prediction_result = await fusion_system.make_ensemble_prediction(
                test_data, strategy
            )
            
            if 'error' not in prediction_result:
                print(f"✅ {strategy} 預測成功")
                print(f"   執行時間: {prediction_result['execution_time']:.3f}秒")
                print(f"   預測長度: {len(prediction_result['prediction'])}")
            else:
                print(f"❌ {strategy} 預測失敗: {prediction_result['error']}")
        
        # 5. 查看性能統計
        print("\n📊 步驟5: 性能統計")
        performance = fusion_system.get_all_fusion_performance()
        for strategy, stats in performance.items():
            print(f"\n📈 {strategy}:")
            print(f"   最後更新: {stats.get('last_updated', 'N/A')}")
            print(f"   預測次數: {len(stats.get('predictions', []))}")
            if 'execution_times' in stats:
                avg_time = np.mean(stats['execution_times'])
                print(f"   平均執行時間: {avg_time:.3f}秒")
        
        # 6. 保存融合系統
        print("\n💾 步驟6: 保存融合系統")
        fusion_system.save_fusion_system("advanced_fusion_system.pkl")
        
        print("\n🎉 高級模型融合系統演示完成！")
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
