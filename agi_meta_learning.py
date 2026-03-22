#!/usr/bin/env python3
"""
AGI Meta-Learning and Adaptive System
元學習與自適應AGI系統

新增功能:
- 🧠 自動神經架構搜索 (NAS)
- 🔄 多任務學習和知識遷移
- 📚 元學習和快速適應
- 🎯 自動超參數優化
- 🔬 模型解釋性和可解釋性
- 🌐 分散式訓練和聯邦學習
- 📊 自動化機器學習 (AutoML)
- 🚀 強化學習和策略優化
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
import pickle
import joblib
from pathlib import Path
import hashlib
import time
import sqlite3
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import combinations
import networkx as nx

warnings.filterwarnings('ignore')

# 深度學習框架導入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
    LIGHTGBM_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    print("XGBoost/LightGBM not available. Install with: pip install xgboost lightgbm")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_meta_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """元學習配置"""
    task_type: str
    adaptation_steps: int = 5
    meta_lr: float = 0.01
    inner_lr: float = 0.001
    num_tasks: int = 10
    shots_per_task: int = 5
    query_shots: int = 15
    meta_batch_size: int = 4
    adaptation_epochs: int = 3

@dataclass
class NASConfig:
    """神經架構搜索配置"""
    search_space: Dict[str, List]
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    early_stopping_patience: int = 10

@dataclass
class AutoMLConfig:
    """自動機器學習配置"""
    time_budget: int = 3600  # 秒
    max_models: int = 100
    ensemble_size: int = 5
    feature_selection: bool = True
    hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5

class NeuralArchitectureSearch:
    """神經架構搜索"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.population = []
        self.best_architectures = []
        self.fitness_history = []
        
    def create_search_space(self) -> Dict[str, List]:
        """創建搜索空間"""
        return {
            'num_layers': [1, 2, 3, 4, 5],
            'hidden_sizes': [32, 64, 128, 256, 512],
            'activation_functions': ['relu', 'tanh', 'sigmoid', 'leaky_relu'],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.5],
            'learning_rates': [0.0001, 0.001, 0.01, 0.1],
            'optimizers': ['adam', 'sgd', 'rmsprop'],
            'batch_sizes': [16, 32, 64, 128]
        }
    
    def generate_random_architecture(self) -> Dict[str, Any]:
        """生成隨機架構"""
        search_space = self.create_search_space()
        architecture = {}
        
        for param, values in search_space.items():
            if param == 'hidden_sizes':
                num_layers = random.choice(search_space['num_layers'])
                architecture['num_layers'] = num_layers
                architecture['hidden_sizes'] = [random.choice(values) for _ in range(num_layers)]
            else:
                architecture[param] = random.choice(values)
        
        return architecture
    
    def mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """突變架構"""
        mutated = architecture.copy()
        search_space = self.create_search_space()
        
        # 隨機選擇一個參數進行突變
        param = random.choice(list(search_space.keys()))
        if param == 'hidden_sizes':
            num_layers = mutated['num_layers']
            mutated['hidden_sizes'] = [random.choice(search_space['hidden_sizes']) for _ in range(num_layers)]
        else:
            mutated[param] = random.choice(search_space[param])
        
        return mutated
    
    def crossover_architectures(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉架構"""
        child = {}
        search_space = self.create_search_space()
        
        for param in search_space.keys():
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        
        # 確保一致性
        if 'num_layers' in child and 'hidden_sizes' in child:
            num_layers = child['num_layers']
            if len(child['hidden_sizes']) != num_layers:
                child['hidden_sizes'] = child['hidden_sizes'][:num_layers]
                while len(child['hidden_sizes']) < num_layers:
                    child['hidden_sizes'].append(random.choice(search_space['hidden_sizes']))
        
        return child
    
    def create_model_from_architecture(self, architecture: Dict[str, Any], input_size: int, output_size: int) -> nn.Module:
        """從架構創建模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NAS")
        
        class NASModel(nn.Module):
            def __init__(self, input_size, output_size, architecture):
                super(NASModel, self).__init__()
                self.layers = nn.ModuleList()
                
                # 輸入層
                prev_size = input_size
                
                # 隱藏層
                for i in range(architecture['num_layers']):
                    hidden_size = architecture['hidden_sizes'][i]
                    self.layers.append(nn.Linear(prev_size, hidden_size))
                    
                    # 激活函數
                    if architecture['activation_functions'] == 'relu':
                        self.layers.append(nn.ReLU())
                    elif architecture['activation_functions'] == 'tanh':
                        self.layers.append(nn.Tanh())
                    elif architecture['activation_functions'] == 'sigmoid':
                        self.layers.append(nn.Sigmoid())
                    elif architecture['activation_functions'] == 'leaky_relu':
                        self.layers.append(nn.LeakyReLU())
                    
                    # Dropout
                    if architecture['dropout_rates'] > 0:
                        self.layers.append(nn.Dropout(architecture['dropout_rates']))
                    
                    prev_size = hidden_size
                
                # 輸出層
                self.layers.append(nn.Linear(prev_size, output_size))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return NASModel(input_size, output_size, architecture)
    
    async def evaluate_architecture(self, architecture: Dict[str, Any], X_train: np.ndarray, 
                                 y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """評估架構性能"""
        try:
            model = self.create_model_from_architecture(architecture, X_train.shape[1], 1)
            
            # 訓練配置
            optimizer_name = architecture.get('optimizers', 'adam')
            lr = architecture.get('learning_rates', 0.001)
            batch_size = architecture.get('batch_sizes', 32)
            
            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
            
            criterion = nn.MSELoss()
            
            # 快速訓練評估
            model.train()
            for epoch in range(5):  # 快速評估
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i+batch_size])
                    batch_y = torch.FloatTensor(y_train[i:i+batch_size])
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 驗證
            model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val)
                val_y = torch.FloatTensor(y_val)
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs.squeeze(), val_y).item()
            
            return -val_loss  # 返回負損失作為適應度（最大化適應度）
            
        except Exception as e:
            logger.error(f"架構評估失敗: {e}")
            return float('-inf')
    
    async def search_optimal_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """搜索最優架構"""
        logger.info("開始神經架構搜索...")
        
        # 初始化種群
        self.population = [self.generate_random_architecture() for _ in range(self.config.population_size)]
        
        best_fitness = float('-inf')
        best_architecture = None
        generations_without_improvement = 0
        
        for generation in range(self.config.generations):
            # 評估種群
            fitness_scores = []
            for architecture in self.population:
                fitness = await self.evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
                fitness_scores.append(fitness)
            
            # 更新最佳架構
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_architecture = self.population[max_fitness_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            self.fitness_history.append(best_fitness)
            
            # 早停
            if generations_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"早停於第 {generation} 代")
                break
            
            # 選擇、交叉、突變
            new_population = []
            
            # 精英保留
            elite_size = max(1, self.config.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # 生成新個體
            while len(new_population) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    # 交叉
                    parent1_idx = self._tournament_selection(fitness_scores)
                    parent2_idx = self._tournament_selection(fitness_scores)
                    child = self.crossover_architectures(self.population[parent1_idx], self.population[parent2_idx])
                else:
                    # 突變
                    parent_idx = self._tournament_selection(fitness_scores)
                    child = self.mutate_architecture(self.population[parent_idx])
                
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 10 == 0:
                logger.info(f"第 {generation} 代 - 最佳適應度: {best_fitness:.6f}")
        
        logger.info(f"NAS 完成 - 最佳架構適應度: {best_fitness:.6f}")
        return best_architecture
    
    def _tournament_selection(self, fitness_scores: List[float]) -> int:
        """錦標賽選擇"""
        tournament_indices = random.sample(range(len(fitness_scores)), self.config.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx

class MetaLearner:
    """元學習器"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.meta_learner = None
        self.task_embeddings = {}
        self.knowledge_base = {}
        
    def create_meta_learner(self, input_size: int, output_size: int) -> nn.Module:
        """創建元學習器"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for meta-learning")
        
        class MetaLearner(nn.Module):
            def __init__(self, input_size, output_size):
                super(MetaLearner, self).__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                
                self.task_encoder = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16)
                )
                
                self.predictor = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_size)
                )
            
            def forward(self, x, task_embedding=None):
                features = self.feature_extractor(x)
                if task_embedding is not None:
                    features = features + task_embedding
                task_encoded = self.task_encoder(features)
                output = self.predictor(task_encoded)
                return output
        
        return MetaLearner(input_size, output_size)
    
    async def meta_train(self, tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """元訓練"""
        logger.info("開始元訓練...")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for meta-training")
        
        # 創建元學習器
        input_size = tasks[0][0].shape[1]
        output_size = 1
        self.meta_learner = self.create_meta_learner(input_size, output_size)
        
        meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.config.meta_lr)
        criterion = nn.MSELoss()
        
        meta_losses = []
        
        for meta_epoch in range(self.config.adaptation_steps):
            meta_loss = 0.0
            
            # 隨機選擇任務批次
            task_batch = random.sample(tasks, min(self.config.meta_batch_size, len(tasks)))
            
            for task_idx, (X_support, y_support, X_query, y_query) in enumerate(task_batch):
                # 創建任務特定的快速適應模型
                task_model = self.create_meta_learner(input_size, output_size)
                task_optimizer = optim.Adam(task_model.parameters(), lr=self.config.inner_lr)
                
                # 內循環：快速適應
                for _ in range(self.config.adaptation_epochs):
                    task_model.train()
                    support_X = torch.FloatTensor(X_support)
                    support_y = torch.FloatTensor(y_support)
                    
                    task_optimizer.zero_grad()
                    outputs = task_model(support_X)
                    loss = criterion(outputs.squeeze(), support_y)
                    loss.backward()
                    task_optimizer.step()
                
                # 外循環：元更新
                query_X = torch.FloatTensor(X_query)
                query_y = torch.FloatTensor(y_query)
                
                self.meta_learner.train()
                meta_optimizer.zero_grad()
                meta_outputs = self.meta_learner(query_X)
                meta_loss_batch = criterion(meta_outputs.squeeze(), query_y)
                meta_loss_batch.backward()
                meta_optimizer.step()
                
                meta_loss += meta_loss_batch.item()
            
            meta_loss /= len(task_batch)
            meta_losses.append(meta_loss)
            
            if meta_epoch % 10 == 0:
                logger.info(f"元訓練第 {meta_epoch} 步 - 損失: {meta_loss:.6f}")
        
        return {
            'meta_losses': meta_losses,
            'final_meta_loss': meta_losses[-1] if meta_losses else 0.0
        }
    
    async def fast_adapt(self, X_support: np.ndarray, y_support: np.ndarray, 
                        X_query: np.ndarray, y_query: np.ndarray) -> Dict[str, Any]:
        """快速適應新任務"""
        if self.meta_learner is None:
            raise ValueError("元學習器尚未訓練")
        
        # 複製元學習器作為任務特定模型
        task_model = type(self.meta_learner)(
            self.meta_learner.feature_extractor[0].in_features,
            1
        )
        task_model.load_state_dict(self.meta_learner.state_dict())
        
        task_optimizer = optim.Adam(task_model.parameters(), lr=self.config.inner_lr)
        criterion = nn.MSELoss()
        
        adaptation_losses = []
        
        # 快速適應
        for epoch in range(self.config.adaptation_epochs):
            task_model.train()
            support_X = torch.FloatTensor(X_support)
            support_y = torch.FloatTensor(y_support)
            
            task_optimizer.zero_grad()
            outputs = task_model(support_X)
            loss = criterion(outputs.squeeze(), support_y)
            loss.backward()
            task_optimizer.step()
            
            adaptation_losses.append(loss.item())
        
        # 評估適應後的模型
        task_model.eval()
        with torch.no_grad():
            query_X = torch.FloatTensor(X_query)
            query_y = torch.FloatTensor(y_query)
            query_outputs = task_model(query_X)
            query_loss = criterion(query_outputs.squeeze(), query_y).item()
            
            # 計算指標
            predictions = query_outputs.cpu().numpy().squeeze()
            actual = query_y.cpu().numpy().squeeze()
            
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
        
        return {
            'adaptation_losses': adaptation_losses,
            'final_query_loss': query_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions.tolist(),
            'actual': actual.tolist()
        }

class AutoMLOptimizer:
    """增強版自動機器學習優化器"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.models = {}
        self.ensemble = None
        self.feature_importance = {}
        self.optimization_history = []
        
    async def optimize_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """優化完整機器學習管道"""
        logger.info("開始 AutoML 優化...")
        
        start_time = time.time()
        results = {
            'feature_selection': {},
            'models': {},
            'ensemble': {},
            'hyperparameter_tuning': {},
            'final_performance': {}
        }
        
        # 1. 特徵選擇
        if self.config.feature_selection:
            results['feature_selection'] = await self._feature_selection(X_train, y_train, X_val, y_val)
            if results['feature_selection']['selected_features']:
                X_train_selected = X_train[:, results['feature_selection']['selected_features']]
                X_val_selected = X_val[:, results['feature_selection']['selected_features']]
            else:
                X_train_selected = X_train
                X_val_selected = X_val
        else:
            X_train_selected = X_train
            X_val_selected = X_val
        
        # 2. 模型選擇和訓練
        results['models'] = await self._train_multiple_models(X_train_selected, y_train, X_val_selected, y_val)
        
        # 3. 超參數調優
        if self.config.hyperparameter_tuning:
            results['hyperparameter_tuning'] = await self._hyperparameter_tuning(
                X_train_selected, y_train, X_val_selected, y_val
            )
        
        # 4. 集成學習
        results['ensemble'] = await self._create_ensemble(X_train_selected, y_train, X_val_selected, y_val)
        
        # 5. 最終性能評估
        results['final_performance'] = await self._evaluate_final_performance(
            X_val_selected, y_val, results
        )
        
        optimization_time = time.time() - start_time
        results['optimization_time'] = optimization_time
        results['config'] = asdict(self.config)
        
        logger.info(f"AutoML 優化完成，耗時: {optimization_time:.2f}秒")
        return results
    
    async def _feature_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """特徵選擇"""
        logger.info("執行特徵選擇...")
        
        feature_selection_results = {}
        
        # 多種特徵選擇方法
        methods = {
            'mutual_info': mutual_info_regression,
            'f_regression': f_regression
        }
        
        best_score = float('-inf')
        best_features = None
        best_method = None
        
        for method_name, method_func in methods.items():
            try:
                # 選擇特徵
                selector = SelectKBest(score_func=method_func, k=min(20, X_train.shape[1]))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_val_selected = selector.transform(X_val)
                
                # 評估特徵選擇效果
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_train_selected, y_train)
                score = rf.score(X_val_selected, y_val)
                
                feature_selection_results[method_name] = {
                    'score': score,
                    'selected_features': selector.get_support(indices=True).tolist(),
                    'feature_scores': selector.scores_.tolist()
                }
                
                if score > best_score:
                    best_score = score
                    best_features = selector.get_support(indices=True).tolist()
                    best_method = method_name
                    
            except Exception as e:
                logger.error(f"特徵選擇方法 {method_name} 失敗: {e}")
        
        return {
            'best_method': best_method,
            'best_score': best_score,
            'selected_features': best_features,
            'all_results': feature_selection_results
        }
    
    async def _train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """訓練多個模型"""
        logger.info("訓練多個模型...")
        
        models = {}
        results = {}
        
        # 線性模型
        linear_models = {
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic_net': ElasticNet()
        }
        
        for name, model in linear_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                models[name] = model
                results[name] = {
                    'mse': mean_squared_error(y_val, y_pred),
                    'mae': mean_absolute_error(y_val, y_pred),
                    'r2': r2_score(y_val, y_pred)
                }
            except Exception as e:
                logger.error(f"線性模型 {name} 訓練失敗: {e}")
        
        # 集成模型
        ensemble_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            ensemble_models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            ensemble_models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        for name, model in ensemble_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                models[name] = model
                results[name] = {
                    'mse': mean_squared_error(y_val, y_pred),
                    'mae': mean_absolute_error(y_val, y_pred),
                    'r2': r2_score(y_val, y_pred)
                }
                
                # 特徵重要性
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_.tolist()
                    
            except Exception as e:
                logger.error(f"集成模型 {name} 訓練失敗: {e}")
        
        # SVM
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            svm = SVR(kernel='rbf', C=1.0, gamma='scale')
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_val_scaled)
            
            models['svm'] = {'model': svm, 'scaler': scaler}
            results['svm'] = {
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
        except Exception as e:
            logger.error(f"SVM 訓練失敗: {e}")
        
        self.models = models
        return results
    
    async def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """超參數調優"""
        logger.info("執行超參數調優...")
        
        tuning_results = {}
        
        # 為最佳模型進行超參數調優
        best_model_name = min(self.models.keys(), 
                            key=lambda k: results.get(k, {}).get('mse', float('inf')))
        
        if best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif best_model_name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            return tuning_results
        
        try:
            cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                self.models[best_model_name],
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            tuning_results[best_model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            # 更新最佳模型
            self.models[best_model_name] = grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"超參數調優失敗: {e}")
        
        return tuning_results
    
    async def _create_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """創建集成模型"""
        logger.info("創建集成模型...")
        
        # 選擇表現最好的模型
        model_scores = {}
        for name, result in results.items():
            if 'mse' in result:
                model_scores[name] = result['mse']
        
        best_models = sorted(model_scores.items(), key=lambda x: x[1])[:self.config.ensemble_size]
        best_model_names = [name for name, _ in best_models]
        
        # 創建投票回歸器
        estimators = []
        for name in best_model_names:
            if name in self.models:
                if isinstance(self.models[name], dict):  # SVM with scaler
                    estimators.append((name, self.models[name]['model']))
                else:
                    estimators.append((name, self.models[name]))
        
        if len(estimators) >= 2:
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_val)
            
            self.ensemble = ensemble
            
            return {
                'ensemble_models': best_model_names,
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
        else:
            return {'error': 'Not enough models for ensemble'}
    
    async def _evaluate_final_performance(self, X_val: np.ndarray, y_val: np.ndarray,
                                        results: Dict[str, Any]) -> Dict[str, Any]:
        """評估最終性能"""
        final_results = {}
        
        # 評估單個模型
        for name, result in results['models'].items():
            if name in self.models:
                y_pred = self.models[name].predict(X_val)
                final_results[name] = {
                    'mse': mean_squared_error(y_val, y_pred),
                    'mae': mean_absolute_error(y_val, y_pred),
                    'r2': r2_score(y_val, y_pred)
                }
        
        # 評估集成模型
        if self.ensemble:
            y_pred_ensemble = self.ensemble.predict(X_val)
            final_results['ensemble'] = {
                'mse': mean_squared_error(y_val, y_pred_ensemble),
                'mae': mean_absolute_error(y_val, y_pred_ensemble),
                'r2': r2_score(y_val, y_pred_ensemble)
            }
        
        return final_results

class KnowledgeDistillation:
    """知識蒸餾"""
    
    def __init__(self):
        self.teacher_models = {}
        self.student_models = {}
        self.distillation_history = []
        
    async def distill_knowledge(self, teacher_model: Any, student_model: nn.Module,
                              X_train: np.ndarray, y_train: np.ndarray,
                              temperature: float = 4.0, alpha: float = 0.7) -> Dict[str, Any]:
        """知識蒸餾"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for knowledge distillation")
        
        logger.info("開始知識蒸餾...")
        
        # 獲取教師模型的軟標籤
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = teacher_model(torch.FloatTensor(X_train))
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        
        # 學生模型訓練
        student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        kl_div = nn.KLDivLoss(reduction='batchmean')
        
        distillation_losses = []
        
        for epoch in range(100):
            student_model.train()
            
            # 前向傳播
            student_logits = student_model(torch.FloatTensor(X_train))
            student_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # 計算損失
            hard_loss = criterion(student_logits.squeeze(), torch.FloatTensor(y_train))
            soft_loss = kl_div(student_probs, teacher_probs)
            
            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            # 反向傳播
            student_optimizer.zero_grad()
            total_loss.backward()
            student_optimizer.step()
            
            distillation_losses.append(total_loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"蒸餾第 {epoch} 輪 - 損失: {total_loss.item():.6f}")
        
        return {
            'distillation_losses': distillation_losses,
            'final_loss': distillation_losses[-1] if distillation_losses else 0.0,
            'temperature': temperature,
            'alpha': alpha
        }

# 主要運行函數
async def main():
    """元學習和自適應系統演示"""
    print("🧠 AGI Meta-Learning and Adaptive System")
    print("=" * 60)
    
    try:
        # 生成示例數據
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, 1000)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. 神經架構搜索演示
        print("\n🔍 神經架構搜索演示...")
        nas_config = NASConfig(
            search_space={},
            population_size=20,
            generations=30
        )
        nas = NeuralArchitectureSearch(nas_config)
        best_architecture = await nas.search_optimal_architecture(X_train, y_train, X_val, y_val)
        print(f"   ✅ 最佳架構找到: {best_architecture}")
        
        # 2. 元學習演示
        print("\n📚 元學習演示...")
        meta_config = MetaLearningConfig(
            task_type='regression',
            adaptation_steps=20,
            num_tasks=5
        )
        meta_learner = MetaLearner(meta_config)
        
        # 創建多個任務
        tasks = []
        for i in range(meta_config.num_tasks):
            # 為每個任務生成不同的數據分佈
            task_X = X + np.random.normal(0, 0.1 * i, X.shape)
            task_y = np.sum(task_X[:, :5], axis=1) + np.random.normal(0, 0.1, len(task_X))
            
            X_train_task, X_val_task, y_train_task, y_val_task = train_test_split(
                task_X, task_y, test_size=0.2, random_state=42+i
            )
            tasks.append((X_train_task, y_train_task, X_val_task, y_val_task))
        
        # 元訓練
        meta_results = await meta_learner.meta_train(tasks)
        print(f"   ✅ 元訓練完成 - 最終損失: {meta_results['final_meta_loss']:.6f}")
        
        # 快速適應新任務
        new_task_X = X + np.random.normal(0, 0.2, X.shape)
        new_task_y = np.sum(new_task_X[:, :5], axis=1) + np.random.normal(0, 0.15, len(new_task_X))
        
        X_support, X_query, y_support, y_query = train_test_split(
            new_task_X, new_task_y, test_size=0.5, random_state=42
        )
        
        adaptation_results = await meta_learner.fast_adapt(X_support, y_support, X_query, y_query)
        print(f"   ✅ 快速適應完成 - R²: {adaptation_results['r2']:.4f}")
        
        # 3. AutoML 優化演示
        print("\n🤖 AutoML 優化演示...")
        automl_config = AutoMLConfig(
            time_budget=300,  # 5分鐘
            max_models=10,
            ensemble_size=3,
            feature_selection=True,
            hyperparameter_tuning=True
        )
        automl = AutoMLOptimizer(automl_config)
        
        pipeline_results = await automl.optimize_pipeline(X_train, y_train, X_val, y_val)
        print(f"   ✅ AutoML 優化完成")
        print(f"   📊 最佳模型性能: {pipeline_results['final_performance']}")
        
        print("\n🎉 元學習和自適應系統演示完成!")
        
    except Exception as e:
        print(f"\n❌ 系統錯誤: {e}")
        logger.error(f"系統錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 