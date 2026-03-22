#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 AutoML/超參數自動調優模組
整合 Optuna, Ray Tune, Bayesian Optimization
"""

import optuna
import ray
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
from datetime import datetime
import joblib
import os

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """超參數優化器"""
    
    def __init__(self, optimization_type: str = "optuna", n_trials: int = 100):
        self.optimization_type = optimization_type
        self.n_trials = n_trials
        self.best_params = {}
        self.best_score = -np.inf
        self.optimization_history = []
        
    def optimize_sklearn_model(self, model_class, X, y, param_space: Dict, 
                              cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict:
        """優化 sklearn 模型"""
        if self.optimization_type == "optuna":
            return self._optimize_with_optuna_sklearn(model_class, X, y, param_space, cv, scoring)
        elif self.optimization_type == "ray_tune":
            return self._optimize_with_ray_tune_sklearn(model_class, X, y, param_space, cv, scoring)
        else:
            raise ValueError(f"不支援的優化類型: {self.optimization_type}")
    
    def _optimize_with_optuna_sklearn(self, model_class, X, y, param_space: Dict, 
                                    cv: int, scoring: str) -> Dict:
        """使用 Optuna 優化 sklearn 模型"""
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_type': 'optuna'
        }
    
    def _optimize_with_ray_tune_sklearn(self, model_class, X, y, param_space: Dict, 
                                      cv: int, scoring: str) -> Dict:
        """使用 Ray Tune 優化 sklearn 模型"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        def train_model(config):
            model = model_class(**config)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            tune.report(score=scores.mean())
        
        # 轉換參數空間格式
        ray_param_space = {}
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                ray_param_space[param_name] = tune.choice(param_config['choices'])
            elif param_config['type'] == 'int':
                ray_param_space[param_name] = tune.randint(param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                ray_param_space[param_name] = tune.uniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_float':
                ray_param_space[param_name] = tune.loguniform(param_config['low'], param_config['high'])
        
        analysis = tune.run(
            train_model,
            config=ray_param_space,
            num_samples=self.n_trials,
            search_alg=OptunaSearch(),
            metric="score",
            mode="max"
        )
        
        self.best_params = analysis.best_config
        self.best_score = analysis.best_result["score"]
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'optimization_type': 'ray_tune'
        }
    
    def optimize_pytorch_model(self, model_class, train_loader, val_loader, 
                              param_space: Dict, epochs: int = 10) -> Dict:
        """優化 PyTorch 模型"""
        if self.optimization_type == "optuna":
            return self._optimize_with_optuna_pytorch(model_class, train_loader, val_loader, param_space, epochs)
        else:
            raise ValueError("PyTorch 模型優化目前只支援 Optuna")
    
    def _optimize_with_optuna_pytorch(self, model_class, train_loader, val_loader, 
                                    param_space: Dict, epochs: int) -> Dict:
        """使用 Optuna 優化 PyTorch 模型"""
        def objective(trial):
            # 建議超參數
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # 創建模型
            model = model_class(**params)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # 訓練模型
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # 驗證模型
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
            
            return -val_loss / len(val_loader)  # 負值因為 Optuna 最大化
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_type': 'optuna'
        }
    
    def optimize_ensemble(self, models_config: List[Dict], X, y, 
                         ensemble_method: str = "voting") -> Dict:
        """優化集成模型"""
        if self.optimization_type == "optuna":
            return self._optimize_ensemble_optuna(models_config, X, y, ensemble_method)
        else:
            raise ValueError("集成模型優化目前只支援 Optuna")
    
    def _optimize_ensemble_optuna(self, models_config: List[Dict], X, y, 
                                ensemble_method: str) -> Dict:
        """使用 Optuna 優化集成模型"""
        def objective(trial):
            models = []
            weights = []
            
            for model_config in models_config:
                # 為每個模型建議參數
                params = {}
                for param_name, param_config in model_config['param_space'].items():
                    if param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(f"{model_config['name']}_{param_name}", param_config['choices'])
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(f"{model_config['name']}_{param_name}", param_config['low'], param_config['high'])
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(f"{model_config['name']}_{param_name}", param_config['low'], param_config['high'])
                
                model = model_config['class'](**params)
                models.append(model)
                
                # 建議權重
                weight = trial.suggest_float(f"{model_config['name']}_weight", 0.0, 1.0)
                weights.append(weight)
            
            # 正規化權重
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # 交叉驗證評估
            scores = []
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 訓練所有模型
                predictions = []
                for model in models:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    predictions.append(pred)
                
                # 集成預測
                predictions = np.array(predictions)
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                
                # 計算分數
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_val, ensemble_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'optimization_type': 'optuna',
            'ensemble_method': ensemble_method
        }
    
    def save_optimization_results(self, filepath: str):
        """保存優化結果"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_type': self.optimization_type,
            'n_trials': self.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"優化結果已保存到: {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """載入優化結果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.optimization_history = results.get('optimization_history', [])
        
        logger.info(f"優化結果已從 {filepath} 載入")

# 預定義的參數空間
PREDEFINED_PARAM_SPACES = {
    'random_forest': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        'bootstrap': {'type': 'categorical', 'choices': [True, False]}
    },
    'xgboost': {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'log_float', 'low': 0.01, 'high': 0.3},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'log_float', 'low': 1e-8, 'high': 1.0},
        'reg_lambda': {'type': 'log_float', 'low': 1e-8, 'high': 1.0}
    },
    'lightgbm': {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'log_float', 'low': 0.01, 'high': 0.3},
        'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'log_float', 'low': 1e-8, 'high': 1.0},
        'reg_lambda': {'type': 'log_float', 'low': 1e-8, 'high': 1.0}
    },
    'pytorch_transformer': {
        'd_model': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
        'nhead': {'type': 'categorical', 'choices': [4, 8, 16]},
        'num_layers': {'type': 'int', 'low': 2, 'high': 8},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
        'learning_rate': {'type': 'log_float', 'low': 1e-5, 'high': 1e-2},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]}
    }
}

def create_optimizer(optimization_type: str = "optuna", n_trials: int = 100) -> HyperparameterOptimizer:
    """創建超參數優化器"""
    return HyperparameterOptimizer(optimization_type, n_trials)

# 使用示例
if __name__ == "__main__":
    # 創建示例數據
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # 創建優化器
    optimizer = create_optimizer("optuna", n_trials=50)
    
    # 優化 Random Forest
    print("🔬 開始優化 Random Forest...")
    rf_params = PREDEFINED_PARAM_SPACES['random_forest']
    rf_results = optimizer.optimize_sklearn_model(
        RandomForestRegressor, X, y, rf_params, cv=5
    )
    print(f"✅ Random Forest 最佳參數: {rf_results['best_params']}")
    print(f"✅ 最佳分數: {rf_results['best_score']:.4f}")
    
    # 優化 XGBoost
    print("\n🔬 開始優化 XGBoost...")
    xgb_params = PREDEFINED_PARAM_SPACES['xgboost']
    xgb_results = optimizer.optimize_sklearn_model(
        xgb.XGBRegressor, X, y, xgb_params, cv=5
    )
    print(f"✅ XGBoost 最佳參數: {xgb_results['best_params']}")
    print(f"✅ 最佳分數: {xgb_results['best_score']:.4f}")
    
    # 保存結果
    optimizer.save_optimization_results("optimization_results.json")
    print("\n💾 優化結果已保存")
