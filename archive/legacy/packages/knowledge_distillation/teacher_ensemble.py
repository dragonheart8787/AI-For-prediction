#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
老師集成模型 - 生成知識蒸餾的軟標籤
實現多種老師模型和集成策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)

class TeacherEnsemble:
    """老師集成模型"""
    
    def __init__(self, models: List[Any], ensemble_method: str = "mean"):
        """
        初始化老師集成
        
        Args:
            models: 老師模型列表
            ensemble_method: 集成方法 ("mean", "weighted", "stacking")
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        self.meta_model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        訓練老師集成模型
        
        Args:
            X: 特徵數據
            y: 目標變量
            cv_folds: 交叉驗證折數
            random_state: 隨機種子
            
        Returns:
            訓練結果
        """
        logger.info(f"開始訓練老師集成模型，包含 {len(self.models)} 個模型")
        
        # 時間序列交叉驗證
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # 存儲每個模型的OOF預測
        oof_predictions = np.zeros((len(X), len(self.models)))
        model_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"訓練第 {fold + 1}/{cv_folds} 折")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_scores = []
            
            for i, model in enumerate(self.models):
                # 訓練模型
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                elif hasattr(model, 'train'):
                    # XGBoost/LightGBM
                    model.train(X_train, y_train)
                
                # 預測
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                elif hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                else:
                    raise ValueError(f"模型 {type(model)} 沒有預測方法")
                
                oof_predictions[val_idx, i] = pred
                
                # 計算分數
                score = self._calculate_score(y_val, pred)
                fold_scores.append(score)
            
            model_scores.append(fold_scores)
        
        # 計算平均分數
        avg_scores = np.mean(model_scores, axis=0)
        logger.info(f"模型平均分數: {avg_scores}")
        
        # 設置集成權重
        if self.ensemble_method == "weighted":
            # 基於性能的權重
            self.weights = self._calculate_weights(avg_scores)
        elif self.ensemble_method == "stacking":
            # 訓練元模型
            self.meta_model = self._train_meta_model(oof_predictions, y)
        
        # 在全部數據上重新訓練所有模型
        for model in self.models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
            elif hasattr(model, 'train'):
                model.train(X, y)
        
        self.is_fitted = True
        
        return {
            'model_scores': avg_scores,
            'weights': self.weights,
            'oof_predictions': oof_predictions
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Args:
            X: 特徵數據
            
        Returns:
            集成預測結果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                raise ValueError(f"模型 {type(model)} 沒有預測方法")
            
            predictions.append(pred)
        
        predictions = np.array(predictions).T  # [n_samples, n_models]
        
        if self.ensemble_method == "mean":
            return np.mean(predictions, axis=1)
        elif self.ensemble_method == "weighted":
            return np.average(predictions, axis=1, weights=self.weights)
        elif self.ensemble_method == "stacking":
            return self.meta_model.predict(predictions)
        else:
            raise ValueError(f"未知的集成方法: {self.ensemble_method}")
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測並返回不確定度
        
        Args:
            X: 特徵數據
            
        Returns:
            (預測均值, 預測方差)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                raise ValueError(f"模型 {type(model)} 沒有預測方法")
            
            predictions.append(pred)
        
        predictions = np.array(predictions).T  # [n_samples, n_models]
        
        # 計算均值和方差
        mean_pred = np.mean(predictions, axis=1)
        var_pred = np.var(predictions, axis=1)
        
        return mean_pred, var_pred
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """計算模型分數（負MSE，越高越好）"""
        mse = np.mean((y_true - y_pred) ** 2)
        return -mse  # 負MSE，越高越好
    
    def _calculate_weights(self, scores: np.ndarray) -> np.ndarray:
        """計算基於性能的權重"""
        # 將分數轉換為正數
        scores = scores - np.min(scores) + 1e-8
        weights = scores / np.sum(scores)
        return weights
    
    def _train_meta_model(self, X_meta: np.ndarray, y: np.ndarray) -> Any:
        """訓練元模型（用於stacking）"""
        from sklearn.linear_model import Ridge
        
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_meta, y)
        return meta_model

class MultiHorizonTeacherEnsemble:
    """多地平線老師集成模型"""
    
    def __init__(self, models: List[Any], horizons: List[int] = [1, 5, 10, 20]):
        """
        初始化多地平線老師集成
        
        Args:
            models: 老師模型列表
            horizons: 預測地平線
        """
        self.models = models
        self.horizons = horizons
        self.teacher_ensembles = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        訓練多地平線老師集成
        
        Args:
            X: 特徵數據
            y: 目標變量 [n_samples, n_horizons]
            cv_folds: 交叉驗證折數
            random_state: 隨機種子
            
        Returns:
            訓練結果
        """
        logger.info(f"開始訓練多地平線老師集成，地平線: {self.horizons}")
        
        results = {}
        
        for i, horizon in enumerate(self.horizons):
            logger.info(f"訓練地平線 {horizon} 的模型")
            
            # 創建該地平線的老師集成
            teacher_ensemble = TeacherEnsemble(self.models, ensemble_method="weighted")
            
            # 訓練
            result = teacher_ensemble.fit(X, y[:, i], cv_folds, random_state)
            
            # 保存
            self.teacher_ensembles[horizon] = teacher_ensemble
            results[f'horizon_{horizon}'] = result
        
        self.is_fitted = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測多地平線
        
        Args:
            X: 特徵數據
            
        Returns:
            預測結果 [n_samples, n_horizons]
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        predictions = []
        for horizon in self.horizons:
            pred = self.teacher_ensembles[horizon].predict(X)
            predictions.append(pred)
        
        return np.array(predictions).T
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測多地平線並返回不確定度
        
        Args:
            X: 特徵數據
            
        Returns:
            (預測均值, 預測方差) [n_samples, n_horizons]
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        means = []
        vars = []
        
        for horizon in self.horizons:
            mean, var = self.teacher_ensembles[horizon].predict_with_uncertainty(X)
            means.append(mean)
            vars.append(var)
        
        return np.array(means).T, np.array(vars).T

class QuantileTeacherEnsemble:
    """分位數老師集成模型"""
    
    def __init__(self, models: List[Any], quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        初始化分位數老師集成
        
        Args:
            models: 老師模型列表
            quantiles: 分位數水平
        """
        self.models = models
        self.quantiles = quantiles
        self.teacher_ensembles = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        訓練分位數老師集成
        
        Args:
            X: 特徵數據
            y: 目標變量
            cv_folds: 交叉驗證折數
            random_state: 隨機種子
            
        Returns:
            訓練結果
        """
        logger.info(f"開始訓練分位數老師集成，分位數: {self.quantiles}")
        
        results = {}
        
        for quantile in self.quantiles:
            logger.info(f"訓練分位數 {quantile} 的模型")
            
            # 創建該分位數的老師集成
            teacher_ensemble = TeacherEnsemble(self.models, ensemble_method="weighted")
            
            # 訓練
            result = teacher_ensemble.fit(X, y, cv_folds, random_state)
            
            # 保存
            self.teacher_ensembles[quantile] = teacher_ensemble
            results[f'quantile_{quantile}'] = result
        
        self.is_fitted = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測分位數
        
        Args:
            X: 特徵數據
            
        Returns:
            預測結果 [n_samples, n_quantiles]
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        predictions = []
        for quantile in self.quantiles:
            pred = self.teacher_ensembles[quantile].predict(X)
            predictions.append(pred)
        
        return np.array(predictions).T

def create_default_teacher_models() -> List[Any]:
    """創建默認的老師模型"""
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        xgb.XGBRegressor(n_estimators=100, random_state=42),
        lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    ]
    return models

def create_advanced_teacher_models() -> List[Any]:
    """創建進階的老師模型"""
    models = [
        # 基礎模型
        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        
        # 線性模型
        Ridge(alpha=0.1),
        Lasso(alpha=0.01),
        
        # 梯度提升
        xgb.XGBRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        
        # 額外的集成模型
        RandomForestRegressor(n_estimators=100, max_depth=15, random_state=123),
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=123)
    ]
    return models

# 示例使用
if __name__ == "__main__":
    # 創建示例數據
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # 創建老師模型
    teacher_models = create_default_teacher_models()
    
    # 創建老師集成
    teacher_ensemble = TeacherEnsemble(teacher_models, ensemble_method="weighted")
    
    # 訓練
    results = teacher_ensemble.fit(X, y, cv_folds=3)
    print(f"訓練完成，模型分數: {results['model_scores']}")
    
    # 預測
    X_test = np.random.randn(100, n_features)
    predictions = teacher_ensemble.predict(X_test)
    print(f"預測結果形狀: {predictions.shape}")
    
    # 預測不確定度
    mean_pred, var_pred = teacher_ensemble.predict_with_uncertainty(X_test)
    print(f"預測均值形狀: {mean_pred.shape}")
    print(f"預測方差形狀: {var_pred.shape}")
