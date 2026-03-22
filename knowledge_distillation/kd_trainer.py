#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知識蒸餾訓練器
整合所有知識蒸餾功能，提供統一的訓練接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
import json
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .probabilistic_student import (
    ProbabilisticStudent, MultiHorizonProbabilisticStudent, QuantileStudent,
    knowledge_distillation_step, multi_horizon_kd_step, quantile_kd_step
)
from .teacher_ensemble import (
    TeacherEnsemble, MultiHorizonTeacherEnsemble, QuantileTeacherEnsemble,
    create_default_teacher_models, create_advanced_teacher_models
)
from .sequence_kd import (
    LSTMStudent, TransformerStudent, MultiAssetStudent, RegimeAwareStudent,
    sequence_kd_step, multi_asset_kd_step, continual_learning_step
)

logger = logging.getLogger(__name__)

class KnowledgeDistillationTrainer:
    """知識蒸餾訓練器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化訓練器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
        # 模型和優化器
        self.teacher_ensemble = None
        self.student_model = None
        self.optimizer = None
        self.scheduler = None
        
        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # 創建輸出目錄
        self.output_dir = config.get('output_dir', 'kd_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        準備數據
        
        Args:
            X: 特徵數據
            y: 目標變量
            test_size: 測試集比例
            random_state: 隨機種子
            
        Returns:
            數據字典
        """
        logger.info("準備數據...")
        
        # 時間序列分割
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 轉換為張量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test)
        
        return {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor,
            'X_train_scaled': X_train_scaled,
            'y_train_scaled': y_train,
            'X_test_scaled': X_test_scaled,
            'y_test_scaled': y_test
        }
    
    def create_teacher_ensemble(self, X: np.ndarray, y: np.ndarray) -> TeacherEnsemble:
        """
        創建老師集成模型
        
        Args:
            X: 特徵數據
            y: 目標變量
            
        Returns:
            老師集成模型
        """
        logger.info("創建老師集成模型...")
        
        # 創建老師模型
        if self.config.get('use_advanced_teachers', False):
            teacher_models = create_advanced_teacher_models()
        else:
            teacher_models = create_default_teacher_models()
        
        # 創建集成
        ensemble_method = self.config.get('ensemble_method', 'weighted')
        teacher_ensemble = TeacherEnsemble(teacher_models, ensemble_method)
        
        # 訓練
        cv_folds = self.config.get('cv_folds', 5)
        results = teacher_ensemble.fit(X, y, cv_folds=cv_folds)
        
        logger.info(f"老師集成訓練完成，模型分數: {results['model_scores']}")
        
        return teacher_ensemble
    
    def create_student_model(self, input_dim: int) -> nn.Module:
        """
        創建學生模型
        
        Args:
            input_dim: 輸入維度
            
        Returns:
            學生模型
        """
        logger.info("創建學生模型...")
        
        model_type = self.config.get('student_model_type', 'probabilistic')
        
        if model_type == 'probabilistic':
            hidden_dim = self.config.get('hidden_dim', 128)
            student_model = ProbabilisticStudent(input_dim, hidden_dim)
            
        elif model_type == 'multi_horizon':
            hidden_dim = self.config.get('hidden_dim', 128)
            horizons = self.config.get('horizons', [1, 5, 10, 20])
            student_model = MultiHorizonProbabilisticStudent(input_dim, hidden_dim, horizons)
            
        elif model_type == 'quantile':
            hidden_dim = self.config.get('hidden_dim', 128)
            quantiles = self.config.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
            student_model = QuantileStudent(input_dim, hidden_dim, quantiles)
            
        elif model_type == 'lstm':
            hidden_dim = self.config.get('hidden_dim', 128)
            num_layers = self.config.get('num_layers', 2)
            student_model = LSTMStudent(input_dim, hidden_dim, num_layers)
            
        elif model_type == 'transformer':
            d_model = self.config.get('d_model', 128)
            nhead = self.config.get('nhead', 8)
            num_layers = self.config.get('num_layers', 4)
            student_model = TransformerStudent(input_dim, d_model, nhead, num_layers)
            
        elif model_type == 'multi_asset':
            hidden_dim = self.config.get('hidden_dim', 128)
            num_assets = self.config.get('num_assets', 3)
            student_model = MultiAssetStudent(input_dim, hidden_dim, num_assets)
            
        else:
            raise ValueError(f"未知的學生模型類型: {model_type}")
        
        return student_model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
        """
        創建優化器和學習率調度器
        
        Args:
            model: 模型
            
        Returns:
            (優化器, 學習率調度器)
        """
        optimizer_type = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"未知的優化器類型: {optimizer_type}")
        
        # 學習率調度器
        scheduler = None
        if self.config.get('use_scheduler', False):
            scheduler_type = self.config.get('scheduler_type', 'step')
            if scheduler_type == 'step':
                step_size = self.config.get('step_size', 30)
                gamma = self.config.get('gamma', 0.1)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'cosine':
                T_max = self.config.get('T_max', 100)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        return optimizer, scheduler
    
    def train_teacher_ensemble(self, data: Dict[str, Any]) -> TeacherEnsemble:
        """
        訓練老師集成模型
        
        Args:
            data: 數據字典
            
        Returns:
            訓練好的老師集成模型
        """
        logger.info("訓練老師集成模型...")
        
        X_train = data['X_train_scaled']
        y_train = data['y_train_scaled']
        
        self.teacher_ensemble = self.create_teacher_ensemble(X_train, y_train)
        
        return self.teacher_ensemble
    
    def generate_teacher_predictions(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        生成老師預測
        
        Args:
            data: 數據字典
            
        Returns:
            老師預測字典
        """
        logger.info("生成老師預測...")
        
        X_train = data['X_train_scaled']
        X_test = data['X_test_scaled']
        
        # 訓練集預測
        y_train_teacher = self.teacher_ensemble.predict(X_train)
        
        # 測試集預測
        y_test_teacher = self.teacher_ensemble.predict(X_test)
        
        # 不確定度預測
        if hasattr(self.teacher_ensemble, 'predict_with_uncertainty'):
            mu_train, var_train = self.teacher_ensemble.predict_with_uncertainty(X_train)
            mu_test, var_test = self.teacher_ensemble.predict_with_uncertainty(X_test)
            
            return {
                'y_train_teacher': y_train_teacher,
                'y_test_teacher': y_test_teacher,
                'mu_train': mu_train,
                'var_train': var_train,
                'mu_test': mu_test,
                'var_test': var_test
            }
        else:
            return {
                'y_train_teacher': y_train_teacher,
                'y_test_teacher': y_test_teacher
            }
    
    def train_student_model(self, data: Dict[str, Any], teacher_predictions: Dict[str, np.ndarray]) -> nn.Module:
        """
        訓練學生模型
        
        Args:
            data: 數據字典
            teacher_predictions: 老師預測字典
            
        Returns:
            訓練好的學生模型
        """
        logger.info("訓練學生模型...")
        
        # 創建學生模型
        input_dim = data['X_train'].shape[1]
        self.student_model = self.create_student_model(input_dim)
        
        # 創建優化器
        self.optimizer, self.scheduler = self.create_optimizer(self.student_model)
        
        # 訓練參數
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        alpha = self.config.get('alpha', 0.5)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # 準備數據
        X_train = data['X_train'].to(self.device)
        y_train = data['y_train'].to(self.device)
        X_test = data['X_test'].to(self.device)
        y_test = data['y_test'].to(self.device)
        
        # 老師預測
        y_train_teacher = torch.FloatTensor(teacher_predictions['y_train_teacher']).to(self.device)
        y_test_teacher = torch.FloatTensor(teacher_predictions['y_test_teacher']).to(self.device)
        
        # 創建數據加載器
        train_dataset = TensorDataset(X_train, y_train, y_train_teacher)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 訓練循環
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 訓練
            train_loss = self._train_epoch(train_loader, alpha, teacher_predictions)
            
            # 驗證
            val_loss = self._validate_epoch(X_test, y_test, y_test_teacher, alpha)
            
            # 記錄歷史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            # 學習率調度
            if self.scheduler:
                self.scheduler.step()
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self._save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停於第 {epoch + 1} 輪")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 加載最佳模型
        self._load_model('best_model.pth')
        
        return self.student_model
    
    def _train_epoch(self, train_loader: DataLoader, alpha: float, teacher_predictions: Dict[str, np.ndarray] = None) -> float:
        """訓練一個epoch"""
        self.student_model.train()
        total_loss = 0.0
        
        for batch_x, batch_y, batch_y_teacher in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_y_teacher = batch_y_teacher.to(self.device)
            
            # 知識蒸餾步驟
            if self.config.get('student_model_type') == 'probabilistic':
                # 機率式模型需要均值和方差
                if teacher_predictions and 'mu_train' in teacher_predictions:
                    mu_teacher = torch.FloatTensor(teacher_predictions['mu_train']).to(self.device)
                    logvar_teacher = torch.log(torch.FloatTensor(teacher_predictions['var_train'])).to(self.device)
                    loss = knowledge_distillation_step(
                        batch_x, batch_y, mu_teacher, logvar_teacher,
                        self.student_model, self.optimizer, alpha
                    )
                else:
                    # 簡化版本，只用均值
                    mu_pred, logvar_pred = self.student_model(batch_x)
                    loss_hard = F.mse_loss(mu_pred, batch_y)
                    loss_soft = F.mse_loss(mu_pred, batch_y_teacher)
                    loss = alpha * loss_hard + (1 - alpha) * loss_soft
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss = loss.item()
            else:
                # 其他模型類型
                y_pred = self.student_model(batch_x)
                loss_hard = F.mse_loss(y_pred, batch_y)
                loss_soft = F.mse_loss(y_pred, batch_y_teacher)
                loss = alpha * loss_hard + (1 - alpha) * loss_soft
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
            
            total_loss += loss
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, X_test: torch.Tensor, y_test: torch.Tensor, 
                       y_test_teacher: torch.Tensor, alpha: float) -> float:
        """驗證一個epoch"""
        self.student_model.eval()
        
        with torch.no_grad():
            if self.config.get('student_model_type') == 'probabilistic':
                mu_pred, logvar_pred = self.student_model(X_test)
                # 確保形狀匹配
                if y_test.dim() == 1:
                    y_test = y_test.unsqueeze(1)
                if y_test_teacher.dim() == 1:
                    y_test_teacher = y_test_teacher.unsqueeze(1)
                loss_hard = F.mse_loss(mu_pred, y_test)
                loss_soft = F.mse_loss(mu_pred, y_test_teacher)
                loss = alpha * loss_hard + (1 - alpha) * loss_soft
            else:
                y_pred = self.student_model(X_test)
                # 確保形狀匹配
                if y_test.dim() == 1:
                    y_test = y_test.unsqueeze(1)
                if y_test_teacher.dim() == 1:
                    y_test_teacher = y_test_teacher.unsqueeze(1)
                loss_hard = F.mse_loss(y_pred, y_test)
                loss_soft = F.mse_loss(y_pred, y_test_teacher)
                loss = alpha * loss_hard + (1 - alpha) * loss_soft
        
        return loss.item()
    
    def evaluate_model(self, data: Dict[str, Any], teacher_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        評估模型
        
        Args:
            data: 數據字典
            teacher_predictions: 老師預測字典
            
        Returns:
            評估指標字典
        """
        logger.info("評估模型...")
        
        self.student_model.eval()
        
        X_test = data['X_test'].to(self.device)
        y_test = data['y_test'].cpu().numpy()
        
        with torch.no_grad():
            if self.config.get('student_model_type') == 'probabilistic':
                mu_pred, logvar_pred = self.student_model(X_test)
                y_pred = mu_pred.cpu().numpy()
            else:
                y_pred = self.student_model(X_test).cpu().numpy()
        
        # 計算指標
        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        
        # 老師模型指標
        y_test_teacher = teacher_predictions['y_test_teacher']
        teacher_mse = float(mean_squared_error(y_test, y_test_teacher))
        teacher_mae = float(mean_absolute_error(y_test, y_test_teacher))
        teacher_r2 = float(r2_score(y_test, y_test_teacher))
        teacher_rmse = float(np.sqrt(teacher_mse))
        
        metrics = {
            'student_mse': mse,
            'student_mae': mae,
            'student_r2': r2,
            'student_rmse': rmse,
            'teacher_mse': teacher_mse,
            'teacher_mae': teacher_mae,
            'teacher_r2': teacher_r2,
            'teacher_rmse': teacher_rmse,
            'mse_ratio': mse / teacher_mse,
            'mae_ratio': mae / teacher_mae,
            'r2_ratio': r2 / teacher_r2
        }
        
        return metrics
    
    def _save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }, model_path)
    
    def _load_model(self, filename: str):
        """加載模型"""
        model_path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_results(self, metrics: Dict[str, float], teacher_predictions: Dict[str, np.ndarray]):
        """保存結果"""
        # 保存指標
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存訓練歷史
        history_path = os.path.join(self.output_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存老師預測
        teacher_pred_path = os.path.join(self.output_dir, 'teacher_predictions.npz')
        np.savez(teacher_pred_path, **teacher_predictions)
        
        logger.info(f"結果已保存到 {self.output_dir}")

def create_default_config() -> Dict[str, Any]:
    """創建默認配置"""
    return {
        'student_model_type': 'probabilistic',
        'hidden_dim': 128,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'use_scheduler': True,
        'scheduler_type': 'step',
        'step_size': 30,
        'gamma': 0.1,
        'alpha': 0.5,
        'early_stopping_patience': 10,
        'cv_folds': 5,
        'ensemble_method': 'weighted',
        'use_advanced_teachers': False,
        'output_dir': 'kd_outputs'
    }

# 示例使用
if __name__ == "__main__":
    # 創建示例數據
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # 創建配置
    config = create_default_config()
    config['epochs'] = 50  # 減少epochs用於測試
    
    # 創建訓練器
    trainer = KnowledgeDistillationTrainer(config)
    
    # 準備數據
    data = trainer.prepare_data(X, y)
    
    # 訓練老師集成
    teacher_ensemble = trainer.train_teacher_ensemble(data)
    
    # 生成老師預測
    teacher_predictions = trainer.generate_teacher_predictions(data)
    
    # 訓練學生模型
    student_model = trainer.train_student_model(data, teacher_predictions)
    
    # 評估模型
    metrics = trainer.evaluate_model(data, teacher_predictions)
    
    # 保存結果
    trainer.save_results(metrics, teacher_predictions)
    
    print("知識蒸餾訓練完成！")
    print(f"學生模型 MSE: {metrics['student_mse']:.4f}")
    print(f"老師模型 MSE: {metrics['teacher_mse']:.4f}")
    print(f"MSE 比率: {metrics['mse_ratio']:.4f}")
