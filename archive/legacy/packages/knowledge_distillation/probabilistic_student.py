#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機率式學生模型 - 知識蒸餾
實現均值+方差的機率式預測模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ProbabilisticStudent(nn.Module):
    """機率式學生模型 - 輸出均值和方差"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 共享特徵提取器
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 均值頭
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)
        
        # 方差頭（輸出 log(σ²)）
        self.logvar_head = nn.Linear(hidden_dim // 2, output_dim)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            
        Returns:
            mu: 預測均值 [batch_size, output_dim]
            logvar: 預測方差對數 [batch_size, output_dim]
        """
        # 特徵提取
        features = self.backbone(x)
        
        # 預測均值和方差
        mu = self.mu_head(features)
        logvar = self.logvar_head(features).clamp(-10, 5)  # 避免極值
        
        return mu, logvar
    
    def predict(self, x: torch.Tensor, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        預測分佈
        
        Args:
            x: 輸入特徵
            num_samples: 採樣數量
            
        Returns:
            預測結果字典
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x)
            var = torch.exp(logvar)
            std = torch.sqrt(var)
            
            # 蒙特卡羅採樣
            samples = torch.randn(num_samples, x.size(0), self.output_dim) * std + mu
            
            return {
                'mean': mu,
                'std': std,
                'var': var,
                'samples': samples,
                'quantiles': self._compute_quantiles(samples)
            }
    
    def _compute_quantiles(self, samples: torch.Tensor, 
                          quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]) -> Dict[float, torch.Tensor]:
        """計算分位數"""
        sorted_samples, _ = torch.sort(samples, dim=0)
        quantile_values = {}
        
        for q in quantiles:
            idx = int(q * (samples.size(0) - 1))
            quantile_values[q] = sorted_samples[idx]
        
        return quantile_values

class MultiHorizonProbabilisticStudent(nn.Module):
    """多地平線機率式學生模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 horizons: list = [1, 5, 10, 20]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizons = horizons
        self.num_horizons = len(horizons)
        
        # 共享特徵提取器
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 每個地平線的均值頭
        self.mu_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(self.num_horizons)
        ])
        
        # 每個地平線的方差頭
        self.logvar_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(self.num_horizons)
        ])
        
        # 地平線權重（指數衰減）
        self.horizon_weights = self._compute_horizon_weights()
        
        self._init_weights()
    
    def _compute_horizon_weights(self, tau: float = 10.0) -> torch.Tensor:
        """計算地平線權重（指數衰減）"""
        weights = torch.exp(-torch.tensor(self.horizons, dtype=torch.float32) / tau)
        return weights / weights.sum()  # 歸一化
    
    def _init_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            
        Returns:
            mu: 預測均值 [batch_size, num_horizons]
            logvar: 預測方差對數 [batch_size, num_horizons]
        """
        # 特徵提取
        features = self.backbone(x)
        
        # 每個地平線的預測
        mu_list = []
        logvar_list = []
        
        for i in range(self.num_horizons):
            mu = self.mu_heads[i](features)
            logvar = self.logvar_heads[i](features).clamp(-10, 5)
            
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        mu = torch.cat(mu_list, dim=1)  # [batch_size, num_horizons]
        logvar = torch.cat(logvar_list, dim=1)  # [batch_size, num_horizons]
        
        return mu, logvar
    
    def predict(self, x: torch.Tensor, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """預測分佈"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.forward(x)
            var = torch.exp(logvar)
            std = torch.sqrt(var)
            
            # 蒙特卡羅採樣
            samples = torch.randn(num_samples, x.size(0), self.num_horizons) * std + mu
            
            return {
                'mean': mu,
                'std': std,
                'var': var,
                'samples': samples,
                'horizons': self.horizons,
                'horizon_weights': self.horizon_weights
            }

class QuantileStudent(nn.Module):
    """分位數學生模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # 共享特徵提取器
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 每個分位數的預測頭
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(self.num_quantiles)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            
        Returns:
            quantiles: 分位數預測 [batch_size, num_quantiles]
        """
        # 特徵提取
        features = self.backbone(x)
        
        # 每個分位數的預測
        quantile_list = []
        for i in range(self.num_quantiles):
            q = self.quantile_heads[i](features)
            quantile_list.append(q)
        
        quantiles = torch.cat(quantile_list, dim=1)  # [batch_size, num_quantiles]
        return quantiles
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """預測分位數"""
        self.eval()
        with torch.no_grad():
            quantiles = self.forward(x)
            
            return {
                'quantiles': quantiles,
                'quantile_values': self.quantiles,
                'median': quantiles[:, self.quantiles.index(0.5)],
                'iqr': quantiles[:, self.quantiles.index(0.75)] - quantiles[:, self.quantiles.index(0.25)]
            }

def nll_gaussian(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    高斯負對數似然損失
    
    Args:
        y: 真實值
        mu: 預測均值
        logvar: 預測方差對數
        
    Returns:
        負對數似然損失
    """
    return 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))

def kl_gaussians(mu_t: torch.Tensor, logvar_t: torch.Tensor, 
                mu_s: torch.Tensor, logvar_s: torch.Tensor) -> torch.Tensor:
    """
    兩個高斯分佈的KL散度
    
    Args:
        mu_t, logvar_t: 老師分佈的均值和方差對數
        mu_s, logvar_s: 學生分佈的均值和方差對數
        
    Returns:
        KL散度
    """
    var_t = torch.exp(logvar_t)
    var_s = torch.exp(logvar_s)
    
    kl = 0.5 * (logvar_s - logvar_t + (var_t + (mu_t - mu_s) ** 2) / var_s - 1.0)
    return kl

def pinball_loss(y: torch.Tensor, quantiles: torch.Tensor, 
                tau: torch.Tensor) -> torch.Tensor:
    """
    Pinball損失（分位數回歸損失）
    
    Args:
        y: 真實值
        quantiles: 預測分位數
        tau: 分位數水平
        
    Returns:
        Pinball損失
    """
    error = y.unsqueeze(-1) - quantiles
    loss = torch.max(tau * error, (tau - 1) * error)
    return loss.mean()

def knowledge_distillation_step(x: torch.Tensor, y: torch.Tensor,
                              mu_t: torch.Tensor, logvar_t: torch.Tensor,
                              student_model: nn.Module, optimizer: torch.optim.Optimizer,
                              alpha: float = 0.5) -> float:
    """
    知識蒸餾訓練步驟
    
    Args:
        x: 輸入特徵
        y: 真實標籤
        mu_t: 老師預測均值
        logvar_t: 老師預測方差對數
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 學生預測
    mu_s, logvar_s = student_model(x)
    
    # 硬標籤損失：對真實y的NLL
    loss_hard = nll_gaussian(y, mu_s, logvar_s).mean()
    
    # 軟標籤損失：對老師分佈的KL散度
    loss_soft = kl_gaussians(mu_t, logvar_t, mu_s, logvar_s).mean()
    
    # 總損失
    loss = alpha * loss_hard + (1 - alpha) * loss_soft
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def multi_horizon_kd_step(x: torch.Tensor, y: torch.Tensor,
                         mu_t: torch.Tensor, logvar_t: torch.Tensor,
                         student_model: nn.Module, optimizer: torch.optim.Optimizer,
                         alpha: float = 0.5) -> float:
    """
    多地平線知識蒸餾訓練步驟
    
    Args:
        x: 輸入特徵
        y: 真實標籤 [batch_size, num_horizons]
        mu_t: 老師預測均值 [batch_size, num_horizons]
        logvar_t: 老師預測方差對數 [batch_size, num_horizons]
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 學生預測
    mu_s, logvar_s = student_model(x)
    
    # 獲取地平線權重
    horizon_weights = student_model.horizon_weights.to(x.device)
    
    # 每個地平線的損失
    total_loss = 0.0
    for h in range(student_model.num_horizons):
        # 硬標籤損失
        loss_hard = nll_gaussian(y[:, h], mu_s[:, h], logvar_s[:, h])
        
        # 軟標籤損失
        loss_soft = kl_gaussians(mu_t[:, h], logvar_t[:, h], 
                                mu_s[:, h], logvar_s[:, h])
        
        # 加權損失
        horizon_loss = alpha * loss_hard + (1 - alpha) * loss_soft
        total_loss += horizon_weights[h] * horizon_loss
    
    # 反向傳播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def quantile_kd_step(x: torch.Tensor, y: torch.Tensor,
                    quantiles_t: torch.Tensor, quantiles_s: torch.Tensor,
                    student_model: nn.Module, optimizer: torch.optim.Optimizer,
                    alpha: float = 0.5) -> float:
    """
    分位數知識蒸餾訓練步驟
    
    Args:
        x: 輸入特徵
        y: 真實標籤
        quantiles_t: 老師分位數預測
        quantiles_s: 學生分位數預測
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 學生預測
    quantiles_pred = student_model(x)
    
    # 分位數水平
    tau = torch.tensor(student_model.quantiles, device=x.device)
    
    # 硬標籤損失：對真實y的分位數損失
    loss_hard = pinball_loss(y, quantiles_pred, tau).mean()
    
    # 軟標籤損失：對老師分位數的損失
    loss_soft = pinball_loss(quantiles_t, quantiles_pred, tau).mean()
    
    # 總損失
    loss = alpha * loss_hard + (1 - alpha) * loss_soft
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 示例使用
if __name__ == "__main__":
    # 創建示例數據
    batch_size, input_dim = 32, 50
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    
    # 老師預測（模擬）
    mu_t = torch.randn(batch_size, 1)
    logvar_t = torch.randn(batch_size, 1)
    
    # 創建學生模型
    student = ProbabilisticStudent(input_dim, hidden_dim=128)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    # 訓練步驟
    loss = knowledge_distillation_step(x, y, mu_t, logvar_t, student, optimizer)
    print(f"知識蒸餾損失: {loss:.4f}")
    
    # 預測
    predictions = student.predict(x)
    print(f"預測均值形狀: {predictions['mean'].shape}")
    print(f"預測方差形狀: {predictions['var'].shape}")
