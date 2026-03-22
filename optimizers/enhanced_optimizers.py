#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強優化器模組
支援 SGD/SGDM 等優化器及其調度策略
"""

import torch
import torch.optim as optim
import torch.nn as nn
import logging
import math
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """優化器配置"""
    # 基本設定
    optimizer_type: str = "adamw"  # adamw, sgd, sgd_momentum, adagrad, rmsprop
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # SGD 特定設定
    momentum: float = 0.9
    nesterov: bool = False
    
    # 調度器設定
    scheduler_type: str = "cosine"  # cosine, step, exponential, plateau, warmup_cosine
    warmup_epochs: int = 5
    total_epochs: int = 100
    
    # 學習率調度參數
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6

class EnhancedOptimizer:
    """增強優化器"""
    
    def __init__(self, model: nn.Module, config: OptimizerConfig):
        self.config = config
        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler()
        self.current_lr = config.learning_rate
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """創建優化器"""
        params = model.parameters()
        
        if self.config.optimizer_type == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0
            )
        elif self.config.optimizer_type == "sgd_momentum":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov
            )
        elif self.config.optimizer_type == "adagrad":
            return optim.Adagrad(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )
        else:
            raise ValueError(f"不支援的優化器類型: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """創建學習率調度器"""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "warmup_cosine":
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.total_epochs,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.gamma,
                patience=10,
                min_lr=self.config.min_lr
            )
        else:
            return None
    
    def step(self):
        """優化器步驟"""
        self.optimizer.step()
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def scheduler_step(self, metric: Optional[float] = None):
        """調度器步驟"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
            
            # 更新當前學習率
            self.current_lr = self.optimizer.param_groups[0]['lr']
    
    def get_lr(self) -> float:
        """獲取當前學習率"""
        return self.current_lr
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """獲取優化器信息"""
        return {
            "optimizer_type": self.config.optimizer_type,
            "current_lr": self.current_lr,
            "scheduler_type": self.config.scheduler_type,
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "momentum": self.config.momentum if "sgd" in self.config.optimizer_type else None
            }
        }

class WarmupCosineScheduler:
    """帶預熱的餘弦退火調度器"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, 
                 total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """調度器步驟"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 預熱階段：線性增長
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # 餘弦退火階段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class GradientClipping:
    """梯度裁剪"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """裁剪梯度"""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

class OptimizerFactory:
    """優化器工廠"""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        scheduler_type: str = "cosine",
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        **kwargs
    ) -> EnhancedOptimizer:
        """創建優化器"""
        config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            scheduler_type=scheduler_type,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs
        )
        
        return EnhancedOptimizer(model, config)

class TrainingLoop:
    """訓練循環"""
    
    def __init__(self, model: nn.Module, optimizer: EnhancedOptimizer, 
                 gradient_clipping: Optional[GradientClipping] = None):
        self.model = model
        self.optimizer = optimizer
        self.gradient_clipping = gradient_clipping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor], loss_fn) -> Dict[str, float]:
        """訓練步驟"""
        self.model.train()
        
        # 轉移數據到設備
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 前向傳播
        outputs = self.model(**batch)
        loss = loss_fn(outputs, batch)
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = 0.0
        if self.gradient_clipping is not None:
            grad_norm = self.gradient_clipping.clip_gradients(self.model)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": self.optimizer.get_lr()
        }
    
    def train_epoch(self, dataloader, loss_fn, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        for batch in dataloader:
            metrics = self.train_step(batch, loss_fn)
            total_loss += metrics["loss"]
            total_grad_norm += metrics["grad_norm"]
            num_batches += 1
        
        # 更新學習率
        self.optimizer.scheduler_step()
        
        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "grad_norm": total_grad_norm / num_batches if num_batches > 0 else 0.0,
            "lr": self.optimizer.get_lr()
        }

# 使用示例
if __name__ == "__main__":
    import torch.nn as nn
    
    # 創建測試模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
            
        def forward(self, x):
            return self.linear(x)
    
    # 測試優化器
    model = TestModel()
    optimizer = OptimizerFactory.create_optimizer(
        model,
        optimizer_type="sgd_momentum",
        learning_rate=1e-2,
        scheduler_type="warmup_cosine"
    )
    
    print("優化器測試完成")
    print(f"優化器信息: {optimizer.get_optimizer_info()}")
