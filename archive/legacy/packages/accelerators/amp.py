#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動混合精度 (AMP) 模組
支援 BF16/FP16 混合精度訓練
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class AMPConfig:
    """AMP 配置"""
    # 精度設定
    enable_amp: bool = True
    dtype: str = "auto"  # auto, bf16, fp16, fp32
    use_grad_scaler: bool = True
    
    # 硬體檢測
    prefer_bf16: bool = True  # 優先使用 BF16
    
    # 優化設定
    use_autocast: bool = True
    cache_enabled: bool = True

class AMPManager:
    """自動混合精度管理器"""
    
    def __init__(self, config: AMPConfig):
        self.config = config
        self.scaler = None
        self.dtype = self._determine_dtype()
        self._setup_scaler()
        
    def _determine_dtype(self) -> torch.dtype:
        """確定最佳精度類型"""
        if self.config.dtype == "auto":
            if self.config.prefer_bf16 and torch.cuda.is_bf16_supported():
                logger.info("使用 BF16 精度")
                return torch.bfloat16
            elif torch.cuda.is_available():
                logger.info("使用 FP16 精度")
                return torch.float16
            else:
                logger.info("使用 FP32 精度")
                return torch.float32
        elif self.config.dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                logger.warning("硬體不支援 BF16，回退到 FP16")
                return torch.float16
            return torch.bfloat16
        elif self.config.dtype == "fp16":
            return torch.float16
        else:
            return torch.float32
    
    def _setup_scaler(self):
        """設置梯度縮放器"""
        if self.config.use_grad_scaler and self.dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=True,
                init_scale=2.**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
            logger.info("已設置 FP16 梯度縮放器")
        else:
            logger.info("不需要梯度縮放器（BF16 或 FP32）")
    
    @contextmanager
    def autocast(self):
        """自動混合精度上下文管理器"""
        if not self.config.enable_amp or not self.config.use_autocast:
            yield
            return
        
        with torch.cuda.amp.autocast(
            enabled=True,
            dtype=self.dtype,
            cache_enabled=self.config.cache_enabled
        ):
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """縮放損失"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """優化器步驟"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_amp_info(self) -> Dict[str, Any]:
        """獲取 AMP 信息"""
        return {
            "enabled": self.config.enable_amp,
            "dtype": str(self.dtype),
            "has_scaler": self.scaler is not None,
            "scaler_scale": self.scaler.get_scale() if self.scaler else None,
            "hardware_support": {
                "cuda_available": torch.cuda.is_available(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "fp16_supported": torch.cuda.is_available()
            }
        }

class OptimizedDataLoader:
    """優化的數據加載器"""
    
    @staticmethod
    def create_optimized_loader(
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        sampler=None
    ):
        """創建優化的數據加載器"""
        if num_workers is None:
            num_workers = min(4, torch.get_num_threads())
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True  # 避免批次大小不一致
        )

def to_device_optimized(
    batch: Union[torch.Tensor, Dict[str, Any]],
    device: torch.device,
    memory_format: Optional[torch.memory_format] = None
) -> Union[torch.Tensor, Dict[str, Any]]:
    """優化的設備轉移"""
    if isinstance(batch, dict):
        return {
            k: v.to(device, non_blocking=True, memory_format=memory_format) 
            if torch.is_tensor(v) else v 
            for k, v in batch.items()
        }
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True, memory_format=memory_format)
    else:
        return batch

def create_amp_config(
    enable_amp: bool = True,
    dtype: str = "auto",
    prefer_bf16: bool = True,
    **kwargs
) -> AMPConfig:
    """創建 AMP 配置"""
    return AMPConfig(
        enable_amp=enable_amp,
        dtype=dtype,
        prefer_bf16=prefer_bf16
    )

class TrainingLoop:
    """優化的訓練循環"""
    
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                 amp_manager: AMPManager, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.amp_manager = amp_manager
        self.device = device
        
    def train_step(self, batch: Dict[str, Any], loss_fn) -> float:
        """單步訓練"""
        # 轉移數據到設備
        batch = to_device_optimized(batch, self.device)
        
        # 前向傳播
        with self.amp_manager.autocast():
            outputs = self.model(**batch)
            loss = loss_fn(outputs, batch)
        
        # 反向傳播
        self.optimizer.zero_grad(set_to_none=True)
        scaled_loss = self.amp_manager.scale_loss(loss)
        scaled_loss.backward()
        self.amp_manager.step_optimizer(self.optimizer)
        
        return loss.item()
    
    def train_epoch(self, dataloader, loss_fn) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch, loss_fn)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

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
    
    # 測試 AMP
    model = TestModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp_config = create_amp_config()
    amp_manager = AMPManager(amp_config)
    
    print("AMP 測試完成")
    print(f"AMP 信息: {amp_manager.get_amp_info()}")
