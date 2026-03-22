#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA加速優化器
實現GPU加速、混合精度訓練、內存優化等高性能計算技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU配置"""
    # 設備設置
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    gpu_memory_fraction: float = 0.9  # GPU內存使用比例
    
    # 混合精度
    use_amp: bool = True  # 自動混合精度
    amp_dtype: str = 'float16'  # 'float16', 'bfloat16'
    use_grad_scaler: bool = True
    
    # 內存優化
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True
    empty_cache_frequency: int = 100  # 每N步清空緩存
    
    # 並行優化
    use_compile: bool = True  # torch.compile
    compile_mode: str = 'max-autotune'  # 'default', 'reduce-overhead', 'max-autotune'
    use_channels_last: bool = True  # 使用channels_last內存格式
    
    # 數據加載優化
    pin_memory: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2

class GPUManager:
    """GPU管理器"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = self._setup_device()
        self.scaler = None
        self._setup_amp()
        
        logger.info(f"GPU Manager initialized on device: {self.device}")
        logger.info(f"GPU Memory: {self._get_gpu_memory_info()}")
    
    def _setup_device(self) -> torch.device:
        """設置設備"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # 設置GPU內存使用比例
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            else:
                device = torch.device('cpu')
                logger.warning("CUDA not available, using CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_amp(self):
        """設置自動混合精度"""
        if self.config.use_amp and self.device.type == 'cuda':
            if self.config.amp_dtype == 'float16':
                self.amp_dtype = torch.float16
            elif self.config.amp_dtype == 'bfloat16':
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            
            # 檢查是否支持bfloat16
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning("bfloat16 not supported, falling back to float16")
                self.amp_dtype = torch.float16
            
            # 設置梯度縮放器
            if self.config.use_grad_scaler and self.amp_dtype == torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()
            
            logger.info(f"AMP enabled with dtype: {self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            logger.info("AMP disabled, using float32")
    
    def _get_gpu_memory_info(self) -> str:
        """獲取GPU內存信息"""
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            cached_memory = torch.cuda.memory_reserved(0) / 1024**3
            return f"Total: {total_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB"
        else:
            return "CPU mode"
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """優化模型"""
        # 移動到GPU
        model = model.to(self.device)
        
        # 使用channels_last內存格式
        if self.config.use_channels_last:
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Model converted to channels_last format")
            except Exception as e:
                logger.warning(f"Failed to convert to channels_last: {e}")
        
        # 啟用梯度檢查點
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        # 編譯模型
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info(f"Model compiled with mode: {self.config.compile_mode}")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    @contextmanager
    def autocast(self):
        """自動混合精度上下文"""
        if self.config.use_amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """縮放損失"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def scale_backward(self, loss: torch.Tensor):
        """縮放反向傳播"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """優化器步驟"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def empty_cache(self):
        """清空GPU緩存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """獲取內存統計"""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(0) / 1024**3,
                'reserved': torch.cuda.memory_reserved(0) / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated(0) / 1024**3,
                'max_reserved': torch.cuda.max_memory_reserved(0) / 1024**3
            }
        else:
            return {}

class DataLoaderOptimizer:
    """數據加載器優化器"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
    
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """優化數據加載器"""
        # 注意：不能修改已初始化的DataLoader屬性
        # 返回原始dataloader，優化應該在創建時進行
        return dataloader
    
    def create_optimized_dataloader(self, dataset, batch_size: int, 
                                  shuffle: bool = True) -> torch.utils.data.DataLoader:
        """創建優化的數據加載器"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )

class MemoryOptimizer:
    """內存優化器"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.step_count = 0
    
    def optimize_training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """優化訓練步驟"""
        self.step_count += 1
        
        # 定期清空緩存
        if self.step_count % self.config.empty_cache_frequency == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def clear_gradients(self, model: nn.Module):
        """清空梯度"""
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
    
    def set_grad_to_none(self, optimizer: torch.optim.Optimizer):
        """設置梯度為None（更高效）"""
        optimizer.zero_grad(set_to_none=True)

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timers = {}
        self.memory_stats = []
    
    def start_timer(self, name: str):
        """開始計時"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """結束計時"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        return 0.0
    
    def record_memory_stats(self, gpu_manager: GPUManager):
        """記錄內存統計"""
        stats = gpu_manager.get_memory_stats()
        stats['timestamp'] = time.time()
        self.memory_stats.append(stats)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """獲取性能報告"""
        if not self.memory_stats:
            return {}
        
        latest_stats = self.memory_stats[-1]
        max_allocated = max(stats['allocated'] for stats in self.memory_stats)
        max_reserved = max(stats['reserved'] for stats in self.memory_stats)
        
        return {
            'current_allocated': latest_stats['allocated'],
            'current_reserved': latest_stats['reserved'],
            'max_allocated': max_allocated,
            'max_reserved': max_reserved,
            'memory_efficiency': latest_stats['allocated'] / max(latest_stats['reserved'], 1e-6)
        }

class OptimizedTrainer:
    """優化訓練器"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.gpu_manager = GPUManager(config)
        self.data_optimizer = DataLoaderOptimizer(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.profiler = PerformanceProfiler()
        
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def setup_training(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                      criterion: nn.Module):
        """設置訓練"""
        # 優化模型
        self.model = self.gpu_manager.optimize_model(model)
        self.optimizer = optimizer
        self.criterion = criterion
        
        logger.info("Training setup completed with GPU optimizations")
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """訓練一個epoch"""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_training first.")
        
        # 優化數據加載器
        train_loader = self.data_optimizer.optimize_dataloader(train_loader)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.profiler.start_timer('epoch')
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 移動數據到GPU
            data = data.to(self.gpu_manager.device, non_blocking=True)
            target = target.to(self.gpu_manager.device, non_blocking=True)
            
            # 使用channels_last格式
            if self.config.use_channels_last and len(data.shape) == 4:
                data = data.to(memory_format=torch.channels_last)
            
            # 清空梯度
            self.memory_optimizer.set_grad_to_none(self.optimizer)
            
            # 前向傳播（使用自動混合精度）
            with self.gpu_manager.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # 反向傳播
            self.gpu_manager.scale_backward(loss)
            self.gpu_manager.step_optimizer(self.optimizer)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 內存優化
            self.memory_optimizer.optimize_training_step(self.model, self.optimizer)
            
            # 記錄內存統計
            if batch_idx % 50 == 0:
                self.profiler.record_memory_stats(self.gpu_manager)
        
        epoch_time = self.profiler.end_timer('epoch')
        avg_loss = total_loss / num_batches
        
        # 獲取性能報告
        performance_report = self.profiler.get_performance_report()
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'batches_per_second': num_batches / epoch_time,
            **performance_report
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """驗證"""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_training first.")
        
        val_loader = self.data_optimizer.optimize_dataloader(val_loader)
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.gpu_manager.device, non_blocking=True)
                target = target.to(self.gpu_manager.device, non_blocking=True)
                
                if self.config.use_channels_last and len(data.shape) == 4:
                    data = data.to(memory_format=torch.channels_last)
                
                with self.gpu_manager.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches
        }
    
    def benchmark(self, model: nn.Module, input_shape: Tuple[int, ...], 
                 num_iterations: int = 100) -> Dict[str, float]:
        """性能基準測試"""
        model = self.gpu_manager.optimize_model(model)
        model.eval()
        
        # 創建測試數據
        test_input = torch.randn(1, *input_shape).to(self.gpu_manager.device)
        if self.config.use_channels_last and len(test_input.shape) == 4:
            test_input = test_input.to(memory_format=torch.channels_last)
        
        # 預熱
        with torch.no_grad():
            for _ in range(10):
                with self.gpu_manager.autocast():
                    _ = model(test_input)
        
        # 基準測試
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                with self.gpu_manager.autocast():
                    _ = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time
        
        return {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'throughput_fps': throughput,
            'memory_stats': self.gpu_manager.get_memory_stats()
        }

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = GPUConfig(
        device='auto',
        use_amp=True,
        amp_dtype='float16',
        use_compile=True,
        compile_mode='max-autotune',
        use_channels_last=True
    )
    
    # 創建優化訓練器
    trainer = OptimizedTrainer(config)
    
    # 創建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 32 * 32, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool2d(x, (32, 32))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 設置訓練
    trainer.setup_training(model, optimizer, criterion)
    
    # 創建示例數據
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 3, 64, 64),
        torch.randint(0, 10, (1000,))
    )
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 訓練一個epoch
    results = trainer.train_epoch(train_loader)
    print("Training results:", results)
    
    # 性能基準測試
    benchmark_results = trainer.benchmark(model, (3, 64, 64))
    print("Benchmark results:", benchmark_results)
