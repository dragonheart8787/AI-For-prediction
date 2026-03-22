#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
內存優化器
實現梯度檢查點、動態批處理、內存池等內存優化技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import psutil
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """內存配置"""
    # 梯度檢查點
    use_gradient_checkpointing: bool = True
    checkpoint_frequency: int = 4  # 每N層設置一個檢查點
    
    # 動態批處理
    use_dynamic_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 64
    memory_threshold: float = 0.8  # 內存使用閾值
    
    # 內存池
    use_memory_pool: bool = True
    pool_size: int = 1000  # 內存池大小
    pool_growth_factor: float = 1.5
    
    # 內存監控
    monitor_memory: bool = True
    memory_check_interval: float = 1.0  # 秒
    auto_cleanup: bool = True
    
    # 優化策略
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4

class MemoryMonitor:
    """內存監控器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_history = deque(maxlen=1000)
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """開始監控"""
        if self.config.monitor_memory:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            try:
                # 系統內存
                system_memory = psutil.virtual_memory()
                
                # GPU內存
                gpu_memory = {}
                if torch.cuda.is_available():
                    gpu_memory = {
                        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                        'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
                        'max_reserved': torch.cuda.max_memory_reserved() / 1024**3     # GB
                    }
                
                memory_info = {
                    'timestamp': time.time(),
                    'system': {
                        'total': system_memory.total / 1024**3,  # GB
                        'available': system_memory.available / 1024**3,  # GB
                        'used': system_memory.used / 1024**3,  # GB
                        'percent': system_memory.percent
                    },
                    'gpu': gpu_memory
                }
                
                self.memory_history.append(memory_info)
                
                # 更新峰值內存
                current_memory = system_memory.used / 1024**3
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # 自動清理
                if self.config.auto_cleanup and system_memory.percent > 90:
                    self._auto_cleanup()
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
            
            time.sleep(self.config.memory_check_interval)
    
    def _auto_cleanup(self):
        """自動清理內存"""
        logger.info("Auto cleanup triggered")
        
        # 清理Python垃圾
        gc.collect()
        
        # 清理GPU緩存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Auto cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取內存統計"""
        if not self.memory_history:
            return {}
        
        latest = self.memory_history[-1]
        
        return {
            'current': latest,
            'peak_memory': self.peak_memory,
            'history_length': len(self.memory_history)
        }
    
    def get_memory_usage(self) -> float:
        """獲取當前內存使用率"""
        if not self.memory_history:
            return 0.0
        
        return self.memory_history[-1]['system']['percent']

class GradientCheckpointer:
    """梯度檢查點器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.checkpoint_count = 0
    
    def enable_checkpointing(self, model: nn.Module):
        """啟用梯度檢查點"""
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
    
    def disable_checkpointing(self, model: nn.Module):
        """禁用梯度檢查點"""
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
    
    @contextmanager
    def checkpoint_forward(self, model: nn.Module, *args, **kwargs):
        """檢查點前向傳播
        注意：torch.utils.checkpoint.checkpoint 返回 tensor，不應在 with 內直接 yield 該值作為 context 變數。
        這裡提供一個上下文，實際前向在外部執行，或改用函式方式。
        """
        yield  # 僅提供上下文，不直接執行

    def forward_with_checkpoint(self, fn: Callable, *args, **kwargs):
        """使用 torch.utils.checkpoint 對函式前向進行檢查點包裝"""
        if self.config.use_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs, use_reentrant=False)
        return fn(*args, **kwargs)

class DynamicBatcher:
    """動態批處理器"""
    
    def __init__(self, config: MemoryConfig, memory_monitor: MemoryMonitor):
        self.config = config
        self.memory_monitor = memory_monitor
        self.current_batch_size = config.min_batch_size
        self.batch_history = deque(maxlen=100)
        
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """獲取最優批大小"""
        if not self.config.use_dynamic_batching:
            return base_batch_size
        
        # 獲取當前內存使用率
        memory_usage = self.memory_monitor.get_memory_usage()
        
        # 根據內存使用率調整批大小
        if memory_usage > self.config.memory_threshold * 100:
            # 內存使用率高，減少批大小
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif memory_usage < self.config.memory_threshold * 50:
            # 內存使用率低，增加批大小
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        
        # 記錄批大小歷史
        self.batch_history.append({
            'batch_size': self.current_batch_size,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        })
        
        return self.current_batch_size
    
    def create_dynamic_dataloader(self, dataset, base_batch_size: int = 32):
        """創建動態數據加載器"""
        class DynamicDataLoader:
            def __init__(self, dataset, batcher, base_batch_size):
                self.dataset = dataset
                self.batcher = batcher
                self.base_batch_size = base_batch_size
                self.index = 0
            
            def __iter__(self):
                self.index = 0
                return self
            
            def __next__(self):
                if self.index >= len(self.dataset):
                    raise StopIteration
                
                # 獲取動態批大小
                batch_size = self.batcher.get_optimal_batch_size(self.base_batch_size)
                
                # 獲取批次數據
                end_index = min(self.index + batch_size, len(self.dataset))
                batch_data = [self.dataset[i] for i in range(self.index, end_index)]
                self.index = end_index
                
                # 將批次數據轉換為張量
                if batch_data:
                    if isinstance(batch_data[0], tuple):
                        # 多個張量
                        return tuple(torch.stack([item[i] for item in batch_data]) for i in range(len(batch_data[0])))
                    else:
                        # 單個張量
                        return torch.stack(batch_data)
                else:
                    raise StopIteration
        
        return DynamicDataLoader(dataset, self, base_batch_size)

class MemoryPool:
    """內存池"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pools = {}
        self.pool_locks = {}
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   device: torch.device = None) -> torch.Tensor:
        """從內存池獲取張量"""
        if not self.config.use_memory_pool:
            return torch.empty(shape, dtype=dtype, device=device)
        
        # 創建池鍵
        pool_key = (shape, dtype, device)
        
        if pool_key not in self.pools:
            self.pools[pool_key] = queue.Queue()
            self.pool_locks[pool_key] = threading.Lock()
        
        with self.pool_locks[pool_key]:
            pool = self.pools[pool_key]
            
            if not pool.empty():
                # 從池中獲取張量
                tensor = pool.get()
                # 重置張量
                tensor.zero_()
                return tensor
            else:
                # 池為空，創建新張量
                return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """將張量返回到內存池"""
        if not self.config.use_memory_pool:
            return
        
        # 創建池鍵
        pool_key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        
        if pool_key in self.pools:
            with self.pool_locks[pool_key]:
                pool = self.pools[pool_key]
                
                if pool.qsize() < self.config.pool_size:
                    # 池未滿，返回張量
                    pool.put(tensor)
    
    def clear_pools(self):
        """清空所有內存池"""
        for pool in self.pools.values():
            while not pool.empty():
                try:
                    pool.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Memory pools cleared")

class GradientAccumulator:
    """梯度累積器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.accumulation_steps = config.accumulation_steps
        self.current_step = 0
    
    def should_accumulate(self) -> bool:
        """是否應該累積梯度"""
        return self.config.use_gradient_accumulation and self.current_step < self.accumulation_steps - 1
    
    def should_step(self) -> bool:
        """是否應該執行優化步驟"""
        return self.current_step >= self.accumulation_steps - 1
    
    def step(self, optimizer: torch.optim.Optimizer):
        """執行優化步驟"""
        if self.should_step():
            optimizer.step()
            optimizer.zero_grad()
            self.current_step = 0
        else:
            self.current_step += 1
    
    def zero_grad(self, optimizer: torch.optim.Optimizer):
        """清零梯度"""
        if self.current_step == 0:
            optimizer.zero_grad()

class MemoryOptimizer:
    """內存優化器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config)
        self.gradient_checkpointer = GradientCheckpointer(config)
        self.dynamic_batcher = DynamicBatcher(config, self.memory_monitor)
        self.memory_pool = MemoryPool(config)
        self.gradient_accumulator = GradientAccumulator(config)
        
        # 內存統計
        self.optimization_stats = {
            'memory_saved': 0,
            'checkpoints_used': 0,
            'dynamic_batches': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
    
    def start_optimization(self):
        """開始內存優化"""
        self.memory_monitor.start_monitoring()
        logger.info("Memory optimization started")
    
    def stop_optimization(self):
        """停止內存優化"""
        self.memory_monitor.stop_monitoring()
        self.memory_pool.clear_pools()
        logger.info("Memory optimization stopped")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """優化模型內存使用"""
        # 啟用梯度檢查點
        self.gradient_checkpointer.enable_checkpointing(model)
        
        # 設置混合精度（不強制更改模型 dtype，改為在步驟中匹配輸入）
        # 保留模型原始 dtype，避免與輸入衝突
        
        logger.info("Model memory optimization applied")
        return model
    
    def optimize_training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                              data, target, criterion: nn.Module) -> Dict[str, Any]:
        """優化訓練步驟"""
        step_stats = {
            'memory_before': self.memory_monitor.get_memory_usage(),
            'batch_size': len(data) if hasattr(data, '__len__') else 1,
            'checkpoint_used': False
        }
        
        # 梯度累積
        self.gradient_accumulator.zero_grad(optimizer)
        
        # 對齊輸入 device 與 dtype
        try:
            first_param = next(model.parameters())
            target_device = first_param.device
            target_dtype = first_param.dtype
        except StopIteration:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            target_dtype = data.dtype if hasattr(data, 'dtype') else torch.float32

        if isinstance(data, torch.Tensor):
            data = data.to(device=target_device, dtype=target_dtype)
            # 若需要梯度，確保 requires_grad=True
            if model.training:
                data.requires_grad_(True)
        if isinstance(target, torch.Tensor):
            target = target.to(device=target_device)

        # 前向傳播（使用檢查點）
        if self.gradient_checkpointer.config.use_gradient_checkpointing:
            step_stats['checkpoint_used'] = True
            self.optimization_stats['checkpoints_used'] += 1
            output = self.gradient_checkpointer.forward_with_checkpoint(model, data)
        else:
            output = model(data)
        
        # 計算損失
        loss = criterion(output, target)
        
        # 梯度累積
        if self.gradient_accumulator.should_accumulate():
            loss = loss / self.gradient_accumulator.accumulation_steps
        
        # 反向傳播
        loss.backward()
        
        # 優化步驟與梯度管理
        if self.gradient_accumulator.should_step():
            self.gradient_accumulator.step(optimizer)
        else:
            # 尚未到達步進時只累積步數
            self.gradient_accumulator.current_step += 1
        
        step_stats['memory_after'] = self.memory_monitor.get_memory_usage()
        step_stats['memory_saved'] = step_stats['memory_before'] - step_stats['memory_after']
        
        return step_stats
    
    def get_tensor_from_pool(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                           device: torch.device = None) -> torch.Tensor:
        """從內存池獲取張量"""
        tensor = self.memory_pool.get_tensor(shape, dtype, device)
        
        if tensor is not None:
            self.optimization_stats['pool_hits'] += 1
        else:
            self.optimization_stats['pool_misses'] += 1
        
        return tensor
    
    def return_tensor_to_pool(self, tensor: torch.Tensor):
        """將張量返回到內存池"""
        self.memory_pool.return_tensor(tensor)
    
    def create_optimized_dataloader(self, dataset, base_batch_size: int = 32):
        """創建優化的數據加載器"""
        if self.config.use_dynamic_batching:
            return self.dynamic_batcher.create_dynamic_dataloader(dataset, base_batch_size)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=base_batch_size, shuffle=True)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """獲取優化報告"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        return {
            'memory_stats': memory_stats,
            'optimization_stats': self.optimization_stats,
            'config': {
                'gradient_checkpointing': self.config.use_gradient_checkpointing,
                'dynamic_batching': self.config.use_dynamic_batching,
                'memory_pool': self.config.use_memory_pool,
                'gradient_accumulation': self.config.use_gradient_accumulation
            }
        }
    
    def benchmark_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...],
                             num_iterations: int = 100) -> Dict[str, Any]:
        """基準測試內存使用"""
        model.eval()
        
        # 記錄初始內存
        initial_memory = self.memory_monitor.get_memory_usage()
        
        # 測試不同批大小
        batch_sizes = [1, 4, 8, 16, 32, 64]
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > self.config.max_batch_size:
                continue
            
            # 創建測試數據
            test_input = torch.randn(batch_size, *input_shape)
            
            # 測試內存使用
            memory_before = self.memory_monitor.get_memory_usage()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(test_input)
            
            memory_after = self.memory_monitor.get_memory_usage()
            
            results[batch_size] = {
                'memory_usage': memory_after - memory_before,
                'peak_memory': self.memory_monitor.peak_memory
            }
        
        return {
            'initial_memory': initial_memory,
            'batch_size_results': results,
            'recommended_batch_size': min(results.keys(), key=lambda x: results[x]['memory_usage'])
        }

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = MemoryConfig(
        use_gradient_checkpointing=True,
        use_dynamic_batching=True,
        use_memory_pool=True,
        use_gradient_accumulation=True,
        accumulation_steps=4
    )
    
    # 創建內存優化器
    optimizer = MemoryOptimizer(config)
    optimizer.start_optimization()
    
    # 創建示例模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 10)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = TestModel()
    optimizer_model = optimizer.optimize_model(model)
    
    # 創建測試數據
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 100),
        torch.randint(0, 10, (1000,))
    )
    
    # 創建優化的數據加載器
    dataloader = optimizer.create_optimized_dataloader(dataset, base_batch_size=32)
    
    # 設置訓練
    train_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 訓練幾個批次
    model.train()
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:  # 只訓練5個批次
            break
        
        step_stats = optimizer.optimize_training_step(
            model, train_optimizer, data, target, criterion
        )
        
        print(f"Batch {i+1}: Memory saved: {step_stats['memory_saved']:.2f}%")
    
    # 基準測試
    benchmark_results = optimizer.benchmark_memory_usage(model, (100,))
    print(f"Recommended batch size: {benchmark_results['recommended_batch_size']}")
    
    # 獲取優化報告
    report = optimizer.get_optimization_report()
    print("Optimization Report:")
    print(f"Checkpoints used: {report['optimization_stats']['checkpoints_used']}")
    print(f"Pool hits: {report['optimization_stats']['pool_hits']}")
    print(f"Pool misses: {report['optimization_stats']['pool_misses']}")
    
    # 停止優化
    optimizer.stop_optimization()
