#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
並行計算引擎
實現多進程、多線程、異步處理等並行計算優化技術
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import queue
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
import psutil
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """並行計算配置"""
    # 多進程配置
    num_processes: int = 4
    use_multiprocessing: bool = True
    
    # 多線程配置
    num_threads: int = 8
    use_multithreading: bool = True
    
    # 異步配置
    use_async: bool = True
    max_concurrent_tasks: int = 10
    
    # 分散式配置
    use_distributed: bool = False
    backend: str = 'nccl'  # 'nccl', 'gloo'
    init_method: str = 'env://'
    
    # 數據並行配置
    use_data_parallel: bool = True
    device_ids: List[int] = None
    
    # 性能配置
    batch_size_per_process: int = 32
    prefetch_factor: int = 2
    pin_memory: bool = True

class ProcessManager:
    """進程管理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.process_pool = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def start_process_pool(self):
        """啟動進程池"""
        if self.config.use_multiprocessing:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.num_processes)
            logger.info(f"Process pool started with {self.config.num_processes} workers")
    
    def stop_process_pool(self):
        """停止進程池"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("Process pool stopped")
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """提交任務到進程池"""
        if self.process_pool:
            return self.process_pool.submit(func, *args, **kwargs)
        else:
            # 如果沒有進程池，直接執行
            return func(*args, **kwargs)
    
    def map_tasks(self, func: Callable, data_list: List[Any]) -> List[Any]:
        """並行處理數據列表"""
        # 使用線程池而不是進程池來避免pickle問題
        if len(data_list) > 1:
            with ThreadPoolExecutor(max_workers=min(len(data_list), 4)) as executor:
                results = list(executor.map(func, data_list))
        else:
            results = [func(data) for data in data_list]
        
        return results

class ThreadManager:
    """線程管理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.thread_pool = None
        
    def start_thread_pool(self):
        """啟動線程池"""
        if self.config.use_multithreading:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
            logger.info(f"Thread pool started with {self.config.num_threads} workers")
    
    def stop_thread_pool(self):
        """停止線程池"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool stopped")
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """提交任務到線程池"""
        if self.thread_pool:
            return self.thread_pool.submit(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def map_tasks(self, func: Callable, data_list: List[Any]) -> List[Any]:
        """並行處理數據列表"""
        if self.thread_pool and len(data_list) > 1:
            results = list(self.thread_pool.map(func, data_list))
        else:
            results = [func(data) for data in data_list]
        
        return results

class AsyncManager:
    """異步管理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
    
    async def async_task(self, func: Callable, *args, **kwargs):
        """異步任務執行"""
        async with self.semaphore:
            # 在線程池中執行CPU密集型任務
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def async_map(self, func: Callable, data_list: List[Any]) -> List[Any]:
        """異步並行處理"""
        tasks = [self.async_task(func, data) for data in data_list]
        results = await asyncio.gather(*tasks)
        return results
    
    async def async_batch_process(self, func: Callable, data_list: List[Any], 
                                batch_size: int = None) -> List[Any]:
        """異步批處理"""
        if batch_size is None:
            batch_size = self.config.max_concurrent_tasks
        
        results = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_results = await self.async_map(func, batch)
            results.extend(batch_results)
        
        return results

class DistributedManager:
    """分散式管理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.initialized = False
        self.rank = 0
        self.world_size = 1
    
    def init_distributed(self):
        """初始化分散式環境"""
        if self.config.use_distributed:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    rank=self.rank,
                    world_size=self.world_size
                )
                
                self.initialized = True
                logger.info(f"Distributed initialized: rank {self.rank}/{self.world_size}")
            else:
                logger.warning("Distributed environment variables not found")
    
    def cleanup_distributed(self):
        """清理分散式環境"""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            logger.info("Distributed environment cleaned up")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """包裝模型用於分散式訓練"""
        if self.initialized:
            model = model.cuda()
            model = DDP(model, device_ids=[self.rank])
            logger.info("Model wrapped with DDP")
        elif self.config.use_data_parallel and torch.cuda.device_count() > 1:
            device_ids = self.config.device_ids or list(range(torch.cuda.device_count()))
            model = DP(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel on devices {device_ids}")
        
        return model

class DataParallelProcessor:
    """數據並行處理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.process_manager = ProcessManager(config)
        self.thread_manager = ThreadManager(config)
        self.async_manager = AsyncManager(config)
    
    def start(self):
        """啟動所有管理器"""
        self.process_manager.start_process_pool()
        self.thread_manager.start_thread_pool()
    
    def stop(self):
        """停止所有管理器"""
        self.process_manager.stop_process_pool()
        self.thread_manager.stop_thread_pool()
    
    def parallel_data_processing(self, data_list: List[Any], 
                               process_func: Callable,
                               use_async: bool = False) -> List[Any]:
        """並行數據處理"""
        if use_async and self.config.use_async:
            # 使用異步處理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self.async_manager.async_map(process_func, data_list)
                )
            finally:
                loop.close()
        else:
            # 使用多進程處理
            results = self.process_manager.map_tasks(process_func, data_list)
        
        return results
    
    def batch_parallel_processing(self, data_list: List[Any],
                                process_func: Callable,
                                batch_size: int = None) -> List[Any]:
        """批並行處理"""
        if batch_size is None:
            batch_size = self.config.batch_size_per_process
        
        results = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            # 使用線程池處理批次
            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                batch_results = list(executor.map(process_func, batch))
            results.extend(batch_results)
        
        return results

class ModelParallelTrainer:
    """模型並行訓練器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.distributed_manager = DistributedManager(config)
        self.data_processor = DataParallelProcessor(config)
    
    def setup_training(self, model: nn.Module, 
                      train_loader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      criterion: nn.Module):
        """設置並行訓練"""
        # 初始化分散式環境
        self.distributed_manager.init_distributed()
        
        # 包裝模型
        self.model = self.distributed_manager.wrap_model(model)
        self.optimizer = optimizer
        self.criterion = criterion
        
        # 設置數據加載器
        if self.distributed_manager.initialized:
            train_loader = self._setup_distributed_dataloader(train_loader)
        
        self.train_loader = train_loader
        
        # 啟動數據處理器
        self.data_processor.start()
        
        logger.info("Parallel training setup completed")
    
    def _setup_distributed_dataloader(self, dataloader: torch.utils.data.DataLoader):
        """設置分散式數據加載器"""
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.distributed_manager.world_size,
            rank=self.distributed_manager.rank
        )
        
        return torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """並行訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 設置epoch（用於分散式採樣器）
        if self.distributed_manager.initialized:
            self.train_loader.sampler.set_epoch(0)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 移動數據到GPU
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            # 前向傳播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def cleanup(self):
        """清理資源"""
        self.data_processor.stop()
        self.distributed_manager.cleanup_distributed()

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'processing_times': []
        }
    
    def start_monitoring(self):
        """開始監控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # 內存使用率
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory.percent)
            
            # GPU使用率（如果可用）
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.metrics['gpu_usage'].append(gpu_memory * 100)
            
            time.sleep(1)
    
    def record_processing_time(self, start_time: float, end_time: float):
        """記錄處理時間"""
        processing_time = end_time - start_time
        self.metrics['processing_times'].append(processing_time)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """獲取性能報告"""
        if not self.metrics['cpu_usage']:
            return {}
        
        return {
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']),
            'max_cpu_usage': np.max(self.metrics['cpu_usage']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'max_memory_usage': np.max(self.metrics['memory_usage']),
            'avg_gpu_usage': np.mean(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0,
            'max_gpu_usage': np.max(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0,
            'avg_processing_time': np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0,
            'total_processing_time': np.sum(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        }

class ParallelEngine:
    """並行計算引擎"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.data_processor = DataParallelProcessor(config)
        self.model_trainer = ModelParallelTrainer(config)
        self.performance_monitor = PerformanceMonitor()
        
        self.is_initialized = False
    
    def initialize(self):
        """初始化引擎"""
        if not self.is_initialized:
            self.data_processor.start()
            self.performance_monitor.start_monitoring()
            self.is_initialized = True
            logger.info("Parallel engine initialized")
    
    def shutdown(self):
        """關閉引擎"""
        if self.is_initialized:
            self.data_processor.stop()
            self.model_trainer.cleanup()
            self.performance_monitor.stop_monitoring()
            self.is_initialized = False
            logger.info("Parallel engine shutdown")
    
    def parallel_inference(self, model: nn.Module, data_list: List[Any],
                          batch_size: int = None) -> List[Any]:
        """並行推理"""
        if batch_size is None:
            batch_size = self.config.batch_size_per_process
        
        def inference_func(data):
            model.eval()
            with torch.no_grad():
                if isinstance(data, torch.Tensor):
                    return model(data)
                else:
                    return model(torch.tensor(data))
        
        return self.data_processor.batch_parallel_processing(
            data_list, inference_func, batch_size
        )
    
    def parallel_training(self, model: nn.Module, 
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         criterion: nn.Module,
                         num_epochs: int = 1) -> Dict[str, List[float]]:
        """並行訓練"""
        self.model_trainer.setup_training(model, train_loader, optimizer, criterion)
        
        training_history = {
            'loss': [],
            'epoch_times': []
        }
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            epoch_results = self.model_trainer.train_epoch()
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            training_history['loss'].append(epoch_results['loss'])
            training_history['epoch_times'].append(epoch_time)
            
            self.performance_monitor.record_processing_time(start_time, end_time)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, "
                       f"Loss: {epoch_results['loss']:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        
        return training_history
    
    def benchmark_parallel_performance(self, func: Callable, data_list: List[Any],
                                     num_iterations: int = 5) -> Dict[str, float]:
        """基準測試並行性能"""
        logger.info("Benchmarking parallel performance...")
        
        # 串行執行
        start_time = time.time()
        for _ in range(num_iterations):
            serial_results = [func(data) for data in data_list]
        serial_time = time.time() - start_time
        
        # 並行執行
        start_time = time.time()
        for _ in range(num_iterations):
            parallel_results = self.data_processor.parallel_data_processing(
                data_list, func
            )
        parallel_time = time.time() - start_time
        
        # 計算加速比
        speedup = serial_time / parallel_time
        
        return {
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': speedup / self.config.num_processes
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'config': {
                'num_processes': self.config.num_processes,
                'num_threads': self.config.num_threads,
                'use_distributed': self.config.use_distributed
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """獲取性能報告"""
        return {
            'system_info': self.get_system_info(),
            'performance_metrics': self.performance_monitor.get_performance_report()
        }

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = ParallelConfig(
        num_processes=4,
        num_threads=8,
        use_async=True,
        max_concurrent_tasks=10
    )
    
    # 創建並行引擎
    engine = ParallelEngine(config)
    engine.initialize()
    
    # 示例函數
    def process_data(data):
        # 模擬數據處理
        time.sleep(0.1)
        return data * 2
    
    # 測試數據
    test_data = list(range(100))
    
    # 基準測試
    benchmark_results = engine.benchmark_parallel_performance(process_data, test_data)
    print("Benchmark Results:")
    print(f"Serial time: {benchmark_results['serial_time']:.2f}s")
    print(f"Parallel time: {benchmark_results['parallel_time']:.2f}s")
    print(f"Speedup: {benchmark_results['speedup']:.2f}x")
    print(f"Efficiency: {benchmark_results['efficiency']:.2f}")
    
    # 獲取性能報告
    performance_report = engine.get_performance_report()
    print("\nPerformance Report:")
    print(performance_report)
    
    # 關閉引擎
    engine.shutdown()
