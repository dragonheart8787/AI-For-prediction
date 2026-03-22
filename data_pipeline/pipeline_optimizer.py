#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
數據管道優化器
實現預取、緩存、壓縮等高效數據處理技術
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import gzip
import threading
import queue
import time
import hashlib
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """數據管道配置"""
    # 預取配置
    use_prefetch: bool = True
    prefetch_size: int = 4  # 預取批次數量
    prefetch_workers: int = 2  # 預取工作線程數
    
    # 緩存配置
    use_cache: bool = True
    cache_size: int = 1000  # 緩存項目數量
    cache_type: str = 'memory'  # 'memory', 'disk', 'hybrid'
    cache_compression: bool = True
    
    # 壓縮配置
    use_compression: bool = True
    compression_level: int = 6  # 1-9
    compression_algorithm: str = 'gzip'  # 'gzip', 'lz4', 'zstd'
    
    # 批處理配置
    use_batching: bool = True
    batch_size: int = 32
    dynamic_batching: bool = True
    max_batch_size: int = 128
    
    # 並行配置
    use_parallel_loading: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # 數據轉換配置
    use_data_augmentation: bool = True
    augmentation_probability: float = 0.5
    use_normalization: bool = True

class DataCache:
    """數據緩存"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.memory_cache = {}
        self.disk_cache_dir = Path("cache")
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0
        }
        self._lock = threading.Lock()
    
    def _get_cache_key(self, data_id: str) -> str:
        """生成緩存鍵"""
        return hashlib.md5(data_id.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """壓縮數據"""
        if not self.config.cache_compression:
            return pickle.dumps(data)
        
        if self.config.compression_algorithm == 'gzip':
            return gzip.compress(pickle.dumps(data), compresslevel=self.config.compression_level)
        else:
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """解壓縮數據"""
        if not self.config.cache_compression:
            return pickle.loads(compressed_data)
        
        if self.config.compression_algorithm == 'gzip':
            return pickle.loads(gzip.decompress(compressed_data))
        else:
            return pickle.loads(compressed_data)
    
    def get(self, data_id: str) -> Optional[Any]:
        """獲取緩存數據"""
        cache_key = self._get_cache_key(data_id)
        
        with self._lock:
            # 檢查內存緩存
            if cache_key in self.memory_cache:
                self.cache_stats['hits'] += 1
                return self.memory_cache[cache_key]
            
            # 檢查磁盤緩存
            if self.config.cache_type in ['disk', 'hybrid']:
                cache_file = self.disk_cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            compressed_data = f.read()
                        data = self._decompress_data(compressed_data)
                        
                        # 如果使用混合緩存，也存到內存
                        if self.config.cache_type == 'hybrid':
                            self._add_to_memory_cache(cache_key, data)
                        
                        self.cache_stats['hits'] += 1
                        return data
                    except Exception as e:
                        logger.warning(f"Failed to load from disk cache: {e}")
            
            self.cache_stats['misses'] += 1
            return None
    
    def put(self, data_id: str, data: Any):
        """存儲數據到緩存"""
        cache_key = self._get_cache_key(data_id)
        
        with self._lock:
            # 存儲到內存緩存
            if self.config.cache_type in ['memory', 'hybrid']:
                self._add_to_memory_cache(cache_key, data)
            
            # 存儲到磁盤緩存
            if self.config.cache_type in ['disk', 'hybrid']:
                try:
                    compressed_data = self._compress_data(data)
                    cache_file = self.disk_cache_dir / f"{cache_key}.cache"
                    with open(cache_file, 'wb') as f:
                        f.write(compressed_data)
                except Exception as e:
                    logger.warning(f"Failed to save to disk cache: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, data: Any):
        """添加到內存緩存"""
        # 檢查緩存大小限制
        if len(self.memory_cache) >= self.config.cache_size:
            # 移除最舊的項目
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = data
        self.cache_stats['size'] = len(self.memory_cache)
    
    def clear(self):
        """清空緩存"""
        with self._lock:
            self.memory_cache.clear()
            self.cache_stats['size'] = 0
            
            # 清空磁盤緩存
            if self.disk_cache_dir.exists():
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'size': self.cache_stats['size'],
            'max_size': self.config.cache_size
        }

class DataPrefetcher:
    """數據預取器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.prefetch_queue = queue.Queue(maxsize=config.prefetch_size)
        self.worker_threads = []
        self.stop_event = threading.Event()
        self.stats = {
            'prefetched': 0,
            'consumed': 0,
            'cache_hits': 0
        }
    
    def start_prefetching(self, data_loader: Iterator, cache: DataCache = None):
        """開始預取"""
        if not self.config.use_prefetch:
            return
        
        self.stop_event.clear()
        
        # 啟動工作線程（為每個工作線程建立各自的 iterator）
        for i in range(self.config.prefetch_workers):
            try:
                data_iter = iter(data_loader)
            except TypeError:
                # 若傳入的本身是 iterator，直接使用
                data_iter = data_loader
            worker = threading.Thread(
                target=self._prefetch_worker,
                args=(data_iter, cache),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Data prefetching started with {self.config.prefetch_workers} workers")
    
    def stop_prefetching(self):
        """停止預取"""
        self.stop_event.set()
        
        # 等待工作線程結束
        for worker in self.worker_threads:
            worker.join()
        
        self.worker_threads.clear()
        logger.info("Data prefetching stopped")
    
    def _prefetch_worker(self, data_loader: Iterator, cache: DataCache = None):
        """預取工作線程"""
        # 建立本地 iterator（若尚未是 iterator）
        try:
            data_iter = iter(data_loader)
        except TypeError:
            data_iter = data_loader

        while not self.stop_event.is_set():
            try:
                # 從數據加載器獲取數據
                batch = next(data_iter)
                
                # 檢查緩存
                if cache:
                    batch_id = str(hash(str(batch)))
                    cached_batch = cache.get(batch_id)
                    if cached_batch:
                        batch = cached_batch
                        self.stats['cache_hits'] += 1
                    else:
                        cache.put(batch_id, batch)
                
                # 將批次放入預取隊列
                self.prefetch_queue.put(batch, timeout=1)
                self.stats['prefetched'] += 1
                
            except StopIteration:
                break
            except queue.Full:
                time.sleep(0.01)  # 隊列滿了，稍等
            except Exception as e:
                logger.warning(f"Prefetch worker error: {e}")
    
    def get_batch(self) -> Any:
        """獲取預取的批次"""
        if not self.config.use_prefetch:
            return None
        
        try:
            batch = self.prefetch_queue.get(timeout=1)
            self.stats['consumed'] += 1
            return batch
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取預取統計"""
        return {
            'prefetched': self.stats['prefetched'],
            'consumed': self.stats['consumed'],
            'cache_hits': self.stats['cache_hits'],
            'queue_size': self.prefetch_queue.qsize()
        }

class DataAugmenter:
    """數據增強器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.augmentation_functions = {
            'noise': self._add_noise,
            'scale': self._scale_data,
            'rotate': self._rotate_data,
            'flip': self._flip_data
        }
    
    def augment(self, data: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """數據增強"""
        if not self.config.use_data_augmentation or np.random.random() > self.config.augmentation_probability:
            return data, target
        
        # 隨機選擇增強方法
        augmentation_methods = list(self.augmentation_functions.keys())
        selected_method = np.random.choice(augmentation_methods)
        
        # 應用增強
        augmented_data = self.augmentation_functions[selected_method](data)
        
        return augmented_data, target
    
    def _add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """添加噪聲"""
        noise = torch.randn_like(data) * 0.01
        return data + noise
    
    def _scale_data(self, data: torch.Tensor) -> torch.Tensor:
        """縮放數據"""
        scale_factor = np.random.uniform(0.9, 1.1)
        return data * scale_factor
    
    def _rotate_data(self, data: torch.Tensor) -> torch.Tensor:
        """旋轉數據（適用於2D數據）"""
        if len(data.shape) >= 2:
            angle = np.random.uniform(-10, 10)  # 度
            # 簡單的旋轉實現
            return torch.roll(data, int(angle), dims=-1)
        return data
    
    def _flip_data(self, data: torch.Tensor) -> torch.Tensor:
        """翻轉數據"""
        if len(data.shape) >= 2:
            return torch.flip(data, dims=[-1])
        return data

class DataNormalizer:
    """數據正規化器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = {}
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """擬合正規化參數"""
        if not self.config.use_normalization:
            return
        
        self.stats = {
            'mean': data.mean(dim=0),
            'std': data.std(dim=0) + 1e-8  # 避免除零
        }
        self.fitted = True
        logger.info("Data normalization parameters fitted")
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """轉換數據"""
        if not self.config.use_normalization or not self.fitted:
            return data
        
        return (data - self.stats['mean']) / self.stats['std']
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """逆轉換數據"""
        if not self.config.use_normalization or not self.fitted:
            return data
        
        return data * self.stats['std'] + self.stats['mean']

class PipelineOptimizer:
    """數據管道優化器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = DataCache(config)
        self.prefetcher = DataPrefetcher(config)
        self.augmenter = DataAugmenter(config)
        self.normalizer = DataNormalizer(config)
        
        self.pipeline_stats = {
            'total_batches': 0,
            'cache_hits': 0,
            'augmented_batches': 0,
            'processing_time': 0
        }
    
    def create_optimized_dataloader(self, dataset, batch_size: int = None) -> torch.utils.data.DataLoader:
        """創建優化的數據加載器"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Windows 平台避免多進程導致 pickling 問題
        import platform
        is_windows = platform.system().lower().startswith('win')
        num_workers = 0 if is_windows else (self.config.num_workers if self.config.use_parallel_loading else 0)
        persistent = (num_workers > 0)

        # 創建數據加載器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=persistent
        )
        
        return dataloader
    
    def optimize_dataset(self, dataset) -> 'OptimizedDataset':
        """優化數據集"""
        return OptimizedDataset(
            dataset,
            self.cache,
            self.augmenter,
            self.normalizer,
            self.config
        )
    
    def start_pipeline(self, dataloader: torch.utils.data.DataLoader):
        """啟動數據管道"""
        # 先擬合正規化器（避免與預取競爭 dataloader 資源）
        if self.config.use_normalization:
            try:
                sample_batch = next(iter(dataloader))
                sample_data = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
                # 兼容 list / ndarray
                if isinstance(sample_data, list):
                    sample_data = torch.stack(sample_data)
                if not isinstance(sample_data, torch.Tensor):
                    sample_data = torch.as_tensor(sample_data)
                self.normalizer.fit(sample_data)
            except Exception as e:
                logger.warning(f"Failed to fit normalizer on sample: {e}")

        # 再開始預取
        self.prefetcher.start_prefetching(dataloader, self.cache)
        logger.info("Data pipeline started")
    
    def stop_pipeline(self):
        """停止數據管道"""
        self.prefetcher.stop_prefetching()
        logger.info("Data pipeline stopped")
    
    def process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """處理批次數據"""
        start_time = time.time()
        
        data, target = batch
        
        # 數據增強
        data, target = self.augmenter.augment(data, target)
        if self.augmenter.config.use_data_augmentation:
            self.pipeline_stats['augmented_batches'] += 1
        
        # 數據正規化
        data = self.normalizer.transform(data)
        
        processing_time = time.time() - start_time
        self.pipeline_stats['processing_time'] += processing_time
        self.pipeline_stats['total_batches'] += 1
        
        return data, target
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """獲取管道統計"""
        cache_stats = self.cache.get_stats()
        prefetch_stats = self.prefetcher.get_stats()
        
        return {
            'pipeline_stats': self.pipeline_stats,
            'cache_stats': cache_stats,
            'prefetch_stats': prefetch_stats,
            'config': {
                'use_prefetch': self.config.use_prefetch,
                'use_cache': self.config.use_cache,
                'use_augmentation': self.config.use_data_augmentation,
                'use_normalization': self.config.use_normalization
            }
        }
    
    def benchmark_pipeline(self, dataset, num_batches: int = 100) -> Dict[str, Any]:
        """基準測試數據管道"""
        logger.info("Benchmarking data pipeline...")
        
        # 創建數據加載器
        dataloader = self.create_optimized_dataloader(dataset)
        
        # 啟動管道
        self.start_pipeline(dataloader)
        
        # 測試處理時間
        start_time = time.time()
        processed_batches = 0
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            processed_batch = self.process_batch(batch)
            processed_batches += 1
        
        total_time = time.time() - start_time
        
        # 停止管道
        self.stop_pipeline()
        
        # 計算性能指標
        batches_per_second = processed_batches / total_time
        avg_processing_time = self.pipeline_stats['processing_time'] / processed_batches
        
        return {
            'total_time': total_time,
            'processed_batches': processed_batches,
            'batches_per_second': batches_per_second,
            'avg_processing_time': avg_processing_time,
            'pipeline_stats': self.get_pipeline_stats()
        }

class OptimizedDataset:
    """優化數據集"""
    
    def __init__(self, base_dataset, cache: DataCache, augmenter: DataAugmenter,
                 normalizer: DataNormalizer, config: PipelineConfig):
        self.base_dataset = base_dataset
        self.cache = cache
        self.augmenter = augmenter
        self.normalizer = normalizer
        self.config = config
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 檢查緩存
        cache_key = f"item_{idx}"
        cached_item = self.cache.get(cache_key)
        
        if cached_item:
            return cached_item
        
        # 從基礎數據集獲取數據
        item = self.base_dataset[idx]
        
        # 數據增強
        if isinstance(item, tuple):
            data, target = item
            data, target = self.augmenter.augment(data, target)
            item = (data, target)
        else:
            item = self.augmenter.augment(item)
        
        # 數據正規化
        if isinstance(item, tuple):
            data, target = item
            data = self.normalizer.transform(data)
            item = (data, target)
        else:
            item = self.normalizer.transform(item)
        
        # 存儲到緩存
        self.cache.put(cache_key, item)
        
        return item

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = PipelineConfig(
        use_prefetch=True,
        use_cache=True,
        use_compression=True,
        use_data_augmentation=True,
        use_normalization=True
    )
    
    # 創建管道優化器
    optimizer = PipelineOptimizer(config)
    
    # 創建示例數據集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.data = torch.randn(size, 100)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    dataset = SimpleDataset(1000)
    
    # 基準測試
    benchmark_results = optimizer.benchmark_pipeline(dataset, num_batches=50)
    
    print("Pipeline Benchmark Results:")
    print(f"Total time: {benchmark_results['total_time']:.2f}s")
    print(f"Batches per second: {benchmark_results['batches_per_second']:.2f}")
    print(f"Average processing time: {benchmark_results['avg_processing_time']:.4f}s")
    
    # 獲取統計信息
    stats = optimizer.get_pipeline_stats()
    print(f"\nCache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    print(f"Augmented batches: {stats['pipeline_stats']['augmented_batches']}")
    print(f"Total batches processed: {stats['pipeline_stats']['total_batches']}")
