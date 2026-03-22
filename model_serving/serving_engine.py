#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型服務引擎
實現批處理推理、模型並行、負載均衡等高效模型服務技術
"""

import torch
import torch.nn as nn
import asyncio
import threading
import queue
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
import uuid
from datetime import datetime, timedelta
import psutil
import gc
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ServingConfig:
    """服務配置"""
    # 批處理配置
    use_batch_processing: bool = True
    max_batch_size: int = 32
    batch_timeout: float = 0.1  # 秒
    min_batch_size: int = 1
    
    # 模型並行配置
    use_model_parallelism: bool = True
    num_model_instances: int = 2
    load_balancing: str = 'round_robin'  # 'round_robin', 'least_loaded', 'random'
    
    # 異步配置
    use_async: bool = True
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0  # 秒
    
    # 緩存配置
    use_response_cache: bool = True
    cache_size: int = 1000
    cache_ttl: float = 300.0  # 秒
    
    # 監控配置
    enable_metrics: bool = True
    metrics_interval: float = 10.0  # 秒
    
    # 性能配置
    use_mixed_precision: bool = True
    use_tensorrt: bool = False
    optimization_level: str = 'O1'  # 'O0', 'O1', 'O2', 'O3'

class Request:
    """請求類"""
    
    def __init__(self, request_id: str, data: Any, metadata: Dict[str, Any] = None):
        self.request_id = request_id
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.status = 'pending'
        self.result = None
        self.error = None
        self.processing_time = 0.0

class ModelInstance:
    """模型實例"""
    
    def __init__(self, model: nn.Module, instance_id: str, device: torch.device):
        self.model = model
        self.instance_id = instance_id
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 性能統計
        self.stats = {
            'total_requests': 0,
            'total_processing_time': 0.0,
            'current_load': 0,
            'last_used': time.time()
        }
        
        # 請求隊列
        self.request_queue = queue.Queue()
        self.processing = False
        self.worker_thread = None
        
        # 啟動處理線程
        self.start_processing()
    
    def start_processing(self):
        """啟動處理線程"""
        if not self.processing:
            self.processing = True
            self.worker_thread = threading.Thread(target=self._process_requests, daemon=True)
            self.worker_thread.start()
    
    def stop_processing(self):
        """停止處理線程"""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _process_requests(self):
        """處理請求"""
        while self.processing:
            try:
                # 獲取請求
                request = self.request_queue.get(timeout=1)
                if request is None:
                    continue
                
                # 處理請求
                start_time = time.time()
                try:
                    with torch.no_grad():
                        result = self.model(request.data.to(self.device))
                    request.result = result.cpu()
                    request.status = 'completed'
                except Exception as e:
                    request.error = str(e)
                    request.status = 'failed'
                    logger.error(f"Model inference error: {e}")
                
                # 更新統計
                processing_time = time.time() - start_time
                request.processing_time = processing_time
                self.stats['total_requests'] += 1
                self.stats['total_processing_time'] += processing_time
                self.stats['last_used'] = time.time()
                
                # 通知請求完成
                if hasattr(request, 'event'):
                    request.event.set()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Request processing error: {e}")
    
    def submit_request(self, request: Request) -> bool:
        """提交請求"""
        try:
            self.request_queue.put(request, timeout=1)
            self.stats['current_load'] = self.request_queue.qsize()
            return True
        except queue.Full:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        avg_processing_time = (
            self.stats['total_processing_time'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )
        
        return {
            'instance_id': self.instance_id,
            'total_requests': self.stats['total_requests'],
            'avg_processing_time': avg_processing_time,
            'current_load': self.stats['current_load'],
            'last_used': self.stats['last_used'],
            'queue_size': self.request_queue.qsize()
        }

class LoadBalancer:
    """負載均衡器"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.model_instances = []
        self.current_index = 0
        self.instance_stats = {}
    
    def add_model_instance(self, model_instance: ModelInstance):
        """添加模型實例"""
        self.model_instances.append(model_instance)
        self.instance_stats[model_instance.instance_id] = {
            'requests': 0,
            'last_used': time.time()
        }
        logger.info(f"Added model instance: {model_instance.instance_id}")
    
    def select_instance(self) -> Optional[ModelInstance]:
        """選擇模型實例"""
        if not self.model_instances:
            return None
        
        if self.config.load_balancing == 'round_robin':
            return self._round_robin_selection()
        elif self.config.load_balancing == 'least_loaded':
            return self._least_loaded_selection()
        elif self.config.load_balancing == 'random':
            return self._random_selection()
        else:
            return self._round_robin_selection()
    
    def _round_robin_selection(self) -> ModelInstance:
        """輪詢選擇"""
        instance = self.model_instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.model_instances)
        return instance
    
    def _least_loaded_selection(self) -> ModelInstance:
        """最少負載選擇"""
        return min(self.model_instances, key=lambda x: x.stats['current_load'])
    
    def _random_selection(self) -> ModelInstance:
        """隨機選擇"""
        return np.random.choice(self.model_instances)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """獲取所有實例統計"""
        return {
            instance.instance_id: instance.get_stats()
            for instance in self.model_instances
        }

class BatchProcessor:
    """批處理器"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.batch_queue = queue.Queue()
        self.batch_timeout = config.batch_timeout
        self.processing = False
        self.worker_thread = None
        
        # 批處理統計
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0
        }
    
    def start_processing(self, model_instances: List[ModelInstance]):
        """啟動批處理"""
        if not self.processing:
            self.processing = True
            self.worker_thread = threading.Thread(
                target=self._process_batches,
                args=(model_instances,),
                daemon=True
            )
            self.worker_thread.start()
            logger.info("Batch processing started")
    
    def stop_processing(self):
        """停止批處理"""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Batch processing stopped")
    
    def _process_batches(self, model_instances: List[ModelInstance]):
        """處理批次"""
        while self.processing:
            try:
                # 收集批次
                batch = self._collect_batch()
                if not batch:
                    continue
                
                # 選擇模型實例
                instance = np.random.choice(model_instances)
                
                # 處理批次
                start_time = time.time()
                self._process_batch(batch, instance)
                processing_time = time.time() - start_time
                
                # 更新統計
                self.stats['total_batches'] += 1
                self.stats['total_requests'] += len(batch)
                self.stats['avg_batch_size'] = (
                    self.stats['total_requests'] / self.stats['total_batches']
                )
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['total_batches'] - 1) + processing_time) /
                    self.stats['total_batches']
                )
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _collect_batch(self) -> List[Request]:
        """收集批次"""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.config.max_batch_size:
            try:
                # 嘗試獲取請求
                request = self.batch_queue.get(timeout=0.01)
                batch.append(request)
                
                # 檢查是否達到最小批次大小和超時
                if (len(batch) >= self.config.min_batch_size and
                    time.time() - start_time >= self.batch_timeout):
                    break
                    
            except queue.Empty:
                # 如果沒有更多請求且已達到最小批次大小，則處理當前批次
                if len(batch) >= self.config.min_batch_size:
                    break
                # 如果沒有請求且未達到最小批次大小，繼續等待
                if time.time() - start_time >= self.batch_timeout:
                    break
        
        return batch
    
    def _process_batch(self, batch: List[Request], instance: ModelInstance):
        """處理批次"""
        try:
            # 合併批次數據
            batch_data = torch.stack([req.data for req in batch])
            
            # 模型推理
            with torch.no_grad():
                batch_result = instance.model(batch_data.to(instance.device))
            
            # 分發結果
            batch_result = batch_result.cpu()
            for i, request in enumerate(batch):
                request.result = batch_result[i]
                request.status = 'completed'
                if hasattr(request, 'event'):
                    request.event.set()
                    
        except Exception as e:
            # 標記所有請求為失敗
            for request in batch:
                request.error = str(e)
                request.status = 'failed'
                if hasattr(request, 'event'):
                    request.event.set()
            logger.error(f"Batch processing error: {e}")
    
    def submit_request(self, request: Request) -> bool:
        """提交請求到批處理隊列"""
        try:
            self.batch_queue.put(request, timeout=1)
            return True
        except queue.Full:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取批處理統計"""
        return {
            'total_batches': self.stats['total_batches'],
            'total_requests': self.stats['total_requests'],
            'avg_batch_size': self.stats['avg_batch_size'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'queue_size': self.batch_queue.qsize()
        }

class ResponseCache:
    """響應緩存"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, cache_key: str) -> Optional[Any]:
        """獲取緩存響應"""
        if not self.config.use_response_cache:
            return None
        
        if cache_key in self.cache:
            # 檢查TTL
            if time.time() - self.cache_timestamps[cache_key] < self.config.cache_ttl:
                self.stats['hits'] += 1
                return self.cache[cache_key]
            else:
                # 過期，移除
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
                self.stats['evictions'] += 1
        
        self.stats['misses'] += 1
        return None
    
    def put(self, cache_key: str, response: Any):
        """存儲響應到緩存"""
        if not self.config.use_response_cache:
            return
        
        # 檢查緩存大小限制
        if len(self.cache) >= self.config.cache_size:
            # 移除最舊的項目
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            self.stats['evictions'] += 1
        
        self.cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
    
    def clear(self):
        """清空緩存"""
        self.cache.clear()
        self.cache_timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'size': len(self.cache),
            'max_size': self.config.cache_size
        }

class ServingEngine:
    """模型服務引擎"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.model_instances = []
        self.load_balancer = LoadBalancer(config)
        self.batch_processor = BatchProcessor(config)
        self.response_cache = ResponseCache(config)
        
        # 服務統計
        self.service_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'start_time': time.time()
        }
        
        # 請求隊列
        self.request_queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self.processing = False
    
    def add_model(self, model: nn.Module, device: torch.device = None) -> str:
        """添加模型"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        instance_id = str(uuid.uuid4())
        model_instance = ModelInstance(model, instance_id, device)
        
        self.model_instances.append(model_instance)
        self.load_balancer.add_model_instance(model_instance)
        
        logger.info(f"Added model instance: {instance_id} on device: {device}")
        return instance_id
    
    def start_serving(self):
        """啟動服務"""
        if not self.model_instances:
            raise ValueError("No model instances available")
        
        self.processing = True
        
        # 啟動批處理
        if self.config.use_batch_processing:
            self.batch_processor.start_processing(self.model_instances)
        
        logger.info("Model serving started")
    
    def stop_serving(self):
        """停止服務"""
        self.processing = False
        
        # 停止批處理
        if self.config.use_batch_processing:
            self.batch_processor.stop_processing()
        
        # 停止模型實例
        for instance in self.model_instances:
            instance.stop_processing()
        
        logger.info("Model serving stopped")
    
    async def predict(self, data: torch.Tensor, request_id: str = None) -> Dict[str, Any]:
        """異步預測"""
        if not self.processing:
            raise RuntimeError("Serving engine not started")
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # 檢查緩存
        cache_key = self._generate_cache_key(data)
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            return {
                'request_id': request_id,
                'result': cached_response,
                'cached': True,
                'processing_time': 0.0
            }
        
        # 創建請求
        request = Request(request_id, data)
        request.event = asyncio.Event()
        
        # 提交請求
        if self.config.use_batch_processing:
            success = self.batch_processor.submit_request(request)
        else:
            instance = self.load_balancer.select_instance()
            if instance:
                success = instance.submit_request(request)
            else:
                success = False
        
        if not success:
            raise RuntimeError("Failed to submit request")
        
        # 等待結果
        try:
            await asyncio.wait_for(request.event.wait(), timeout=self.config.request_timeout)
        except asyncio.TimeoutError:
            request.status = 'timeout'
            raise TimeoutError("Request timeout")
        
        # 更新統計
        self._update_stats(request)
        
        # 緩存響應
        if request.status == 'completed':
            self.response_cache.put(cache_key, request.result)
        
        return {
            'request_id': request_id,
            'result': request.result,
            'status': request.status,
            'error': request.error,
            'processing_time': request.processing_time,
            'cached': False
        }
    
    def predict_sync(self, data: torch.Tensor, request_id: str = None) -> Dict[str, Any]:
        """同步預測"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.predict(data, request_id))
        finally:
            loop.close()
    
    def _generate_cache_key(self, data: torch.Tensor) -> str:
        """生成緩存鍵"""
        # 使用數據的哈希值作為緩存鍵
        data_hash = hashlib.md5(data.numpy().tobytes()).hexdigest()
        return f"cache_{data_hash}"
    
    def _update_stats(self, request: Request):
        """更新統計"""
        self.service_stats['total_requests'] += 1
        
        if request.status == 'completed':
            self.service_stats['successful_requests'] += 1
        else:
            self.service_stats['failed_requests'] += 1
        
        # 更新平均響應時間
        total_time = self.service_stats['avg_response_time'] * (self.service_stats['total_requests'] - 1)
        self.service_stats['avg_response_time'] = (total_time + request.processing_time) / self.service_stats['total_requests']
    
    def get_service_stats(self) -> Dict[str, Any]:
        """獲取服務統計"""
        uptime = time.time() - self.service_stats['start_time']
        
        return {
            'service_stats': self.service_stats,
            'load_balancer_stats': self.load_balancer.get_all_stats(),
            'batch_processor_stats': self.batch_processor.get_stats(),
            'cache_stats': self.response_cache.get_stats(),
            'uptime': uptime,
            'requests_per_second': self.service_stats['total_requests'] / uptime if uptime > 0 else 0
        }
    
    def benchmark(self, test_data: List[torch.Tensor], num_requests: int = 100) -> Dict[str, Any]:
        """基準測試"""
        logger.info(f"Starting benchmark with {num_requests} requests")
        
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            data = test_data[i % len(test_data)]
            try:
                result = self.predict_sync(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark request {i} failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 計算統計
        successful_requests = len([r for r in results if r['status'] == 'completed'])
        failed_requests = len(results) - successful_requests
        avg_processing_time = np.mean([r['processing_time'] for r in results if r['processing_time'] > 0])
        
        return {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time,
            'avg_processing_time': avg_processing_time,
            'success_rate': successful_requests / num_requests
        }

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = ServingConfig(
        use_batch_processing=True,
        max_batch_size=16,
        num_model_instances=2,
        use_response_cache=True
    )
    
    # 創建服務引擎
    engine = ServingEngine(config)
    
    # 創建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.fc(x)
    
    # 添加模型實例
    model1 = SimpleModel()
    model2 = SimpleModel()
    
    engine.add_model(model1)
    engine.add_model(model2)
    
    # 啟動服務
    engine.start_serving()
    
    # 創建測試數據
    test_data = [torch.randn(100) for _ in range(50)]
    
    # 基準測試
    benchmark_results = engine.benchmark(test_data, num_requests=100)
    
    print("Benchmark Results:")
    print(f"Total requests: {benchmark_results['total_requests']}")
    print(f"Successful requests: {benchmark_results['successful_requests']}")
    print(f"Failed requests: {benchmark_results['failed_requests']}")
    print(f"Requests per second: {benchmark_results['requests_per_second']:.2f}")
    print(f"Average processing time: {benchmark_results['avg_processing_time']:.4f}s")
    print(f"Success rate: {benchmark_results['success_rate']:.2%}")
    
    # 獲取服務統計
    stats = engine.get_service_stats()
    print(f"\nService Stats:")
    print(f"Uptime: {stats['uptime']:.2f}s")
    print(f"Requests per second: {stats['requests_per_second']:.2f}")
    print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    
    # 停止服務
    engine.stop_serving()
