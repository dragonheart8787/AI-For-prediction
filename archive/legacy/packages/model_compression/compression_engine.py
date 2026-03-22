#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型壓縮引擎
實現量化、剪枝、蒸餾等模型壓縮技術，大幅提升推理速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import torch.jit as jit
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
import copy
import time
from collections import OrderedDict
import pickle

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """壓縮配置"""
    # 量化配置
    quantization_enabled: bool = True
    quantization_type: str = 'dynamic'  # 'dynamic', 'static', 'qat'
    quantization_bits: int = 8  # 8, 16
    calibration_samples: int = 100
    
    # 剪枝配置
    pruning_enabled: bool = True
    pruning_ratio: float = 0.3  # 剪枝比例
    pruning_type: str = 'magnitude'  # 'magnitude', 'gradient', 'random'
    pruning_frequency: int = 10  # 每N個epoch剪枝一次
    
    # 蒸餾配置
    distillation_enabled: bool = True
    distillation_alpha: float = 0.7  # 硬標籤權重
    distillation_temperature: float = 3.0  # 溫度參數
    
    # 其他配置
    save_compressed_model: bool = True
    compression_ratio_target: float = 0.5  # 目標壓縮比

class QuantizationEngine:
    """量化引擎"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """動態量化"""
        logger.info("Applying dynamic quantization...")
        
        # 動態量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.GRU}, 
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization completed")
        return quantized_model
    
    def static_quantization(self, model: nn.Module, calibration_data: List[torch.Tensor]) -> nn.Module:
        """靜態量化"""
        logger.info("Applying static quantization...")
        
        # 設置量化配置
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 準備量化
        model_prepared = torch.quantization.prepare(model)
        
        # 校準
        logger.info(f"Calibrating with {len(calibration_data)} samples...")
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
        
        # 轉換為量化模型
        quantized_model = torch.quantization.convert(model_prepared)
        
        logger.info("Static quantization completed")
        return quantized_model
    
    def quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """量化感知訓練"""
        logger.info("Setting up quantization aware training...")
        
        # 設置QAT配置
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 準備QAT
        model_prepared = torch.quantization.prepare_qat(model)
        
        logger.info("QAT setup completed")
        return model_prepared
    
    def convert_qat_model(self, model: nn.Module) -> nn.Module:
        """轉換QAT模型為量化模型"""
        logger.info("Converting QAT model to quantized model...")
        
        model.eval()
        quantized_model = torch.quantization.convert(model)
        
        logger.info("QAT conversion completed")
        return quantized_model

class PruningEngine:
    """剪枝引擎"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.pruning_scheduler = None
    
    def magnitude_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """基於幅度的剪枝"""
        logger.info(f"Applying magnitude pruning with ratio {pruning_ratio}...")
        
        # 創建剪枝配置
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))
        
        # 應用剪枝（簡化版本）
        for module, param_name in parameters_to_prune:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                # 簡單的幅度剪枝
                threshold = torch.quantile(torch.abs(param), pruning_ratio)
                mask = torch.abs(param) > threshold
                setattr(module, param_name, param * mask.float())
        
        logger.info("Magnitude pruning completed")
        return model
    
    def structured_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """結構化剪枝"""
        logger.info(f"Applying structured pruning with ratio {pruning_ratio}...")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 簡單的結構化剪枝
                weight = module.weight
                # 計算每行的L2範數
                row_norms = torch.norm(weight, dim=1)
                # 選擇要剪枝的行
                num_prune = int(pruning_ratio * weight.size(0))
                _, indices = torch.topk(row_norms, num_prune, largest=False)
                # 將選中的行設為零
                weight[indices] = 0
        
        logger.info("Structured pruning completed")
        return model
    
    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """移除剪枝（永久化）"""
        logger.info("Pruning already permanent in simplified version")
        return model
    
    def get_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """獲取剪枝統計"""
        total_params = 0
        pruned_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                total_params += weight.numel()
                pruned_params += (weight == 0).sum().item()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'pruning_ratio': pruning_ratio,
            'remaining_parameters': total_params - pruned_params
        }

class DistillationEngine:
    """蒸餾引擎"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def create_student_model(self, teacher_model: nn.Module, 
                           student_architecture: nn.Module) -> nn.Module:
        """創建學生模型"""
        logger.info("Creating student model...")
        
        # 複製學生架構
        student_model = copy.deepcopy(student_architecture)
        
        # 初始化權重
        for module in student_model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        logger.info("Student model created")
        return student_model
    
    def distillation_loss(self, student_output: torch.Tensor, 
                         teacher_output: torch.Tensor, 
                         target: torch.Tensor, 
                         alpha: float, 
                         temperature: float) -> torch.Tensor:
        """蒸餾損失"""
        # 軟標籤損失
        soft_loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 硬標籤損失
        hard_loss = F.cross_entropy(student_output, target)
        
        # 組合損失
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
        
        return total_loss
    
    def train_student(self, student_model: nn.Module, 
                     teacher_model: nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     num_epochs: int = 10) -> nn.Module:
        """訓練學生模型"""
        logger.info("Training student model with distillation...")
        
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                
                # 學生模型輸出
                student_output = student_model(data)
                
                # 老師模型輸出（不計算梯度）
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                
                # 計算蒸餾損失
                loss = self.distillation_loss(
                    student_output, teacher_output, target,
                    self.config.distillation_alpha,
                    self.config.distillation_temperature
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Student model training completed")
        return student_model

class ModelCompressor:
    """模型壓縮器"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.quantization_engine = QuantizationEngine(config)
        self.pruning_engine = PruningEngine(config)
        self.distillation_engine = DistillationEngine(config)
        
        self.compression_stats = {}
    
    def compress_model(self, model: nn.Module, 
                      compression_methods: List[str] = None) -> nn.Module:
        """壓縮模型"""
        if compression_methods is None:
            compression_methods = []
            if self.config.quantization_enabled:
                compression_methods.append('quantization')
            if self.config.pruning_enabled:
                compression_methods.append('pruning')
        
        logger.info(f"Starting model compression with methods: {compression_methods}")
        
        compressed_model = copy.deepcopy(model)
        
        # 記錄原始模型大小
        original_size = self._get_model_size(model)
        self.compression_stats['original_size'] = original_size
        
        # 應用壓縮方法
        for method in compression_methods:
            if method == 'quantization':
                compressed_model = self._apply_quantization(compressed_model)
            elif method == 'pruning':
                compressed_model = self._apply_pruning(compressed_model)
            elif method == 'distillation':
                compressed_model = self._apply_distillation(compressed_model)
        
        # 記錄壓縮後模型大小
        compressed_size = self._get_model_size(compressed_model)
        self.compression_stats['compressed_size'] = compressed_size
        self.compression_stats['compression_ratio'] = compressed_size / original_size
        
        logger.info(f"Model compression completed. "
                   f"Size: {original_size:.2f}MB -> {compressed_size:.2f}MB "
                   f"(Ratio: {self.compression_stats['compression_ratio']:.2f})")
        
        return compressed_model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """應用量化"""
        if self.config.quantization_type == 'dynamic':
            return self.quantization_engine.dynamic_quantization(model)
        elif self.config.quantization_type == 'static':
            # 需要校準數據，這裡使用模擬數據
            calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(self.config.calibration_samples)]
            return self.quantization_engine.static_quantization(model, calibration_data)
        else:
            logger.warning(f"Unsupported quantization type: {self.config.quantization_type}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """應用剪枝"""
        if self.config.pruning_type == 'magnitude':
            return self.pruning_engine.magnitude_pruning(model, self.config.pruning_ratio)
        elif self.config.pruning_type == 'structured':
            return self.pruning_engine.structured_pruning(model, self.config.pruning_ratio)
        else:
            logger.warning(f"Unsupported pruning type: {self.config.pruning_type}")
            return model
    
    def _apply_distillation(self, model: nn.Module) -> nn.Module:
        """應用蒸餾"""
        # 這裡需要老師模型和訓練數據，暫時跳過
        logger.warning("Distillation requires teacher model and training data")
        return model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """獲取模型大小（MB）"""
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 估算大小（假設float32，4字節）
        size_mb = total_params * 4 / (1024 * 1024)
        
        return size_mb
    
    def benchmark_models(self, original_model: nn.Module, 
                        compressed_model: nn.Module,
                        input_shape: Tuple[int, ...],
                        num_iterations: int = 100) -> Dict[str, Any]:
        """基準測試模型"""
        logger.info("Benchmarking models...")
        
        # 準備測試數據
        test_input = torch.randn(1, *input_shape)
        
        # 測試原始模型
        original_model.eval()
        with torch.no_grad():
            # 預熱
            for _ in range(10):
                _ = original_model(test_input)
            
            # 計時
            start_time = time.time()
            for _ in range(num_iterations):
                _ = original_model(test_input)
            original_time = time.time() - start_time
        
        # 測試壓縮模型
        compressed_model.eval()
        with torch.no_grad():
            # 預熱
            for _ in range(10):
                _ = compressed_model(test_input)
            
            # 計時
            start_time = time.time()
            for _ in range(num_iterations):
                _ = compressed_model(test_input)
            compressed_time = time.time() - start_time
        
        # 計算加速比
        speedup = original_time / compressed_time
        
        results = {
            'original_inference_time': original_time / num_iterations,
            'compressed_inference_time': compressed_time / num_iterations,
            'speedup': speedup,
            'compression_stats': self.compression_stats
        }
        
        logger.info(f"Benchmark completed. Speedup: {speedup:.2f}x")
        return results
    
    def save_compressed_model(self, model: nn.Module, filepath: str):
        """保存壓縮模型"""
        if self.config.save_compressed_model:
            torch.save(model.state_dict(), filepath)
            logger.info(f"Compressed model saved to {filepath}")
    
    def get_compression_report(self) -> Dict[str, Any]:
        """獲取壓縮報告"""
        return {
            'compression_stats': self.compression_stats,
            'config': {
                'quantization_enabled': self.config.quantization_enabled,
                'pruning_enabled': self.config.pruning_enabled,
                'distillation_enabled': self.config.distillation_enabled,
                'quantization_type': self.config.quantization_type,
                'pruning_ratio': self.config.pruning_ratio
            }
        }

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = CompressionConfig(
        quantization_enabled=True,
        quantization_type='dynamic',
        pruning_enabled=True,
        pruning_ratio=0.3,
        pruning_type='magnitude'
    )
    
    # 創建模型壓縮器
    compressor = ModelCompressor(config)
    
    # 創建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool2d(x, (32, 32))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 創建原始模型
    original_model = SimpleModel()
    
    # 壓縮模型
    compressed_model = compressor.compress_model(original_model)
    
    # 基準測試
    benchmark_results = compressor.benchmark_models(
        original_model, compressed_model, (3, 64, 64)
    )
    
    print("Compression Results:")
    print(f"Original size: {benchressor.compression_stats['original_size']:.2f}MB")
    print(f"Compressed size: {compressor.compression_stats['compressed_size']:.2f}MB")
    print(f"Compression ratio: {compressor.compression_stats['compression_ratio']:.2f}")
    print(f"Speedup: {benchmark_results['speedup']:.2f}x")
    
    # 保存壓縮模型
    compressor.save_compressed_model(compressed_model, 'compressed_model.pth')
    
    # 獲取壓縮報告
    report = compressor.get_compression_report()
    print("\nCompression Report:")
    print(report)
