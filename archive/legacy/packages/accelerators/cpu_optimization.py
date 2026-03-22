#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU 最佳化模組
支援 Intel IPEX、OpenVINO、ONNX Runtime 等 CPU 最佳化
"""

import torch
import logging
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CPUOptimizationConfig:
    """CPU 最佳化配置"""
    # 基本設定
    enable_cpu_optimization: bool = True
    num_threads: int = 0  # 0 表示自動檢測
    num_interop_threads: int = 2
    
    # Intel 特定設定
    enable_ipex: bool = True
    ipex_dtype: str = "auto"  # auto, bf16, fp32
    
    # OpenVINO 設定
    enable_openvino: bool = False
    openvino_precision: str = "INT8"  # INT8, BF16, FP16, FP32
    
    # ONNX Runtime 設定
    enable_onnx: bool = False
    onnx_providers: list = None
    
    # 環境變數設定
    set_omp_threads: bool = True
    set_kmp_affinity: bool = True

class CPUOptimizer:
    """CPU 最佳化器"""
    
    def __init__(self, config: CPUOptimizationConfig):
        self.config = config
        self.optimized_model = None
        self.ipex_available = False
        self.openvino_available = False
        self.onnx_available = False
        
        self._check_dependencies()
        self._setup_environment()
    
    def _check_dependencies(self):
        """檢查依賴項"""
        # 檢查 Intel IPEX
        try:
            import intel_extension_for_pytorch as ipex
            self.ipex_available = True
            logger.info("Intel IPEX 可用")
        except ImportError:
            logger.info("Intel IPEX 不可用")
        
        # 檢查 OpenVINO
        try:
            import openvino
            self.openvino_available = True
            logger.info("OpenVINO 可用")
        except ImportError:
            logger.info("OpenVINO 不可用")
        
        # 檢查 ONNX Runtime
        try:
            import onnxruntime as ort
            self.onnx_available = True
            logger.info("ONNX Runtime 可用")
        except ImportError:
            logger.info("ONNX Runtime 不可用")
    
    def _setup_environment(self):
        """設置環境變數"""
        if not self.config.enable_cpu_optimization:
            return
        
        # 設置執行緒數
        if self.config.num_threads == 0:
            self.config.num_threads = torch.get_num_threads()
        
        torch.set_num_threads(self.config.num_threads)
        torch.set_num_interop_threads(self.config.num_interop_threads)
        
        # 設置 OMP 執行緒
        if self.config.set_omp_threads:
            os.environ["OMP_NUM_THREADS"] = str(self.config.num_threads)
        
        # 設置 KMP 親和性
        if self.config.set_kmp_affinity:
            os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        
        logger.info(f"CPU 執行緒設置: {self.config.num_threads} threads, "
                   f"{self.config.num_interop_threads} interop threads")
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """最佳化模型"""
        if not self.config.enable_cpu_optimization:
            return model
        
        optimized_model = model
        
        # Intel IPEX 最佳化
        if self.config.enable_ipex and self.ipex_available:
            optimized_model = self._optimize_with_ipex(optimized_model)
        
        # OpenVINO 最佳化
        if self.config.enable_openvino and self.openvino_available:
            optimized_model = self._optimize_with_openvino(optimized_model)
        
        self.optimized_model = optimized_model
        return optimized_model
    
    def _optimize_with_ipex(self, model: torch.nn.Module) -> torch.nn.Module:
        """使用 Intel IPEX 最佳化"""
        try:
            import intel_extension_for_pytorch as ipex
            
            # 確定精度類型
            if self.config.ipex_dtype == "auto":
                if torch.cpu.is_available() and hasattr(torch.cpu, 'is_bf16_supported'):
                    dtype = torch.bfloat16 if torch.cpu.is_bf16_supported() else torch.float32
                else:
                    dtype = torch.float32
            elif self.config.ipex_dtype == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # 最佳化模型
            model.eval()
            optimized_model = ipex.optimize(model, dtype=dtype)
            
            logger.info(f"模型已使用 Intel IPEX 最佳化，精度: {dtype}")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Intel IPEX 最佳化失敗: {e}")
            return model
    
    def _optimize_with_openvino(self, model: torch.nn.Module) -> torch.nn.Module:
        """使用 OpenVINO 最佳化"""
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            
            # 這裡需要根據實際模型類型進行轉換
            # 暫時返回原模型
            logger.info("OpenVINO 最佳化功能待實現")
            return model
            
        except Exception as e:
            logger.error(f"OpenVINO 最佳化失敗: {e}")
            return model
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """獲取最佳化信息"""
        return {
            "enabled": self.config.enable_cpu_optimization,
            "num_threads": self.config.num_threads,
            "num_interop_threads": self.config.num_interop_threads,
            "dependencies": {
                "ipex_available": self.ipex_available,
                "openvino_available": self.openvino_available,
                "onnx_available": self.onnx_available
            },
            "config": {
                "enable_ipex": self.config.enable_ipex,
                "enable_openvino": self.config.enable_openvino,
                "enable_onnx": self.config.enable_onnx
            }
        }

class OpenVINOManager:
    """OpenVINO 管理器"""
    
    def __init__(self, precision: str = "INT8"):
        self.precision = precision
        self.available = False
        
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            self.available = True
            logger.info("OpenVINO 管理器已初始化")
        except ImportError:
            logger.warning("OpenVINO 不可用")
    
    def convert_model(self, model_path: str, output_path: str) -> bool:
        """轉換模型為 OpenVINO 格式"""
        if not self.available:
            logger.error("OpenVINO 不可用")
            return False
        
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            from transformers import AutoTokenizer
            
            # 載入模型和 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path,
                export=True,
                compile=True,
                ov_config={"INFERENCE_PRECISION": self.precision}
            )
            
            # 保存轉換後的模型
            ov_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"模型已轉換為 OpenVINO 格式: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"OpenVINO 轉換失敗: {e}")
            return False
    
    def load_model(self, model_path: str):
        """載入 OpenVINO 模型"""
        if not self.available:
            return None
        
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = OVModelForCausalLM.from_pretrained(model_path)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"載入 OpenVINO 模型失敗: {e}")
            return None

class ONNXRuntimeManager:
    """ONNX Runtime 管理器"""
    
    def __init__(self, providers: list = None):
        self.providers = providers or ["CPUExecutionProvider"]
        self.available = False
        
        try:
            import onnxruntime as ort
            self.available = True
            logger.info("ONNX Runtime 管理器已初始化")
        except ImportError:
            logger.warning("ONNX Runtime 不可用")
    
    def create_session(self, model_path: str, num_threads: int = 4) -> Optional[Any]:
        """創建 ONNX Runtime 會話"""
        if not self.available:
            logger.error("ONNX Runtime 不可用")
            return None
        
        try:
            import onnxruntime as ort
            
            # 設置會話選項
            so = ort.SessionOptions()
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = 2
            
            # 創建會話
            session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=self.providers
            )
            
            logger.info(f"ONNX Runtime 會話已創建: {model_path}")
            return session
            
        except Exception as e:
            logger.error(f"創建 ONNX Runtime 會話失敗: {e}")
            return None

def create_cpu_optimization_config(
    enable_cpu_optimization: bool = True,
    num_threads: int = 0,
    enable_ipex: bool = True,
    enable_openvino: bool = False,
    enable_onnx: bool = False
) -> CPUOptimizationConfig:
    """創建 CPU 最佳化配置"""
    return CPUOptimizationConfig(
        enable_cpu_optimization=enable_cpu_optimization,
        num_threads=num_threads,
        enable_ipex=enable_ipex,
        enable_openvino=enable_openvino,
        enable_onnx=enable_onnx
    )

def optimize_for_cpu_training(
    model: torch.nn.Module,
    config: Optional[CPUOptimizationConfig] = None
) -> torch.nn.Module:
    """為 CPU 訓練最佳化模型"""
    if config is None:
        config = create_cpu_optimization_config()
    
    optimizer = CPUOptimizer(config)
    optimized_model = optimizer.optimize_model(model)
    
    return optimized_model

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
    
    # 測試 CPU 最佳化
    model = TestModel()
    config = create_cpu_optimization_config()
    optimizer = CPUOptimizer(config)
    optimized_model = optimizer.optimize_model(model)
    
    print("CPU 最佳化測試完成")
    print(f"最佳化信息: {optimizer.get_optimization_info()}")
