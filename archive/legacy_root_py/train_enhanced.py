#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強訓練腳本
整合編譯器、VLM、優化器、分散式訓練等功能
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 添加模組路徑
sys.path.append(str(Path(__file__).parent))

from accelerators.compile import CompilerManager, create_compiler_config
from accelerators.amp import AMPManager, create_amp_config, TrainingLoop as AMPTrainingLoop
from accelerators.ddp import DistributedManager, create_distributed_config, setup_nccl_environment
from optimizers.enhanced_optimizers import OptimizerFactory, GradientClipping
from models.vlm_models import VLM, create_vlm_config
from export.to_hf import HFExporter

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """增強訓練器"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.compiler_manager = None
        self.amp_manager = None
        self.distributed_manager = None
        self.training_loop = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """載入配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """設置設備"""
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"使用設備: {device}")
        return device
    
    def _create_model(self) -> nn.Module:
        """創建模型"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'transformer')
        
        if model_type == 'vlm':
            # 創建 VLM 模型
            vlm_config = create_vlm_config(**self.config.get('vlm', {}))
            model = VLM(vlm_config)
        else:
            # 創建標準模型（這裡需要根據實際需求實現）
            model = self._create_standard_model(model_config)
        
        return model
    
    def _create_standard_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """創建標準模型"""
        # 這裡需要根據實際需求實現
        # 暫時返回一個簡單的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        return SimpleModel()
    
    def _setup_compiler(self):
        """設置編譯器"""
        compiler_config = self.config.get('compiler', {})
        config = create_compiler_config(**compiler_config)
        self.compiler_manager = CompilerManager(config)
        logger.info("編譯器已設置")
    
    def _setup_amp(self):
        """設置 AMP"""
        amp_config = self.config.get('amp', {})
        config = create_amp_config(**amp_config)
        self.amp_manager = AMPManager(config)
        logger.info("AMP 已設置")
    
    def _setup_distributed(self):
        """設置分散式訓練"""
        distributed_config = self.config.get('distributed', {})
        config = create_distributed_config(**distributed_config)
        self.distributed_manager = DistributedManager(config)
        
        if config.enable_ddp or config.enable_fsdp:
            setup_nccl_environment()
            self.distributed_manager.initialize()
            logger.info("分散式訓練已設置")
    
    def _setup_optimizer(self):
        """設置優化器"""
        optimizer_config = self.config.get('optimizer', {})
        scheduler_config = self.config.get('scheduler', {})
        
        # 合併配置，避免重複參數
        combined_config = {**optimizer_config, **scheduler_config}
        
        self.optimizer = OptimizerFactory.create_optimizer(
            self.model,
            **combined_config
        )
        logger.info("優化器已設置")
    
    def _setup_training_loop(self):
        """設置訓練循環"""
        gradient_config = self.config.get('training', {}).get('gradient_clipping', {})
        gradient_clipping = None
        
        if gradient_config.get('enabled', False):
            gradient_clipping = GradientClipping(
                max_norm=gradient_config.get('max_norm', 1.0),
                norm_type=gradient_config.get('norm_type', 2.0)
            )
        
        self.training_loop = AMPTrainingLoop(
            self.model,
            self.optimizer,
            gradient_clipping,
            self.device
        )
        logger.info("訓練循環已設置")
    
    def setup(self):
        """設置訓練環境"""
        logger.info("開始設置訓練環境...")
        
        # 創建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 設置各個組件
        self._setup_compiler()
        self._setup_amp()
        self._setup_distributed()
        
        # 編譯模型
        if self.compiler_manager:
            self.model = self.compiler_manager.compile_model(self.model, "main_model")
        
        # 包裝模型（分散式）
        if self.distributed_manager and self.distributed_manager.initialized:
            self.model = self.distributed_manager.wrap_model(self.model)
        
        # 設置優化器和訓練循環
        self._setup_optimizer()
        self._setup_training_loop()
        
        logger.info("訓練環境設置完成")
    
    def train(self, dataloader, loss_fn, num_epochs: Optional[int] = None):
        """訓練模型"""
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('epochs', 100)
        
        logger.info(f"開始訓練，共 {num_epochs} 個 epoch")
        
        for epoch in range(num_epochs):
            # 訓練一個 epoch
            metrics = self.training_loop.train_epoch(dataloader, loss_fn, epoch)
            
            # 記錄指標
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Loss={metrics['loss']:.4f}, "
                       f"LR={metrics['lr']:.6f}")
            
            # 保存檢查點
            if (epoch + 1) % self.config.get('training', {}).get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
        
        logger.info("訓練完成")
    
    def save_checkpoint(self, epoch: int):
        """保存檢查點"""
        output_dir = Path(self.config.get('output', {}).get('model_dir', './checkpoints'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
        if self.distributed_manager and self.distributed_manager.initialized:
            self.distributed_manager.save_model(self.model, str(checkpoint_path))
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
        
        logger.info(f"檢查點已保存: {checkpoint_path}")
    
    def export_model(self, export_format: str = "hf"):
        """匯出模型"""
        output_dir = Path(self.config.get('output', {}).get('model_dir', './checkpoints'))
        export_path = output_dir / "exported_model"
        
        if export_format == "hf":
            exporter = HFExporter(str(export_path))
            exporter.export_from_pytorch(self.model)
            logger.info(f"模型已匯出為 HF 格式: {export_path}")
        else:
            logger.warning(f"不支援的匯出格式: {export_format}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """獲取訓練信息"""
        info = {
            "device": str(self.device),
            "model_type": self.config.get('model', {}).get('type', 'unknown')
        }
        
        if self.compiler_manager:
            info["compiler"] = self.compiler_manager.get_compiler_info()
        
        if self.amp_manager:
            info["amp"] = self.amp_manager.get_amp_info()
        
        if self.distributed_manager:
            info["distributed"] = self.distributed_manager.get_distributed_info()
        
        if self.optimizer:
            info["optimizer"] = self.optimizer.get_optimizer_info()
        
        return info

def create_sample_dataloader():
    """創建示例數據加載器"""
    from torch.utils.data import DataLoader, TensorDataset
    
    # 創建示例數據
    x = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    return dataloader

def create_sample_loss_fn():
    """創建示例損失函數"""
    def loss_fn(outputs, batch):
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs)
        else:
            logits = outputs
        
        if isinstance(batch, dict):
            labels = batch.get('labels', batch.get('y'))
        else:
            labels = batch[1] if isinstance(batch, (list, tuple)) else batch
        
        return torch.nn.functional.cross_entropy(logits, labels)
    
    return loss_fn

def main():
    parser = argparse.ArgumentParser(description="增強訓練腳本")
    parser.add_argument("--config", default="configs/train.yaml", help="配置文件路徑")
    parser.add_argument("--epochs", type=int, help="訓練輪數")
    parser.add_argument("--export", action="store_true", help="訓練後匯出模型")
    parser.add_argument("--info", action="store_true", help="顯示訓練信息")
    
    args = parser.parse_args()
    
    # 創建訓練器
    trainer = EnhancedTrainer(args.config)
    
    # 設置訓練環境
    trainer.setup()
    
    # 顯示訓練信息
    if args.info:
        info = trainer.get_training_info()
        print("訓練信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # 創建示例數據和損失函數
    dataloader = create_sample_dataloader()
    loss_fn = create_sample_loss_fn()
    
    # 開始訓練
    trainer.train(dataloader, loss_fn, args.epochs)
    
    # 匯出模型
    if args.export:
        export_format = trainer.config.get('output', {}).get('export_format', 'hf')
        trainer.export_model(export_format)

if __name__ == "__main__":
    main()
