#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分散式訓練模組
支援 DDP、FSDP 等分散式訓練策略
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """分散式訓練配置"""
    # 基本設定
    enable_ddp: bool = False
    enable_fsdp: bool = False
    
    # DDP 設定
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    
    # FSDP 設定
    fsdp_min_num_params: int = 1_000_000
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_use_orig_params: bool = True
    
    # 環境設定
    backend: str = "nccl"  # nccl, gloo
    init_method: str = "env://"

class DistributedManager:
    """分散式訓練管理器"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.initialized = False
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        
    def initialize(self):
        """初始化分散式環境"""
        if not self.config.enable_ddp and not self.config.enable_fsdp:
            logger.info("分散式訓練未啟用")
            return
        
        try:
            # 從環境變數獲取分散式信息
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # 初始化進程組
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method
                )
                logger.info(f"分散式進程組已初始化: rank={self.local_rank}, world_size={self.world_size}")
            
            # 設置設備
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"分散式初始化失敗: {e}")
            self.initialized = False
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """包裝模型用於分散式訓練"""
        if not self.initialized:
            return model
        
        if self.config.enable_fsdp:
            return self._wrap_fsdp(model)
        elif self.config.enable_ddp:
            return self._wrap_ddp(model)
        else:
            return model
    
    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """使用 DDP 包裝模型"""
        logger.info("使用 DDP 包裝模型")
        return DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
            gradient_as_bucket_view=self.config.ddp_gradient_as_bucket_view
        )
    
    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """使用 FSDP 包裝模型"""
        logger.info("使用 FSDP 包裝模型")
        
        # 創建自動包裝策略
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=self.config.fsdp_min_num_params
        )
        
        # 選擇分片策略
        if self.config.fsdp_sharding_strategy == "FULL_SHARD":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD
        
        return FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=self.config.fsdp_use_orig_params
        )
    
    def create_sampler(self, dataset) -> Optional[DistributedSampler]:
        """創建分散式採樣器"""
        if not self.initialized:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True
        )
    
    def save_model(self, model: nn.Module, save_path: str):
        """保存模型（處理分散式包裝）"""
        if not self.initialized:
            torch.save(model.state_dict(), save_path)
            return
        
        # 處理 FSDP 模型保存
        if self.config.enable_fsdp:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig()):
                state_dict = model.state_dict()
                if self.local_rank == 0:  # 只在主進程保存
                    torch.save(state_dict, save_path)
        else:
            # 處理 DDP 模型保存
            if hasattr(model, "module"):  # DDP unwrap
                model_to_save = model.module
            else:
                model_to_save = model
            
            if self.local_rank == 0:  # 只在主進程保存
                torch.save(model_to_save.state_dict(), save_path)
    
    def cleanup(self):
        """清理分散式環境"""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("分散式進程組已清理")
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """獲取分散式信息"""
        return {
            "initialized": self.initialized,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": str(self.device) if self.device else None,
            "config": {
                "enable_ddp": self.config.enable_ddp,
                "enable_fsdp": self.config.enable_fsdp,
                "backend": self.config.backend
            }
        }

def setup_nccl_environment():
    """設置 NCCL 環境變數"""
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_IB_TIMEOUT", "22")
    # os.environ.setdefault("NCCL_DEBUG", "INFO")  # 除錯時開啟

def create_distributed_config(
    enable_ddp: bool = False,
    enable_fsdp: bool = False,
    backend: str = "nccl",
    **kwargs
) -> DistributedConfig:
    """創建分散式配置"""
    return DistributedConfig(
        enable_ddp=enable_ddp,
        enable_fsdp=enable_fsdp,
        backend=backend
    )

class DistributedTrainingLoop:
    """分散式訓練循環"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 distributed_manager: DistributedManager, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.distributed_manager = distributed_manager
        self.dataloader = dataloader
        self.sampler = None
        
        # 設置分散式採樣器
        if self.distributed_manager.initialized:
            self.sampler = self.distributed_manager.create_sampler(dataloader.dataset)
    
    def train_epoch(self, loss_fn, epoch: int = 0):
        """訓練一個 epoch"""
        self.model.train()
        
        # 設置採樣器 epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.dataloader:
            # 前向傳播
            outputs = self.model(**batch)
            loss = loss_fn(outputs, batch)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
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
    
    # 測試分散式訓練
    model = TestModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = create_distributed_config()
    distributed_manager = DistributedManager(config)
    
    print("分散式訓練測試完成")
    print(f"分散式信息: {distributed_manager.get_distributed_info()}")
