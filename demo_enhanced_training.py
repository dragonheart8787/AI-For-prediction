#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強訓練系統演示腳本
展示編譯器、VLM、優化器、分散式訓練等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_model():
    """創建演示模型"""
    class DemoModel(nn.Module):
        def __init__(self, input_size=100, hidden_size=64, output_size=10):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = F.relu(self.linear2(x))
            x = self.dropout(x)
            x = self.linear3(x)
            return x
    
    return DemoModel()

def create_demo_data():
    """創建演示數據"""
    # 生成隨機數據
    x = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader

def demo_compiler():
    """演示編譯器功能"""
    print("\n🧱 編譯器功能演示")
    print("=" * 50)
    
    from accelerators.compile import CompilerManager, create_compiler_config
    
    # 創建模型
    model = create_demo_model()
    
    # 創建編譯器配置
    config = create_compiler_config(
        enable_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="max-autotune"
    )
    
    # 編譯模型
    compiler_manager = CompilerManager(config)
    optimized_model = compiler_manager.compile_model(model, "demo_model")
    
    # 顯示編譯信息
    info = compiler_manager.get_compiler_info()
    print(f"✅ 編譯器信息: {info}")
    
    return optimized_model

def demo_amp():
    """演示 AMP 功能"""
    print("\n⚡ AMP 功能演示")
    print("=" * 50)
    
    from accelerators.amp import AMPManager, create_amp_config
    
    # 創建 AMP 配置
    config = create_amp_config(
        enable_amp=True,
        dtype="auto",
        prefer_bf16=True
    )
    
    # 創建 AMP 管理器
    amp_manager = AMPManager(config)
    
    # 顯示 AMP 信息
    info = amp_manager.get_amp_info()
    print(f"✅ AMP 信息: {info}")
    
    return amp_manager

def demo_optimizer():
    """演示優化器功能"""
    print("\n🔁 優化器功能演示")
    print("=" * 50)
    
    from optimizers.enhanced_optimizers import OptimizerFactory
    
    # 創建模型
    model = create_demo_model()
    
    # 創建優化器
    optimizer = OptimizerFactory.create_optimizer(
        model,
        optimizer_type="sgd_momentum",
        learning_rate=0.01,
        scheduler_type="warmup_cosine",
        total_epochs=10
    )
    
    # 顯示優化器信息
    info = optimizer.get_optimizer_info()
    print(f"✅ 優化器信息: {info}")
    
    return optimizer

def demo_vlm():
    """演示 VLM 功能"""
    print("\n🖼️ VLM 功能演示")
    print("=" * 50)
    
    from models.vlm_models import VLM, create_vlm_config
    
    # 創建 VLM 配置
    config = create_vlm_config(
        vision_encoder_type="resnet",
        text_encoder_type="transformer",
        fusion_type="cross_attention"
    )
    
    # 創建 VLM 模型
    model = VLM(config)
    
    # 測試前向傳播
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    
    try:
        outputs = model(images, input_ids, attention_mask)
        print(f"✅ VLM 輸出形狀: {outputs['logits'].shape}")
    except Exception as e:
        print(f"⚠️ VLM 測試跳過（依賴項未安裝）: {e}")
    
    return model

def demo_cpu_optimization():
    """演示 CPU 最佳化功能"""
    print("\n🖥️ CPU 最佳化功能演示")
    print("=" * 50)
    
    from accelerators.cpu_optimization import CPUOptimizer, create_cpu_optimization_config
    
    # 創建 CPU 最佳化配置
    config = create_cpu_optimization_config(
        enable_cpu_optimization=True,
        enable_ipex=False,  # 避免依賴問題
        num_threads=4
    )
    
    # 創建 CPU 最佳化器
    optimizer = CPUOptimizer(config)
    
    # 顯示最佳化信息
    info = optimizer.get_optimization_info()
    print(f"✅ CPU 最佳化信息: {info}")
    
    return optimizer

def demo_training_loop():
    """演示完整訓練循環"""
    print("\n🚀 完整訓練循環演示")
    print("=" * 50)
    
    # 創建模型和數據
    model = create_demo_model()
    dataloader = create_demo_data()
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 創建優化器和損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 訓練幾個 epoch
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成，平均損失: {avg_loss:.4f}")
    
    print("✅ 訓練循環演示完成")

def demo_export():
    """演示模型匯出功能"""
    print("\n📤 模型匯出功能演示")
    print("=" * 50)
    
    from export.to_hf import HFExporter
    
    # 創建模型
    model = create_demo_model()
    
    # 創建匯出器
    exporter = HFExporter("./demo_export")
    
    # 匯出模型
    try:
        output_path = exporter.export_from_pytorch(model)
        print(f"✅ 模型已匯出到: {output_path}")
        
        # 創建模型卡片
        exporter.create_model_card("demo_model", "這是一個演示模型")
        print("✅ 模型卡片已創建")
        
    except Exception as e:
        print(f"⚠️ 匯出功能測試跳過: {e}")

def main():
    """主演示函數"""
    print("🎉 增強訓練系統功能演示")
    print("=" * 60)
    
    try:
        # 演示各個功能模組
        demo_compiler()
        demo_amp()
        demo_optimizer()
        demo_vlm()
        demo_cpu_optimization()
        demo_training_loop()
        demo_export()
        
        print("\n🎊 所有功能演示完成！")
        print("=" * 60)
        print("您現在可以:")
        print("1. 使用 python train_enhanced.py --config configs/train.yaml 進行完整訓練")
        print("2. 查看 ENHANCED_TRAINING_GUIDE.md 了解詳細使用方法")
        print("3. 使用各個模組進行自定義開發")
        
    except Exception as e:
        print(f"❌ 演示過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
