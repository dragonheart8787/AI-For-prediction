#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試優化功能
測試所有新實現的優化技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_acceleration():
    """測試GPU加速"""
    logger.info("測試GPU加速...")
    
    try:
        from gpu_acceleration.cuda_optimizer import GPUConfig, OptimizedTrainer
        
        # 創建配置
        config = GPUConfig(device='auto', use_amp=True)
        
        # 創建簡單模型
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # 創建訓練器
        trainer = OptimizedTrainer(config)
        
        # 創建測試數據
        X = torch.randn(1000, 100)
        y = torch.randint(0, 10, (1000,))
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 設置訓練
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        trainer.setup_training(model, optimizer, criterion)
        
        # 訓練一個epoch
        results = trainer.train_epoch(dataloader)
        
        logger.info(f"✅ GPU加速測試成功 - 損失: {results['loss']:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU加速測試失敗: {e}")
        return False

def test_model_compression():
    """測試模型壓縮"""
    logger.info("測試模型壓縮...")
    
    try:
        from model_compression.compression_engine import CompressionConfig, ModelCompressor
        
        # 創建配置
        config = CompressionConfig(
            quantization_enabled=True,
            quantization_type='dynamic',
            pruning_enabled=True,
            pruning_ratio=0.2
        )
        
        # 創建簡單模型
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # 創建壓縮器
        compressor = ModelCompressor(config)
        
        # 壓縮模型
        compressed_model = compressor.compress_model(model)
        
        # 基準測試
        benchmark_results = compressor.benchmark_models(
            model, compressed_model, (100,)
        )
        
        logger.info(f"✅ 模型壓縮測試成功 - 壓縮比: {compressor.compression_stats['compression_ratio']:.2f}")
        logger.info(f"   加速比: {benchmark_results['speedup']:.2f}x")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型壓縮測試失敗: {e}")
        return False

def test_parallel_computing():
    """測試並行計算"""
    logger.info("測試並行計算...")
    
    try:
        from parallel_computing.parallel_engine import ParallelConfig, ParallelEngine
        
        # 創建配置
        config = ParallelConfig(
            num_processes=2,
            num_threads=4,
            use_async=True
        )
        
        # 創建並行引擎
        engine = ParallelEngine(config)
        engine.initialize()
        
        # 測試函數
        def process_data(data):
            # 模擬計算
            result = torch.randn(10, 10)
            for _ in range(50):
                result = torch.mm(result, torch.randn(10, 10))
            return result.sum().item()
        
        # 測試數據
        test_data = [torch.randn(10, 10) for _ in range(10)]
        
        # 基準測試
        benchmark_results = engine.benchmark_parallel_performance(process_data, test_data)
        
        engine.shutdown()
        
        logger.info(f"✅ 並行計算測試成功 - 加速比: {benchmark_results['speedup']:.2f}x")
        return True
        
    except Exception as e:
        logger.error(f"❌ 並行計算測試失敗: {e}")
        return False

def test_neural_architecture_search():
    """測試神經架構搜索"""
    logger.info("測試神經架構搜索...")
    
    try:
        from neural_architecture_search.nas_engine import NASConfig, NASEngine
        
        # 創建配置
        config = NASConfig(
            max_layers=4,
            min_layers=2,
            population_size=5,
            generations=2,
            epochs_per_architecture=1
        )
        
        # 創建NAS引擎
        nas_engine = NASEngine(config)
        
        # 創建測試數據
        X_train = torch.randn(100, 10)
        y_train = torch.randn(100, 1)
        X_val = torch.randn(20, 10)
        y_val = torch.randn(20, 1)
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 執行搜索
        best_architecture = nas_engine.search(
            (10,), (1,), train_loader, val_loader, torch.device('cpu')
        )
        
        logger.info(f"✅ 神經架構搜索測試成功 - 最佳適應度: {best_architecture.fitness:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 神經架構搜索測試失敗: {e}")
        return False

def test_reinforcement_learning():
    """測試強化學習"""
    logger.info("測試強化學習...")
    
    try:
        from reinforcement_learning.rl_engine import RLConfig, RLEngine
        
        # 創建配置
        config = RLConfig(
            state_dim=5,
            action_dim=3,
            max_episodes=10,
            max_steps_per_episode=20
        )
        
        # 創建RL引擎
        rl_engine = RLEngine(config)
        
        # 生成測試數據
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.1)
        
        # 訓練
        training_results = rl_engine.train(prices)
        
        logger.info(f"✅ 強化學習測試成功 - 最佳獎勵: {training_results['best_reward']:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 強化學習測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🧪 快速測試優化功能")
    print("=" * 40)
    
    tests = [
        ("GPU加速", test_gpu_acceleration),
        ("模型壓縮", test_model_compression),
        ("並行計算", test_parallel_computing),
        ("神經架構搜索", test_neural_architecture_search),
        ("強化學習", test_reinforcement_learning)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 測試 {test_name}...")
        start_time = time.time()
        
        try:
            success = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'success': success,
                'time': end_time - start_time
            }
            
            if success:
                passed_tests += 1
                print(f"✅ {test_name} 測試通過 ({end_time - start_time:.2f}s)")
            else:
                print(f"❌ {test_name} 測試失敗")
                
        except Exception as e:
            end_time = time.time()
            results[test_name] = {
                'success': False,
                'time': end_time - start_time,
                'error': str(e)
            }
            print(f"❌ {test_name} 測試異常: {e}")
    
    # 生成測試報告
    print("\n" + "=" * 40)
    print("📊 測試報告")
    print("=" * 40)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result['success'] else "❌ 失敗"
        print(f"{test_name}: {status} ({result['time']:.2f}s)")
        if not result['success'] and 'error' in result:
            print(f"  錯誤: {result['error']}")
    
    print(f"\n總計: {passed_tests}/{total_tests} 測試通過")
    
    if passed_tests == total_tests:
        print("🎉 所有測試都通過了！系統運行正常。")
    else:
        print("⚠️  部分測試失敗，請檢查相關模組。")
    
    return results

if __name__ == "__main__":
    main()
