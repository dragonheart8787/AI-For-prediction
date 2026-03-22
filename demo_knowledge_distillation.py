#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知識蒸餾演示腳本
展示各種知識蒸餾技術的應用
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 導入知識蒸餾模組
from knowledge_distillation.kd_trainer import KnowledgeDistillationTrainer, create_default_config
from knowledge_distillation.probabilistic_student import (
    ProbabilisticStudent, MultiHorizonProbabilisticStudent, QuantileStudent
)
from knowledge_distillation.teacher_ensemble import (
    TeacherEnsemble, create_default_teacher_models
)
from knowledge_distillation.sequence_kd import LSTMStudent, TransformerStudent

def create_synthetic_data(n_samples: int = 1000, n_features: int = 20, 
                         noise_level: float = 0.1, random_state: int = 42) -> tuple:
    """
    創建合成數據
    
    Args:
        n_samples: 樣本數量
        n_features: 特徵數量
        noise_level: 噪聲水平
        random_state: 隨機種子
        
    Returns:
        (X, y) 特徵和目標變量
    """
    np.random.seed(random_state)
    
    # 創建特徵
    X = np.random.randn(n_samples, n_features)
    
    # 創建非線性目標變量
    y = (X[:, 0] ** 2 + 
         X[:, 1] * X[:, 2] + 
         np.sin(X[:, 3]) + 
         X[:, 4] * X[:, 5] + 
         np.random.randn(n_samples) * noise_level)
    
    return X, y

def create_time_series_data(n_samples: int = 1000, n_features: int = 20, 
                           seq_len: int = 10, random_state: int = 42) -> tuple:
    """
    創建時間序列數據
    
    Args:
        n_samples: 樣本數量
        n_features: 特徵數量
        seq_len: 序列長度
        random_state: 隨機種子
        
    Returns:
        (X, y) 序列特徵和目標變量
    """
    np.random.seed(random_state)
    
    # 創建序列特徵
    X = np.random.randn(n_samples, seq_len, n_features)
    
    # 創建序列目標變量（基於最後一個時間步）
    y = (X[:, -1, 0] ** 2 + 
         X[:, -1, 1] * X[:, -1, 2] + 
         np.sin(X[:, -1, 3]) + 
         np.random.randn(n_samples) * 0.1)
    
    return X, y

def demo_single_step_kd():
    """演示單步回歸知識蒸餾"""
    print("\n" + "="*60)
    print("🚀 演示 1: 單步回歸知識蒸餾")
    print("="*60)
    
    # 創建數據
    X, y = create_synthetic_data(n_samples=1000, n_features=20)
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    
    # 創建配置
    config = create_default_config()
    config.update({
        'student_model_type': 'probabilistic',
        'epochs': 50,
        'batch_size': 32,
        'alpha': 0.5,
        'output_dir': 'kd_demo_single_step'
    })
    
    # 創建訓練器
    trainer = KnowledgeDistillationTrainer(config)
    
    # 準備數據
    data = trainer.prepare_data(X, y)
    
    # 訓練老師集成
    print("📚 訓練老師集成模型...")
    teacher_ensemble = trainer.train_teacher_ensemble(data)
    
    # 生成老師預測
    print("🔮 生成老師預測...")
    teacher_predictions = trainer.generate_teacher_predictions(data)
    
    # 訓練學生模型
    print("🎓 訓練學生模型...")
    student_model = trainer.train_student_model(data, teacher_predictions)
    
    # 評估模型
    print("📊 評估模型...")
    metrics = trainer.evaluate_model(data, teacher_predictions)
    
    # 保存結果
    trainer.save_results(metrics, teacher_predictions)
    
    # 打印結果
    print("\n📈 單步回歸知識蒸餾結果:")
    print(f"  學生模型 MSE: {metrics['student_mse']:.4f}")
    print(f"  老師模型 MSE: {metrics['teacher_mse']:.4f}")
    print(f"  MSE 比率: {metrics['mse_ratio']:.4f}")
    print(f"  學生模型 R²: {metrics['student_r2']:.4f}")
    print(f"  老師模型 R²: {metrics['teacher_r2']:.4f}")
    
    return metrics

def demo_multi_horizon_kd():
    """演示多地平線知識蒸餾"""
    print("\n" + "="*60)
    print("🚀 演示 2: 多地平線知識蒸餾")
    print("="*60)
    
    # 創建數據
    X, y = create_synthetic_data(n_samples=1000, n_features=20)
    
    # 創建多地平線目標變量
    horizons = [1, 5, 10, 20]
    y_multi = np.zeros((len(y), len(horizons)))
    
    for i, h in enumerate(horizons):
        # 模擬不同地平線的預測目標
        y_multi[:, i] = y + np.random.randn(len(y)) * 0.1 * h
    
    print(f"數據形狀: X={X.shape}, y_multi={y_multi.shape}")
    print(f"地平線: {horizons}")
    
    # 創建配置
    config = create_default_config()
    config.update({
        'student_model_type': 'multi_horizon',
        'horizons': horizons,
        'epochs': 50,
        'batch_size': 32,
        'alpha': 0.5,
        'output_dir': 'kd_demo_multi_horizon'
    })
    
    # 創建訓練器
    trainer = KnowledgeDistillationTrainer(config)
    
    # 準備數據
    data = trainer.prepare_data(X, y_multi)
    
    # 訓練老師集成（簡化版本）
    print("📚 訓練老師集成模型...")
    teacher_models = create_default_teacher_models()
    teacher_ensemble = TeacherEnsemble(teacher_models, 'weighted')
    teacher_ensemble.fit(data['X_train_scaled'], data['y_train_scaled'], cv_folds=3)
    
    # 生成老師預測
    print("🔮 生成老師預測...")
    y_train_teacher = teacher_ensemble.predict(data['X_train_scaled'])
    y_test_teacher = teacher_ensemble.predict(data['X_test_scaled'])
    
    teacher_predictions = {
        'y_train_teacher': y_train_teacher,
        'y_test_teacher': y_test_teacher
    }
    
    # 訓練學生模型
    print("🎓 訓練學生模型...")
    student_model = trainer.train_student_model(data, teacher_predictions)
    
    # 評估模型
    print("📊 評估模型...")
    metrics = trainer.evaluate_model(data, teacher_predictions)
    
    # 保存結果
    trainer.save_results(metrics, teacher_predictions)
    
    # 打印結果
    print("\n📈 多地平線知識蒸餾結果:")
    print(f"  學生模型 MSE: {metrics['student_mse']:.4f}")
    print(f"  老師模型 MSE: {metrics['teacher_mse']:.4f}")
    print(f"  MSE 比率: {metrics['mse_ratio']:.4f}")
    
    return metrics

def demo_quantile_kd():
    """演示分位數知識蒸餾"""
    print("\n" + "="*60)
    print("🚀 演示 3: 分位數知識蒸餾")
    print("="*60)
    
    # 創建數據
    X, y = create_synthetic_data(n_samples=1000, n_features=20)
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    
    # 創建配置
    config = create_default_config()
    config.update({
        'student_model_type': 'quantile',
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
        'epochs': 50,
        'batch_size': 32,
        'alpha': 0.5,
        'output_dir': 'kd_demo_quantile'
    })
    
    # 創建訓練器
    trainer = KnowledgeDistillationTrainer(config)
    
    # 準備數據
    data = trainer.prepare_data(X, y)
    
    # 訓練老師集成
    print("📚 訓練老師集成模型...")
    teacher_ensemble = trainer.train_teacher_ensemble(data)
    
    # 生成老師預測
    print("🔮 生成老師預測...")
    teacher_predictions = trainer.generate_teacher_predictions(data)
    
    # 訓練學生模型
    print("🎓 訓練學生模型...")
    student_model = trainer.train_student_model(data, teacher_predictions)
    
    # 評估模型
    print("📊 評估模型...")
    metrics = trainer.evaluate_model(data, teacher_predictions)
    
    # 保存結果
    trainer.save_results(metrics, teacher_predictions)
    
    # 打印結果
    print("\n📈 分位數知識蒸餾結果:")
    print(f"  學生模型 MSE: {metrics['student_mse']:.4f}")
    print(f"  老師模型 MSE: {metrics['teacher_mse']:.4f}")
    print(f"  MSE 比率: {metrics['mse_ratio']:.4f}")
    
    return metrics

def demo_sequence_kd():
    """演示序列到序列知識蒸餾"""
    print("\n" + "="*60)
    print("🚀 演示 4: 序列到序列知識蒸餾")
    print("="*60)
    
    # 創建時間序列數據
    X, y = create_time_series_data(n_samples=1000, n_features=20, seq_len=10)
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    
    # 創建配置
    config = create_default_config()
    config.update({
        'student_model_type': 'lstm',
        'hidden_dim': 128,
        'num_layers': 2,
        'epochs': 50,
        'batch_size': 32,
        'alpha': 0.5,
        'output_dir': 'kd_demo_sequence'
    })
    
    # 創建訓練器
    trainer = KnowledgeDistillationTrainer(config)
    
    # 準備數據（簡化版本）
    X_flat = X.reshape(X.shape[0], -1)  # 展平序列
    data = trainer.prepare_data(X_flat, y)
    
    # 訓練老師集成
    print("📚 訓練老師集成模型...")
    teacher_ensemble = trainer.train_teacher_ensemble(data)
    
    # 生成老師預測
    print("🔮 生成老師預測...")
    teacher_predictions = trainer.generate_teacher_predictions(data)
    
    # 訓練學生模型
    print("🎓 訓練學生模型...")
    student_model = trainer.train_student_model(data, teacher_predictions)
    
    # 評估模型
    print("📊 評估模型...")
    metrics = trainer.evaluate_model(data, teacher_predictions)
    
    # 保存結果
    trainer.save_results(metrics, teacher_predictions)
    
    # 打印結果
    print("\n📈 序列到序列知識蒸餾結果:")
    print(f"  學生模型 MSE: {metrics['student_mse']:.4f}")
    print(f"  老師模型 MSE: {metrics['teacher_mse']:.4f}")
    print(f"  MSE 比率: {metrics['mse_ratio']:.4f}")
    
    return metrics

def demo_probabilistic_prediction():
    """演示機率式預測"""
    print("\n" + "="*60)
    print("🚀 演示 5: 機率式預測")
    print("="*60)
    
    # 創建數據
    X, y = create_synthetic_data(n_samples=1000, n_features=20)
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    
    # 創建機率式學生模型
    student_model = ProbabilisticStudent(input_dim=20, hidden_dim=128)
    
    # 創建示例數據
    X_tensor = torch.FloatTensor(X[:100])
    
    # 預測
    print("🔮 進行機率式預測...")
    predictions = student_model.predict(X_tensor)
    
    print(f"預測均值形狀: {predictions['mean'].shape}")
    print(f"預測方差形狀: {predictions['var'].shape}")
    print(f"預測樣本形狀: {predictions['samples'].shape}")
    
    # 計算分位數
    quantiles = predictions['quantiles']
    print(f"分位數: {list(quantiles.keys())}")
    
    # 打印統計信息
    mean_pred = predictions['mean'].mean().item()
    std_pred = predictions['std'].mean().item()
    print(f"平均預測均值: {mean_pred:.4f}")
    print(f"平均預測標準差: {std_pred:.4f}")
    
    return predictions

def create_comparison_plot(results: dict):
    """創建比較圖表"""
    print("\n📊 創建比較圖表...")
    
    # 準備數據
    model_types = list(results.keys())
    mse_ratios = [results[model]['mse_ratio'] for model in model_types]
    r2_ratios = [results[model]['r2_ratio'] for model in model_types]
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE比率
    ax1.bar(model_types, mse_ratios, color='skyblue', alpha=0.7)
    ax1.set_title('MSE比率 (學生/老師)')
    ax1.set_ylabel('MSE比率')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='相等性能')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # R²比率
    ax2.bar(model_types, r2_ratios, color='lightgreen', alpha=0.7)
    ax2.set_title('R²比率 (學生/老師)')
    ax2.set_ylabel('R²比率')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='相等性能')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('knowledge_distillation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 比較圖表已保存為 'knowledge_distillation_comparison.png'")

def main():
    """主函數"""
    print("🚀 SuperFusionAGI 知識蒸餾演示")
    print("="*80)
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 存儲結果
    results = {}
    
    try:
        # 演示 1: 單步回歸知識蒸餾
        results['單步回歸'] = demo_single_step_kd()
        
        # 演示 2: 多地平線知識蒸餾
        results['多地平線'] = demo_multi_horizon_kd()
        
        # 演示 3: 分位數知識蒸餾
        results['分位數'] = demo_quantile_kd()
        
        # 演示 4: 序列到序列知識蒸餾
        results['序列到序列'] = demo_sequence_kd()
        
        # 演示 5: 機率式預測
        demo_probabilistic_prediction()
        
        # 創建比較圖表
        create_comparison_plot(results)
        
        # 保存結果
        results_file = f"knowledge_distillation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ 結果已保存到: {results_file}")
        
        # 打印總結
        print("\n" + "="*80)
        print("🎉 知識蒸餾演示完成！")
        print("="*80)
        
        print("\n📊 總結:")
        for model_type, metrics in results.items():
            print(f"  {model_type}:")
            print(f"    MSE比率: {metrics['mse_ratio']:.4f}")
            print(f"    R²比率: {metrics['r2_ratio']:.4f}")
        
        print("\n🚀 下一步:")
        print("  1. 查看生成的輸出目錄了解詳細結果")
        print("  2. 調整配置參數優化性能")
        print("  3. 在實際數據上測試知識蒸餾")
        print("  4. 集成到 SuperFusionAGI 系統中")
        
    except Exception as e:
        logger.error(f"演示過程中出現錯誤: {e}")
        print(f"\n❌ 演示失敗: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 知識蒸餾演示成功完成！")
    else:
        print("\n❌ 知識蒸餾演示失敗")
