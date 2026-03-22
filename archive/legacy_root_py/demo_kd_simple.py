#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版知識蒸餾演示腳本
只演示單步回歸知識蒸餾
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 導入知識蒸餾模組
from knowledge_distillation.kd_trainer import KnowledgeDistillationTrainer, create_default_config
from knowledge_distillation.probabilistic_student import ProbabilisticStudent

def create_synthetic_data(n_samples: int = 1000, n_features: int = 20, 
                         noise_level: float = 0.1, random_state: int = 42) -> tuple:
    """創建合成數據"""
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

def demo_single_step_kd():
    """演示單步回歸知識蒸餾"""
    print("\n" + "="*60)
    print("🚀 單步回歸知識蒸餾演示")
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
        'output_dir': 'kd_demo_simple'
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

def demo_probabilistic_prediction():
    """演示機率式預測"""
    print("\n" + "="*60)
    print("🚀 機率式預測演示")
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

def create_visualization(metrics: dict):
    """創建可視化圖表"""
    print("\n📊 創建可視化圖表...")
    
    # 創建圖表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE比較
    categories = ['學生模型', '老師模型']
    mse_values = [metrics['student_mse'], metrics['teacher_mse']]
    ax1.bar(categories, mse_values, color=['skyblue', 'lightgreen'], alpha=0.7)
    ax1.set_title('MSE比較')
    ax1.set_ylabel('MSE')
    for i, v in enumerate(mse_values):
        ax1.text(i, v + 0.1, f'{v:.4f}', ha='center', va='bottom')
    
    # R²比較
    r2_values = [metrics['student_r2'], metrics['teacher_r2']]
    ax2.bar(categories, r2_values, color=['skyblue', 'lightgreen'], alpha=0.7)
    ax2.set_title('R²比較')
    ax2.set_ylabel('R²')
    for i, v in enumerate(r2_values):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # 比率比較
    ratio_categories = ['MSE比率', 'R²比率']
    ratio_values = [metrics['mse_ratio'], metrics['r2_ratio']]
    ax3.bar(ratio_categories, ratio_values, color=['orange', 'purple'], alpha=0.7)
    ax3.set_title('性能比率 (學生/老師)')
    ax3.set_ylabel('比率')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='相等性能')
    ax3.legend()
    for i, v in enumerate(ratio_values):
        ax3.text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom')
    
    # 性能總結
    ax4.text(0.1, 0.8, f"學生模型 MSE: {metrics['student_mse']:.4f}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f"老師模型 MSE: {metrics['teacher_mse']:.4f}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f"MSE 比率: {metrics['mse_ratio']:.4f}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f"學生模型 R²: {metrics['student_r2']:.4f}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f"老師模型 R²: {metrics['teacher_r2']:.4f}", fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f"R² 比率: {metrics['r2_ratio']:.4f}", fontsize=12, transform=ax4.transAxes)
    
    # 性能評估
    if metrics['mse_ratio'] < 1.2:
        performance = "優秀"
        color = "green"
    elif metrics['mse_ratio'] < 1.5:
        performance = "良好"
        color = "orange"
    else:
        performance = "需要改進"
        color = "red"
    
    ax4.text(0.1, 0.2, f"性能評估: {performance}", fontsize=14, color=color, transform=ax4.transAxes)
    ax4.set_title('性能總結')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('knowledge_distillation_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 可視化圖表已保存為 'knowledge_distillation_simple.png'")

def main():
    """主函數"""
    print("🚀 SuperFusionAGI 知識蒸餾簡化演示")
    print("="*80)
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # 演示 1: 單步回歸知識蒸餾
        metrics = demo_single_step_kd()
        
        # 演示 2: 機率式預測
        predictions = demo_probabilistic_prediction()
        
        # 創建可視化
        create_visualization(metrics)
        
        # 保存結果
        results = {
            'single_step_kd': metrics,
            'probabilistic_prediction': {
                'mean_pred': float(predictions['mean'].mean().item()),
                'std_pred': float(predictions['std'].mean().item()),
                'quantiles': {str(k): float(v.mean().item()) for k, v in predictions['quantiles'].items()}
            }
        }
        
        results_file = f"kd_simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ 結果已保存到: {results_file}")
        
        # 打印總結
        print("\n" + "="*80)
        print("🎉 知識蒸餾簡化演示完成！")
        print("="*80)
        
        print("\n📊 總結:")
        print(f"  單步回歸知識蒸餾:")
        print(f"    MSE比率: {metrics['mse_ratio']:.4f}")
        print(f"    R²比率: {metrics['r2_ratio']:.4f}")
        
        print(f"\n  機率式預測:")
        print(f"    平均預測均值: {predictions['mean'].mean().item():.4f}")
        print(f"    平均預測標準差: {predictions['std'].mean().item():.4f}")
        
        print("\n🚀 下一步:")
        print("  1. 查看生成的輸出目錄了解詳細結果")
        print("  2. 調整配置參數優化性能")
        print("  3. 在實際數據上測試知識蒸餾")
        print("  4. 集成到 SuperFusionAGI 系統中")
        
        return True
        
    except Exception as e:
        logger.error(f"演示過程中出現錯誤: {e}")
        print(f"\n❌ 演示失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 知識蒸餾簡化演示成功完成！")
    else:
        print("\n❌ 知識蒸餾簡化演示失敗")
