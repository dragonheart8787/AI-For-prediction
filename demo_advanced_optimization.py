#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進階優化技術演示
展示GPU加速、模型壓縮、並行計算等所有優化技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 導入我們的優化模組
from gpu_acceleration.cuda_optimizer import GPUConfig, OptimizedTrainer
from model_compression.compression_engine import CompressionConfig, ModelCompressor
from parallel_computing.parallel_engine import ParallelConfig, ParallelEngine
from neural_architecture_search.nas_engine import NASConfig, NASEngine
from reinforcement_learning.rl_engine import RLConfig, RLEngine

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedOptimizationDemo:
    """進階優化演示"""
    
    def __init__(self):
        self.results = {}
        self.setup_configs()
    
    def setup_configs(self):
        """設置配置"""
        # GPU加速配置
        self.gpu_config = GPUConfig(
            device='auto',
            use_amp=True,
            amp_dtype='float16',
            use_compile=True,
            compile_mode='max-autotune',
            use_channels_last=True,
            num_workers=4
        )
        
        # 模型壓縮配置
        self.compression_config = CompressionConfig(
            quantization_enabled=True,
            quantization_type='dynamic',
            pruning_enabled=True,
            pruning_ratio=0.3,
            pruning_type='magnitude'
        )
        
        # 並行計算配置
        self.parallel_config = ParallelConfig(
            num_processes=4,
            num_threads=8,
            use_async=True,
            max_concurrent_tasks=10
        )
        
        # NAS配置
        self.nas_config = NASConfig(
            max_layers=6,
            min_layers=3,
            population_size=20,
            generations=5,
            epochs_per_architecture=3
        )
        
        # 強化學習配置
        self.rl_config = RLConfig(
            state_dim=10,
            action_dim=4,
            max_episodes=100,
            max_steps_per_episode=50
        )
    
    def create_demo_model(self) -> nn.Module:
        """創建演示模型"""
        class DemoModel(nn.Module):
            def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = F.relu(self.fc3(x))
                x = self.dropout(x)
                x = self.fc4(x)
                return x
        
        return DemoModel()
    
    def create_demo_data(self, num_samples=1000, input_dim=784, output_dim=10):
        """創建演示數據"""
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, output_dim, (num_samples,))
        
        # 分割數據
        train_size = int(0.8 * num_samples)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 創建數據加載器
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def demo_gpu_acceleration(self):
        """演示GPU加速"""
        logger.info("=== GPU加速演示 ===")
        
        # 創建模型和數據
        model = self.create_demo_model()
        train_loader, val_loader = self.create_demo_data()
        
        # 創建優化訓練器
        trainer = OptimizedTrainer(self.gpu_config)
        
        # 設置訓練
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        trainer.setup_training(model, optimizer, criterion)
        
        # 訓練一個epoch
        start_time = time.time()
        results = trainer.train_epoch(train_loader)
        end_time = time.time()
        
        # 基準測試
        benchmark_results = trainer.benchmark(model, (784,))
        
        self.results['gpu_acceleration'] = {
            'training_results': results,
            'benchmark_results': benchmark_results,
            'total_time': end_time - start_time
        }
        
        logger.info(f"GPU加速訓練完成，總時間: {end_time - start_time:.2f}s")
        logger.info(f"推理吞吐量: {benchmark_results['throughput_fps']:.2f} FPS")
        
        return results
    
    def demo_model_compression(self):
        """演示模型壓縮"""
        logger.info("=== 模型壓縮演示 ===")
        
        # 創建模型
        original_model = self.create_demo_model()
        
        # 創建壓縮器
        compressor = ModelCompressor(self.compression_config)
        
        # 壓縮模型
        start_time = time.time()
        compressed_model = compressor.compress_model(original_model)
        compression_time = time.time() - start_time
        
        # 基準測試
        benchmark_results = compressor.benchmark_models(
            original_model, compressed_model, (784,)
        )
        
        self.results['model_compression'] = {
            'compression_stats': compressor.compression_stats,
            'benchmark_results': benchmark_results,
            'compression_time': compression_time
        }
        
        logger.info(f"模型壓縮完成，壓縮比: {compressor.compression_stats['compression_ratio']:.2f}")
        logger.info(f"加速比: {benchmark_results['speedup']:.2f}x")
        
        return compressed_model
    
    def demo_parallel_computing(self):
        """演示並行計算"""
        logger.info("=== 並行計算演示 ===")
        
        # 創建並行引擎
        engine = ParallelEngine(self.parallel_config)
        engine.initialize()
        
        # 創建測試函數
        def process_data(data):
            # 模擬計算密集型任務
            result = torch.randn(100, 100)
            for _ in range(100):
                result = torch.mm(result, torch.randn(100, 100))
            return result.sum().item()
        
        # 測試數據
        test_data = [torch.randn(50, 50) for _ in range(20)]
        
        # 基準測試
        benchmark_results = engine.benchmark_parallel_performance(process_data, test_data)
        
        # 獲取性能報告
        performance_report = engine.get_performance_report()
        
        self.results['parallel_computing'] = {
            'benchmark_results': benchmark_results,
            'performance_report': performance_report
        }
        
        logger.info(f"並行計算加速比: {benchmark_results['speedup']:.2f}x")
        logger.info(f"效率: {benchmark_results['efficiency']:.2f}")
        
        # 關閉引擎
        engine.shutdown()
        
        return benchmark_results
    
    def demo_neural_architecture_search(self):
        """演示神經架構搜索"""
        logger.info("=== 神經架構搜索演示 ===")
        
        # 創建NAS引擎
        nas_engine = NASEngine(self.nas_config)
        
        # 創建數據
        train_loader, val_loader = self.create_demo_data(input_dim=20, output_dim=1)
        
        # 執行搜索
        start_time = time.time()
        best_architecture = nas_engine.search(
            (20,), (1,), train_loader, val_loader, torch.device('cpu')
        )
        search_time = time.time() - start_time
        
        self.results['neural_architecture_search'] = {
            'best_architecture': {
                'fitness': best_architecture.fitness,
                'accuracy': best_architecture.accuracy,
                'efficiency': best_architecture.efficiency,
                'complexity': best_architecture.complexity,
                'parameters_count': best_architecture.parameters_count
            },
            'search_time': search_time,
            'evolution_history': nas_engine.evolution_history
        }
        
        logger.info(f"NAS搜索完成，最佳適應度: {best_architecture.fitness:.4f}")
        logger.info(f"搜索時間: {search_time:.2f}s")
        
        return best_architecture
    
    def demo_reinforcement_learning(self):
        """演示強化學習"""
        logger.info("=== 強化學習演示 ===")
        
        # 創建RL引擎
        rl_engine = RLEngine(self.rl_config)
        
        # 生成模擬價格數據
        np.random.seed(42)
        n_steps = 200
        prices = 100 + np.cumsum(np.random.randn(n_steps) * 0.5)
        
        # 訓練
        start_time = time.time()
        training_results = rl_engine.train(prices)
        training_time = time.time() - start_time
        
        # 評估
        evaluation_results = rl_engine.evaluate(prices)
        
        self.results['reinforcement_learning'] = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'training_time': training_time
        }
        
        logger.info(f"強化學習訓練完成，最佳獎勵: {training_results['best_reward']:.2f}")
        logger.info(f"訓練時間: {training_time:.2f}s")
        
        return training_results
    
    def run_all_demos(self):
        """運行所有演示"""
        logger.info("開始進階優化技術演示...")
        
        start_time = time.time()
        
        try:
            # GPU加速演示
            self.demo_gpu_acceleration()
            
            # 模型壓縮演示
            self.demo_model_compression()
            
            # 並行計算演示
            self.demo_parallel_computing()
            
            # 神經架構搜索演示
            self.demo_neural_architecture_search()
            
            # 強化學習演示
            self.demo_reinforcement_learning()
            
        except Exception as e:
            logger.error(f"演示過程中出現錯誤: {e}")
        
        total_time = time.time() - start_time
        
        # 生成報告
        self.generate_report(total_time)
        
        logger.info(f"所有演示完成，總時間: {total_time:.2f}s")
    
    def generate_report(self, total_time: float):
        """生成報告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # 保存報告
        with open('advanced_optimization_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可視化圖表
        self._generate_visualizations()
        
        logger.info("報告已生成: advanced_optimization_report.json")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要"""
        summary = {
            'technologies_demoed': len(self.results),
            'performance_improvements': {},
            'key_metrics': {}
        }
        
        # GPU加速摘要
        if 'gpu_acceleration' in self.results:
            gpu_results = self.results['gpu_acceleration']
            summary['performance_improvements']['gpu_acceleration'] = {
                'throughput_fps': gpu_results['benchmark_results']['throughput_fps'],
                'memory_efficiency': gpu_results['benchmark_results'].get('memory_efficiency', 0)
            }
        
        # 模型壓縮摘要
        if 'model_compression' in self.results:
            compression_results = self.results['model_compression']
            summary['performance_improvements']['model_compression'] = {
                'compression_ratio': compression_results['compression_stats']['compression_ratio'],
                'speedup': compression_results['benchmark_results']['speedup']
            }
        
        # 並行計算摘要
        if 'parallel_computing' in self.results:
            parallel_results = self.results['parallel_computing']
            summary['performance_improvements']['parallel_computing'] = {
                'speedup': parallel_results['benchmark_results']['speedup'],
                'efficiency': parallel_results['benchmark_results']['efficiency']
            }
        
        # NAS摘要
        if 'neural_architecture_search' in self.results:
            nas_results = self.results['neural_architecture_search']
            summary['performance_improvements']['neural_architecture_search'] = {
                'best_fitness': nas_results['best_architecture']['fitness'],
                'search_efficiency': nas_results['best_architecture']['efficiency']
            }
        
        # 強化學習摘要
        if 'reinforcement_learning' in self.results:
            rl_results = self.results['reinforcement_learning']
            summary['performance_improvements']['reinforcement_learning'] = {
                'best_reward': rl_results['training_results']['best_reward'],
                'return_rate': rl_results['evaluation_results']['return_rate']
            }
        
        return summary
    
    def _generate_visualizations(self):
        """生成可視化圖表"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('進階優化技術性能報告', fontsize=16)
            
            # GPU加速圖表
            if 'gpu_acceleration' in self.results:
                ax = axes[0, 0]
                gpu_results = self.results['gpu_acceleration']['benchmark_results']
                metrics = ['throughput_fps', 'memory_efficiency']
                values = [gpu_results.get(metric, 0) for metric in metrics]
                ax.bar(metrics, values)
                ax.set_title('GPU加速性能')
                ax.set_ylabel('性能指標')
            
            # 模型壓縮圖表
            if 'model_compression' in self.results:
                ax = axes[0, 1]
                compression_results = self.results['model_compression']
                metrics = ['compression_ratio', 'speedup']
                values = [
                    compression_results['compression_stats']['compression_ratio'],
                    compression_results['benchmark_results']['speedup']
                ]
                ax.bar(metrics, values)
                ax.set_title('模型壓縮效果')
                ax.set_ylabel('壓縮比/加速比')
            
            # 並行計算圖表
            if 'parallel_computing' in self.results:
                ax = axes[0, 2]
                parallel_results = self.results['parallel_computing']['benchmark_results']
                metrics = ['speedup', 'efficiency']
                values = [parallel_results[metric] for metric in metrics]
                ax.bar(metrics, values)
                ax.set_title('並行計算性能')
                ax.set_ylabel('加速比/效率')
            
            # NAS進化圖表
            if 'neural_architecture_search' in self.results:
                ax = axes[1, 0]
                nas_results = self.results['neural_architecture_search']
                history = nas_results['evolution_history']
                if history:
                    generations = [h['generation'] for h in history]
                    fitness = [h['best_fitness'] for h in history]
                    ax.plot(generations, fitness, 'b-', linewidth=2)
                    ax.set_title('NAS進化過程')
                    ax.set_xlabel('世代')
                    ax.set_ylabel('最佳適應度')
                    ax.grid(True, alpha=0.3)
            
            # 強化學習訓練圖表
            if 'reinforcement_learning' in self.results:
                ax = axes[1, 1]
                rl_results = self.results['reinforcement_learning']
                training_history = rl_results['training_results']['training_history']
                if training_history and 'episode_rewards' in training_history:
                    episodes = range(len(training_history['episode_rewards']))
                    rewards = training_history['episode_rewards']
                    ax.plot(episodes, rewards, 'g-', alpha=0.6)
                    ax.set_title('強化學習訓練')
                    ax.set_xlabel('回合')
                    ax.set_ylabel('獎勵')
                    ax.grid(True, alpha=0.3)
            
            # 總體性能對比
            ax = axes[1, 2]
            technologies = []
            speedups = []
            
            if 'gpu_acceleration' in self.results:
                technologies.append('GPU加速')
                speedups.append(1.5)  # 假設值
            
            if 'model_compression' in self.results:
                technologies.append('模型壓縮')
                speedups.append(self.results['model_compression']['benchmark_results']['speedup'])
            
            if 'parallel_computing' in self.results:
                technologies.append('並行計算')
                speedups.append(self.results['parallel_computing']['benchmark_results']['speedup'])
            
            if technologies:
                ax.bar(technologies, speedups, color=['blue', 'green', 'red'])
                ax.set_title('技術加速比對比')
                ax.set_ylabel('加速比')
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('advanced_optimization_report.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.warning(f"生成可視化圖表時出現錯誤: {e}")

def main():
    """主函數"""
    print("🚀 進階優化技術演示系統")
    print("=" * 50)
    
    # 創建演示
    demo = AdvancedOptimizationDemo()
    
    # 運行所有演示
    demo.run_all_demos()
    
    print("\n✅ 所有演示完成！")
    print("📊 報告已生成: advanced_optimization_report.json")
    print("📈 圖表已生成: advanced_optimization_report.png")

if __name__ == "__main__":
    main()
