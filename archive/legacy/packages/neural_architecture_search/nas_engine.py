#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神經架構搜索 (Neural Architecture Search) 引擎
自動設計最佳神經網絡架構的進階AI技術
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
import itertools

logger = logging.getLogger(__name__)

@dataclass
class NASConfig:
    """NAS配置"""
    # 搜索空間
    max_layers: int = 10
    min_layers: int = 3
    layer_types: List[str] = field(default_factory=lambda: [
        'conv1d', 'conv2d', 'lstm', 'gru', 'transformer', 'linear', 'attention'
    ])
    activation_functions: List[str] = field(default_factory=lambda: [
        'relu', 'leaky_relu', 'gelu', 'swish', 'mish', 'tanh', 'sigmoid'
    ])
    
    # 進化參數
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # 訓練參數
    epochs_per_architecture: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # 目標函數權重
    accuracy_weight: float = 0.7
    efficiency_weight: float = 0.2
    complexity_weight: float = 0.1
    
    # 早停參數
    patience: int = 5
    min_delta: float = 0.001

@dataclass
class LayerConfig:
    """層配置"""
    layer_type: str
    parameters: Dict[str, Any]
    activation: str
    dropout: float = 0.0
    batch_norm: bool = False

@dataclass
class Architecture:
    """神經網絡架構"""
    layers: List[LayerConfig]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    fitness: float = 0.0
    accuracy: float = 0.0
    efficiency: float = 0.0
    complexity: float = 0.0
    training_time: float = 0.0
    parameters_count: int = 0

class LayerFactory:
    """層工廠"""
    
    @staticmethod
    def create_layer(config: LayerConfig, input_dim: int, output_dim: int) -> nn.Module:
        """創建層"""
        if config.layer_type == 'linear':
            layer = nn.Linear(input_dim, output_dim)
        elif config.layer_type == 'conv1d':
            layer = nn.Conv1d(
                input_dim, 
                output_dim, 
                kernel_size=config.parameters.get('kernel_size', 3),
                padding=config.parameters.get('padding', 1)
            )
        elif config.layer_type == 'conv2d':
            layer = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=config.parameters.get('kernel_size', 3),
                padding=config.parameters.get('padding', 1)
            )
        elif config.layer_type == 'lstm':
            layer = nn.LSTM(
                input_dim,
                output_dim,
                num_layers=config.parameters.get('num_layers', 1),
                batch_first=True,
                dropout=config.dropout
            )
        elif config.layer_type == 'gru':
            layer = nn.GRU(
                input_dim,
                output_dim,
                num_layers=config.parameters.get('num_layers', 1),
                batch_first=True,
                dropout=config.dropout
            )
        elif config.layer_type == 'transformer':
            layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=config.parameters.get('nhead', 8),
                dim_feedforward=output_dim,
                dropout=config.dropout,
                batch_first=True
            )
        elif config.layer_type == 'attention':
            layer = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=config.parameters.get('num_heads', 8),
                dropout=config.dropout,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")
        
        return layer
    
    @staticmethod
    def get_activation(activation_name: str) -> nn.Module:
        """獲取激活函數"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation_name, nn.ReLU())

class ArchitectureGenerator:
    """架構生成器"""
    
    def __init__(self, config: NASConfig):
        self.config = config
    
    def generate_random_architecture(self, input_shape: Tuple[int, ...], 
                                   output_shape: Tuple[int, ...]) -> Architecture:
        """生成隨機架構"""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        layers = []
        
        current_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
        
        for i in range(num_layers):
            # 選擇層類型
            layer_type = random.choice(self.config.layer_types)
            
            # 生成參數
            parameters = self._generate_layer_parameters(layer_type)
            
            # 選擇激活函數
            activation = random.choice(self.config.activation_functions)
            
            # 生成dropout
            dropout = random.uniform(0.0, 0.5)
            
            # 是否使用batch norm
            batch_norm = random.choice([True, False])
            
            # 計算輸出維度
            if i == num_layers - 1:
                output_dim = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
            else:
                output_dim = random.randint(32, 512)
            
            layer_config = LayerConfig(
                layer_type=layer_type,
                parameters=parameters,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            )
            
            layers.append(layer_config)
            current_dim = output_dim
        
        architecture = Architecture(
            layers=layers,
            input_shape=input_shape,
            output_shape=output_shape
        )
        
        return architecture
    
    def _generate_layer_parameters(self, layer_type: str) -> Dict[str, Any]:
        """生成層參數"""
        if layer_type in ['conv1d', 'conv2d']:
            return {
                'kernel_size': random.choice([3, 5, 7]),
                'padding': random.choice([0, 1, 2])
            }
        elif layer_type in ['lstm', 'gru']:
            return {
                'num_layers': random.randint(1, 3)
            }
        elif layer_type == 'transformer':
            return {
                'nhead': random.choice([4, 8, 16])
            }
        elif layer_type == 'attention':
            return {
                'num_heads': random.choice([4, 8, 16])
            }
        else:
            return {}

class ArchitectureEvaluator:
    """架構評估器"""
    
    def __init__(self, config: NASConfig):
        self.config = config
    
    def evaluate_architecture(self, architecture: Architecture, 
                            train_loader, val_loader, 
                            device: torch.device) -> Dict[str, float]:
        """評估架構"""
        try:
            # 創建模型
            model = self._build_model(architecture)
            model = model.to(device)
            
            # 計算參數數量
            parameters_count = sum(p.numel() for p in model.parameters())
            architecture.parameters_count = parameters_count
            
            # 訓練模型
            start_time = time.time()
            accuracy = self._train_model(model, train_loader, val_loader, device)
            training_time = time.time() - start_time
            
            # 計算效率分數
            efficiency = self._calculate_efficiency(architecture, training_time)
            
            # 計算複雜度分數
            complexity = self._calculate_complexity(architecture)
            
            # 計算總體適應度
            fitness = (
                self.config.accuracy_weight * accuracy +
                self.config.efficiency_weight * efficiency +
                self.config.complexity_weight * (1.0 - complexity)
            )
            
            return {
                'fitness': fitness,
                'accuracy': accuracy,
                'efficiency': efficiency,
                'complexity': complexity,
                'training_time': training_time,
                'parameters_count': parameters_count
            }
            
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return {
                'fitness': 0.0,
                'accuracy': 0.0,
                'efficiency': 0.0,
                'complexity': 1.0,
                'training_time': 0.0,
                'parameters_count': 0
            }
    
    def _build_model(self, architecture: Architecture) -> nn.Module:
        """構建模型"""
        class NASModel(nn.Module):
            def __init__(self, arch: Architecture):
                super().__init__()
                self.architecture = arch
                self.layers = nn.ModuleList()
                self.activations = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                
                current_dim = arch.input_shape[-1] if len(arch.input_shape) > 1 else arch.input_shape[0]
                
                for i, layer_config in enumerate(arch.layers):
                    # 計算輸出維度
                    if i == len(arch.layers) - 1:
                        output_dim = arch.output_shape[-1] if len(arch.output_shape) > 1 else arch.output_shape[0]
                    else:
                        output_dim = random.randint(32, 512)
                    
                    # 創建層
                    layer = LayerFactory.create_layer(layer_config, current_dim, output_dim)
                    self.layers.append(layer)
                    
                    # 創建激活函數
                    activation = LayerFactory.get_activation(layer_config.activation)
                    self.activations.append(activation)
                    
                    # 創建dropout
                    if layer_config.dropout > 0:
                        self.dropouts.append(nn.Dropout(layer_config.dropout))
                    else:
                        self.dropouts.append(nn.Identity())
                    
                    # 創建batch norm
                    if layer_config.batch_norm:
                        if layer_config.layer_type in ['conv1d', 'conv2d']:
                            self.batch_norms.append(nn.BatchNorm1d(output_dim))
                        else:
                            self.batch_norms.append(nn.BatchNorm1d(output_dim))
                    else:
                        self.batch_norms.append(nn.Identity())
                    
                    current_dim = output_dim
            
            def forward(self, x):
                for layer, activation, dropout, batch_norm in zip(
                    self.layers, self.activations, self.dropouts, self.batch_norms
                ):
                    x = layer(x)
                    x = batch_norm(x)
                    x = activation(x)
                    x = dropout(x)
                return x
        
        return NASModel(architecture)
    
    def _train_model(self, model: nn.Module, train_loader, val_loader, 
                    device: torch.device) -> float:
        """訓練模型"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs_per_architecture):
            # 訓練
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 驗證
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break
        
        # 返回準確率（1 - 正規化損失）
        return max(0.0, 1.0 - best_val_loss)
    
    def _calculate_efficiency(self, architecture: Architecture, training_time: float) -> float:
        """計算效率分數"""
        # 基於訓練時間和參數數量的效率
        time_score = 1.0 / (1.0 + training_time / 100.0)  # 正規化到0-1
        param_score = 1.0 / (1.0 + architecture.parameters_count / 1000000.0)  # 正規化到0-1
        return (time_score + param_score) / 2.0
    
    def _calculate_complexity(self, architecture: Architecture) -> float:
        """計算複雜度分數"""
        # 基於層數和層類型的複雜度
        layer_complexity = {
            'linear': 1.0,
            'conv1d': 2.0,
            'conv2d': 3.0,
            'lstm': 4.0,
            'gru': 3.5,
            'transformer': 5.0,
            'attention': 4.5
        }
        
        total_complexity = sum(
            layer_complexity.get(layer.layer_type, 1.0) 
            for layer in architecture.layers
        )
        
        # 正規化到0-1
        max_complexity = len(architecture.layers) * 5.0
        return min(1.0, total_complexity / max_complexity)

class GeneticAlgorithm:
    """遺傳算法"""
    
    def __init__(self, config: NASConfig):
        self.config = config
    
    def evolve(self, population: List[Architecture], 
              evaluator: ArchitectureEvaluator,
              train_loader, val_loader, device: torch.device) -> List[Architecture]:
        """進化一代"""
        # 評估適應度
        for arch in population:
            if arch.fitness == 0.0:  # 未評估的架構
                results = evaluator.evaluate_architecture(arch, train_loader, val_loader, device)
                arch.fitness = results['fitness']
                arch.accuracy = results['accuracy']
                arch.efficiency = results['efficiency']
                arch.complexity = results['complexity']
                arch.training_time = results['training_time']
                arch.parameters_count = results['parameters_count']
        
        # 排序
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 選擇精英
        elite = population[:self.config.elite_size]
        
        # 生成新個體
        new_population = elite.copy()
        
        while len(new_population) < self.config.population_size:
            # 選擇父母
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # 交叉
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # 變異
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 確保不超過種群大小
        return new_population[:self.config.population_size]
    
    def _tournament_selection(self, population: List[Architecture], 
                            tournament_size: int = 3) -> Architecture:
        """錦標賽選擇"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Tuple[Architecture, Architecture]:
        """交叉操作"""
        # 單點交叉
        min_layers = min(len(parent1.layers), len(parent2.layers))
        if min_layers < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(1, min_layers - 1)
        
        child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        
        child1 = Architecture(
            layers=child1_layers,
            input_shape=parent1.input_shape,
            output_shape=parent1.output_shape
        )
        
        child2 = Architecture(
            layers=child2_layers,
            input_shape=parent2.input_shape,
            output_shape=parent2.output_shape
        )
        
        return child1, child2
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """變異操作"""
        mutated = copy.deepcopy(architecture)
        
        mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'change_activation'])
        
        if mutation_type == 'add_layer' and len(mutated.layers) < self.config.max_layers:
            # 添加層
            layer_config = self._generate_random_layer()
            insert_pos = random.randint(0, len(mutated.layers))
            mutated.layers.insert(insert_pos, layer_config)
            
        elif mutation_type == 'remove_layer' and len(mutated.layers) > self.config.min_layers:
            # 移除層
            remove_pos = random.randint(0, len(mutated.layers) - 1)
            mutated.layers.pop(remove_pos)
            
        elif mutation_type == 'modify_layer' and mutated.layers:
            # 修改層
            modify_pos = random.randint(0, len(mutated.layers) - 1)
            mutated.layers[modify_pos] = self._generate_random_layer()
            
        elif mutation_type == 'change_activation' and mutated.layers:
            # 改變激活函數
            change_pos = random.randint(0, len(mutated.layers) - 1)
            mutated.layers[change_pos].activation = random.choice(self.config.activation_functions)
        
        return mutated
    
    def _generate_random_layer(self) -> LayerConfig:
        """生成隨機層"""
        layer_type = random.choice(self.config.layer_types)
        parameters = self._generate_layer_parameters(layer_type)
        activation = random.choice(self.config.activation_functions)
        dropout = random.uniform(0.0, 0.5)
        batch_norm = random.choice([True, False])
        
        return LayerConfig(
            layer_type=layer_type,
            parameters=parameters,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )
    
    def _generate_layer_parameters(self, layer_type: str) -> Dict[str, Any]:
        """生成層參數"""
        if layer_type in ['conv1d', 'conv2d']:
            return {
                'kernel_size': random.choice([3, 5, 7]),
                'padding': random.choice([0, 1, 2])
            }
        elif layer_type in ['lstm', 'gru']:
            return {
                'num_layers': random.randint(1, 3)
            }
        elif layer_type == 'transformer':
            return {
                'nhead': random.choice([4, 8, 16])
            }
        elif layer_type == 'attention':
            return {
                'num_heads': random.choice([4, 8, 16])
            }
        else:
            return {}

class NASEngine:
    """神經架構搜索引擎"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.generator = ArchitectureGenerator(config)
        self.evaluator = ArchitectureEvaluator(config)
        self.genetic_algorithm = GeneticAlgorithm(config)
        
        self.population = []
        self.best_architecture = None
        self.evolution_history = []
        
    def search(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
              train_loader, val_loader, device: torch.device) -> Architecture:
        """執行神經架構搜索"""
        logger.info("Starting Neural Architecture Search...")
        
        # 初始化種群
        self.population = [
            self.generator.generate_random_architecture(input_shape, output_shape)
            for _ in range(self.config.population_size)
        ]
        
        # 進化
        for generation in range(self.config.generations):
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # 進化一代
            self.population = self.genetic_algorithm.evolve(
                self.population, self.evaluator, train_loader, val_loader, device
            )
            
            # 記錄最佳架構
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_architecture is None or current_best.fitness > self.best_architecture.fitness:
                self.best_architecture = current_best
            
            # 記錄進化歷史
            generation_stats = {
                'generation': generation + 1,
                'best_fitness': current_best.fitness,
                'avg_fitness': np.mean([arch.fitness for arch in self.population]),
                'best_accuracy': current_best.accuracy,
                'best_efficiency': current_best.efficiency,
                'best_complexity': current_best.complexity
            }
            self.evolution_history.append(generation_stats)
            
            logger.info(f"Best fitness: {current_best.fitness:.4f}, "
                       f"Accuracy: {current_best.accuracy:.4f}, "
                       f"Efficiency: {current_best.efficiency:.4f}")
        
        logger.info("Neural Architecture Search completed!")
        return self.best_architecture
    
    def get_evolution_plot(self) -> plt.Figure:
        """獲取進化圖表"""
        if not self.evolution_history:
            return None
        
        generations = [stats['generation'] for stats in self.evolution_history]
        best_fitness = [stats['best_fitness'] for stats in self.evolution_history]
        avg_fitness = [stats['avg_fitness'] for stats in self.evolution_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 適應度進化
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 多目標進化
        best_accuracy = [stats['best_accuracy'] for stats in self.evolution_history]
        best_efficiency = [stats['best_efficiency'] for stats in self.evolution_history]
        best_complexity = [stats['best_complexity'] for stats in self.evolution_history]
        
        ax2.plot(generations, best_accuracy, 'g-', label='Accuracy', linewidth=2)
        ax2.plot(generations, best_efficiency, 'm-', label='Efficiency', linewidth=2)
        ax2.plot(generations, best_complexity, 'c-', label='Complexity', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Score')
        ax2.set_title('Multi-Objective Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filepath: str):
        """保存結果"""
        results = {
            'config': {
                'max_layers': self.config.max_layers,
                'min_layers': self.config.min_layers,
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate
            },
            'best_architecture': {
                'layers': [
                    {
                        'layer_type': layer.layer_type,
                        'parameters': layer.parameters,
                        'activation': layer.activation,
                        'dropout': layer.dropout,
                        'batch_norm': layer.batch_norm
                    }
                    for layer in self.best_architecture.layers
                ],
                'input_shape': self.best_architecture.input_shape,
                'output_shape': self.best_architecture.output_shape,
                'fitness': self.best_architecture.fitness,
                'accuracy': self.best_architecture.accuracy,
                'efficiency': self.best_architecture.efficiency,
                'complexity': self.best_architecture.complexity,
                'training_time': self.best_architecture.training_time,
                'parameters_count': self.best_architecture.parameters_count
            },
            'evolution_history': self.evolution_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = NASConfig(
        max_layers=8,
        min_layers=3,
        population_size=20,
        generations=10,
        epochs_per_architecture=5
    )
    
    # 創建NAS引擎
    nas_engine = NASEngine(config)
    
    # 模擬數據
    input_shape = (100, 20)  # (batch_size, features)
    output_shape = (100, 1)  # (batch_size, output)
    
    # 創建模擬數據加載器
    X_train = torch.randn(1000, 20)
    y_train = torch.randn(1000, 1)
    X_val = torch.randn(200, 20)
    y_val = torch.randn(200, 1)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 執行搜索
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_architecture = nas_engine.search(
        input_shape, output_shape, train_loader, val_loader, device
    )
    
    print(f"Best architecture found:")
    print(f"Fitness: {best_architecture.fitness:.4f}")
    print(f"Accuracy: {best_architecture.accuracy:.4f}")
    print(f"Efficiency: {best_architecture.efficiency:.4f}")
    print(f"Complexity: {best_architecture.complexity:.4f}")
    print(f"Parameters: {best_architecture.parameters_count}")
    
    # 保存結果
    nas_engine.save_results('nas_results.json')
    
    # 生成進化圖表
    fig = nas_engine.get_evolution_plot()
    if fig:
        fig.savefig('nas_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
