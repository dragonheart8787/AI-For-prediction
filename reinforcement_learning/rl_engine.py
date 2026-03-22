#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強化學習引擎
實現策略優化和環境適應的進階強化學習系統
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gym
from gym import spaces
import pickle

logger = logging.getLogger(__name__)

# 經驗回放緩存
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class RLConfig:
    """強化學習配置"""
    # 環境參數
    state_dim: int = 10
    action_dim: int = 4
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    
    # 網絡參數
    hidden_dim: int = 128
    learning_rate: float = 0.001
    batch_size: int = 64
    
    # 強化學習參數
    gamma: float = 0.99  # 折扣因子
    epsilon_start: float = 1.0  # 初始探索率
    epsilon_end: float = 0.01  # 最終探索率
    epsilon_decay: int = 500  # 探索率衰減
    tau: float = 0.005  # 軟更新參數
    
    # 經驗回放
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # 訓練參數
    update_frequency: int = 4
    target_update_frequency: int = 100
    
    # 算法選擇
    algorithm: str = 'dqn'  # dqn, ddpg, ppo, sac

class TradingEnvironment:
    """交易環境"""
    
    def __init__(self, data: np.ndarray, config: RLConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0.0  # 持倉
        self.entry_price = 0.0
        
        # 動作空間：0=持有, 1=買入, 2=賣出, 3=平倉
        self.action_space = spaces.Discrete(4)
        
        # 狀態空間：[價格, 餘額, 持倉, 技術指標...]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.state_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """重置環境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """執行動作"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {}
        
        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        
        reward = 0.0
        info = {}
        
        # 執行動作
        if action == 1:  # 買入
            if self.balance > current_price:
                self.position = self.balance / current_price
                self.balance = 0.0
                self.entry_price = current_price
                reward = -0.01  # 交易成本
        elif action == 2:  # 賣出
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0.0
                reward = -0.01  # 交易成本
        elif action == 3:  # 平倉
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0.0
                reward = -0.01  # 交易成本
        
        # 計算獎勵
        if self.position > 0:
            # 持倉收益
            price_change = (next_price - self.entry_price) / self.entry_price
            reward += price_change * 10  # 放大收益信號
        
        # 更新狀態
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= len(self.data) - 1
        
        # 計算總資產
        total_assets = self.balance + (self.position * next_price if self.position > 0 else 0)
        info['total_assets'] = total_assets
        info['balance'] = self.balance
        info['position'] = self.position
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """獲取當前狀態"""
        if self.current_step >= len(self.data):
            return np.zeros(self.config.state_dim, dtype=np.float32)
        
        # 基本狀態
        current_price = self.data[self.current_step]
        price_change = (current_price - self.data[max(0, self.current_step - 1)]) / self.data[max(0, self.current_step - 1)]
        
        # 技術指標
        if self.current_step >= 20:
            prices_20 = self.data[self.current_step - 19:self.current_step + 1]
            sma_20 = np.mean(prices_20)
            rsi = self._calculate_rsi(prices_20)
            volatility = np.std(prices_20) / np.mean(prices_20)
        else:
            sma_20 = current_price
            rsi = 50.0
            volatility = 0.0
        
        # 構建狀態向量
        state = np.array([
            current_price / 1000.0,  # 正規化價格
            self.balance / self.initial_balance,  # 正規化餘額
            self.position / 100.0,  # 正規化持倉
            price_change,  # 價格變化率
            sma_20 / 1000.0,  # 正規化移動平均
            rsi / 100.0,  # 正規化RSI
            volatility,  # 波動率
            self.current_step / len(self.data),  # 時間進度
            (self.balance + (self.position * current_price if self.position > 0 else 0)) / self.initial_balance,  # 總資產比例
            float(self.position > 0)  # 是否持倉
        ], dtype=np.float32)
        
        # 確保狀態維度正確
        if len(state) < self.config.state_dim:
            state = np.pad(state, (0, self.config.state_dim - len(state)), 'constant')
        elif len(state) > self.config.state_dim:
            state = state[:self.config.state_dim]
        
        return state
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """計算RSI指標"""
        if len(prices) < 2:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class DQNNetwork(nn.Module):
    """DQN網絡"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DDPGNetwork(nn.Module):
    """DDPG網絡"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class ReplayBuffer:
    """經驗回放緩存"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加經驗"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """採樣經驗"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN智能體"""
    
    def __init__(self, config: RLConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # 網絡
        self.q_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.target_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # 探索參數
        self.epsilon = config.epsilon_start
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay
        
        # 訓練計數器
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """選擇動作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """訓練網絡"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return
        
        if self.step_count % self.config.update_frequency != 0:
            return
        
        # 採樣經驗
        experiences = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # 計算當前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 計算目標Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # 計算損失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目標網絡
        if self.step_count % self.config.target_update_frequency == 0:
            self._soft_update_target_network()
        
        # 更新探索率
        self.epsilon = max(self.config.epsilon_end, self.epsilon - self.epsilon_decay)
        
        self.step_count += 1
    
    def _soft_update_target_network(self):
        """軟更新目標網絡"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

class DDPGAgent:
    """DDPG智能體"""
    
    def __init__(self, config: RLConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Actor網絡
        self.actor = DDPGNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.actor_target = DDPGNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        
        # Critic網絡
        self.critic = DDPGNetwork(config.state_dim + config.action_dim, 1, config.hidden_dim).to(device)
        self.critic_target = DDPGNetwork(config.state_dim + config.action_dim, 1, config.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # 噪聲
        self.noise_std = 0.1
        
        # 訓練計數器
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """選擇動作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if training:
                # 添加噪聲
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """訓練網絡"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return
        
        if self.step_count % self.config.update_frequency != 0:
            return
        
        # 採樣經驗
        experiences = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # 訓練Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
        
        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 訓練Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 軟更新目標網絡
        if self.step_count % self.config.target_update_frequency == 0:
            self._soft_update_target_network()
        
        self.step_count += 1
    
    def _soft_update_target_network(self):
        """軟更新目標網絡"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

class RLEngine:
    """強化學習引擎"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 創建智能體
        if config.algorithm == 'dqn':
            self.agent = DQNAgent(config, self.device)
        elif config.algorithm == 'ddpg':
            self.agent = DDPGAgent(config, self.device)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        
        # 訓練歷史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_assets': [],
            'epsilon_history': [],
            'loss_history': []
        }
    
    def train(self, data: np.ndarray) -> Dict[str, Any]:
        """訓練強化學習智能體"""
        logger.info(f"Starting RL training with {self.config.algorithm} algorithm...")
        
        # 創建環境
        env = TradingEnvironment(data, self.config)
        
        best_reward = float('-inf')
        best_assets = 0.0
        
        for episode in range(self.config.max_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.config.max_steps_per_episode):
                # 選擇動作
                if self.config.algorithm == 'dqn':
                    action = self.agent.select_action(state, training=True)
                else:  # ddpg
                    action = self.agent.select_action(state, training=True)
                    action = int(np.argmax(action))  # 轉換為離散動作
                
                # 執行動作
                next_state, reward, done, info = env.step(action)
                
                # 存儲經驗
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # 訓練智能體
                self.agent.train()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 記錄訓練歷史
            final_assets = info.get('total_assets', env.balance)
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['episode_assets'].append(final_assets)
            
            if hasattr(self.agent, 'epsilon'):
                self.training_history['epsilon_history'].append(self.agent.epsilon)
            
            # 更新最佳結果
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_assets = final_assets
            
            # 打印進度
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_assets = np.mean(self.training_history['episode_assets'][-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                           f"Avg Assets: {avg_assets:.2f}, Best Assets: {best_assets:.2f}")
        
        logger.info("RL training completed!")
        
        return {
            'best_reward': best_reward,
            'best_assets': best_assets,
            'final_assets': final_assets,
            'training_history': self.training_history
        }
    
    def evaluate(self, data: np.ndarray) -> Dict[str, Any]:
        """評估智能體"""
        env = TradingEnvironment(data, self.config)
        state = env.reset()
        
        total_reward = 0.0
        actions_taken = []
        assets_history = []
        
        for step in range(self.config.max_steps_per_episode):
            # 選擇動作（無探索）
            if self.config.algorithm == 'dqn':
                action = self.agent.select_action(state, training=False)
            else:  # ddpg
                action = self.agent.select_action(state, training=False)
                action = int(np.argmax(action))
            
            # 執行動作
            next_state, reward, done, info = env.step(action)
            
            actions_taken.append(action)
            assets_history.append(info.get('total_assets', env.balance))
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'final_assets': assets_history[-1] if assets_history else env.balance,
            'return_rate': (assets_history[-1] - env.initial_balance) / env.initial_balance if assets_history else 0.0,
            'actions_taken': actions_taken,
            'assets_history': assets_history
        }
    
    def get_training_plots(self) -> plt.Figure:
        """獲取訓練圖表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 獎勵歷史
        episodes = range(len(self.training_history['episode_rewards']))
        ax1.plot(episodes, self.training_history['episode_rewards'], alpha=0.6)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # 資產歷史
        ax2.plot(episodes, self.training_history['episode_assets'], alpha=0.6)
        ax2.set_title('Episode Assets')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Assets')
        ax2.grid(True, alpha=0.3)
        
        # 探索率歷史
        if self.training_history['epsilon_history']:
            ax3.plot(episodes, self.training_history['epsilon_history'])
            ax3.set_title('Epsilon Decay')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            ax3.grid(True, alpha=0.3)
        
        # 移動平均獎勵
        window = 100
        if len(self.training_history['episode_rewards']) >= window:
            moving_avg = np.convolve(self.training_history['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(self.training_history['episode_rewards'])), moving_avg)
            ax4.set_title(f'Moving Average Reward (window={window})')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Reward')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'config': {
                'state_dim': self.config.state_dim,
                'action_dim': self.config.action_dim,
                'algorithm': self.config.algorithm,
                'hidden_dim': self.config.hidden_dim,
                'learning_rate': self.config.learning_rate
            },
            'training_history': self.training_history
        }
        
        if self.config.algorithm == 'dqn':
            model_data['q_network_state_dict'] = self.agent.q_network.state_dict()
            model_data['target_network_state_dict'] = self.agent.target_network.state_dict()
        elif self.config.algorithm == 'ddpg':
            model_data['actor_state_dict'] = self.agent.actor.state_dict()
            model_data['critic_state_dict'] = self.agent.critic.state_dict()
        
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str):
        """加載模型"""
        model_data = torch.load(filepath, map_location=self.device)
        
        if self.config.algorithm == 'dqn':
            self.agent.q_network.load_state_dict(model_data['q_network_state_dict'])
            self.agent.target_network.load_state_dict(model_data['target_network_state_dict'])
        elif self.config.algorithm == 'ddpg':
            self.agent.actor.load_state_dict(model_data['actor_state_dict'])
            self.agent.critic.load_state_dict(model_data['critic_state_dict'])
        
        self.training_history = model_data.get('training_history', {})

# 示例使用
if __name__ == "__main__":
    # 創建配置
    config = RLConfig(
        state_dim=10,
        action_dim=4,
        max_episodes=500,
        max_steps_per_episode=100,
        algorithm='dqn'
    )
    
    # 創建RL引擎
    rl_engine = RLEngine(config)
    
    # 生成模擬價格數據
    np.random.seed(42)
    n_steps = 1000
    prices = 100 + np.cumsum(np.random.randn(n_steps) * 0.5)
    
    # 訓練
    training_results = rl_engine.train(prices)
    
    print(f"Training completed!")
    print(f"Best reward: {training_results['best_reward']:.2f}")
    print(f"Best assets: {training_results['best_assets']:.2f}")
    print(f"Final assets: {training_results['final_assets']:.2f}")
    
    # 評估
    evaluation_results = rl_engine.evaluate(prices)
    print(f"\nEvaluation results:")
    print(f"Total reward: {evaluation_results['total_reward']:.2f}")
    print(f"Final assets: {evaluation_results['final_assets']:.2f}")
    print(f"Return rate: {evaluation_results['return_rate']:.2%}")
    
    # 保存模型
    rl_engine.save_model('rl_model.pth')
    
    # 生成訓練圖表
    fig = rl_engine.get_training_plots()
    fig.savefig('rl_training.png', dpi=300, bbox_inches='tight')
    plt.show()
