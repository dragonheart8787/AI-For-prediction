#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列到序列知識蒸餾
實現LSTM/Transformer學生的序列預測知識蒸餾
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class LSTMStudent(nn.Module):
    """LSTM學生模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 設置遺忘門偏置為1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            預測結果 [batch_size, seq_len, output_dim]
        """
        # LSTM前向傳播
        lstm_out, _ = self.lstm(x)
        
        # 輸出層
        output = self.output_layer(lstm_out)
        
        return output
    
    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """獲取隱藏狀態（用於表徵蒸餾）"""
        lstm_out, _ = self.lstm(x)
        return lstm_out

class TransformerStudent(nn.Module):
    """Transformer學生模型"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 輸入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置編碼
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            預測結果 [batch_size, seq_len, output_dim]
        """
        # 輸入投影
        x = self.input_projection(x)
        
        # 位置編碼
        x = self.pos_encoding(x)
        
        # Transformer編碼
        transformer_out = self.transformer(x)
        
        # 輸出層
        output = self.output_layer(transformer_out)
        
        return output
    
    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """獲取隱藏狀態（用於表徵蒸餾）"""
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        transformer_out = self.transformer(x)
        return transformer_out

class PositionalEncoding(nn.Module):
    """位置編碼"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class MultiAssetStudent(nn.Module):
    """多資產學生模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_assets: int = 3, asset_types: List[str] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        
        if asset_types is None:
            asset_types = ['stock', 'etf', 'crypto']
        self.asset_types = asset_types
        
        # 資產類型嵌入
        self.asset_embedding = nn.Embedding(num_assets, hidden_dim // 4)
        
        # 特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            asset_ids: 資產ID [batch_size]
            
        Returns:
            預測結果 [batch_size, 1]
        """
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 資產嵌入
        asset_emb = self.asset_embedding(asset_ids)
        
        # 特徵融合
        combined = torch.cat([features, asset_emb], dim=1)
        
        # 預測
        output = self.fusion_layer(combined)
        
        return output

class RegimeAwareStudent(nn.Module):
    """Regime感知學生模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_regimes: int = 3, regime_types: List[str] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        if regime_types is None:
            regime_types = ['bull', 'bear', 'sideways']
        self.regime_types = regime_types
        
        # Regime嵌入
        self.regime_embedding = nn.Embedding(num_regimes, hidden_dim // 4)
        
        # 特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 條件化融合層
        self.conditional_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_regimes)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, x: torch.Tensor, regime_ids: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            regime_ids: Regime ID [batch_size]
            
        Returns:
            預測結果 [batch_size, 1]
        """
        # 特徵提取
        features = self.feature_extractor(x)
        
        # Regime嵌入
        regime_emb = self.regime_embedding(regime_ids)
        
        # 特徵融合
        combined = torch.cat([features, regime_emb], dim=1)
        
        # 條件化預測
        outputs = []
        for i in range(self.num_regimes):
            mask = (regime_ids == i)
            if mask.any():
                regime_output = self.conditional_fusion[i](combined[mask])
                outputs.append((regime_output, mask))
        
        # 組合輸出
        final_output = torch.zeros(x.size(0), 1, device=x.device)
        for output, mask in outputs:
            final_output[mask] = output
        
        return final_output

def sequence_kd_step(x: torch.Tensor, y: torch.Tensor, y_teacher: torch.Tensor,
                    student_model: nn.Module, optimizer: torch.optim.Optimizer,
                    alpha: float = 0.5, temperature: float = 3.0) -> float:
    """
    序列知識蒸餾訓練步驟
    
    Args:
        x: 輸入序列 [batch_size, seq_len, input_dim]
        y: 真實標籤 [batch_size, seq_len, output_dim]
        y_teacher: 老師預測 [batch_size, seq_len, output_dim]
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        temperature: 溫度參數
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 學生預測
    y_student = student_model(x)
    
    # 硬標籤損失（MSE）
    loss_hard = F.mse_loss(y_student, y)
    
    # 軟標籤損失（MSE with temperature）
    loss_soft = F.mse_loss(y_student / temperature, y_teacher / temperature) * (temperature ** 2)
    
    # 總損失
    loss = alpha * loss_hard + (1 - alpha) * loss_soft
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def hint_loss_step(x: torch.Tensor, h_teacher: torch.Tensor,
                  student_model: nn.Module, optimizer: torch.optim.Optimizer,
                  hint_weight: float = 0.1) -> float:
    """
    表徵蒸餾（Hint Loss）訓練步驟
    
    Args:
        x: 輸入序列
        h_teacher: 老師隱藏狀態
        student_model: 學生模型
        optimizer: 優化器
        hint_weight: Hint損失權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 獲取學生隱藏狀態
    h_student = student_model.get_hidden_states(x)
    
    # Hint損失（MSE）
    hint_loss = F.mse_loss(h_student, h_teacher)
    
    # 反向傳播
    optimizer.zero_grad()
    hint_loss.backward()
    optimizer.step()
    
    return hint_loss.item() * hint_weight

def multi_asset_kd_step(x: torch.Tensor, y: torch.Tensor, y_teacher: torch.Tensor,
                       asset_ids: torch.Tensor, student_model: nn.Module,
                       optimizer: torch.optim.Optimizer, alpha: float = 0.5) -> float:
    """
    多資產知識蒸餾訓練步驟
    
    Args:
        x: 輸入特徵
        y: 真實標籤
        y_teacher: 老師預測
        asset_ids: 資產ID
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 學生預測
    y_student = student_model(x, asset_ids)
    
    # 硬標籤損失
    loss_hard = F.mse_loss(y_student, y)
    
    # 軟標籤損失
    loss_soft = F.mse_loss(y_student, y_teacher)
    
    # 總損失
    loss = alpha * loss_hard + (1 - alpha) * loss_soft
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def continual_learning_step(x_new: torch.Tensor, y_new: torch.Tensor,
                           x_old: torch.Tensor, y_old: torch.Tensor,
                           y_old_teacher: torch.Tensor, student_model: nn.Module,
                           optimizer: torch.optim.Optimizer, alpha: float = 0.5,
                           beta: float = 0.3) -> float:
    """
    持續學習訓練步驟（不遺忘）
    
    Args:
        x_new: 新數據特徵
        y_new: 新數據標籤
        x_old: 舊數據特徵
        y_old: 舊數據標籤
        y_old_teacher: 舊數據老師預測
        student_model: 學生模型
        optimizer: 優化器
        alpha: 硬標籤權重
        beta: 舊數據權重
        
    Returns:
        損失值
    """
    student_model.train()
    
    # 新數據預測
    y_new_pred = student_model(x_new)
    loss_new = F.mse_loss(y_new_pred, y_new)
    
    # 舊數據預測
    y_old_pred = student_model(x_old)
    loss_old_hard = F.mse_loss(y_old_pred, y_old)
    loss_old_soft = F.mse_loss(y_old_pred, y_old_teacher)
    loss_old = alpha * loss_old_hard + (1 - alpha) * loss_old_soft
    
    # 總損失
    loss = loss_new + beta * loss_old
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 示例使用
if __name__ == "__main__":
    # 創建示例數據
    batch_size, seq_len, input_dim = 32, 10, 20
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, seq_len, 1)
    y_teacher = torch.randn(batch_size, seq_len, 1)
    
    # 創建LSTM學生模型
    lstm_student = LSTMStudent(input_dim, hidden_dim=128, num_layers=2)
    optimizer = torch.optim.Adam(lstm_student.parameters(), lr=0.001)
    
    # 序列知識蒸餾訓練
    loss = sequence_kd_step(x, y, y_teacher, lstm_student, optimizer)
    print(f"序列知識蒸餾損失: {loss:.4f}")
    
    # 創建Transformer學生模型
    transformer_student = TransformerStudent(input_dim, d_model=128, nhead=8, num_layers=4)
    optimizer = torch.optim.Adam(transformer_student.parameters(), lr=0.001)
    
    # 序列知識蒸餾訓練
    loss = sequence_kd_step(x, y, y_teacher, transformer_student, optimizer)
    print(f"Transformer知識蒸餾損失: {loss:.4f}")
    
    # 多資產模型示例
    batch_size, input_dim = 32, 20
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    y_teacher = torch.randn(batch_size, 1)
    asset_ids = torch.randint(0, 3, (batch_size,))
    
    multi_asset_student = MultiAssetStudent(input_dim, num_assets=3)
    optimizer = torch.optim.Adam(multi_asset_student.parameters(), lr=0.001)
    
    # 多資產知識蒸餾訓練
    loss = multi_asset_kd_step(x, y, y_teacher, asset_ids, multi_asset_student, optimizer)
    print(f"多資產知識蒸餾損失: {loss:.4f}")
