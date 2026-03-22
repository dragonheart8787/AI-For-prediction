#!/usr/bin/env python3
"""完美預測模型 - 整合多種AI技術和模型邏輯"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLSTM(nn.Module):
    """高級LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多層LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # 全連接層
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # 批標準化
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函數
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # LSTM前向傳播
        lstm_out, _ = self.lstm(x)
        
        # 注意力機制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最後一個時間步的輸出
        out = attn_out[:, -1, :]
        
        # 全連接層
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.tanh(out)
        
        return out

class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, output_size: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 輸入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 輸出投影
        self.output_projection = nn.Linear(d_model, output_size)
        
        # 批標準化和Dropout
        self.bn = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 輸入投影
        x = self.input_projection(x)
        
        # 位置編碼
        x = self.pos_encoder(x)
        
        # Transformer編碼
        x = self.transformer_encoder(x)
        
        # 取最後一個時間步
        x = x[:, -1, :]
        
        # 輸出投影
        x = self.output_projection(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        return x

class PositionalEncoding(nn.Module):
    """位置編碼"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class EnsembleModel:
    """集成模型"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.model_types = [
            'lstm', 'transformer', 'random_forest', 'gradient_boosting',
            'xgboost', 'lightgbm', 'svr', 'mlp'
        ]
    
    def add_model(self, model_name: str, model: Any, scaler: Any = None, weight: float = 1.0):
        """添加模型到集成"""
        self.models[model_name] = model
        if scaler:
            self.scalers[model_name] = scaler
        self.weights[model_name] = weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """集成預測"""
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X
            
            # 根據模型類型進行預測
            if model_name in ['lstm', 'transformer']:
                # PyTorch模型
                X_tensor = torch.FloatTensor(X_scaled)
                with torch.no_grad():
                    pred = model(X_tensor).numpy()
            else:
                # Scikit-learn模型
                pred = model.predict(X_scaled)
            
            weight = self.weights[model_name]
            predictions.append(pred * weight)
            total_weight += weight
        
        # 加權平均
        if total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def get_model_weights(self) -> Dict[str, float]:
        """獲取模型權重"""
        return self.weights.copy()

class PerfectPredictionModel:
    """完美預測模型"""
    
    def __init__(self, model_dir: str = "./agi_storage/models"):
        self.model_dir = model_dir
        self.ensemble = EnsembleModel()
        self.feature_importance = {}
        self.model_performance = {}
        self.training_history = {}
        
        # 模型配置
        self.model_configs = {
            'lstm': {
                'input_size': 50,
                'hidden_size': 128,
                'num_layers': 3,
                'output_size': 1,
                'dropout': 0.2
            },
            'transformer': {
                'input_size': 50,
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'output_size': 1,
                'dropout': 0.1
            }
        }
    
    def prepare_data(self, data: pd.DataFrame, target_col: str, 
                    sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """準備序列資料"""
        # 特徵工程
        features = self._engineer_features(data)
        
        # 創建序列
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(data[target_col].iloc[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """特徵工程"""
        features = data.copy()
        
        # 技術指標
        if 'price' in features.columns:
            # 移動平均
            features['sma_5'] = features['price'].rolling(5).mean()
            features['sma_10'] = features['price'].rolling(10).mean()
            features['sma_20'] = features['price'].rolling(20).mean()
            
            # 相對強弱指數
            features['rsi'] = self._calculate_rsi(features['price'])
            
            # 布林帶
            features['bb_upper'] = features['sma_20'] + 2 * features['price'].rolling(20).std()
            features['bb_lower'] = features['sma_20'] - 2 * features['price'].rolling(20).std()
            
            # 價格變化
            features['price_change'] = features['price'].pct_change()
            features['price_change_2'] = features['price'].pct_change(2)
            features['price_change_5'] = features['price'].pct_change(5)
        
        # 時間特徵
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
        
        # 統計特徵
        for col in features.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                features[f'{col}_rolling_mean'] = features[col].rolling(10).mean()
                features[f'{col}_rolling_std'] = features[col].rolling(10).std()
                features[f'{col}_rolling_skew'] = features[col].rolling(10).skew()
        
        # 處理缺失值
        features = features.fillna(method='ffill').fillna(0)
        
        # 選擇數值特徵
        numeric_features = features.select_dtypes(include=[np.number])
        return numeric_features.values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 100, batch_size: int = 32) -> nn.Module:
        """訓練LSTM模型"""
        config = self.model_configs['lstm']
        model = AdvancedLSTM(**config)
        
        # 資料轉換
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # 資料載入器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 優化器和損失函數
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # 訓練
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 驗證
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            # 學習率調度
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, "
                          f"Val Loss: {val_loss:.6f}")
        
        return model
    
    def train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               epochs: int = 100, batch_size: int = 32) -> nn.Module:
        """訓練Transformer模型"""
        config = self.model_configs['transformer']
        model = TransformerModel(**config)
        
        # 資料轉換
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # 資料載入器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 優化器和損失函數
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        
        # 訓練
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 驗證
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            # 學習率調度
            scheduler.step()
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, "
                          f"Val Loss: {val_loss:.6f}")
        
        return model
    
    def train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """訓練傳統機器學習模型"""
        models = {}
        scalers = {}
        
        # 資料標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 隨機森林
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, 
                                       random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        scalers['random_forest'] = scaler
        
        # 梯度提升
        gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, 
                                           learning_rate=0.1, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
        scalers['gradient_boosting'] = scaler
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, 
                                    learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model
        scalers['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, 
                                     learning_rate=0.1, random_state=42)
        lgb_model.fit(X_train_scaled, y_train)
        models['lightgbm'] = lgb_model
        scalers['lightgbm'] = scaler
        
        # SVR
        svr_model = SVR(kernel='rbf', C=100, gamma='scale')
        svr_model.fit(X_train_scaled, y_train)
        models['svr'] = svr_model
        scalers['svr'] = scaler
        
        # MLP
        mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, 
                                random_state=42)
        mlp_model.fit(X_train_scaled, y_train)
        models['mlp'] = mlp_model
        scalers['mlp'] = mlp_model
        
        return models, scalers
    
    def train_all_models(self, data: pd.DataFrame, target_col: str,
                        test_size: float = 0.2, sequence_length: int = 50) -> Dict[str, Any]:
        """訓練所有模型"""
        logger.info("開始訓練所有模型")
        
        # 準備資料
        X, y = self.prepare_data(data, target_col, sequence_length)
        
        # 分割資料
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 進一步分割驗證集
        val_split_idx = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        results = {}
        
        # 訓練LSTM
        logger.info("訓練LSTM模型...")
        lstm_model = self.train_lstm_model(X_train_final, y_train_final, X_val, y_val)
        self.ensemble.add_model('lstm', lstm_model, weight=1.0)
        results['lstm'] = self._evaluate_model(lstm_model, X_test, y_test, 'lstm')
        
        # 訓練Transformer
        logger.info("訓練Transformer模型...")
        transformer_model = self.train_transformer_model(X_train_final, y_train_final, X_val, y_val)
        self.ensemble.add_model('transformer', transformer_model, weight=1.0)
        results['transformer'] = self._evaluate_model(transformer_model, X_test, y_test, 'transformer')
        
        # 訓練傳統模型
        logger.info("訓練傳統機器學習模型...")
        traditional_models, traditional_scalers = self.train_traditional_models(X_train_final, y_train_final)
        
        for name, model in traditional_models.items():
            self.ensemble.add_model(name, model, traditional_scalers[name], weight=1.0)
            results[name] = self._evaluate_model(model, X_test, y_test, name, traditional_scalers[name])
        
        # 集成預測
        logger.info("執行集成預測...")
        ensemble_pred = self.ensemble.predict(X_test)
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred)
        results['ensemble'] = ensemble_metrics
        
        # 保存結果
        self.model_performance = results
        self._save_models()
        
        return results
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str, scaler: Any = None) -> Dict[str, float]:
        """評估模型性能"""
        if scaler:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # 預測
        if model_name in ['lstm', 'transformer']:
            X_tensor = torch.FloatTensor(X_test_scaled)
            with torch.no_grad():
                y_pred = model(X_tensor).numpy().flatten()
        else:
            y_pred = model.predict(X_test_scaled)
        
        # 計算指標
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # 特徵重要性
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """計算評估指標"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def _save_models(self):
        """保存模型"""
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 保存PyTorch模型
        for name, model in self.ensemble.models.items():
            if name in ['lstm', 'transformer']:
                torch.save(model.state_dict(), 
                          os.path.join(self.model_dir, f"{name}_model.pth"))
        
        # 保存Scikit-learn模型
        for name, model in self.ensemble.models.items():
            if name not in ['lstm', 'transformer']:
                joblib.dump(model, 
                           os.path.join(self.model_dir, f"{name}_model.pkl"))
        
        # 保存Scaler
        for name, scaler in self.ensemble.scalers.items():
            joblib.dump(scaler, 
                       os.path.join(self.model_dir, f"{name}_scaler.pkl"))
        
        # 保存性能指標
        with open(os.path.join(self.model_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        
        logger.info(f"模型已保存到 {self.model_dir}")
    
    def load_models(self):
        """載入已保存的模型"""
        import os
        
        # 載入PyTorch模型
        for name in ['lstm', 'transformer']:
            model_path = os.path.join(self.model_dir, f"{name}_model.pth")
            if os.path.exists(model_path):
                config = self.model_configs[name]
                if name == 'lstm':
                    model = AdvancedLSTM(**config)
                else:
                    model = TransformerModel(**config)
                
                model.load_state_dict(torch.load(model_path))
                model.eval()
                self.ensemble.add_model(name, model, weight=1.0)
                logger.info(f"載入 {name} 模型")
        
        # 載入Scikit-learn模型
        for name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'svr', 'mlp']:
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.ensemble.add_model(name, model, weight=1.0)
                logger.info(f"載入 {name} 模型")
        
        # 載入Scaler
        for name in self.ensemble.models.keys():
            if name not in ['lstm', 'transformer']:
                scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.ensemble.scalers[name] = scaler
        
        # 載入性能指標
        metrics_path = os.path.join(self.model_dir, 'performance_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.model_performance = json.load(f)
        
        logger.info("所有模型載入完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用集成模型進行預測"""
        return self.ensemble.predict(X)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """獲取模型摘要"""
        summary = {
            'total_models': len(self.ensemble.models),
            'model_types': list(self.ensemble.models.keys()),
            'performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'weights': self.ensemble.get_model_weights()
        }
        return summary

def main():
    """測試完美預測模型"""
    # 生成示例資料
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # 模擬股票價格資料
    price = 100
    prices = []
    for _ in range(1000):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    # 創建資料框
    data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.randint(1000000, 10000000, 1000),
        'volatility': np.random.uniform(0.1, 0.3, 1000)
    })
    
    # 創建預測模型
    model = PerfectPredictionModel()
    
    # 訓練所有模型
    results = model.train_all_models(data, 'price')
    
    # 顯示結果
    print("=== 模型訓練結果 ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # 模型摘要
    summary = model.get_model_summary()
    print(f"\n=== 模型摘要 ===")
    print(f"總模型數: {summary['total_models']}")
    print(f"模型類型: {summary['model_types']}")
    
    # 測試預測
    test_data = data.tail(100).copy()
    X_test, y_test = model.prepare_data(test_data, 'price')
    
    predictions = model.predict(X_test)
    
    print(f"\n=== 預測結果 ===")
    print(f"預測樣本數: {len(predictions)}")
    print(f"預測值範圍: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"實際值範圍: {y_test.min():.4f} - {y_test.max():.4f}")

if __name__ == "__main__":
    main()
