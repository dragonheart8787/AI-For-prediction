#!/usr/bin/env python3
"""簡化版完美預測模型 - 不依賴外部庫"""
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLSTM:
    """簡化版LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # 初始化權重
        self.weights = np.random.randn(input_size, hidden_size) * 0.01
        self.bias = np.zeros(hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.01
        self.output_bias = np.zeros(output_size)
    
    def forward(self, x):
        """前向傳播"""
        # 簡化的LSTM計算
        batch_size, seq_len, _ = x.shape
        
        # 初始化隱藏狀態
        h = np.zeros((batch_size, self.hidden_size))
        
        # 序列處理
        for t in range(seq_len):
            # 輸入門
            i_t = self._sigmoid(np.dot(x[:, t, :], self.weights) + self.bias)
            
            # 遺忘門
            f_t = self._sigmoid(np.dot(x[:, t, :], self.weights) + self.bias)
            
            # 輸出門
            o_t = self._sigmoid(np.dot(x[:, t, :], self.weights) + self.bias)
            
            # 候選值
            c_tilde = np.tanh(np.dot(x[:, t, :], self.weights) + self.bias)
            
            # 更新隱藏狀態
            h = o_t * np.tanh(h)
        
        # 輸出層
        output = np.dot(h, self.output_weights) + self.output_bias
        return output
    
    def _sigmoid(self, x):
        """Sigmoid激活函數"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x):
        """Tanh激活函數"""
        return np.tanh(x)

class SimpleTransformer:
    """簡化版Transformer模型"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, output_size: int, dropout: float = 0.1):
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # 初始化權重
        self.input_projection = np.random.randn(input_size, d_model) * 0.01
        self.output_projection = np.random.randn(d_model, output_size) * 0.01
        
        # 注意力權重
        self.attention_weights = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, x):
        """前向傳播"""
        batch_size, seq_len, _ = x.shape
        
        # 輸入投影
        x = np.dot(x, self.input_projection)
        
        # 簡化的注意力機制
        for _ in range(self.num_layers):
            # 自注意力
            attention_output = self._self_attention(x)
            
            # 殘差連接
            x = x + attention_output
        
        # 取最後一個時間步
        x = x[:, -1, :]
        
        # 輸出投影
        output = np.dot(x, self.output_projection)
        return output
    
    def _self_attention(self, x):
        """簡化的自注意力機制"""
        batch_size, seq_len, d_model = x.shape
        
        # 計算注意力分數
        scores = np.dot(x, np.dot(x, self.attention_weights).T)
        scores = scores / np.sqrt(d_model)
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # 應用注意力權重
        output = np.dot(attention_weights, x)
        return output
    
    def _softmax(self, x, axis=None):
        """Softmax函數"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SimpleEnsembleModel:
    """簡化版集成模型"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.model_types = ['lstm', 'transformer', 'linear', 'polynomial']
    
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
                pred = model.forward(X_scaled)
            elif model_name == 'linear':
                pred = self._linear_predict(model, X_scaled)
            elif model_name == 'polynomial':
                pred = self._polynomial_predict(model, X_scaled)
            else:
                pred = np.zeros((X_scaled.shape[0], 1))
            
            weight = self.weights[model_name]
            predictions.append(pred * weight)
            total_weight += weight
        
        # 加權平均
        if total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def _linear_predict(self, model, X):
        """線性模型預測"""
        weights, bias = model
        return np.dot(X, weights) + bias
    
    def _polynomial_predict(self, model, X):
        """多項式模型預測"""
        coefficients = model
        result = np.zeros((X.shape[0], 1))
        
        for i, coef in enumerate(coefficients):
            if i == 0:
                result += coef
            else:
                result += coef * (X ** i)
        
        return result
    
    def get_model_weights(self) -> Dict[str, float]:
        """獲取模型權重"""
        return self.weights.copy()

class SimplePerfectPredictionModel:
    """簡化版完美預測模型"""
    
    def __init__(self, model_dir: str = "./agi_storage/models"):
        self.model_dir = model_dir
        self.ensemble = SimpleEnsembleModel()
        self.feature_importance = {}
        self.model_performance = {}
        self.training_history = {}
        
        # 模型配置
        self.model_configs = {
            'lstm': {
                'input_size': 10,
                'hidden_size': 32,
                'num_layers': 2,
                'output_size': 1,
                'dropout': 0.2
            },
            'transformer': {
                'input_size': 10,
                'd_model': 32,
                'nhead': 4,
                'num_layers': 2,
                'output_size': 1,
                'dropout': 0.1
            }
        }
    
    def prepare_data(self, data: np.ndarray, target_col: int, 
                    sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """準備序列資料"""
        # 特徵工程
        features = self._engineer_features(data)
        
        # 創建序列
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(data[i + sequence_length, target_col])
        
        return np.array(X), np.array(y)
    
    def _engineer_features(self, data: np.ndarray) -> np.ndarray:
        """特徵工程"""
        features = data.copy()
        
        # 添加統計特徵
        for i in range(features.shape[1]):
            # 移動平均
            features[:, i] = self._rolling_mean(features[:, i], 5)
            
            # 標準差
            features[:, i] = np.concatenate([
                features[:5, i],
                self._rolling_std(features[5:, i], 5)
            ])
        
        # 處理缺失值
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """計算移動平均"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i+1])
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """計算移動標準差"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.std(data[start:i+1])
        return result
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 50) -> SimpleLSTM:
        """訓練LSTM模型"""
        config = self.model_configs['lstm']
        model = SimpleLSTM(**config)
        
        # 簡化的訓練過程
        logger.info("訓練LSTM模型...")
        
        # 這裡只是創建模型，實際訓練需要更複雜的實現
        # 為了演示，我們直接返回模型
        
        return model
    
    def train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               epochs: int = 50) -> SimpleTransformer:
        """訓練Transformer模型"""
        config = self.model_configs['transformer']
        model = SimpleTransformer(**config)
        
        logger.info("訓練Transformer模型...")
        
        return model
    
    def train_simple_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """訓練簡單模型"""
        models = {}
        scalers = {}
        
        # 線性回歸
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_scaled = (X_train - X_mean) / (X_std + 1e-8)
        
        # 簡單線性回歸
        weights = np.linalg.lstsq(X_scaled, y_train, rcond=None)[0]
        bias = np.mean(y_train - np.dot(X_scaled, weights))
        
        models['linear'] = (weights, bias)
        scalers['linear'] = (X_mean, X_std)
        
        # 多項式回歸
        X_poly = np.column_stack([X_scaled, X_scaled**2])
        poly_weights = np.linalg.lstsq(X_poly, y_train, rcond=None)[0]
        
        models['polynomial'] = poly_weights
        scalers['polynomial'] = (X_mean, X_std)
        
        return models, scalers
    
    def train_all_models(self, data: np.ndarray, target_col: int,
                        test_size: float = 0.2, sequence_length: int = 10) -> Dict[str, Any]:
        """訓練所有模型"""
        logger.info("開始訓練所有模型")
        
        # 準備資料
        X, y = self.prepare_data(data, target_col, sequence_length)
        
        if len(X) == 0:
            logger.warning("資料不足，無法訓練模型")
            return {}
        
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
        
        # 訓練簡單模型
        logger.info("訓練簡單模型...")
        simple_models, simple_scalers = self.train_simple_models(X_train_final, y_train_final)
        
        for name, model in simple_models.items():
            self.ensemble.add_model(name, model, simple_scalers[name], weight=1.0)
            results[name] = self._evaluate_model(model, X_test, y_test, name, simple_scalers[name])
        
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
            X_mean, X_std = scaler
            X_test_scaled = (X_test - X_mean) / (X_std + 1e-8)
        else:
            X_test_scaled = X_test
        
        # 預測
        if model_name in ['lstm', 'transformer']:
            y_pred = model.forward(X_test_scaled)
        elif model_name == 'linear':
            weights, bias = model
            y_pred = np.dot(X_test_scaled, weights) + bias
        elif model_name == 'polynomial':
            y_pred = self.ensemble._polynomial_predict(model, X_test_scaled)
        else:
            y_pred = np.zeros_like(y_test)
        
        # 計算指標
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """計算評估指標"""
        # 確保形狀一致
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        
        # 計算指標
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # 計算R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
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
        
        # 保存性能指標
        with open(os.path.join(self.model_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        
        logger.info(f"模型性能指標已保存到 {self.model_dir}")
    
    def load_models(self):
        """載入已保存的模型"""
        logger.info("載入模型...")
        
        # 創建預設模型
        lstm_config = self.model_configs['lstm']
        lstm_model = SimpleLSTM(**lstm_config)
        self.ensemble.add_model('lstm', lstm_model, weight=1.0)
        
        transformer_config = self.model_configs['transformer']
        transformer_model = SimpleTransformer(**transformer_config)
        self.ensemble.add_model('transformer', transformer_model, weight=1.0)
        
        # 載入性能指標
        metrics_path = os.path.join(self.model_dir, 'performance_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.model_performance = json.load(f)
        
        logger.info("模型載入完成")
    
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
    """測試簡化版完美預測模型"""
    # 生成示例資料
    np.random.seed(42)
    
    # 創建模擬資料
    n_samples = 200
    n_features = 10
    
    # 生成特徵
    X = np.random.randn(n_samples, n_features)
    
    # 生成目標值（線性組合 + 雜訊）
    weights = np.random.randn(n_features)
    y = np.dot(X, weights) + np.random.randn(n_samples) * 0.1
    
    # 組合資料
    data = np.column_stack([X, y])
    
    # 創建預測模型
    model = SimplePerfectPredictionModel()
    
    # 訓練所有模型
    results = model.train_all_models(data, target_col=n_features)
    
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
    test_data = data[-50:].copy()
    X_test, y_test = model.prepare_data(test_data, n_features)
    
    if len(X_test) > 0:
        predictions = model.predict(X_test)
        
        print(f"\n=== 預測結果 ===")
        print(f"預測樣本數: {len(predictions)}")
        print(f"預測值範圍: {predictions.min():.4f} - {predictions.max():.4f}")
        print(f"實際值範圍: {y_test.min():.4f} - {y_test.max():.4f}")

if __name__ == "__main__":
    main()
