#!/usr/bin/env python3
"""
真實LSTM時間序列預測模型
使用PyTorch實現
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM時間序列預測模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM前向傳播
        lstm_out, _ = self.lstm(x)
        
        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]
        
        # 全連接層
        output = self.fc(last_output)
        
        return output

class LSTMTimeSeriesPredictor:
    """LSTM時間序列預測器"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 learning_rate: float = 0.001, device: str = 'auto'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # 設備選擇
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🚀 使用設備: {self.device}")
        
        # 初始化模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        # 優化器和損失函數
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 數據標準化器
        self.scaler = StandardScaler()
        
        # 訓練歷史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close',
                    sequence_length: int = 60, test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """準備訓練數據"""
        try:
            # 選擇特徵列
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in feature_columns if col in data.columns]
            
            if not available_columns:
                available_columns = ['Close']
            
            # 標準化數據
            scaled_data = self.scaler.fit_transform(data[available_columns])
            
            # 創建序列數據
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, available_columns.index(target_column)])
            
            X = np.array(X)
            y = np.array(y)
            
            # 分割訓練和測試數據
            split_idx = int(len(X) * (1 - test_split))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"✅ 數據準備完成: 訓練集 {len(X_train)}, 測試集 {len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ 數據準備失敗: {e}")
            return None, None, None, None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, 
              early_stopping_patience: int = 10) -> Dict[str, Any]:
        """訓練模型"""
        try:
            logger.info("🧠 開始訓練LSTM模型...")
            
            # 轉換為PyTorch張量
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                has_validation = True
            else:
                has_validation = False
            
            # 訓練循環
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # 訓練模式
                self.model.train()
                
                # 批次訓練
                total_train_loss = 0
                num_batches = 0
                
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    # 前向傳播
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)
                    
                    # 反向傳播
                    loss.backward()
                    self.optimizer.step()
                    
                    total_train_loss += loss.item()
                    num_batches += 1
                
                avg_train_loss = total_train_loss / num_batches
                
                # 驗證
                if has_validation:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val_tensor).squeeze()
                        val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    
                    # 早停檢查
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                    else:
                        patience_counter += 1
                    
                    # 記錄訓練歷史
                    self.training_history['train_loss'].append(avg_train_loss)
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['epochs'].append(epoch)
                    
                    if epoch % 10 == 0:
                        logger.info(f"📊 Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # 早停
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"🛑 早停觸發，在 Epoch {epoch}")
                        break
                else:
                    if epoch % 10 == 0:
                        logger.info(f"📊 Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.6f}")
            
            logger.info("✅ LSTM模型訓練完成")
            
            # 載入最佳模型
            if has_validation:
                self.model.load_state_dict(torch.load('best_lstm_model.pth'))
            
            return {
                'success': True,
                'epochs_completed': epoch + 1,
                'final_train_loss': avg_train_loss,
                'final_val_loss': val_loss if has_validation else None,
                'best_val_loss': best_val_loss if has_validation else None
            }
            
        except Exception as e:
            logger.error(f"❌ LSTM模型訓練失敗: {e}")
            return {'error': str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """進行預測"""
        try:
            self.model.eval()
            
            # 轉換為PyTorch張量
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            return predictions.squeeze()
            
        except Exception as e:
            logger.error(f"❌ 預測失敗: {e}")
            return None
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """評估模型性能"""
        try:
            # 進行預測
            y_pred = self.predict(X_test)
            
            if y_pred is None:
                return {'error': '預測失敗'}
            
            # 計算評估指標
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # 計算R²分數
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            evaluation_results = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'test_samples': len(y_test)
            }
            
            logger.info(f"✅ 模型評估完成: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ 模型評估失敗: {e}")
            return {'error': str(e)}
    
    def plot_training_history(self, save_path: str = None):
        """繪製訓練歷史"""
        try:
            if not self.training_history['epochs']:
                logger.warning("⚠️ 沒有訓練歷史數據可繪製")
                return
            
            plt.figure(figsize=(12, 6))
            
            # 訓練損失
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history['epochs'], self.training_history['train_loss'], 
                    label='Training Loss', color='blue')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 驗證損失
            if self.training_history['val_loss']:
                plt.subplot(1, 2, 2)
                plt.plot(self.training_history['epochs'], self.training_history['val_loss'], 
                        label='Validation Loss', color='red')
                plt.title('Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 訓練歷史圖表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ 繪製訓練歷史失敗: {e}")
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'learning_rate': self.learning_rate
                },
                'training_history': self.training_history
            }, filepath)
            
            logger.info(f"💾 模型已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 模型保存失敗: {e}")
    
    def load_model(self, filepath: str):
        """載入模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 載入模型狀態
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 載入其他組件
            self.scaler = checkpoint['scaler']
            self.training_history = checkpoint.get('training_history', {})
            
            logger.info(f"✅ 模型已載入: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}")

# 使用示例
if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建預測器
    predictor = LSTMTimeSeriesPredictor(
        input_size=5,  # 5個特徵
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001
    )
    
    print("🚀 LSTM時間序列預測模型創建成功！")
    print(f"📱 使用設備: {predictor.device}")
    print(f"🔧 模型參數: 輸入維度={predictor.input_size}, 隱藏層大小={predictor.hidden_size}, 層數={predictor.num_layers}")
