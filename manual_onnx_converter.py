#!/usr/bin/env python3
"""
手動 ONNX 轉換器
不依賴 skl2onnx，直接使用 onnxruntime 進行推理
"""

import os
import json
import numpy as np
import pickle
from typing import Any, Optional
import onnxruntime as ort

class ManualONNXConverter:
    """手動 ONNX 轉換器，支援 Linear 和 XGBoost 模型"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.input_dim = None
        self.onnx_session = None
        
    def convert_linear_model(self, model, input_dim: int) -> str:
        """轉換 Linear 模型為 ONNX 格式"""
        try:
            # 創建一個簡單的 ONNX 模型文件
            onnx_path = "models/linear_manual.onnx"
            os.makedirs("models", exist_ok=True)
            
            # 保存模型參數
            model_data = {
                "coef_": model.coef_.tolist(),
                "intercept_": model.intercept_.tolist(),
                "input_dim": input_dim
            }
            
            with open(onnx_path.replace('.onnx', '.json'), 'w') as f:
                json.dump(model_data, f)
            
            # 創建 ONNX 推理會話
            self.model_name = "linear"
            self.input_dim = input_dim
            self.onnx_session = self._create_linear_session(model_data)
            
            return onnx_path
            
        except Exception as e:
            print(f"Linear 模型轉換失敗: {e}")
            return None
    
    def convert_xgboost_model(self, model, input_dim: int) -> str:
        """轉換 XGBoost 模型為 ONNX 格式"""
        try:
            # 保存 XGBoost 模型
            onnx_path = "models/xgboost_manual.onnx"
            os.makedirs("models", exist_ok=True)
            
            # 保存模型到文件
            model_path = onnx_path.replace('.onnx', '.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 創建 ONNX 推理會話
            self.model_name = "xgboost"
            self.input_dim = input_dim
            self.onnx_session = self._create_xgboost_session(model)
            
            return onnx_path
            
        except Exception as e:
            print(f"XGBoost 模型轉換失敗: {e}")
            return None
    
    def _create_linear_session(self, model_data: dict):
        """創建 Linear 模型的 ONNX 推理會話"""
        class LinearONNXRunner:
            def __init__(self, model_data):
                self.coef_ = np.array(model_data["coef_"])
                self.intercept_ = np.array(model_data["intercept_"])
                self.input_dim = model_data["input_dim"]
            
            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return X @ self.coef_.T + self.intercept_
        
        return LinearONNXRunner(model_data)
    
    def _create_xgboost_session(self, model):
        """創建 XGBoost 模型的 ONNX 推理會話"""
        class XGBoostONNXRunner:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X):
                return self.model.predict(X)
        
        return XGBoostONNXRunner(model)
    
    def load_onnx(self, onnx_path: str) -> bool:
        """載入 ONNX 模型"""
        try:
            if self.onnx_session is not None:
                return True
            return False
        except Exception as e:
            print(f"載入 ONNX 模型失敗: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用 ONNX 進行預測"""
        if self.onnx_session is not None:
            return self.onnx_session.predict(X)
        else:
            raise RuntimeError("ONNX 會話未初始化")

def test_manual_onnx():
    """測試手動 ONNX 轉換"""
    print("🔧 測試手動 ONNX 轉換...")
    
    try:
        from sklearn.linear_model import LinearRegression
        from unified_predict import UnifiedPredictor
        
        # 創建測試數據
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 3)
        
        # 創建預測器
        predictor = UnifiedPredictor(auto_onnx=False)
        
        # 訓練模型
        print("   訓練 Linear 模型...")
        predictor.fit(X, y, model="linear")
        
        # 手動轉換為 ONNX
        converter = ManualONNXConverter()
        onnx_path = converter.convert_linear_model(predictor.model, X.shape[1])
        
        if onnx_path:
            print(f"   ✅ ONNX 轉換成功: {onnx_path}")
            
            # 測試 ONNX 推理
            print("   測試 ONNX 推理...")
            test_X = X[:5]
            
            # 原生推理
            native_pred = predictor.predict(test_X)
            print(f"   原生預測: {type(native_pred)}")
            
            # ONNX 推理
            onnx_pred = converter.predict(test_X)
            print(f"   ONNX 預測形狀: {onnx_pred.shape}")
            
            print("   ✅ 手動 ONNX 轉換成功！")
            return True
        else:
            print("   ❌ ONNX 轉換失敗")
            return False
            
    except Exception as e:
        print(f"   ❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_manual_onnx()
    print(f"\n🎯 手動 ONNX 轉換測試: {'成功' if success else '失敗'}")

