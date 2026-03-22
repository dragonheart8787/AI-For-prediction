#!/usr/bin/env python3
"""
簡單的 ONNX 測試
"""

import numpy as np
from unified_predict import UnifiedPredictor

def simple_test():
    """簡單測試"""
    print("🔍 簡單 ONNX 測試...")
    
    # 創建測試數據
    X = np.random.randn(20, 5)
    y = np.random.randn(20)  # 單維度 y
    
    # 創建預測器
    predictor = UnifiedPredictor(auto_onnx=True)
    
    print(f"   初始 auto_onnx: {predictor.auto_onnx}")
    print(f"   初始 model_name: {predictor.model_name}")
    print(f"   初始 onnx_runner: {predictor.onnx_runner}")
    
    # 訓練模型
    print("   訓練模型...")
    predictor.fit(X, y, model="linear")
    
    print(f"   訓練後 model_name: {predictor.model_name}")
    print(f"   訓練後 onnx_runner: {predictor.onnx_runner}")
    
    # 測試預測
    result = predictor.predict_many(X[:5], domain="custom")
    print(f"   預測結果 model: {result.get('model')}")
    
    return predictor.model_name.startswith('onnx')

if __name__ == "__main__":
    success = simple_test()
    print(f"\n🎯 結果: {'成功' if success else '失敗'}")

