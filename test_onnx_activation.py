#!/usr/bin/env python3
"""
測試 ONNX 是否真的被啟用
"""

import json
import numpy as np
from unified_predict import UnifiedPredictor

def test_onnx_activation():
    """測試 ONNX 是否被啟用"""
    print("🔍 測試 ONNX 啟用狀態...")
    
    # 創建測試數據
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 3)
    
    # 創建預測器並啟用 ONNX
    predictor = UnifiedPredictor(auto_onnx=True)
    
    print("   訓練模型...")
    predictor.fit(X, y, model="linear")
    
    print(f"   模型名稱: {predictor.model_name}")
    print(f"   ONNX Runner: {predictor.onnx_runner is not None}")
    
    # 測試預測
    print("   測試預測...")
    result = predictor.predict_many(X[:10], domain="custom")
    
    print(f"   返回的模型名稱: {result.get('model')}")
    print(f"   ONNX 狀態: {'onnx' in result.get('model', '').lower()}")
    
    # 檢查是否真的使用了 ONNX
    if predictor.onnx_runner is not None:
        print("   ✅ ONNX Runner 已啟用")
        print(f"   ONNX Runner 類型: {type(predictor.onnx_runner).__name__}")
    else:
        print("   ❌ ONNX Runner 未啟用")
    
    return result.get('model', '').lower().startswith('onnx')

def test_different_models():
    """測試不同模型的 ONNX 啟用"""
    print("\n🔍 測試不同模型的 ONNX 啟用...")
    
    X = np.random.randn(50, 5)
    y = np.random.randn(50, 2)
    
    models = ["linear", "xgboost", "lightgbm"]
    results = {}
    
    for model in models:
        try:
            print(f"\n   測試 {model} 模型...")
            predictor = UnifiedPredictor(auto_onnx=True)
            predictor.fit(X, y, model=model)
            
            result = predictor.predict_many(X[:5], domain="custom")
            model_name = result.get('model', '')
            onnx_enabled = 'onnx' in model_name.lower()
            
            results[model] = {
                "model_name": model_name,
                "onnx_enabled": onnx_enabled,
                "onnx_runner": predictor.onnx_runner is not None
            }
            
            print(f"     模型名稱: {model_name}")
            print(f"     ONNX 啟用: {onnx_enabled}")
            print(f"     ONNX Runner: {predictor.onnx_runner is not None}")
            
        except Exception as e:
            print(f"     ❌ {model} 測試失敗: {e}")
            results[model] = {"error": str(e)}
    
    return results

def main():
    """主測試函數"""
    print("🚀 ONNX 啟用測試開始...")
    print("=" * 60)
    
    # 基本測試
    onnx_activated = test_onnx_activation()
    
    # 不同模型測試
    model_results = test_different_models()
    
    print("\n" + "=" * 60)
    print("📊 測試結果摘要:")
    print(f"   基本 ONNX 啟用: {'✅' if onnx_activated else '❌'}")
    
    for model, result in model_results.items():
        if "error" in result:
            print(f"   {model}: ❌ {result['error']}")
        else:
            status = "✅" if result["onnx_enabled"] else "❌"
            print(f"   {model}: {status} ({result['model_name']})")
    
    if onnx_activated:
        print("\n🎉 ONNX 已成功啟用！")
    else:
        print("\n⚠️  ONNX 未啟用，需要進一步調試")

if __name__ == "__main__":
    main()

