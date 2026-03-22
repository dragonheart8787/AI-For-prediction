#!/usr/bin/env python3
"""
簡化的 ONNX 測試腳本
測試 onnxruntime 是否可用，並嘗試手動實現 ONNX 轉換
"""

import json
import time
import sys

def test_onnxruntime():
    """測試 onnxruntime 是否可用"""
    try:
        import onnxruntime as ort
        print("✅ onnxruntime 可用")
        print(f"   版本: {ort.__version__}")
        return True
    except ImportError as e:
        print(f"❌ onnxruntime 不可用: {e}")
        return False

def test_skl2onnx():
    """測試 skl2onnx 是否可用"""
    try:
        import skl2onnx
        print("✅ skl2onnx 可用")
        print(f"   版本: {skl2onnx.__version__}")
        return True
    except ImportError as e:
        print(f"❌ skl2onnx 不可用: {e}")
        return False

def test_onnx():
    """測試 onnx 是否可用"""
    try:
        import onnx
        print("✅ onnx 可用")
        print(f"   版本: {onnx.__version__}")
        return True
    except ImportError as e:
        print(f"❌ onnx 不可用: {e}")
        return False

def test_unified_predictor_onnx():
    """測試 UnifiedPredictor 的 ONNX 功能"""
    try:
        from unified_predict import UnifiedPredictor
        print("✅ UnifiedPredictor 可用")
        
        # 創建一個簡單的測試
        predictor = UnifiedPredictor(auto_onnx=True)  # 啟用自動 ONNX
        print("✅ UnifiedPredictor 支援 auto_onnx")
        return True
    except Exception as e:
        print(f"❌ UnifiedPredictor 測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔍 測試 ONNX 環境...")
    print("=" * 50)
    
    results = {
        "onnxruntime": test_onnxruntime(),
        "skl2onnx": test_skl2onnx(),
        "onnx": test_onnx(),
        "unified_predictor": test_unified_predictor_onnx()
    }
    
    print("=" * 50)
    print("📊 測試結果摘要:")
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component}")
    
    # 檢查是否可以啟用 ONNX
    onnx_ready = results["onnxruntime"] and results["unified_predictor"]
    
    if onnx_ready:
        print("\n🎉 ONNX 環境準備就緒！")
        print("   可以執行: python demo_backtest_all.py --model auto --batch 8192")
    else:
        print("\n⚠️  ONNX 環境未完全準備就緒")
        print("   建議:")
        if not results["onnxruntime"]:
            print("   - 重新安裝 onnxruntime: pip install onnxruntime")
        if not results["unified_predictor"]:
            print("   - 檢查 unified_predict.py 是否正確")
    
    return onnx_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
