#!/usr/bin/env python3
"""
調試 ONNX 環境
"""

def debug_onnx_environment():
    """調試 ONNX 環境"""
    print("🔍 調試 ONNX 環境...")
    
    # 檢查 onnxruntime
    try:
        import onnxruntime as ort
        print(f"✅ onnxruntime: {ort.__version__}")
    except ImportError as e:
        print(f"❌ onnxruntime: {e}")
        return False
    
    # 檢查 skl2onnx
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        print("✅ skl2onnx: 可用")
        skl2onnx_available = True
    except ImportError as e:
        print(f"❌ skl2onnx: {e}")
        skl2onnx_available = False
    
    # 檢查 UnifiedPredictor
    try:
        from unified_predict import UnifiedPredictor
        predictor = UnifiedPredictor(auto_onnx=True)
        print(f"✅ UnifiedPredictor: auto_onnx={predictor.auto_onnx}")
        print(f"   ort is not None: {ort is not None}")
        print(f"   convert_sklearn is not None: {skl2onnx_available}")
        
        # 檢查條件
        condition1 = predictor.auto_onnx
        condition2 = ort is not None
        print(f"   條件1 (auto_onnx): {condition1}")
        print(f"   條件2 (ort): {condition2}")
        print(f"   總條件: {condition1 and condition2}")
        
    except Exception as e:
        print(f"❌ UnifiedPredictor: {e}")
        return False
    
    return True

if __name__ == "__main__":
    debug_onnx_environment()

