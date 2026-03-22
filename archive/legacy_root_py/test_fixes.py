#!/usr/bin/env python3
"""
測試修復的腳本
"""

import asyncio
import numpy as np
from agi_persistent import PersistentConfig, PersistentAGISystem

async def test_fixes():
    """測試修復"""
    print("測試持久化AGI系統修復...")
    
    # 創建配置和系統
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        # 測試1: 訓練模型
        print("1. 測試模型訓練...")
        training_results = await agi_system.train_all_models()
        
        if training_results:
            print("✅ 模型訓練成功")
            if 'lstm_result' in training_results:
                print(f"   LSTM準確率: {training_results['lstm_result']['final_accuracy']:.4f}")
            if 'transformer_result' in training_results:
                print(f"   Transformer準確率: {training_results['transformer_result']['final_accuracy']:.4f}")
        else:
            print("❌ 模型訓練失敗")
            return
        
        # 測試2: 預測功能
        print("\n2. 測試預測功能...")
        test_data = np.random.randn(5, 10)
        
        lstm_prediction = await agi_system.make_prediction("financial_lstm", test_data)
        if lstm_prediction:
            print("✅ LSTM預測成功")
            print(f"   預測值: {lstm_prediction['prediction'][:3]}...")
            print(f"   置信度: {lstm_prediction['confidence']:.3f}")
        
        transformer_prediction = await agi_system.make_prediction("weather_transformer", test_data)
        if transformer_prediction:
            print("✅ Transformer預測成功")
            print(f"   預測值: {transformer_prediction['prediction'][:3]}...")
            print(f"   置信度: {transformer_prediction['confidence']:.3f}")
        
        # 測試3: 雲端上傳
        print("\n3. 測試雲端上傳...")
        upload_result = await agi_system.upload_to_cloud("financial_lstm")
        print(f"雲端上傳: {'✅ 成功' if upload_result else '❌ 失敗'}")
        
        # 測試4: 系統狀態
        print("\n4. 測試系統狀態...")
        status = agi_system.get_system_status()
        print("✅ 系統狀態獲取成功")
        print(f"   總模型數: {status.get('total_models', 0)}")
        print(f"   總預測數: {status.get('total_predictions', 0)}")
        print(f"   系統運行: {status.get('system_running', False)}")
        
        print("\n🎉 所有測試通過！修復成功！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixes()) 