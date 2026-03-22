#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試增強版真實AI預測系統
"""

import asyncio
import json
from pathlib import Path
from start_enhanced_ai_system import EnhancedRealAISystem

async def test_enhanced_system():
    """測試增強系統"""
    print("🧪 開始測試增強版真實AI預測系統...")
    
    try:
        system = EnhancedRealAISystem()
        
        # 測試模型下載器
        print("🔍 測試模型下載器...")
        downloader = system.downloader
        
        # 先安裝本地模型
        print("🏠 安裝本地統計模型...")
        local_results = downloader.install_local_models()
        print(f"✅ 本地模型安裝結果: {local_results}")
        
        available_models = downloader.get_available_models()
        print(f"✅ 可用模型: {available_models}")
        
        # 測試數據收集器
        print("🔍 測試數據收集器...")
        collector = system.collector
        all_data = collector.collect_enhanced_data()
        print(f"✅ 收集數據: {len(all_data)} 個數據集")
        
        # 測試模型訓練器
        print("🔍 測試模型訓練器...")
        trainer = system.trainer
        training_results = {}
        
        for model_key in available_models:
            if all_data:
                first_data = list(all_data.values())[0]
                print(f"🚀 訓練模型: {model_key}")
                result = trainer.train_enhanced_model(model_key, first_data)
                training_results[model_key] = result
                print(f"✅ 模型 {model_key} 訓練完成")
        
        # 測試模型融合
        print("🔍 測試模型融合...")
        fusion = system.fusion
        fusion_result = fusion.create_enhanced_fusion(training_results)
        print(f"✅ 融合完成: {len(fusion_result.get('base_models', []))} 個模型")
        
        # 測試最終預測
        print("🔍 測試最終預測...")
        final_prediction = system._generate_enhanced_prediction(fusion_result)
        print(f"✅ 預測完成: {final_prediction.get('prediction_horizon', 0)} 個預測點")
        
        # 顯示結果摘要
        print("\n🎯 測試結果摘要:")
        print(f"   - 可用模型數: {len(available_models)}")
        print(f"   - 訓練成功數: {len(training_results)}")
        print(f"   - 參與融合模型: {fusion_result.get('base_models', [])}")
        print(f"   - 預測點數: {final_prediction.get('prediction_horizon', 0)}")
        
        if 'model_details' in fusion_result:
            print("📊 模型融合詳情:")
            for model_name, details in fusion_result['model_details'].items():
                print(f"   - {model_name}: {details.get('type', 'Unknown')} (權重: {details.get('weight', 0):.3f})")
        
        print("\n🎉 所有測試通過！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_system())
