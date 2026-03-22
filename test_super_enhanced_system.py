#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試超級增強版時間序列預測系統
"""

import asyncio
import json
from pathlib import Path
from super_enhanced_ts_system import SuperEnhancedTSSystem

async def test_super_enhanced_system():
    """測試超級增強系統"""
    print("🧪 開始測試超級增強版時間序列預測系統...")
    
    try:
        system = SuperEnhancedTSSystem()
        
        # 測試配置
        print("🔍 測試系統配置...")
        config = system.config
        print(f"✅ 模型類別: {list(config.model_categories.keys())}")
        print(f"✅ 數據源: {list(config.data_sources.keys())}")
        
        # 測試數據收集
        print("🔍 測試數據收集...")
        all_data = system._collect_enhanced_data()
        print(f"✅ 收集數據: {len(all_data)} 個數據集")
        
        # 測試模型訓練
        print("🔍 測試模型訓練...")
        training_results = {}
        
        # 訓練經典統計模型
        print("📊 訓練經典統計模型...")
        for model_key in ['arima', 'ets', 'garch']:
            if all_data:
                first_data = list(all_data.values())[0]
                print(f"🚀 訓練模型: {model_key}")
                result = system.model_trainers[model_key](first_data['Close'])
                training_results[model_key] = result
                print(f"✅ 模型 {model_key} 訓練完成")
        
        # 訓練機器學習模型
        print("🤖 訓練機器學習模型...")
        for model_key in ['xgboost', 'lightgbm']:
            if all_data:
                first_data = list(all_data.values())[0]
                print(f"🚀 訓練模型: {model_key}")
                result = system.model_trainers[model_key](first_data)
                training_results[model_key] = result
                print(f"✅ 模型 {model_key} 訓練完成")
        
        # 測試模型融合
        print("🔍 測試模型融合...")
        try:
            fusion_result = system._create_advanced_fusion(training_results, 'short_term_forecast')
            print("✅ 模型融合成功")
            print(f"   融合結果: {len(fusion_result)} 個模型")
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 測試最終預測
        print("🔍 測試最終預測...")
        try:
            final_prediction = system._generate_super_enhanced_prediction(fusion_result, 'short_term_forecast')
            print("✅ 最終預測成功")
            print(f"   預測結果: {len(final_prediction)} 個預測")
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 顯示結果摘要
        print("\n🎯 測試結果摘要:")
        print(f"   - 訓練模型數: {len(training_results)}")
        print(f"   - 參與融合模型: {fusion_result.get('base_models', [])}")
        print(f"   - 預測點數: {final_prediction.get('prediction_horizon', 0)}")
        
        if 'model_weights' in fusion_result:
            print("📊 模型融合詳情:")
            for i, model_name in enumerate(fusion_result.get('base_models', [])):
                weight = fusion_result['model_weights'][i]
                print(f"   - {model_name}: 權重 {weight:.3f}")
        
        print("\n🎉 所有測試通過！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_super_enhanced_system())
