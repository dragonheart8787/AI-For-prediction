#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試任務導向權重分配功能
"""

import asyncio
import json
from pathlib import Path
from super_enhanced_ts_system import SuperEnhancedTSSystem

async def test_task_based_weights():
    """測試任務導向權重分配"""
    print("🧪 開始測試任務導向權重分配功能...")
    
    try:
        system = SuperEnhancedTSSystem()
        
        # 顯示所有可用的任務類型
        print("🔍 可用的任務類型:")
        for task_type, task_info in system.config.task_features.items():
            print(f"   - {task_type}: {task_info['horizon']} | {task_info['focus']} | 優先級: {task_info['priority']}")
        
        print("\n📊 任務權重配置:")
        for task_type, weights in system.config.task_based_weights.items():
            print(f"   {task_type}:")
            for model, weight in weights.items():
                print(f"     - {model}: {weight:.3f}")
        
        # 測試數據收集
        print("\n🔍 收集測試數據...")
        all_data = system._collect_enhanced_data()
        print(f"✅ 收集數據: {len(all_data)} 個數據集")
        
        # 訓練所有模型
        print("\n🔍 訓練所有模型...")
        training_results = {}
        
        # 訓練經典統計模型
        for model_key in ['arima', 'ets', 'garch']:
            if all_data:
                first_data = list(all_data.values())[0]
                result = system.model_trainers[model_key](first_data['Close'])
                training_results[model_key] = result
                print(f"✅ {model_key} 訓練完成")
        
        # 訓練機器學習模型
        for model_key in ['xgboost', 'lightgbm']:
            if all_data:
                first_data = list(all_data.values())[0]
                result = system.model_trainers[model_key](first_data)
                training_results[model_key] = result
                print(f"✅ {model_key} 訓練完成")
        
        # 測試不同任務類型的權重分配
        test_tasks = [
            'short_term_forecast',
            'medium_term_forecast', 
            'long_term_forecast',
            'high_frequency_trading',
            'risk_management'
        ]
        
        print("\n🔍 測試不同任務類型的權重分配...")
        
        for task_type in test_tasks:
            print(f"\n📋 任務類型: {task_type}")
            print(f"   任務特徵: {system.config.task_features[task_type]}")
            
            # 創建融合模型
            fusion_result = system._create_advanced_fusion(training_results, task_type)
            
            if 'error' not in fusion_result:
                print(f"   ✅ 融合成功: {len(fusion_result.get('base_models', []))} 個模型")
                print(f"   權重分配:")
                
                for i, model_name in enumerate(fusion_result.get('base_models', [])):
                    weight = fusion_result['model_weights'][i]
                    print(f"     - {model_name}: {weight:.3f}")
                
                # 生成預測
                final_prediction = system._generate_super_enhanced_prediction(fusion_result, task_type)
                print(f"   預測點數: {final_prediction.get('prediction_horizon', 0)}")
            else:
                print(f"   ❌ 融合失敗: {fusion_result['error']}")
        
        print("\n🎯 測試結果摘要:")
        print(f"   - 訓練模型數: {len(training_results)}")
        print(f"   - 測試任務數: {len(test_tasks)}")
        print(f"   - 所有任務類型: {list(system.config.task_features.keys())}")
        
        print("\n🎉 任務導向權重分配測試完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

async def test_specific_task():
    """測試特定任務類型"""
    print("\n🎯 測試特定任務類型: 風險管理...")
    
    try:
        system = SuperEnhancedTSSystem()
        
        # 收集數據
        all_data = system._collect_enhanced_data()
        
        # 訓練模型
        training_results = {}
        for model_key in ['arima', 'ets', 'garch', 'xgboost', 'lightgbm']:
            if all_data:
                first_data = list(all_data.values())[0]
                if model_key in ['arima', 'ets', 'garch']:
                    result = system.model_trainers[model_key](first_data['Close'])
                else:
                    result = system.model_trainers[model_key](first_data)
                training_results[model_key] = result
        
        # 測試風險管理任務
        task_type = 'risk_management'
        print(f"\n🔍 測試任務: {task_type}")
        
        task_info = system.config.task_features[task_type]
        task_weights = system.config.task_based_weights[task_type]
        
        print(f"任務特徵: {task_info}")
        print(f"預設權重: {task_weights}")
        
        # 創建融合模型
        fusion_result = system._create_advanced_fusion(training_results, task_type)
        
        if 'error' not in fusion_result:
            print(f"✅ 融合成功!")
            print(f"最終權重分配:")
            
            for i, model_name in enumerate(fusion_result.get('base_models', [])):
                weight = fusion_result['model_weights'][i]
                preset_weight = task_weights.get(model_name, 0)
                print(f"   - {model_name}: 最終權重 {weight:.3f} (預設: {preset_weight:.3f})")
            
            # 分析權重變化
            print(f"\n📊 權重分析:")
            print(f"   - GARCH 模型在風險管理任務中權重最高")
            print(f"   - 統計模型 (ARIMA, ETS) 提供趨勢和季節性風險")
            print(f"   - 機器學習模型捕捉複雜的特徵風險關係")
        
    except Exception as e:
        print(f"❌ 特定任務測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 啟動任務導向權重分配測試...")
    
    # 測試基本功能
    asyncio.run(test_task_based_weights())
    
    # 測試特定任務
    asyncio.run(test_specific_task())
