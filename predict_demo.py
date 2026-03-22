#!/usr/bin/env python3
"""
AGI預測演示腳本
展示如何使用當前的增強模型進行預測
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from agi_new_features import EnhancedAPI, ModelPerformance

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_financial_prediction():
    """演示金融預測"""
    print("\n💰 金融預測演示")
    print("-" * 30)
    
    api = EnhancedAPI()
    
    # 設置模型性能
    api.model_selector.update_performance('financial_lstm', 
        ModelPerformance('financial_lstm', 0.85, 0.8, 0.5, 0.9, datetime.now()))
    
    # 模擬金融數據（股價、交易量等）
    financial_data = np.random.randn(1, 20)  # 20個特徵
    print(f"📊 輸入數據形狀: {financial_data.shape}")
    print(f"📈 數據範圍: {financial_data.min():.3f} ~ {financial_data.max():.3f}")
    
    # 進行預測
    result = await api.smart_predict('financial', financial_data, use_fusion=True)
    
    if 'error' not in result:
        print(f"🎯 選擇模型: {result.get('model_name', 'N/A')}")
        print(f"📊 預測結果: {len(result.get('prediction', []))} 個值")
        print(f"🎯 置信度: {result.get('confidence', 0):.3f}")
        print(f"⏱️ 處理時間: {result.get('processing_time', 0):.3f}秒")
        
        if 'fusion' in result:
            fusion = result['fusion']
            print(f"🧠 融合模型數: {fusion.get('model_count', 0)}")
            print(f"🎯 融合置信度: {fusion.get('confidence', 0):.3f}")
    else:
        print(f"❌ 預測失敗: {result['error']}")
    
    return result

async def demo_weather_prediction():
    """演示天氣預測"""
    print("\n🌤️ 天氣預測演示")
    print("-" * 30)
    
    api = EnhancedAPI()
    
    # 設置模型性能
    api.model_selector.update_performance('weather_transformer', 
        ModelPerformance('weather_transformer', 0.92, 0.9, 0.3, 0.95, datetime.now()))
    
    # 模擬天氣數據（溫度、濕度、氣壓等）
    weather_data = np.random.randn(1, 15)  # 15個特徵
    print(f"📊 輸入數據形狀: {weather_data.shape}")
    print(f"🌡️ 數據範圍: {weather_data.min():.3f} ~ {weather_data.max():.3f}")
    
    # 進行預測
    result = await api.smart_predict('weather', weather_data, use_fusion=True)
    
    if 'error' not in result:
        print(f"🎯 選擇模型: {result.get('model_name', 'N/A')}")
        print(f"📊 預測結果: {len(result.get('prediction', []))} 個值")
        print(f"🎯 置信度: {result.get('confidence', 0):.3f}")
        print(f"⏱️ 處理時間: {result.get('processing_time', 0):.3f}秒")
        
        if 'fusion' in result:
            fusion = result['fusion']
            print(f"🧠 融合模型數: {fusion.get('model_count', 0)}")
            print(f"🎯 融合置信度: {fusion.get('confidence', 0):.3f}")
    else:
        print(f"❌ 預測失敗: {result['error']}")
    
    return result

async def demo_medical_prediction():
    """演示醫療預測"""
    print("\n🏥 醫療預測演示")
    print("-" * 30)
    
    api = EnhancedAPI()
    
    # 設置模型性能
    api.model_selector.update_performance('medical_cnn', 
        ModelPerformance('medical_cnn', 0.88, 0.85, 0.4, 0.92, datetime.now()))
    
    # 模擬醫療數據（體溫、血壓、心率等）
    medical_data = np.random.randn(1, 12)  # 12個特徵
    print(f"📊 輸入數據形狀: {medical_data.shape}")
    print(f"💓 數據範圍: {medical_data.min():.3f} ~ {medical_data.max():.3f}")
    
    # 進行預測
    result = await api.smart_predict('medical', medical_data, use_fusion=True)
    
    if 'error' not in result:
        print(f"🎯 選擇模型: {result.get('model_name', 'N/A')}")
        print(f"📊 預測結果: {len(result.get('prediction', []))} 個值")
        print(f"🎯 置信度: {result.get('confidence', 0):.3f}")
        print(f"⏱️ 處理時間: {result.get('processing_time', 0):.3f}秒")
        
        if 'fusion' in result:
            fusion = result['fusion']
            print(f"🧠 融合模型數: {fusion.get('model_count', 0)}")
            print(f"🎯 融合置信度: {fusion.get('confidence', 0):.3f}")
    else:
        print(f"❌ 預測失敗: {result['error']}")
    
    return result

async def demo_energy_prediction():
    """演示能源預測"""
    print("\n⚡ 能源預測演示")
    print("-" * 30)
    
    api = EnhancedAPI()
    
    # 設置模型性能
    api.model_selector.update_performance('energy_hybrid', 
        ModelPerformance('energy_hybrid', 0.90, 0.87, 0.6, 0.93, datetime.now()))
    
    # 模擬能源數據（用電量、發電量、負載等）
    energy_data = np.random.randn(1, 18)  # 18個特徵
    print(f"📊 輸入數據形狀: {energy_data.shape}")
    print(f"⚡ 數據範圍: {energy_data.min():.3f} ~ {energy_data.max():.3f}")
    
    # 進行預測
    result = await api.smart_predict('energy', energy_data, use_fusion=True)
    
    if 'error' not in result:
        print(f"🎯 選擇模型: {result.get('model_name', 'N/A')}")
        print(f"📊 預測結果: {len(result.get('prediction', []))} 個值")
        print(f"🎯 置信度: {result.get('confidence', 0):.3f}")
        print(f"⏱️ 處理時間: {result.get('processing_time', 0):.3f}秒")
        
        if 'fusion' in result:
            fusion = result['fusion']
            print(f"🧠 融合模型數: {fusion.get('model_count', 0)}")
            print(f"🎯 融合置信度: {fusion.get('confidence', 0):.3f}")
    else:
        print(f"❌ 預測失敗: {result['error']}")
    
    return result

async def demo_system_status():
    """演示系統狀態"""
    print("\n📊 系統狀態演示")
    print("-" * 30)
    
    api = EnhancedAPI()
    
    # 進行一些預測來生成數據
    await api.smart_predict('financial', np.random.randn(1, 10))
    await api.smart_predict('weather', np.random.randn(1, 15))
    await api.smart_predict('medical', np.random.randn(1, 12))
    
    # 獲取系統狀態
    status = api.get_system_status()
    
    print("📈 性能摘要:")
    for metric, data in status.get('performance_summary', {}).items():
        if isinstance(data, dict):
            print(f"  {metric}: 當前={data.get('current', 0):.3f}, 平均={data.get('average', 0):.3f}")
    
    print(f"\n🔄 恢復歷史: {len(status.get('recovery_history', []))} 條記錄")
    print(f"🧠 融合歷史: {len(status.get('fusion_history', []))} 條記錄")
    print(f"🎯 模型選擇: {len(status.get('model_selections', []))} 條記錄")
    
    return status

async def main():
    """主演示函數"""
    print("🚀 AGI預測系統演示")
    print("=" * 50)
    print("展示修復後的增強AGI系統的預測能力")
    print("=" * 50)
    
    # 運行所有演示
    demos = [
        ("金融預測", demo_financial_prediction),
        ("天氣預測", demo_weather_prediction),
        ("醫療預測", demo_medical_prediction),
        ("能源預測", demo_energy_prediction),
        ("系統狀態", demo_system_status)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🎬 開始 {demo_name} 演示...")
            results[demo_name] = await demo_func()
            print(f"✅ {demo_name} 演示完成")
        except Exception as e:
            print(f"❌ {demo_name} 演示失敗: {e}")
            results[demo_name] = None
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 演示總結")
    print("=" * 50)
    
    successful_demos = sum(1 for result in results.values() if result is not None)
    total_demos = len(results)
    
    print(f"✅ 成功演示: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("🎉 所有演示成功！系統運行正常。")
    else:
        print("⚠️ 部分演示失敗，請檢查相關功能。")
    
    # 保存演示結果
    demo_report = {
        'timestamp': datetime.now().isoformat(),
        'total_demos': total_demos,
        'successful_demos': successful_demos,
        'results': {name: str(result) for name, result in results.items()}
    }
    
    with open('demo_report.json', 'w', encoding='utf-8') as f:
        json.dump(demo_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 演示報告已保存到: demo_report.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 