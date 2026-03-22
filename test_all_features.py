#!/usr/bin/env python3
"""
AGI系統全功能測試腳本
測試所有新功能和修復
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

async def test_smart_model_selection():
    """測試智能模型選擇"""
    print("\n🎯 測試智能模型選擇...")
    
    api = EnhancedAPI()
    
    # 設置模型性能數據
    api.model_selector.update_performance('financial_lstm', 
        ModelPerformance('financial_lstm', 0.85, 0.8, 0.5, 0.9, datetime.now()))
    api.model_selector.update_performance('weather_transformer', 
        ModelPerformance('weather_transformer', 0.92, 0.9, 0.3, 0.95, datetime.now()))
    
    # 測試不同任務類型
    test_cases = [
        ('financial', '金融預測任務'),
        ('weather', '天氣預測任務'),
        ('medical', '醫療預測任務')
    ]
    
    for task_type, description in test_cases:
        selected_model = api.model_selector.select_best_model(task_type, 10)
        print(f"  {description}: 選擇模型 {selected_model}")
    
    return True

async def test_performance_monitoring():
    """測試性能監控"""
    print("\n📊 測試性能監控...")
    
    api = EnhancedAPI()
    
    # 模擬性能數據
    test_metrics = [
        ('accuracy', 0.85, 'financial_lstm'),
        ('confidence', 0.8, 'financial_lstm'),
        ('processing_time', 0.5, 'financial_lstm'),
        ('accuracy', 0.92, 'weather_transformer'),
        ('confidence', 0.9, 'weather_transformer'),
        ('processing_time', 0.3, 'weather_transformer')
    ]
    
    for metric_name, value, model_name in test_metrics:
        api.performance_monitor.record_metric(metric_name, value, model_name)
        print(f"  記錄 {metric_name}: {value} ({model_name})")
    
    # 獲取性能摘要
    summary = api.performance_monitor.get_performance_summary()
    print(f"  性能摘要: {len(summary)} 個指標")
    
    return True

async def test_auto_recovery():
    """測試自動故障恢復"""
    print("\n🔄 測試自動故障恢復...")
    
    api = EnhancedAPI()
    
    # 模擬不同類型的故障
    test_failures = [
        ('financial_lstm', 'connection_error', '連接錯誤'),
        ('weather_transformer', 'memory_error', '記憶體錯誤'),
        ('medical_model', 'model_error', '模型錯誤'),
        ('energy_model', 'general_error', '通用錯誤')
    ]
    
    for model_name, error_type, description in test_failures:
        success = api.auto_recovery.handle_failure(model_name, error_type)
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {description}: {status}")
    
    return True

async def test_model_fusion():
    """測試模型融合"""
    print("\n🧠 測試模型融合...")
    
    api = EnhancedAPI()
    
    # 模擬多個預測結果
    predictions = [
        {
            'model_name': 'financial_lstm',
            'prediction': [0.85, 0.92, 0.78],
            'confidence': 0.8,
            'processing_time': 0.5
        },
        {
            'model_name': 'weather_transformer',
            'prediction': [0.82, 0.89, 0.81],
            'confidence': 0.9,
            'processing_time': 0.3
        },
        {
            'model_name': 'medical_model',
            'prediction': [0.88, 0.91, 0.85],
            'confidence': 0.85,
            'processing_time': 0.4
        }
    ]
    
    # 進行融合
    fusion_result = api.model_fusion.fuse_predictions(predictions)
    
    if fusion_result:
        print(f"  融合成功: {len(fusion_result.get('weights', []))} 個模型")
        print(f"  融合置信度: {fusion_result.get('confidence', 0):.3f}")
        print(f"  權重分配: {[f'{w:.3f}' for w in fusion_result.get('weights', [])]}")
    else:
        print("  ❌ 融合失敗")
    
    return True

async def test_smart_prediction():
    """測試智能預測"""
    print("\n🚀 測試智能預測...")
    
    api = EnhancedAPI()
    
    # 設置模型性能
    api.model_selector.update_performance('financial_lstm', 
        ModelPerformance('financial_lstm', 0.85, 0.8, 0.5, 0.9, datetime.now()))
    api.model_selector.update_performance('weather_transformer', 
        ModelPerformance('weather_transformer', 0.92, 0.9, 0.3, 0.95, datetime.now()))
    
    # 測試不同類型的預測
    test_cases = [
        ('financial', np.random.randn(1, 10), '金融預測'),
        ('weather', np.random.randn(1, 15), '天氣預測'),
        ('medical', np.random.randn(1, 8), '醫療預測')
    ]
    
    for task_type, input_data, description in test_cases:
        result = await api.smart_predict(task_type, input_data, use_fusion=True)
        
        if 'error' not in result:
            print(f"  {description}: ✅ 成功")
            print(f"    選擇模型: {result.get('model_name', 'N/A')}")
            print(f"    置信度: {result.get('confidence', 0):.3f}")
            if 'fusion' in result:
                print(f"    融合模型數: {result['fusion'].get('model_count', 0)}")
        else:
            print(f"  {description}: ❌ 失敗 - {result['error']}")
    
    return True

async def test_system_status():
    """測試系統狀態"""
    print("\n📈 測試系統狀態...")
    
    api = EnhancedAPI()
    
    # 模擬一些操作
    await api.smart_predict('financial', np.random.randn(1, 10))
    await api.smart_predict('weather', np.random.randn(1, 15))
    
    # 獲取系統狀態
    status = api.get_system_status()
    
    print(f"  性能摘要: {len(status.get('performance_summary', {}))} 個指標")
    print(f"  恢復歷史: {len(status.get('recovery_history', []))} 條記錄")
    print(f"  融合歷史: {len(status.get('fusion_history', []))} 條記錄")
    print(f"  模型選擇: {len(status.get('model_selections', []))} 條記錄")
    
    return True

async def run_all_tests():
    """運行所有測試"""
    print("🧪 AGI系統全功能測試")
    print("=" * 50)
    
    tests = [
        ("智能模型選擇", test_smart_model_selection),
        ("性能監控", test_performance_monitoring),
        ("自動故障恢復", test_auto_recovery),
        ("模型融合", test_model_fusion),
        ("智能預測", test_smart_prediction),
        ("系統狀態", test_system_status)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
            status = "✅ 通過" if results[test_name] else "❌ 失敗"
            print(f"{status} - {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ 異常 - {test_name}: {e}")
    
    # 總結
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 測試總結: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統運行正常。")
    else:
        print("⚠️ 部分測試失敗，請檢查相關功能。")
    
    return results

async def main():
    """主函數"""
    try:
        results = await run_all_tests()
        
        # 生成測試報告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'passed_tests': sum(results.values()),
            'failed_tests': len(results) - sum(results.values()),
            'results': results
        }
        
        # 保存測試報告
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 測試報告已保存到: test_report.json")
        
    except Exception as e:
        logger.error(f"❌ 測試執行失敗: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 