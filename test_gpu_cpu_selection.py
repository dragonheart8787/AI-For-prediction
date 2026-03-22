#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試GPU/CPU選擇功能
"""

import asyncio
import logging
from super_enhanced_ts_system import SuperEnhancedTSSystem, SuperEnhancedTSConfig

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gpu_cpu_selection():
    """測試GPU/CPU選擇功能"""
    
    logger.info("🧪 開始測試GPU/CPU選擇功能")
    logger.info("=" * 60)
    
    # 測試1: 自動檢測模式
    logger.info("📋 測試1: 自動檢測模式")
    system1 = SuperEnhancedTSSystem()
    status1 = system1.get_device_status()
    logger.info(f"   當前設備: {status1['current_device']}")
    logger.info(f"   後端: {status1['backend']}")
    logger.info(f"   強制CPU: {status1['force_cpu']}")
    logger.info(f"   強制GPU: {status1['force_gpu']}")
    logger.info(f"   GPU偏好: {status1['gpu_preference']}")
    
    # 測試2: 強制CPU模式
    logger.info("\n📋 測試2: 強制CPU模式")
    system2 = SuperEnhancedTSSystem(force_cpu=True)
    status2 = system2.get_device_status()
    logger.info(f"   當前設備: {status2['current_device']}")
    logger.info(f"   後端: {status2['backend']}")
    logger.info(f"   強制CPU: {status2['force_cpu']}")
    logger.info(f"   強制GPU: {status2['force_gpu']}")
    
    # 測試3: 強制GPU模式
    logger.info("\n📋 測試3: 強制GPU模式")
    system3 = SuperEnhancedTSSystem(force_gpu=True)
    status3 = system3.get_device_status()
    logger.info(f"   當前設備: {status3['current_device']}")
    logger.info(f"   後端: {status3['backend']}")
    logger.info(f"   強制CPU: {status3['force_cpu']}")
    logger.info(f"   強制GPU: {status3['force_gpu']}")
    
    # 測試4: 指定GPU後端偏好
    logger.info("\n📋 測試4: 指定GPU後端偏好 (PyTorch)")
    system4 = SuperEnhancedTSSystem(gpu_preference='pytorch')
    status4 = system4.get_device_status()
    logger.info(f"   當前設備: {status4['current_device']}")
    logger.info(f"   後端: {status4['backend']}")
    logger.info(f"   GPU偏好: {status4['gpu_preference']}")
    
    # 測試5: 動態切換設備
    logger.info("\n📋 測試5: 動態切換設備")
    system5 = SuperEnhancedTSSystem()
    logger.info(f"   初始設備: {system5.get_device_status()['current_device']}")
    
    # 切換到CPU
    system5.switch_computation_device('cpu')
    logger.info(f"   切換到CPU後: {system5.get_device_status()['current_device']}")
    
    # 切換到GPU
    system5.switch_computation_device('gpu')
    logger.info(f"   切換到GPU後: {system5.get_device_status()['current_device']}")
    
    # 測試6: 顯示所有可用的GPU後端
    logger.info("\n📋 測試6: 顯示所有可用的GPU後端")
    available_backends = system5.gpu_config.get_available_backends()
    logger.info("   可用的GPU後端:")
    for backend_name, backend_info in available_backends.items():
        status = "✅" if backend_info['available'] else "❌"
        logger.info(f"     {status} {backend_name}: {backend_info['name']}")
    
    # 測試7: 運行簡單的預測任務
    logger.info("\n📋 測試7: 運行簡單的預測任務")
    try:
        # 創建測試數據
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 運行預測
        results = await system5.run_super_enhanced_system(['short_term_forecast'])
        logger.info("✅ 預測任務完成")
        
        # 顯示結果摘要
        if 'fusion_results' in results:
            for task_type, fusion_result in results['fusion_results'].items():
                logger.info(f"   任務: {task_type}")
                logger.info(f"   融合類型: {fusion_result.get('fusion_type', 'N/A')}")
                logger.info(f"   模型數量: {len(fusion_result.get('model_weights', {}))}")
                
    except Exception as e:
        logger.error(f"❌ 預測任務失敗: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 GPU/CPU選擇功能測試完成")

def test_device_switching():
    """測試設備切換功能"""
    
    logger.info("\n🔧 測試設備切換功能")
    logger.info("=" * 60)
    
    # 創建系統實例
    system = SuperEnhancedTSSystem()
    
    # 顯示初始狀態
    initial_status = system.get_device_status()
    logger.info(f"初始狀態: {initial_status['current_device']} ({initial_status['backend']})")
    
    # 測試切換到CPU
    logger.info("\n🔄 切換到CPU...")
    system.switch_computation_device('cpu')
    cpu_status = system.get_device_status()
    logger.info(f"CPU狀態: {cpu_status['current_device']} ({cpu_status['backend']})")
    
    # 測試切換到GPU (自動選擇)
    logger.info("\n🔄 切換到GPU (自動選擇)...")
    system.switch_computation_device('gpu')
    gpu_status = system.get_device_status()
    logger.info(f"GPU狀態: {gpu_status['current_device']} ({gpu_status['backend']})")
    
    # 測試切換到特定GPU後端
    available_backends = system.gpu_config.get_available_backends()
    for backend_name, backend_info in available_backends.items():
        if backend_info['available']:
            logger.info(f"\n🔄 切換到 {backend_name}...")
            system.switch_computation_device('gpu', backend_name)
            specific_status = system.get_device_status()
            logger.info(f"{backend_name}狀態: {specific_status['current_device']} ({specific_status['backend']})")
            break
    
    logger.info("\n✅ 設備切換測試完成")

if __name__ == "__main__":
    # 運行測試
    asyncio.run(test_gpu_cpu_selection())
    test_device_switching()
