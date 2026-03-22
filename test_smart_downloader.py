#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能下載器測試腳本
測試增強模型下載器的各種下載策略
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from start_enhanced_ai_system import EnhancedModelDownloader

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_downloader_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_smart_downloader():
    """測試智能下載器"""
    print("🧪 開始測試智能下載器...")
    
    # 創建下載器
    downloader = EnhancedModelDownloader("./test_enhanced_models")
    
    # 測試單個模型下載
    print("\n📥 測試單個模型下載:")
    
    # 測試timegpt (應該會嘗試多個策略)
    print("\n🔍 測試 TimeGPT 下載:")
    result1 = await downloader.smart_download_model('timegpt')
    print(f"TimeGPT 下載結果: {result1}")
    
    # 測試chronos (應該會成功)
    print("\n🔍 測試 Chronos 下載:")
    result2 = await downloader.smart_download_model('chronos')
    print(f"Chronos 下載結果: {result2}")
    
    # 測試patchtst (應該會嘗試多個策略)
    print("\n🔍 測試 PatchTST 下載:")
    result3 = await downloader.smart_download_model('patchtst')
    print(f"PatchTST 下載結果: {result3}")
    
    # 測試所有模型下載
    print("\n🚀 測試所有模型下載:")
    all_results = await downloader.download_all_models()
    
    print("\n📊 下載結果總結:")
    print(f"總模型數: {all_results['total_models']}")
    print(f"成功下載: {all_results['successful_downloads']}")
    print(f"失敗數: {all_results['failed_downloads']}")
    
    if 'download_stats' in all_results:
        stats = all_results['download_stats']
        print(f"直接下載成功: {stats['successful_downloads']}")
        print(f"本地備用創建: {stats['fallback_created']}")
        print(f"策略使用情況: {stats['strategy_used']}")
    
    # 檢查可用模型
    print("\n📋 可用模型列表:")
    available_models = downloader.get_available_models()
    for model in available_models:
        print(f"  ✅ {model}")
    
    print("\n🎉 智能下載器測試完成！")

async def test_individual_strategies():
    """測試個別下載策略"""
    print("\n🔧 測試個別下載策略...")
    
    downloader = EnhancedModelDownloader("./test_enhanced_models")
    
    # 測試配置下載
    print("\n📥 測試配置下載策略:")
    result = await downloader._try_download_config('test_model', 'microsoft/DialoGPT-medium')
    print(f"配置下載測試結果: {result}")
    
    # 測試替代模型搜索
    print("\n🔍 測試替代模型搜索:")
    alt_model = await downloader._search_alternative_model('timegpt', {
        'type': 'zero_shot',
        'name': 'TimeGPT'
    })
    print(f"找到的替代模型: {alt_model}")
    
    # 測試模型可用性
    print("\n✅ 測試模型可用性:")
    is_available = await downloader._test_model_availability('microsoft/DialoGPT-medium')
    print(f"模型可用性: {is_available}")
    
    print("\n🔧 個別策略測試完成！")

async def main():
    """主函數"""
    try:
        # 測試智能下載器
        await test_smart_downloader()
        
        # 測試個別策略
        await test_individual_strategies()
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
