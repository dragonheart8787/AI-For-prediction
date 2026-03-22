#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定義插件演示腳本
展示如何使用新創建的爬蟲插件
"""

import asyncio
import logging
import sys
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('custom_plugins_demo.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def demo_social_media_trends():
    """演示社交媒體趨勢插件"""
    logger.info("🔍 測試社交媒體趨勢插件...")
    
    try:
        # 動態導入插件
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "social_media_trends_plugin", 
            "social_media_trends_plugin.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 創建插件實例
        plugin = module.SocialMediaTrendsPlugin()
        
        # 執行爬取
        result = await plugin.crawl({'test': True})
        
        if result['success']:
            logger.info("✅ 社交媒體趨勢插件測試成功！")
            logger.info(f"📊 數據類型: {result['data_type']}")
            logger.info(f"📈 總記錄數: {result['metadata']['total_records']}")
            logger.info(f"🔍 分析結果: {result['analysis']}")
        else:
            logger.error(f"❌ 插件測試失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        logger.error(f"❌ 測試社交媒體趨勢插件時出錯: {e}")

async def demo_news_sentiment():
    """演示新聞情緒分析插件"""
    logger.info("🔍 測試新聞情緒分析插件...")
    
    try:
        # 動態導入插件
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "real_time_news_sentiment_plugin", 
            "real_time_news_sentiment_plugin.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 創建插件實例
        plugin = module.RealTimeNewsSentimentPlugin()
        
        # 執行爬取
        result = await plugin.crawl({'test': True})
        
        if result['success']:
            logger.info("✅ 新聞情緒分析插件測試成功！")
            logger.info(f"📊 數據類型: {result['data_type']}")
            logger.info(f"📰 總新聞數: {result['sentiment_analysis']['total_news']}")
            logger.info(f"😊 正面情緒: {result['sentiment_analysis']['sentiment_distribution']['positive']}")
            logger.info(f"😞 負面情緒: {result['sentiment_analysis']['sentiment_distribution']['negative']}")
            logger.info(f"🔥 熱門關鍵詞: {result['top_keywords'][:5]}")
        else:
            logger.error(f"❌ 插件測試失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        logger.error(f"❌ 測試新聞情緒分析插件時出錯: {e}")

async def demo_supply_chain():
    """演示供應鏈物流插件"""
    logger.info("🔍 測試供應鏈物流插件...")
    
    try:
        # 動態導入插件
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "supply_chain_logistics_plugin", 
            "supply_chain_logistics_plugin.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 創建插件實例
        plugin = module.SupplyChainLogisticsPlugin()
        
        # 執行爬取
        result = await plugin.crawl({'test': True})
        
        if result['success']:
            logger.info("✅ 供應鏈物流插件測試成功！")
            logger.info(f"📊 數據類型: {result['data_type']}")
            logger.info(f"🔗 總供應鏈數: {result['analysis']['total_chains']}")
            logger.info(f"🚢 海運數量: {result['analysis']['transport_modes']['海運']}")
            logger.info(f"✈️ 空運數量: {result['analysis']['transport_modes']['空運']}")
            logger.info(f"💰 總成本: ${result['analysis']['cost_analysis']['total_cost']:,.2f}")
            logger.info(f"🌍 地理分布: {result['geographic_distribution']['origin_countries']}")
        else:
            logger.error(f"❌ 插件測試失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        logger.error(f"❌ 測試供應鏈物流插件時出錯: {e}")

async def demo_plugin_integration():
    """演示插件集成使用"""
    logger.info("🔧 測試插件集成...")
    
    try:
        # 導入所有插件
        plugins = []
        
        # 社交媒體趨勢插件
        import importlib.util
        spec1 = importlib.util.spec_from_file_location(
            "social_media_trends_plugin", 
            "social_media_trends_plugin.py"
        )
        module1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(module1)
        plugins.append(("社交媒體趨勢", module1.SocialMediaTrendsPlugin()))
        
        # 新聞情緒分析插件
        spec2 = importlib.util.spec_from_file_location(
            "real_time_news_sentiment_plugin", 
            "real_time_news_sentiment_plugin.py"
        )
        module2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(module2)
        plugins.append(("新聞情緒分析", module2.RealTimeNewsSentimentPlugin()))
        
        # 供應鏈物流插件
        spec3 = importlib.util.spec_from_file_location(
            "supply_chain_logistics_plugin", 
            "supply_chain_logistics_plugin.py"
        )
        module3 = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(module3)
        plugins.append(("供應鏈物流", module3.SupplyChainLogisticsPlugin()))
        
        # 執行所有插件
        all_results = {}
        for name, plugin in plugins:
            logger.info(f"🚀 執行 {name} 插件...")
            result = await plugin.crawl({'test': True})
            all_results[name] = result
            
            if result['success']:
                logger.info(f"✅ {name} 插件執行成功")
            else:
                logger.warning(f"⚠️ {name} 插件執行失敗: {result.get('error', '未知錯誤')}")
        
        # 統計結果
        successful_plugins = sum(1 for r in all_results.values() if r['success'])
        total_plugins = len(plugins)
        
        logger.info(f"📊 插件集成測試完成！")
        logger.info(f"✅ 成功: {successful_plugins}/{total_plugins}")
        logger.info(f"📈 總數據記錄: {sum(r['metadata']['total_records'] for r in all_results.values() if r['success'])}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ 插件集成測試時出錯: {e}")
        return {}

async def main():
    """主函數"""
    logger.info("🎉 開始自定義插件演示！")
    logger.info("=" * 50)
    
    # 檢查插件文件是否存在
    plugin_files = [
        "social_media_trends_plugin.py",
        "real_time_news_sentiment_plugin.py", 
        "supply_chain_logistics_plugin.py"
    ]
    
    missing_files = []
    for file in plugin_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ 缺少以下插件文件: {missing_files}")
        logger.error("請先創建這些插件文件")
        return
    
    logger.info("✅ 所有插件文件檢查完成")
    
    # 執行單個插件測試
    await demo_social_media_trends()
    logger.info("-" * 30)
    
    await demo_news_sentiment()
    logger.info("-" * 30)
    
    await demo_supply_chain()
    logger.info("-" * 30)
    
    # 執行插件集成測試
    await demo_plugin_integration()
    
    logger.info("=" * 50)
    logger.info("🎉 自定義插件演示完成！")
    logger.info("💡 這些插件展示了系統可以自己建立相關爬蟲的能力")
    logger.info("🔧 您可以根據需要修改這些插件或創建新的插件")

if __name__ == "__main__":
    asyncio.run(main())
