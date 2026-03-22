#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示：如何將新創建的插件集成到系統中
"""

import asyncio
import os
import sys
from plugin_manager import PluginManager

# 設置編碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

async def demo_plugin_integration():
    """演示插件集成功能"""
    print("🚀 演示：如何將新創建的插件集成到系統中")
    print("=" * 70)
    
    # 創建插件管理器
    plugin_manager = PluginManager()
    
    # 1. 創建一個新的爬蟲插件
    print("\n📝 步驟 1: 創建一個新的爬蟲插件")
    print("-" * 50)
    
    new_plugin_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
電商數據爬蟲插件
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class EcommerceDataPlugin:
    """電商數據爬蟲插件"""
    
    def __init__(self):
        self.name = "ecommerce_data"
        self.version = "1.0.0"
        self.description = "爬取電商平台數據"
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("🚀 開始爬取電商數據...")
        
        try:
            # 模擬爬取電商數據
            ecommerce_data = [
                {
                    'platform': 'Amazon',
                    'product_category': '電子產品',
                    'sales_volume': random.randint(1000, 10000),
                    'price_range': '100-500',
                    'trending_score': random.uniform(0.7, 1.0),
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'platform': '淘寶',
                    'product_category': '服裝',
                    'sales_volume': random.randint(5000, 20000),
                    'price_range': '50-300',
                    'trending_score': random.uniform(0.8, 1.0),
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'platform': '京東',
                    'product_category': '家電',
                    'sales_volume': random.randint(800, 8000),
                    'price_range': '200-1000',
                    'trending_score': random.uniform(0.6, 0.9),
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return {
                'success': True,
                'data_type': 'ecommerce_data',
                'data': ecommerce_data,
                'metadata': {
                    'total_platforms': len(ecommerce_data),
                    'crawled_at': datetime.now().isoformat(),
                    'categories': list(set(item['product_category'] for item in ecommerce_data))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 爬取電商數據失敗: {{e}}")
            return {
                'success': False,
                'error': str(e),
                'data_type': 'ecommerce_data'
            }
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['ecommerce_data', 'sales_analytics', 'product_trends']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {
            'requests': '>=2.25.0',
            'beautifulsoup4': '>=4.9.0'
        }

# 測試代碼
if __name__ == "__main__":
    async def test_plugin():
        plugin = EcommerceDataPlugin()
        result = await plugin.crawl({'test': True})
        print(f"測試結果: {{result}}")
    
    asyncio.run(test_plugin())
'''
    
    # 保存新插件
    plugin_filename = "ecommerce_data_plugin.py"
    with open(plugin_filename, 'w', encoding='utf-8') as f:
        f.write(new_plugin_code)
    
    print(f"✅ 成功創建新插件: {plugin_filename}")
    
    # 2. 動態加載插件
    print("\n🔌 步驟 2: 動態加載新插件")
    print("-" * 50)
    
    try:
        # 動態導入插件
        import importlib.util
        spec = importlib.util.spec_from_file_location("ecommerce_plugin", plugin_filename)
        ecommerce_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ecommerce_module)
        
        # 創建插件實例
        plugin_instance = ecommerce_module.EcommerceDataPlugin()
        
        print(f"✅ 成功加載插件: {plugin_instance.name}")
        print(f"📋 插件描述: {plugin_instance.description}")
        print(f"🔧 支持類型: {plugin_instance.get_supported_types()}")
        print(f"📦 依賴要求: {plugin_instance.get_requirements()}")
        
    except Exception as e:
        print(f"❌ 加載插件失敗: {e}")
        return
    
    # 3. 測試插件功能
    print("\n🧪 步驟 3: 測試插件功能")
    print("-" * 50)
    
    try:
        # 執行爬取
        result = await plugin_instance.crawl({'test': True, 'platform': 'all'})
        
        if result['success']:
            print("✅ 插件測試成功！")
            print(f"📊 數據類型: {result['data_type']}")
            print(f"📈 爬取記錄數: {result['metadata']['total_platforms']}")
            print(f"🏷️  產品類別: {', '.join(result['metadata']['categories'])}")
            
            # 顯示詳細數據
            print("\n📋 爬取數據詳情:")
            for i, item in enumerate(result['data'], 1):
                print(f"  {i}. {item['platform']} - {item['product_category']}")
                print(f"     銷量: {item['sales_volume']}, 趨勢分數: {item['trending_score']:.2f}")
        else:
            print(f"❌ 插件測試失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        print(f"❌ 測試插件失敗: {e}")
    
    # 4. 創建插件配置文件
    print("\n⚙️  步驟 4: 創建插件配置文件")
    print("-" * 50)
    
    plugin_config = {
        "ecommerce_data_plugin": {
            "enabled": True,
            "priority": 1,
            "schedule": "daily",
            "config": {
                "platforms": ["Amazon", "淘寶", "京東"],
                "categories": ["電子產品", "服裝", "家電"],
                "update_interval": 3600
            }
        }
    }
    
    config_filename = "plugin_config.json"
    import json
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(plugin_config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 成功創建插件配置: {config_filename}")
    
    # 5. 展示插件管理功能
    print("\n🔧 步驟 5: 展示插件管理功能")
    print("-" * 50)
    
    print("📊 插件管理器功能:")
    print("  • 自動發現新插件")
    print("  • 動態加載/卸載")
    print("  • 插件配置管理")
    print("  • 插件性能監控")
    print("  • 插件依賴檢查")
    print("  • 插件版本管理")
    
    # 6. 創建使用說明文檔
    print("\n📚 步驟 6: 創建使用說明文檔")
    print("-" * 50)
    
    readme_content = f"""# 電商數據爬蟲插件使用說明

## 概述
這是一個自動生成的電商數據爬蟲插件，可以爬取多個電商平台的數據。

## 功能特點
- 🚀 支持多平台數據爬取
- 📊 自動數據分析和處理
- 🔄 可配置的更新頻率
- 📈 趨勢分析和評分

## 使用方法
1. 將插件文件放入 plugins/ 目錄
2. 在配置文件中啟用插件
3. 運行系統即可自動使用

## 配置選項
- platforms: 要爬取的電商平台
- categories: 產品類別
- update_interval: 更新間隔（秒）

## 數據格式
插件返回標準化的JSON格式數據，包含：
- 平台信息
- 產品類別
- 銷量數據
- 趨勢評分
- 時間戳

## 依賴要求
- requests >= 2.25.0
- beautifulsoup4 >= 4.9.0

## 注意事項
- 請遵守各平台的爬蟲政策
- 建議設置適當的爬取間隔
- 定期檢查插件更新
"""
    
    readme_filename = "PLUGIN_README.md"
    with open(readme_filename, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ 成功創建使用說明: {readme_filename}")
    
    # 7. 總結
    print("\n🎯 總結：系統如何自己建立相關爬蟲")
    print("=" * 70)
    print("1. 📝 自動生成插件代碼模板")
    print("2. 🔌 動態加載和測試插件")
    print("3. ⚙️  自動生成配置文件")
    print("4. 📚 創建完整使用文檔")
    print("5. 🧪 即時測試和驗證")
    print("6. 🔄 支持熱插拔和更新")
    print("7. 📊 完整的插件管理系統")
    
    print(f"\n✅ 本次演示創建了以下文件:")
    print(f"   • {plugin_filename} - 新爬蟲插件")
    print(f"   • {config_filename} - 插件配置")
    print(f"   • {readme_filename} - 使用說明")
    
    print("\n🎉 系統成功展示了完整的插件創建和集成流程！")
    print("   用戶現在可以：")
    print("   • 使用模板快速創建新爬蟲")
    print("   • 動態加載和管理插件")
    print("   • 自動配置和文檔生成")
    print("   • 即時測試和驗證功能")

if __name__ == "__main__":
    asyncio.run(demo_plugin_integration())
