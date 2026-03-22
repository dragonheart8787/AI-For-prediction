#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示系統如何自己建立相關爬蟲
"""

import asyncio
import os
import sys
from plugin_manager import PluginManager

# 設置編碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

async def demo_plugin_creation():
    """演示插件創建功能"""
    print("🚀 演示：系統如何自己建立相關爬蟲")
    print("=" * 60)
    
    # 創建插件管理器
    plugin_manager = PluginManager()
    
    # 1. 創建一個新的爬蟲插件模板
    print("\n📝 步驟 1: 創建新爬蟲插件模板")
    print("-" * 40)
    
    plugin_name = "custom_social_media"
    plugin_type = "social_media"
    
    template = plugin_manager.create_plugin_template(plugin_name, plugin_type)
    
    # 保存模板到文件
    template_filename = f"{plugin_name}_plugin.py"
    with open(template_filename, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ 成功創建插件模板: {template_filename}")
    print(f"📁 模板已保存到: {os.path.abspath(template_filename)}")
    
    # 2. 顯示模板內容
    print("\n📋 插件模板內容預覽:")
    print("-" * 40)
    lines = template.split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2d}: {line}")
    print("   ... (更多內容)")
    
    # 3. 創建另一個插件模板
    print("\n📝 步驟 2: 創建另一個爬蟲插件模板")
    print("-" * 40)
    
    plugin_name2 = "real_estate_data"
    plugin_type2 = "real_estate"
    
    template2 = plugin_manager.create_plugin_template(plugin_name2, plugin_type2)
    
    template_filename2 = f"{plugin_name2}_plugin.py"
    with open(template_filename2, 'w', encoding='utf-8') as f:
        f.write(template2)
    
    print(f"✅ 成功創建插件模板: {template_filename2}")
    
    # 4. 展示插件管理器的其他功能
    print("\n🔧 步驟 3: 展示插件管理器功能")
    print("-" * 40)
    
    print("📊 可用功能:")
    print("  • create_plugin_template() - 創建插件模板")
    print("  • discover_plugins() - 自動發現插件")
    print("  • load_plugin() - 動態加載插件")
    print("  • unload_plugin() - 卸載插件")
    print("  • get_plugin_info() - 獲取插件信息")
    
    # 5. 創建一個完整的示例插件
    print("\n📝 步驟 4: 創建一個完整的示例插件")
    print("-" * 40)
    
    complete_plugin = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整示例：社交媒體趨勢爬蟲插件
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import aiohttp
import json

logger = logging.getLogger(__name__)

class SocialMediaTrendsPlugin:
    """社交媒體趨勢爬蟲插件"""
    
    def __init__(self):
        self.name = "social_media_trends"
        self.version = "1.0.0"
        self.description = "爬取社交媒體趨勢數據"
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("🚀 開始爬取社交媒體趨勢數據...")
        
        try:
            # 模擬爬取社交媒體數據
            trends_data = [
                {{
                    'platform': 'Twitter',
                    'trending_topic': 'AI技術發展',
                    'mentions': 15420,
                    'sentiment': 'positive',
                    'timestamp': datetime.now().isoformat()
                }},
                {{
                    'platform': 'Instagram',
                    'trending_topic': '可持續發展',
                    'mentions': 8920,
                    'sentiment': 'neutral',
                    'timestamp': datetime.now().isoformat()
                }},
                {{
                    'platform': 'LinkedIn',
                    'trending_topic': '遠程工作',
                    'mentions': 5670,
                    'sentiment': 'positive',
                    'timestamp': datetime.now().isoformat()
                }}
            ]
            
            return {{
                'success': True,
                'data_type': 'social_media_trends',
                'data': trends_data,
                'metadata': {{
                    'total_trends': len(trends_data),
                    'crawled_at': datetime.now().isoformat(),
                    'platforms': ['Twitter', 'Instagram', 'LinkedIn']
                }}
            }}
            
        except Exception as e:
            logger.error(f"❌ 爬取社交媒體趨勢數據失敗: {{e}}")
            return {{
                'success': False,
                'error': str(e),
                'data_type': 'social_media_trends'
            }}
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['social_media_trends', 'trending_topics', 'sentiment_analysis']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {{
            'aiohttp': '>=3.8.0',
            'pandas': '>=1.3.0'
        }}

# 測試代碼
if __name__ == "__main__":
    async def test_plugin():
        plugin = SocialMediaTrendsPlugin()
        result = await plugin.crawl({{'test': True}})
        print(f"測試結果: {{result}}")
    
    asyncio.run(test_plugin())
'''
    
    complete_filename = "complete_social_media_plugin.py"
    with open(complete_filename, 'w', encoding='utf-8') as f:
        f.write(complete_plugin)
    
    print(f"✅ 成功創建完整示例插件: {complete_filename}")
    
    # 6. 總結
    print("\n🎯 總結：系統如何自己建立相關爬蟲")
    print("=" * 60)
    print("1. 📝 自動生成插件模板")
    print("2. 🔧 提供標準化的插件結構")
    print("3. 📚 包含完整的示例代碼")
    print("4. 🚀 支持即時測試和驗證")
    print("5. 🔄 可以動態加載和卸載")
    print("6. 📊 自動發現和管理插件")
    
    print(f"\n✅ 本次演示創建了以下文件:")
    print(f"   • {template_filename}")
    print(f"   • {template_filename2}")
    print(f"   • {complete_filename}")
    
    print("\n🎉 系統成功展示了如何自己建立相關爬蟲！")
    print("   用戶只需要運行這個腳本，就能獲得完整的爬蟲插件模板。")

if __name__ == "__main__":
    asyncio.run(demo_plugin_creation())
