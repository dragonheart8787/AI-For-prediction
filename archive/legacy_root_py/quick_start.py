#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ SuperFusionAGI 快速啟動器
快速啟動核心功能
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# 設置編碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class QuickStarter:
    """快速啟動器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.status = {}
    
    def print_banner(self):
        """打印橫幅"""
        print("=" * 60)
        print("⚡ SuperFusionAGI 快速啟動器")
        print("🚀 快速啟動核心功能")
        print("=" * 60)
        print(f"⏰ 啟動時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    async def start_plugin_system(self):
        """啟動插件系統"""
        print("\n🔌 啟動插件系統...")
        
        try:
            # 檢查插件管理器是否存在
            if os.path.exists("plugin_manager.py"):
                from plugin_manager import PluginManager
                plugin_manager = PluginManager()
                
                # 創建插件目錄
                if not os.path.exists("plugins"):
                    os.makedirs("plugins")
                    print("📁 創建 plugins/ 目錄")
                
                self.status['plugin_system'] = "✅ 已啟動"
                print("✅ 插件系統啟動成功")
            else:
                self.status['plugin_system'] = "⚠️  插件管理器不存在"
                print("⚠️  插件管理器不存在，跳過")
                
        except Exception as e:
            self.status['plugin_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 插件系統啟動失敗: {e}")
    
    async def start_data_crawler(self):
        """啟動數據爬取系統"""
        print("\n📊 啟動數據爬取系統...")
        
        try:
            # 檢查爬蟲系統是否存在
            if os.path.exists("enhanced_comprehensive_crawler.py"):
                from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler
                crawler = EnhancedComprehensiveDataCrawler()
                
                self.status['data_crawler'] = "✅ 已啟動"
                print("✅ 數據爬取系統啟動成功")
            else:
                self.status['data_crawler'] = "⚠️  爬蟲系統不存在"
                print("⚠️  爬蟲系統不存在，跳過")
                
        except Exception as e:
            self.status['data_crawler'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 數據爬取系統啟動失敗: {e}")
    
    async def start_demo_system(self):
        """啟動演示系統"""
        print("\n🎬 啟動演示系統...")
        
        try:
            # 檢查演示腳本是否存在
            if os.path.exists("demo_plugin_creation.py"):
                self.status['demo_system'] = "✅ 已準備"
                print("✅ 演示系統準備完成")
            else:
                self.status['demo_system'] = "⚠️  演示腳本不存在"
                print("⚠️  演示腳本不存在，跳過")
                
        except Exception as e:
            self.status['demo_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 演示系統啟動失敗: {e}")
    
    async def create_sample_plugins(self):
        """創建示例插件"""
        print("\n📝 創建示例插件...")
        
        try:
            # 創建一個簡單的示例插件
            sample_plugin = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例爬蟲插件
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class SamplePlugin:
    """示例爬蟲插件"""
    
    def __init__(self):
        self.name = "sample_plugin"
        self.version = "1.0.0"
        self.description = "示例爬蟲插件"
    
    async def crawl(self, config: Dict) -> Dict[str, Any]:
        """執行爬取操作"""
        logger.info("🚀 開始示例爬取...")
        
        try:
            # 模擬數據
            sample_data = [
                {
                    'id': 1,
                    'name': '示例數據1',
                    'value': random.randint(100, 1000),
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'id': 2,
                    'name': '示例數據2',
                    'value': random.randint(100, 1000),
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return {
                'success': True,
                'data_type': 'sample_data',
                'data': sample_data,
                'metadata': {
                    'total_records': len(sample_data),
                    'crawled_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 示例爬取失敗: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_type': 'sample_data'
            }
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ['sample_data']
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {}

# 測試代碼
if __name__ == "__main__":
    async def test_plugin():
        plugin = SamplePlugin()
        result = await plugin.crawl({'test': True})
        print(f"測試結果: {result}")
    
    asyncio.run(test_plugin())
'''
            
            # 保存示例插件
            with open("sample_plugin.py", "w", encoding='utf-8') as f:
                f.write(sample_plugin)
            
            self.status['sample_plugins'] = "✅ 已創建"
            print("✅ 示例插件創建成功")
            
        except Exception as e:
            self.status['sample_plugins'] = f"❌ 創建失敗: {e}"
            print(f"❌ 示例插件創建失敗: {e}")
    
    async def start_all_core_systems(self):
        """啟動所有核心系統"""
        print("\n🚀 開始啟動核心系統...")
        
        # 並行啟動核心系統
        await asyncio.gather(
            self.start_plugin_system(),
            self.start_data_crawler(),
            self.start_demo_system(),
            self.create_sample_plugins()
        )
    
    def show_status(self):
        """顯示狀態"""
        print("\n" + "=" * 60)
        print("📊 系統狀態")
        print("=" * 60)
        
        for system, status in self.status.items():
            print(f"{system.replace('_', ' ').title()}: {status}")
        
        print("=" * 60)
        
        # 計算啟動時間
        end_time = datetime.now()
        startup_time = (end_time - self.start_time).total_seconds()
        print(f"⏱️  啟動時間: {startup_time:.2f} 秒")
    
    def show_quick_guide(self):
        """顯示快速指南"""
        print("\n" + "=" * 60)
        print("📖 快速使用指南")
        print("=" * 60)
        
        guide = """
🎯 可用功能:

1. 🔌 插件系統
   - 自動發現和管理爬蟲插件
   - 支持動態加載

2. 📊 數據爬取系統
   - 多源數據自動爬取
   - 實時數據更新

3. 🎬 演示系統
   - 插件創建演示
   - 系統功能展示

4. 📝 示例插件
   - 自動生成的示例代碼
   - 學習和測試用途

🚀 快速開始:
- 運行 python demo_plugin_creation.py 查看插件創建
- 運行 python start_enhanced_comprehensive_crawler.py 啟動爬蟲
- 運行 python sample_plugin.py 測試示例插件

📞 更多功能:
- 查看 SYSTEM_SELF_EXTENDING_DEMO.md 了解詳細功能
- 運行 python start_all_systems.py 啟動完整系統
        """
        
        print(guide)
    
    async def run_quick_demo(self):
        """運行快速演示"""
        print("\n🎬 運行快速演示...")
        
        try:
            # 測試示例插件
            print("📝 測試示例插件...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("sample_plugin", "sample_plugin.py")
            sample_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sample_module)
            
            plugin = sample_module.SamplePlugin()
            result = await plugin.crawl({'test': True})
            
            if result['success']:
                print("✅ 示例插件測試成功！")
                print(f"📊 數據類型: {result['data_type']}")
                print(f"📈 記錄數: {result['metadata']['total_records']}")
            else:
                print(f"❌ 示例插件測試失敗: {result.get('error', '未知錯誤')}")
            
        except Exception as e:
            print(f"❌ 快速演示失敗: {e}")
    
    def save_quick_report(self):
        """保存快速報告"""
        report = {
            "startup_time": self.start_time.isoformat(),
            "status": self.status,
            "total_systems": len(self.status),
            "successful_systems": len([s for s in self.status.values() if "✅" in s]),
            "warning_systems": len([s for s in self.status.values() if "⚠️" in s]),
            "failed_systems": len([s for s in self.status.values() if "❌" in s])
        }
        
        with open("quick_startup_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 快速啟動報告已保存到: quick_startup_report.json")

async def main():
    """主函數"""
    starter = QuickStarter()
    
    # 打印橫幅
    starter.print_banner()
    
    # 啟動核心系統
    await starter.start_all_core_systems()
    
    # 顯示狀態
    starter.show_status()
    
    # 顯示指南
    starter.show_quick_guide()
    
    # 保存報告
    starter.save_quick_report()
    
    # 詢問是否運行演示
    print("\n" + "=" * 60)
    print("🎬 是否要運行快速演示？(y/n): ", end="")
    
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '是']:
            await starter.run_quick_demo()
    except KeyboardInterrupt:
        print("\n👋 用戶取消演示")
    
    print("\n🎉 SuperFusionAGI 快速啟動完成！")
    print("🚀 核心功能已就緒！")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 快速啟動被用戶中斷")
    except Exception as e:
        print(f"\n❌ 快速啟動失敗: {e}")
        import traceback
        traceback.print_exc()
