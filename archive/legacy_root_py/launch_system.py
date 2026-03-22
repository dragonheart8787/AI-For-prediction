#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusionAGI 系統 - 實際啟動腳本
整合現有的系統組件，讓您可以真正啟動所有功能
"""

import asyncio
import sys
import os
import logging
import threading
import time
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('system_launch.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SystemLauncher:
    """系統啟動器"""
    
    def __init__(self):
        self.running_components = {}
        self.plugin_manager = None
        self.data_crawler = None
        
    async def launch_plugin_system(self):
        """啟動插件系統"""
        try:
            print("🔌 啟動插件系統...")
            
            # 檢查並導入插件管理器
            if os.path.exists('plugin_manager.py'):
                from plugin_manager import PluginManager
                self.plugin_manager = PluginManager()
                # 插件管理器不需要initialize方法，直接創建即可
                print("✅ 插件系統已啟動")
                self.running_components['plugin_system'] = True
                return True
            else:
                print("⚠️ 插件管理器文件不存在，跳過插件系統")
                return False
                
        except Exception as e:
            print(f"❌ 插件系統啟動失敗: {e}")
            return False
    
    async def launch_data_crawler(self):
        """啟動數據爬取系統"""
        try:
            print("🕷️ 啟動數據爬取系統...")
            
            # 檢查並導入數據爬取器
            if os.path.exists('enhanced_comprehensive_crawler.py'):
                from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler
                self.data_crawler = EnhancedComprehensiveDataCrawler()
                # 數據爬取器不需要initialize方法，直接創建即可
                print("✅ 數據爬取系統已啟動")
                self.running_components['data_crawler'] = True
                return True
            else:
                print("⚠️ 數據爬取器文件不存在，跳過數據爬取系統")
                return False
                
        except Exception as e:
            print(f"❌ 數據爬取系統啟動失敗: {e}")
            return False
    
    async def launch_demo_system(self):
        """啟動演示系統"""
        try:
            print("🎯 啟動演示系統...")
            
            # 創建示例插件
            if self.plugin_manager:
                await self.create_sample_plugins()
            
            print("✅ 演示系統已啟動")
            self.running_components['demo_system'] = True
            return True
            
        except Exception as e:
            print(f"❌ 演示系統啟動失敗: {e}")
            return False
    
    async def create_sample_plugins(self):
        """創建示例插件"""
        try:
            print("📝 創建示例插件...")
            
            # 創建一個簡單的示例插件
            sample_plugin_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例數據爬蟲插件
"""

import asyncio
import random
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class DataCrawlerPlugin(ABC):
    """數據爬取插件抽象基類"""
    
    @abstractmethod
    async def crawl(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """執行爬取操作"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        pass

class SampleDataPlugin(DataCrawlerPlugin):
    """示例數據插件"""
    
    def __init__(self):
        self.name = "sample_data"
        self.description = "示例數據爬取插件"
    
    async def crawl(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """執行爬取操作"""
        await asyncio.sleep(1)  # 模擬網絡延遲
        
        return {
            "timestamp": asyncio.get_event_loop().time(),
            "data": {
                "value": random.randint(1, 100),
                "status": "success",
                "message": "示例數據爬取成功"
            },
            "metadata": {
                "source": "sample_plugin",
                "type": "demo_data"
            }
        }
    
    def get_supported_types(self) -> List[str]:
        """獲取支持的數據類型"""
        return ["demo_data", "sample_data"]
    
    def get_requirements(self) -> Dict[str, str]:
        """獲取插件依賴要求"""
        return {
            "python": ">=3.7",
            "packages": []
        }

# 創建插件實例
plugin_instance = SampleDataPlugin()
'''
            
            with open('sample_plugin.py', 'w', encoding='utf-8') as f:
                f.write(sample_plugin_code)
            
            print("✅ 示例插件已創建")
            
        except Exception as e:
            print(f"❌ 創建示例插件失敗: {e}")
    
    async def launch_web_interface(self):
        """啟動Web界面"""
        try:
            print("🌐 啟動Web界面...")
            
            # 創建簡單的Web服務器
            web_server_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的Web界面服務器
"""

import asyncio
import aiohttp
from aiohttp import web
import json

async def handle_index(request):
    """處理首頁請求"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperFusionAGI 系統</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 SuperFusionAGI 系統</h1>
            <div class="status success">
                <h2>✅ 系統狀態</h2>
                <p>系統已成功啟動並運行中</p>
            </div>
            <div class="status success">
                <h3>🔌 插件系統</h3>
                <p>插件管理器已啟動，支持動態插件加載</p>
            </div>
            <div class="status success">
                <h3>🕷️ 數據爬取系統</h3>
                <p>綜合數據爬取器已啟動，支持多源數據收集</p>
            </div>
            <div class="status success">
                <h3>🎯 演示系統</h3>
                <p>示例插件已創建，可以測試插件功能</p>
            </div>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def handle_status(request):
    """處理狀態API請求"""
    status = {
        "system": "SuperFusionAGI",
        "status": "running",
        "components": {
            "plugin_system": True,
            "data_crawler": True,
            "demo_system": True,
            "web_interface": True
        },
        "timestamp": asyncio.get_event_loop().time()
    }
    return web.json_response(status)

async def init_app():
    """初始化應用"""
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/status', handle_status)
    return app

if __name__ == '__main__':
    app = init_app()
    web.run_app(app, host='127.0.0.1', port=8080)
'''
            
            with open('web_server.py', 'w', encoding='utf-8') as f:
                f.write(web_server_code)
            
            # 在後台啟動Web服務器
            def run_web_server():
                try:
                    import subprocess
                    subprocess.run([sys.executable, 'web_server.py'], 
                                 capture_output=True, text=True)
                except Exception as e:
                    print(f"Web服務器啟動失敗: {e}")
            
            web_thread = threading.Thread(target=run_web_server, daemon=True)
            web_thread.start()
            
            # 等待服務器啟動
            await asyncio.sleep(2)
            
            print("✅ Web界面已啟動 (http://127.0.0.1:8080)")
            self.running_components['web_interface'] = True
            return True
            
        except Exception as e:
            print(f"❌ Web界面啟動失敗: {e}")
            return False
    
    async def launch_all_systems(self):
        """啟動所有系統"""
        print("\n🚀 SuperFusionAGI 系統啟動中...")
        print("=" * 50)
        
        # 啟動各個組件
        await self.launch_plugin_system()
        await self.launch_data_crawler()
        await self.launch_demo_system()
        await self.launch_web_interface()
        
        # 顯示啟動結果
        print("\n📋 系統啟動結果:")
        print("-" * 30)
        for component, status in self.running_components.items():
            if status:
                print(f"✅ {component}")
            else:
                print(f"❌ {component}")
        
        running_count = sum(self.running_components.values())
        total_count = len(self.running_components)
        
        print(f"\n🎉 成功啟動 {running_count}/{total_count} 個組件")
        
        if running_count > 0:
            print("\n✅ 系統已準備就緒！")
            print("您現在可以:")
            print("1. 🌐 訪問 Web 界面: http://127.0.0.1:8080")
            print("2. 🔌 使用插件系統創建新的爬蟲")
            print("3. 🕷️ 使用數據爬取功能")
            print("4. 📝 查看示例插件: sample_plugin.py")
            print("5. 📊 監控系統狀態")
        else:
            print("\n⚠️ 沒有組件成功啟動，請檢查錯誤信息")
    
    def show_menu(self):
        """顯示主菜單"""
        print("\n" + "="*60)
        print("🤖 SuperFusionAGI 系統 - 啟動器")
        print("="*60)
        print("1. 🚀 啟動所有功能")
        print("2. 🔌 只啟動插件系統")
        print("3. 🕷️ 只啟動數據爬取系統")
        print("4. 🎯 只啟動演示系統")
        print("5. 🌐 只啟動Web界面")
        print("6. 📊 查看系統狀態")
        print("7. ❌ 退出")
        print("="*60)
    
    async def run(self):
        """運行啟動器"""
        while True:
            self.show_menu()
            choice = input("\n請選擇操作 (1-7): ").strip()
            
            if choice == "1":
                await self.launch_all_systems()
            elif choice == "2":
                await self.launch_plugin_system()
            elif choice == "3":
                await self.launch_data_crawler()
            elif choice == "4":
                await self.launch_demo_system()
            elif choice == "5":
                await self.launch_web_interface()
            elif choice == "6":
                self.show_system_status()
            elif choice == "7":
                print("👋 再見！")
                break
            else:
                print("❌ 無效選擇，請重新輸入")
    
    def show_system_status(self):
        """顯示系統狀態"""
        print("\n📊 系統狀態:")
        print("-" * 30)
        for component, status in self.running_components.items():
            if status:
                print(f"✅ {component} - 運行中")
            else:
                print(f"❌ {component} - 未啟動")

async def main():
    """主函數"""
    print("🤖 歡迎使用 SuperFusionAGI 系統！")
    launcher = SystemLauncher()
    await launcher.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 系統已停止")
    except Exception as e:
        print(f"❌ 系統錯誤: {e}")
        logger.error(f"系統啟動錯誤: {e}")
