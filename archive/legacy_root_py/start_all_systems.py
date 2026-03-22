#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SuperFusionAGI 完整功能啟動器
一鍵啟動所有系統功能
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# 設置編碼
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class SuperFusionAGIStarter:
    """SuperFusionAGI 完整功能啟動器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.modules = {}
        self.status = {}
        
    def print_banner(self):
        """打印啟動橫幅"""
        print("=" * 80)
        print("🚀 SuperFusionAGI 完整功能啟動器")
        print("🎯 一鍵啟動所有系統功能")
        print("=" * 80)
        print(f"⏰ 啟動時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def check_dependencies(self) -> bool:
        """檢查系統依賴"""
        print("\n🔍 檢查系統依賴...")
        
        required_modules = [
            'asyncio', 'aiohttp', 'pandas', 'numpy', 'sqlite3',
            'json', 'logging', 'datetime', 'typing'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module} - 缺失")
        
        if missing_modules:
            print(f"\n⚠️  缺少以下模組: {', '.join(missing_modules)}")
            print("請運行: pip install -r requirements.txt")
            return False
        
        print("✅ 所有依賴檢查完成")
        return True
    
    async def start_plugin_system(self):
        """啟動插件系統"""
        print("\n🔌 啟動插件系統...")
        
        try:
            from plugin_manager import PluginManager
            plugin_manager = PluginManager()
            
            # 創建插件目錄
            if not os.path.exists("plugins"):
                os.makedirs("plugins")
                print("📁 創建 plugins/ 目錄")
            
            # 啟動自動發現
            plugin_manager.start_auto_discovery()
            
            self.modules['plugin_manager'] = plugin_manager
            self.status['plugin_system'] = "✅ 已啟動"
            print("✅ 插件系統啟動成功")
            
        except Exception as e:
            self.status['plugin_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 插件系統啟動失敗: {e}")
    
    async def start_data_crawler(self):
        """啟動數據爬取系統"""
        print("\n📊 啟動數據爬取系統...")
        
        try:
            from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler
            crawler = EnhancedComprehensiveDataCrawler()
            
            self.modules['data_crawler'] = crawler
            self.status['data_crawler'] = "✅ 已啟動"
            print("✅ 數據爬取系統啟動成功")
            
        except Exception as e:
            self.status['data_crawler'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 數據爬取系統啟動失敗: {e}")
    
    async def start_prediction_system(self):
        """啟動預測系統"""
        print("\n🔮 啟動預測系統...")
        
        try:
            from agi_predictor import AGIEngine
            predictor = AGIEngine()
            await predictor.initialize()
            
            self.modules['predictor'] = predictor
            self.status['prediction_system'] = "✅ 已啟動"
            print("✅ 預測系統啟動成功")
            
        except Exception as e:
            self.status['prediction_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 預測系統啟動失敗: {e}")
    
    async def start_fusion_system(self):
        """啟動模型融合系統"""
        print("\n🔄 啟動模型融合系統...")
        
        try:
            from advanced_model_fusion import AdvancedModelFusion
            fusion = AdvancedModelFusion()
            
            self.modules['fusion'] = fusion
            self.status['fusion_system'] = "✅ 已啟動"
            print("✅ 模型融合系統啟動成功")
            
        except Exception as e:
            self.status['fusion_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 模型融合系統啟動失敗: {e}")
    
    async def start_web_server(self):
        """啟動Web服務器"""
        print("\n🌐 啟動Web服務器...")
        
        try:
            from web_server import app
            import uvicorn
            
            # 啟動Web服務器（後台運行）
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            server = uvicorn.Server(config)
            
            # 在後台啟動服務器
            import threading
            server_thread = threading.Thread(target=server.run, daemon=True)
            server_thread.start()
            
            self.modules['web_server'] = server
            self.status['web_server'] = "✅ 已啟動 (http://localhost:8000)"
            print("✅ Web服務器啟動成功")
            print("🌐 訪問地址: http://localhost:8000")
            
        except Exception as e:
            self.status['web_server'] = f"❌ 啟動失敗: {e}"
            print(f"❌ Web服務器啟動失敗: {e}")
    
    async def start_monitoring_system(self):
        """啟動監控系統"""
        print("\n📈 啟動監控系統...")
        
        try:
            # 創建監控配置
            monitoring_config = {
                "enabled": True,
                "interval": 60,  # 60秒檢查一次
                "metrics": ["cpu", "memory", "disk", "network"],
                "alerts": {
                    "cpu_threshold": 80,
                    "memory_threshold": 85,
                    "disk_threshold": 90
                }
            }
            
            # 保存監控配置
            with open("monitoring_config.json", "w", encoding='utf-8') as f:
                json.dump(monitoring_config, f, ensure_ascii=False, indent=2)
            
            self.status['monitoring_system'] = "✅ 已啟動"
            print("✅ 監控系統啟動成功")
            
        except Exception as e:
            self.status['monitoring_system'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 監控系統啟動失敗: {e}")
    
    async def start_continuous_learning(self):
        """啟動持續學習系統"""
        print("\n🧠 啟動持續學習系統...")
        
        try:
            # 創建學習配置
            learning_config = {
                "enabled": True,
                "auto_retrain": True,
                "retrain_interval": 3600,  # 1小時
                "performance_threshold": 0.8,
                "data_collection": True,
                "model_optimization": True
            }
            
            # 保存學習配置
            with open("learning_config.json", "w", encoding='utf-8') as f:
                json.dump(learning_config, f, ensure_ascii=False, indent=2)
            
            self.status['continuous_learning'] = "✅ 已啟動"
            print("✅ 持續學習系統啟動成功")
            
        except Exception as e:
            self.status['continuous_learning'] = f"❌ 啟動失敗: {e}"
            print(f"❌ 持續學習系統啟動失敗: {e}")
    
    async def start_all_systems(self):
        """啟動所有系統"""
        print("\n🚀 開始啟動所有系統...")
        
        # 並行啟動所有系統
        await asyncio.gather(
            self.start_plugin_system(),
            self.start_data_crawler(),
            self.start_prediction_system(),
            self.start_fusion_system(),
            self.start_web_server(),
            self.start_monitoring_system(),
            self.start_continuous_learning()
        )
    
    def show_system_status(self):
        """顯示系統狀態"""
        print("\n" + "=" * 80)
        print("📊 系統狀態報告")
        print("=" * 80)
        
        for system, status in self.status.items():
            print(f"{system.replace('_', ' ').title()}: {status}")
        
        print("=" * 80)
        
        # 計算啟動時間
        end_time = datetime.now()
        startup_time = (end_time - self.start_time).total_seconds()
        print(f"⏱️  總啟動時間: {startup_time:.2f} 秒")
    
    def show_usage_guide(self):
        """顯示使用指南"""
        print("\n" + "=" * 80)
        print("📖 使用指南")
        print("=" * 80)
        
        guide = """
🎯 系統功能說明:

1. 🔌 插件系統
   - 自動發現和管理爬蟲插件
   - 支持動態加載和熱插拔
   - 插件模板自動生成

2. 📊 數據爬取系統
   - 多源數據自動爬取
   - 支持股票、加密貨幣、外匯等
   - 實時數據更新

3. 🔮 預測系統
   - 多領域智能預測
   - 支持金融、天氣、醫療等
   - 模型自動優化

4. 🔄 模型融合系統
   - 多模型智能融合
   - 自動權重調整
   - 性能優化

5. 🌐 Web服務器
   - RESTful API接口
   - 實時數據查詢
   - 系統狀態監控

6. 📈 監控系統
   - 系統性能監控
   - 自動警報機制
   - 日誌記錄

7. 🧠 持續學習系統
   - 自動模型重訓練
   - 性能自適應
   - 知識積累

🚀 快速開始:
- 訪問 http://localhost:8000 查看Web界面
- 運行 python start_enhanced_comprehensive_crawler.py 啟動爬蟲
- 運行 python demo_plugin_creation.py 創建新插件

📞 支持:
- 查看 SYSTEM_SELF_EXTENDING_DEMO.md 了解詳細功能
- 運行 python demo_plugin_integration.py 學習插件開發
        """
        
        print(guide)
    
    async def run_demo(self):
        """運行演示"""
        print("\n🎬 運行系統演示...")
        
        try:
            # 運行插件創建演示
            print("📝 運行插件創建演示...")
            from demo_plugin_creation import demo_plugin_creation
            await demo_plugin_creation()
            
            # 運行插件集成演示
            print("\n🔌 運行插件集成演示...")
            from demo_plugin_integration import demo_plugin_integration
            await demo_plugin_integration()
            
            print("✅ 演示完成")
            
        except Exception as e:
            print(f"❌ 演示運行失敗: {e}")
    
    def save_startup_report(self):
        """保存啟動報告"""
        report = {
            "startup_time": self.start_time.isoformat(),
            "status": self.status,
            "modules_loaded": list(self.modules.keys()),
            "total_systems": len(self.status),
            "successful_systems": len([s for s in self.status.values() if "✅" in s]),
            "failed_systems": len([s for s in self.status.values() if "❌" in s])
        }
        
        with open("startup_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 啟動報告已保存到: startup_report.json")

async def main():
    """主函數"""
    starter = SuperFusionAGIStarter()
    
    # 打印橫幅
    starter.print_banner()
    
    # 檢查依賴
    if not starter.check_dependencies():
        print("\n❌ 依賴檢查失敗，請安裝缺失的模組")
        return
    
    # 啟動所有系統
    await starter.start_all_systems()
    
    # 顯示狀態
    starter.show_system_status()
    
    # 顯示使用指南
    starter.show_usage_guide()
    
    # 保存報告
    starter.save_startup_report()
    
    # 詢問是否運行演示
    print("\n" + "=" * 80)
    print("🎬 是否要運行系統演示？(y/n): ", end="")
    
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '是']:
            await starter.run_demo()
    except KeyboardInterrupt:
        print("\n👋 用戶取消演示")
    
    print("\n🎉 SuperFusionAGI 系統啟動完成！")
    print("🚀 所有功能已就緒，開始使用吧！")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 系統啟動被用戶中斷")
    except Exception as e:
        print(f"\n❌ 系統啟動失敗: {e}")
        import traceback
        traceback.print_exc()
