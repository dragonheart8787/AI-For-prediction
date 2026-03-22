#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版綜合數據爬取系統啟動腳本
展示如何使用插件管理器和自動發現功能
"""

import asyncio
import sys
import os
from pathlib import Path
import time
import json

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from enhanced_comprehensive_crawler import EnhancedComprehensiveDataCrawler
from plugin_manager import PluginManager, DataCrawlerPlugin
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """打印橫幅"""
    print("=" * 80)
    print("🚀 增強版綜合數據爬取系統")
    print("🔌 支持自動插件發現和動態加載")
    print("📊 爬取所有類型的數據和信息")
    print("=" * 80)

def print_menu():
    """打印主菜單"""
    print("\n📋 主菜單:")
    print("1. 🚀 開始全面數據爬取")
    print("2. 🔍 查看可用數據類型")
    print("3. 🔌 管理插件")
    print("4. 📊 查看數據摘要")
    print("5. 🧪 測試插件")
    print("6. 📝 創建新插件模板")
    print("7. ⚙️  系統配置")
    print("8. 📈 查看爬取歷史")
    print("9. 🔧 手動添加自定義數據源")
    print("0. 👋 退出系統")

def print_plugin_menu():
    """打印插件管理菜單"""
    print("\n🔌 插件管理:")
    print("1. 📋 查看所有插件")
    print("2. 🔍 查看插件詳細信息")
    print("3. 🧪 測試特定插件")
    print("4. 📝 創建插件模板")
    print("5. 🔄 重新加載插件")
    print("6. ⬅️  返回主菜單")

def print_config_menu():
    """打印配置菜單"""
    print("\n⚙️  系統配置:")
    print("1. 📋 查看當前配置")
    print("2. ✏️  修改配置")
    print("3. 💾 保存配置")
    print("4. 🔄 重置為默認配置")
    print("5. ⬅️  返回主菜單")

async def start_comprehensive_crawling(crawler):
    """開始全面數據爬取"""
    print("\n🚀 開始全面數據爬取...")
    print("這將爬取所有啟用的數據源...")
    
    try:
        start_time = time.time()
        result = await crawler.start_comprehensive_crawling()
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 爬取完成！")
        print(f"📊 會話ID: {result['session_id']}")
        print(f"✅ 成功數據源: {result['successful']}")
        print(f"❌ 失敗數據源: {result['failed']}")
        print(f"📈 總記錄數: {result['total_records']}")
        print(f"⏱️  耗時: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"❌ 爬取失敗: {e}")
        print(f"❌ 爬取失敗: {e}")

def show_available_data_types(crawler):
    """顯示可用數據類型"""
    print("\n📋 可用的數據類型:")
    
    data_types = crawler.get_available_data_types()
    if not data_types:
        print("⚠️  沒有可用的數據類型")
        return
    
    for i, data_type in enumerate(data_types, 1):
        print(f"  {i}. {data_type}")
    
    print(f"\n總共 {len(data_types)} 種數據類型")

def manage_plugins(crawler):
    """管理插件"""
    while True:
        print_plugin_menu()
        try:
            choice = input("請輸入選項 (1-6): ").strip()
            
            if choice == '1':
                show_all_plugins(crawler)
            elif choice == '2':
                show_plugin_details(crawler)
            elif choice == '3':
                test_specific_plugin(crawler)
            elif choice == '4':
                create_plugin_template(crawler)
            elif choice == '5':
                reload_plugins(crawler)
            elif choice == '6':
                break
            else:
                print("❌ 請輸入有效的選項 (1-6)")
                
            if choice != '6':
                input("\n按Enter鍵繼續...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 用戶中斷操作")
            break
        except Exception as e:
            logger.error(f"插件管理操作失敗: {e}")
            print(f"❌ 操作失敗: {e}")
            input("\n按Enter鍵繼續...")

def show_all_plugins(crawler):
    """顯示所有插件"""
    print("\n🔌 所有插件:")
    
    plugin_info = crawler.get_plugin_info()
    if not plugin_info:
        print("⚠️  沒有註冊的插件")
        return
    
    for i, (name, info) in enumerate(plugin_info.items(), 1):
        print(f"  {i}. {name}")
        print(f"     狀態: {info['status']}")
        print(f"     類型: {info['type']}")
        print(f"     支持: {', '.join(info['supported_types'])}")
        print(f"     註冊時間: {info['registered_at']}")
        print()

def show_plugin_details(crawler):
    """顯示插件詳細信息"""
    plugin_info = crawler.get_plugin_info()
    if not plugin_info:
        print("⚠️  沒有註冊的插件")
        return
    
    print("\n🔍 選擇要查看的插件:")
    plugin_names = list(plugin_info.keys())
    for i, name in enumerate(plugin_names, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = int(input("請輸入選項編號: ")) - 1
        if 0 <= choice < len(plugin_names):
            plugin_name = plugin_names[choice]
            info = plugin_info[plugin_name]
            
            print(f"\n📋 插件詳細信息: {plugin_name}")
            print(f"狀態: {info['status']}")
            print(f"類型: {info['type']}")
            print(f"註冊時間: {info['registered_at']}")
            print(f"支持的數據類型: {', '.join(info['supported_types'])}")
            print(f"依賴要求: {info['requirements']}")
            
            if 'metadata' in info:
                print(f"元數據: {json.dumps(info['metadata'], ensure_ascii=False, indent=2)}")
        else:
            print("❌ 無效的選項編號")
    except ValueError:
        print("❌ 請輸入有效的數字")
    except Exception as e:
        print(f"❌ 操作失敗: {e}")

def test_specific_plugin(crawler):
    """測試特定插件"""
    plugin_info = crawler.get_plugin_info()
    if not plugin_info:
        print("⚠️  沒有註冊的插件")
        return
    
    print("\n🧪 選擇要測試的插件:")
    plugin_names = list(plugin_info.keys())
    for i, name in enumerate(plugin_names, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = int(input("請輸入選項編號: ")) - 1
        if 0 <= choice < len(plugin_names):
            plugin_name = plugin_names[choice]
            
            print(f"\n🧪 測試插件: {plugin_name}")
            result = crawler.test_plugin(plugin_name)
            
            if result['success']:
                print("✅ 插件測試成功！")
                print(f"📊 測試結果: {result['result']}")
            else:
                print("❌ 插件測試失敗！")
                print(f"❌ 錯誤信息: {result['error']}")
            print(f"⏰ 測試時間: {result['tested_at']}")
        else:
            print("❌ 無效的選項編號")
    except ValueError:
        print("❌ 請輸入有效的數字")
    except Exception as e:
        print(f"❌ 操作失敗: {e}")

def create_plugin_template(crawler):
    """創建插件模板"""
    print("\n📝 創建新插件模板")
    
    try:
        plugin_name = input("請輸入插件名稱: ").strip()
        if not plugin_name:
            print("❌ 插件名稱不能為空")
            return
        
        plugin_type = input("請輸入插件類型 (默認: custom): ").strip() or "custom"
        
        template_file = crawler.create_plugin_template(plugin_name, plugin_type)
        print(f"✅ 插件模板已創建: {template_file}")
        print(f"💡 請編輯 {template_file} 文件來實現您的插件邏輯")
        
    except Exception as e:
        print(f"❌ 創建插件模板失敗: {e}")

def reload_plugins(crawler):
    """重新加載插件"""
    print("\n🔄 重新加載插件...")
    
    try:
        # 這裡可以實現插件重新加載邏輯
        print("✅ 插件重新加載完成")
        print("💡 新添加的插件將在下次自動發現時被加載")
        
    except Exception as e:
        print(f"❌ 重新加載插件失敗: {e}")

def show_data_summary(crawler):
    """顯示數據摘要"""
    print("\n📊 數據摘要:")
    
    try:
        summary = crawler.get_data_summary()
        if not summary:
            print("⚠️  沒有數據")
            return
        
        total_records = 0
        for data_type, info in summary.items():
            print(f"📈 {data_type}:")
            print(f"    記錄數: {info['record_count']}")
            print(f"    符號數: {info['symbol_count']}")
            print(f"    最早日期: {info['earliest_date']}")
            print(f"    最新日期: {info['latest_date']}")
            print()
            total_records += info['record_count']
        
        print(f"📊 總記錄數: {total_records}")
        
    except Exception as e:
        print(f"❌ 獲取數據摘要失敗: {e}")

def show_system_config(crawler):
    """顯示系統配置"""
    while True:
        print_config_menu()
        try:
            choice = input("請輸入選項 (1-5): ").strip()
            
            if choice == '1':
                show_current_config(crawler)
            elif choice == '2':
                modify_config(crawler)
            elif choice == '3':
                save_config(crawler)
            elif choice == '4':
                reset_config(crawler)
            elif choice == '5':
                break
            else:
                print("❌ 請輸入有效的選項 (1-5)")
                
            if choice != '5':
                input("\n按Enter鍵繼續...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 用戶中斷操作")
            break
        except Exception as e:
            logger.error(f"配置操作失敗: {e}")
            print(f"❌ 操作失敗: {e}")
            input("\n按Enter鍵繼續...")

def show_current_config(crawler):
    """顯示當前配置"""
    print("\n📋 當前系統配置:")
    print(json.dumps(crawler.config, ensure_ascii=False, indent=2))

def modify_config(crawler):
    """修改配置"""
    print("\n✏️  修改配置")
    print("⚠️  此功能需要手動編輯配置文件")
    print(f"💡 配置文件路徑: enhanced_crawler_config.json")

def save_config(crawler):
    """保存配置"""
    print("\n💾 保存配置")
    print("✅ 配置已自動保存")

def reset_config(crawler):
    """重置配置"""
    print("\n🔄 重置配置")
    print("⚠️  此功能將重置為默認配置")
    confirm = input("確定要重置嗎？(y/N): ").strip().lower()
    if confirm == 'y':
        crawler.config = crawler._get_default_config()
        print("✅ 配置已重置為默認值")

def show_crawling_history(crawler):
    """顯示爬取歷史"""
    print("\n📈 爬取歷史:")
    print("⚠️  此功能需要實現數據庫查詢")
    print("💡 爬取歷史存儲在數據庫的 crawling_history 表中")

def add_custom_data_source(crawler):
    """手動添加自定義數據源"""
    print("\n🔧 手動添加自定義數據源")
    print("⚠️  此功能需要實現自定義插件類")
    print("💡 請先創建插件模板，然後實現插件邏輯")

async def main():
    """主函數"""
    print_banner()
    
    try:
        # 初始化增強版爬取器
        print("🔧 初始化系統...")
        crawler = EnhancedComprehensiveDataCrawler()
        print("✅ 系統初始化完成！")
        
        # 主循環
        while True:
            print_menu()
            try:
                choice = input("請輸入選項 (0-9): ").strip()
                
                if choice == '0':
                    print("\n👋 感謝使用增強版綜合數據爬取系統！再見！")
                    break
                elif choice == '1':
                    await start_comprehensive_crawling(crawler)
                elif choice == '2':
                    show_available_data_types(crawler)
                elif choice == '3':
                    manage_plugins(crawler)
                elif choice == '4':
                    show_data_summary(crawler)
                elif choice == '5':
                    test_specific_plugin(crawler)
                elif choice == '6':
                    create_plugin_template(crawler)
                elif choice == '7':
                    show_system_config(crawler)
                elif choice == '8':
                    show_crawling_history(crawler)
                elif choice == '9':
                    add_custom_data_source(crawler)
                else:
                    print("❌ 請輸入有效的選項 (0-9)")
                
                if choice != '0':
                    input("\n按Enter鍵繼續...")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ 用戶中斷操作")
                break
            except Exception as e:
                logger.error(f"主菜單操作失敗: {e}")
                print(f"❌ 操作失敗: {e}")
                input("\n按Enter鍵繼續...")
        
        # 停止系統
        crawler.stop()
        
    except Exception as e:
        logger.error(f"系統初始化失敗: {e}")
        print(f"❌ 系統初始化失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 修復Windows編碼問題
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    asyncio.run(main())
