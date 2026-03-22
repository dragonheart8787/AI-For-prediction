#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面數據爬取器啟動腳本
提供用戶友好的界面來使用全面數據爬取器
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_data_crawler import ComprehensiveDataCrawler, DataCrawlerPlugin

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_crawler_ui.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印歡迎橫幅"""
    print("=" * 80)
    print("🚀 全面數據爬取器 - Comprehensive Data Crawler")
    print("=" * 80)
    print("🌟 功能特色:")
    print("   • 支持25+種數據類型爬取")
    print("   • 插件化架構，可動態擴展")
    print("   • 智能重試和錯誤處理")
    print("   • 並行爬取，高效快速")
    print("   • 數據持久化存儲")
    print("   • 可自定義數據源配置")
    print("=" * 80)

def print_menu():
    """打印主菜單"""
    print("\n📋 主菜單:")
    print("1. 🚀 開始全面數據爬取")
    print("2. 📊 查看可用數據類型")
    print("3. 📈 查看數據摘要")
    print("4. ⚙️  管理數據源配置")
    print("5. 🔌 管理插件")
    print("6. 📁 查看數據文件")
    print("7. 🔍 搜索特定數據")
    print("8. 📋 查看爬取歷史")
    print("9. 🆕 添加自定義數據源")
    print("0. 👋 退出")
    print("-" * 50)

def print_data_types_menu():
    """打印數據類型菜單"""
    print("\n📊 數據類型菜單:")
    print("1. 📈 股票數據 (Stocks)")
    print("2. 🪙 加密貨幣 (Cryptocurrency)")
    print("3. 💱 外匯數據 (Forex)")
    print("4. 🛢️  商品數據 (Commodities)")
    print("5. 📊 市場指數 (Market Indices)")
    print("6. 📊 經濟指標 (Economic Indicators)")
    print("7. 📰 新聞情感 (News Sentiment)")
    print("8. 📱 社交媒體 (Social Media)")
    print("9. 🌤️ 天氣數據 (Weather Data)")
    print("10. 🌍 地緣政治事件 (Geopolitical Events)")
    print("11. 🏦 央行公告 (Central Bank Announcements)")
    print("12. 📅 財報日曆 (Earnings Calendar)")
    print("13. 👥 內幕交易 (Insider Trading)")
    print("14. 📊 期權流動 (Options Flow)")
    print("15. 📉 空頭利息 (Short Interest)")
    print("16. 🏢 機構持股 (Institutional Holdings)")
    print("17. 🚚 供應鏈數據 (Supply Chain Data)")
    print("18. ⚡ 能源市場 (Energy Markets)")
    print("19. 🏠 房地產數據 (Real Estate)")
    print("0. 🔙 返回主菜單")

def print_config_menu():
    """打印配置管理菜單"""
    print("\n⚙️  配置管理菜單:")
    print("1. 📋 查看當前配置")
    print("2. ✏️  編輯配置")
    print("3. 🔄 重新加載配置")
    print("4. 💾 保存配置")
    print("5. 🔙 重置為默認配置")
    print("0. 🔙 返回主菜單")

def print_plugin_menu():
    """打印插件管理菜單"""
    print("\n🔌 插件管理菜單:")
    print("1. 📋 查看已安裝插件")
    print("2. ✅ 啟用/禁用插件")
    print("3. 📥 安裝新插件")
    print("4. 🗑️  卸載插件")
    print("5. 🔍 查看插件詳情")
    print("0. 🔙 返回主菜單")

async def start_comprehensive_crawling(crawler):
    """開始全面數據爬取"""
    print("\n🚀 開始全面數據爬取...")
    print("⏳ 這可能需要幾分鐘時間，請耐心等待...")
    
    try:
        start_time = datetime.now()
        results = await crawler.start_comprehensive_crawling()
        end_time = datetime.now()
        
        print(f"\n✅ 爬取完成！耗時: {end_time - start_time}")
        print(f"📊 爬取結果:")
        print(f"   總計數據源: {results['total_sources']}")
        print(f"   成功: {results['successful_sources']}")
        print(f"   失敗: {results['failed_sources']}")
        
        if results['failed_sources'] > 0:
            print(f"\n⚠️  失敗的數據源:")
            for i, result in enumerate(results['results']):
                if isinstance(result, Exception) or not result.get('success', False):
                    print(f"   - {result}")
        
        return results
        
    except Exception as e:
        logger.error(f"爬取過程失敗: {e}")
        print(f"❌ 爬取失敗: {e}")
        return None

def show_available_data_types(crawler):
    """顯示可用數據類型"""
    print("\n📊 可用數據類型:")
    available_types = crawler.get_available_data_types()
    
    if not available_types:
        print("   ❌ 沒有可用的數據類型")
        return
    
    for i, data_type in enumerate(available_types, 1):
        print(f"   {i:2d}. {data_type}")
    
    print(f"\n總計: {len(available_types)} 種數據類型")

def show_data_summary(crawler):
    """顯示數據摘要"""
    print("\n📈 數據摘要:")
    data_summary = crawler.get_data_summary()
    
    if not data_summary:
        print("   ❌ 沒有數據")
        return
    
    total_records = 0
    total_symbols = 0
    
    for data_type, stats in data_summary.items():
        record_count = stats.get('record_count', 0)
        symbol_count = stats.get('symbol_count', 0)
        total_records += record_count
        total_symbols += symbol_count
        
        print(f"   📊 {data_type}:")
        print(f"      記錄數: {record_count:,}")
        print(f"      符號數: {symbol_count}")
        if stats.get('earliest_date'):
            print(f"      最早日期: {stats['earliest_date']}")
        if stats.get('latest_date'):
            print(f"      最新日期: {stats['latest_date']}")
        print()
    
    print(f"📊 總計:")
    print(f"   記錄數: {total_records:,}")
    print(f"   符號數: {total_symbols}")

def show_current_config(crawler):
    """顯示當前配置"""
    print("\n⚙️  當前配置:")
    config = crawler.config
    
    print(f"📁 數據庫: {config['storage']['database']}")
    print(f"🔄 最大並發: {config['crawling']['max_concurrent']}")
    print(f"⏱️  請求間隔: {config['crawling']['rate_limit']} 秒")
    print(f"🔄 重試次數: {config['crawling']['retry_attempts']}")
    print(f"⏰ 超時時間: {config['crawling']['timeout']} 秒")
    
    print(f"\n📊 數據源配置:")
    for source_name, source_config in config['data_sources'].items():
        enabled = "✅" if source_config.get('enabled', False) else "❌"
        priority = source_config.get('priority', 999)
        print(f"   {enabled} {source_name} (優先級: {priority})")

def show_installed_plugins(crawler):
    """顯示已安裝插件"""
    print("\n🔌 已安裝插件:")
    plugins = crawler.plugins
    
    if not plugins:
        print("   ❌ 沒有已安裝的插件")
        return
    
    for plugin_name, plugin in plugins.items():
        supported_types = plugin.get_supported_types()
        requirements = plugin.get_requirements()
        
        print(f"   🔌 {plugin_name}:")
        print(f"      支持類型: {', '.join(supported_types)}")
        print(f"      依賴要求: {requirements}")
        print()

def show_data_files():
    """顯示數據文件"""
    print("\n📁 數據文件:")
    
    # 檢查數據庫文件
    db_files = [
        "comprehensive_financial_data.db",
        "enhanced_financial_data.db"
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            size_mb = size / (1024 * 1024)
            print(f"   💾 {db_file}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {db_file}: 不存在")
    
    # 檢查日誌文件
    log_files = [
        "comprehensive_crawler.log",
        "comprehensive_crawler_ui.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            size_kb = size / 1024
            print(f"   📝 {log_file}: {size_kb:.2f} KB")
        else:
            print(f"   ❌ {log_file}: 不存在")

def search_specific_data(crawler):
    """搜索特定數據"""
    print("\n🔍 搜索特定數據:")
    print("⚠️  此功能待實現")
    print("   您可以通過以下方式搜索:")
    print("   1. 直接查詢數據庫")
    print("   2. 使用 get_data_for_training 方法")
    print("   3. 查看數據摘要了解可用數據")

def show_crawling_history():
    """顯示爬取歷史"""
    print("\n📋 爬取歷史:")
    print("⚠️  此功能待實現")
    print("   您可以通過以下方式查看歷史:")
    print("   1. 查看日誌文件")
    print("   2. 查看數據庫中的時間戳")
    print("   3. 查看數據源狀態表")

def add_custom_data_source(crawler):
    """添加自定義數據源"""
    print("\n🆕 添加自定義數據源:")
    print("⚠️  此功能需要編程實現")
    print("   您可以通過以下方式添加:")
    print("   1. 創建新的插件類")
    print("   2. 繼承 DataCrawlerPlugin")
    print("   3. 實現 crawl 方法")
    print("   4. 使用 add_custom_data_source 方法")

async def main():
    """主函數"""
    print_banner()
    
    try:
        # 初始化爬取器
        print("🔧 正在初始化全面數據爬取器...")
        crawler = ComprehensiveDataCrawler()
        print("✅ 初始化完成！")
        
        while True:
            print_menu()
            try:
                choice = input("請輸入選項 (0-9): ").strip()
                
                if choice == '0':
                    print("\n👋 感謝使用全面數據爬取器！再見！")
                    break
                    
                elif choice == '1':
                    await start_comprehensive_crawling(crawler)
                    
                elif choice == '2':
                    show_available_data_types(crawler)
                    
                elif choice == '3':
                    show_data_summary(crawler)
                    
                elif choice == '4':
                    while True:
                        print_config_menu()
                        config_choice = input("請輸入選項 (0-5): ").strip()
                        
                        if config_choice == '0':
                            break
                        elif config_choice == '1':
                            show_current_config(crawler)
                        elif config_choice == '2':
                            print("✏️  編輯配置功能待實現")
                        elif config_choice == '3':
                            print("🔄 重新加載配置功能待實現")
                        elif config_choice == '4':
                            print("💾 保存配置功能待實現")
                        elif config_choice == '5':
                            print("🔄 重置配置功能待實現")
                        else:
                            print("❌ 請輸入有效的選項")
                        
                        if config_choice != '0':
                            input("\n按Enter鍵繼續...")
                    
                elif choice == '5':
                    while True:
                        print_plugin_menu()
                        plugin_choice = input("請輸入選項 (0-5): ").strip()
                        
                        if plugin_choice == '0':
                            break
                        elif plugin_choice == '1':
                            show_installed_plugins(crawler)
                        elif plugin_choice == '2':
                            print("✅ 啟用/禁用插件功能待實現")
                        elif plugin_choice == '3':
                            print("📥 安裝新插件功能待實現")
                        elif plugin_choice == '4':
                            print("🗑️  卸載插件功能待實現")
                        elif plugin_choice == '5':
                            print("🔍 查看插件詳情功能待實現")
                        else:
                            print("❌ 請輸入有效的選項")
                        
                        if plugin_choice != '0':
                            input("\n按Enter鍵繼續...")
                    
                elif choice == '6':
                    show_data_files()
                    
                elif choice == '7':
                    search_specific_data(crawler)
                    
                elif choice == '8':
                    show_crawling_history()
                    
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
                logger.error(f"菜單操作失敗: {e}")
                print(f"❌ 操作失敗: {e}")
                input("\n按Enter鍵繼續...")
                
    except Exception as e:
        logger.error(f"系統初始化失敗: {e}")
        print(f"❌ 系統初始化失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        logger.error(f"程序執行失敗: {e}")
        print(f"❌ 程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
