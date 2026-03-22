#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合AI系統啟動腳本
提供簡單的菜單界面來運行不同的功能
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印系統橫幅"""
    print("=" * 70)
    print("🚀 整合AI系統 - 智能金融預測平台")
    print("=" * 70)
    print("🌟 功能特色:")
    print("   📊 自動爬取大量金融數據 (股票、加密貨幣、外匯、商品)")
    print("   🤖 智能AI模型訓練 (統計、機器學習、深度學習)")
    print("   🔗 模型融合與集成預測")
    print("   🎯 任務導向權重分配")
    print("   🚀 GPU加速支持")
    print("   📈 自動模型評估與選擇")
    print("=" * 70)

def print_menu():
    """打印主菜單"""
    print("\n📋 請選擇操作:")
    print("1. 🚀 運行完整工作流程 (爬取 + 訓練 + 評估)")
    print("2. 📊 只爬取數據")
    print("3. 🤖 只訓練AI模型")
    print("4. 📈 模型評估與選擇")
    print("5. 🔍 查看系統狀態")
    print("6. 📚 查看可用模型")
    print("7. ⚙️  系統配置")
    print("8. 🚪 退出")
    print("-" * 50)

def check_dependencies():
    """檢查依賴項"""
    print("🔍 檢查系統依賴項...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'statsmodels',
        'yfinance', 'aiohttp', 'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (未安裝)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依賴項: {', '.join(missing_packages)}")
        print("請運行: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依賴項檢查完成")
    return True

async def run_complete_workflow():
    """運行完整工作流程"""
    print("🚀 啟動完整工作流程...")
    
    try:
        from integrated_ai_system import IntegratedAISystem
        
        system = IntegratedAISystem()
        results = await system.run_complete_workflow()
        
        if results and 'summary' in results:
            summary = results['summary']
            print("\n🎯 工作流程完成摘要:")
            print(f"   數據爬取成功率: {summary['data_crawling']['success_rate']:.2%}")
            print(f"   模型訓練成功率: {summary['model_training']['success_rate']:.2%}")
            print(f"   系統健康狀態: {summary['system_health']}")
        
        return True
        
    except Exception as e:
        logger.error(f"完整工作流程失敗: {e}")
        print(f"❌ 工作流程失敗: {e}")
        return False

async def run_data_crawling_only():
    """只運行數據爬取"""
    print("📊 啟動數據爬取...")
    
    try:
        from enhanced_data_crawler import EnhancedDataCrawler
        
        crawler = EnhancedDataCrawler()
        results = await crawler.start_crawling()
        
        print(f"\n✅ 數據爬取完成:")
        print(f"   總計符號: {results['total']}")
        print(f"   成功: {results['successful']}")
        print(f"   失敗: {results['failed']}")
        
        return True
        
    except Exception as e:
        logger.error(f"數據爬取失敗: {e}")
        print(f"❌ 數據爬取失敗: {e}")
        return False

def run_ai_training_only():
    """只運行AI模型訓練"""
    print("🤖 啟動AI模型訓練...")
    
    try:
        from enhanced_ai_trainer import EnhancedAITrainer
        
        trainer = EnhancedAITrainer()
        
        # 獲取可用的訓練數據
        training_data = trainer.get_training_data(
            data_type="stocks",
            min_data_points=200
        )
        
        if not training_data:
            print("❌ 沒有可用的訓練數據，請先運行數據爬取")
            return False
        
        print(f"📊 找到 {len(training_data)} 個符號的數據")
        
        # 選擇前3個符號進行訓練
        selected_symbols = list(training_data.keys())[:3]
        selected_data = {symbol: training_data[symbol] for symbol in selected_symbols}
        
        print(f"🎯 選擇訓練符號: {selected_symbols}")
        
        # 開始訓練
        results = trainer.train_models_for_multiple_symbols(selected_data)
        
        print(f"\n✅ 模型訓練完成:")
        print(f"   成功訓練符號: {len(results)}")
        print(f"   總計模型數: {sum(len(symbol_results) for symbol_results in results.values())}")
        
        return True
        
    except Exception as e:
        logger.error(f"AI訓練失敗: {e}")
        print(f"❌ AI訓練失敗: {e}")
        return False

def show_system_status():
    """顯示系統狀態"""
    print("🔍 系統狀態檢查...")
    
    try:
        # 檢查數據庫
        db_path = "enhanced_financial_data.db"
        if os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 檢查數據表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"   📊 數據庫: {db_path}")
            print(f"   📋 數據表: {len(tables)} 個")
            
            # 檢查數據量
            if 'financial_data' in [table[0] for table in tables]:
                cursor.execute("SELECT COUNT(*) FROM financial_data")
                data_count = cursor.fetchone()[0]
                print(f"   📈 數據記錄: {data_count:,} 條")
            
            conn.close()
        else:
            print("   ❌ 數據庫不存在")
        
        # 檢查模型目錄
        models_path = Path("trained_models")
        if models_path.exists():
            model_count = 0
            symbol_count = 0
            
            for symbol_dir in models_path.iterdir():
                if symbol_dir.is_dir():
                    symbol_count += 1
                    for model_file in symbol_dir.glob("*.pkl"):
                        model_count += 1
            
            print(f"   🤖 訓練模型: {model_count} 個")
            print(f"   🎯 符號數量: {symbol_count} 個")
        else:
            print("   ❌ 模型目錄不存在")
        
        # 檢查結果目錄
        results_path = Path("integrated_results")
        if results_path.exists():
            result_files = list(results_path.glob("*.json"))
            print(f"   📋 結果文件: {len(result_files)} 個")
        
        print("✅ 系統狀態檢查完成")
        
    except Exception as e:
        logger.error(f"系統狀態檢查失敗: {e}")
        print(f"❌ 系統狀態檢查失敗: {e}")

def show_available_models():
    """顯示可用模型"""
    print("📚 檢查可用模型...")
    
    try:
        models_path = Path("trained_models")
        if not models_path.exists():
            print("❌ 模型目錄不存在")
            return
        
        available_models = {}
        
        for symbol_dir in models_path.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                models = []
                
                for model_file in symbol_dir.glob("*.pkl"):
                    model_name = model_file.stem
                    models.append(model_name)
                
                if models:
                    available_models[symbol] = models
        
        if available_models:
            print(f"\n✅ 找到 {len(available_models)} 個符號的模型:")
            for symbol, models in available_models.items():
                print(f"   📊 {symbol}: {', '.join(models)}")
        else:
            print("❌ 沒有找到可用的模型")
        
    except Exception as e:
        logger.error(f"檢查可用模型失敗: {e}")
        print(f"❌ 檢查可用模型失敗: {e}")

def show_system_config():
    """顯示系統配置"""
    print("⚙️  系統配置...")
    
    try:
        config_path = "integrated_config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print("📋 當前配置:")
            for section, settings in config.items():
                print(f"   {section}:")
                for key, value in settings.items():
                    print(f"     {key}: {value}")
        else:
            print("📋 使用默認配置")
            print("   工作流程: 自動爬取和訓練")
            print("   數據管理: 最小200個數據點")
            print("   模型管理: 性能評估和集成")
            print("   性能: GPU加速和並行處理")
        
        print("✅ 配置檢查完成")
        
    except Exception as e:
        logger.error(f"配置檢查失敗: {e}")
        print(f"❌ 配置檢查失敗: {e}")

async def main():
    """主函數"""
    print_banner()
    
    # 檢查依賴項
    if not check_dependencies():
        print("\n❌ 依賴項檢查失敗，請安裝缺少的包後重試")
        return
    
    print("\n✅ 系統準備就緒！")
    
    while True:
        print_menu()
        
        try:
            choice = input("請輸入選項 (1-8): ").strip()
            
            if choice == '1':
                print("\n🚀 選擇: 運行完整工作流程")
                success = await run_complete_workflow()
                if success:
                    print("✅ 完整工作流程執行成功！")
                else:
                    print("❌ 完整工作流程執行失敗")
                
            elif choice == '2':
                print("\n📊 選擇: 只爬取數據")
                success = await run_data_crawling_only()
                if success:
                    print("✅ 數據爬取成功！")
                else:
                    print("❌ 數據爬取失敗")
                
            elif choice == '3':
                print("\n🤖 選擇: 只訓練AI模型")
                success = run_ai_training_only()
                if success:
                    print("✅ AI模型訓練成功！")
                else:
                    print("❌ AI模型訓練失敗")
                
            elif choice == '4':
                print("\n📈 選擇: 模型評估與選擇")
                print("⚠️ 此功能需要先運行模型訓練")
                print("請先選擇選項3進行模型訓練")
                
            elif choice == '5':
                print("\n🔍 選擇: 查看系統狀態")
                show_system_status()
                
            elif choice == '6':
                print("\n📚 選擇: 查看可用模型")
                show_available_models()
                
            elif choice == '7':
                print("\n⚙️  選擇: 系統配置")
                show_system_config()
                
            elif choice == '8':
                print("\n👋 感謝使用整合AI系統！再見！")
                break
                
            else:
                print("❌ 請輸入有效的選項 (1-8)")
            
            if choice != '8':
                input("\n按Enter鍵繼續...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 用戶中斷操作")
            break
        except Exception as e:
            logger.error(f"菜單操作失敗: {e}")
            print(f"❌ 操作失敗: {e}")
            input("\n按Enter鍵繼續...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 系統已退出")
    except Exception as e:
        logger.error(f"系統運行失敗: {e}")
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
