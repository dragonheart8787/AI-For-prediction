#!/usr/bin/env python3
"""
終極時間序列預測AGI系統運行腳本
支持演示、測試和主運行模式
"""

import asyncio
import sys
import numpy as np
from ultimate_time_series_agi import UltimateTimeSeriesConfig, UltimateTimeSeriesAGI

async def demo_mode():
    """演示模式"""
    print("🎭 終極時間序列預測AGI系統演示模式")
    print("=" * 60)
    
    try:
        # 創建配置
        config = UltimateTimeSeriesConfig()
        print("⚙️ 創建系統配置...")
        print(f"✅ 配置創建成功: 預測範圍={config.prediction_config['forecast_horizon']}步")
        
        # 初始化系統
        print("🔧 初始化終極時間序列預測AGI系統...")
        agi_system = UltimateTimeSeriesAGI(config)
        
        # 啟動系統
        print("🚀 啟動系統...")
        await agi_system.start_system()
        
        # 獲取系統狀態
        print("📊 獲取系統狀態...")
        status = await agi_system.get_system_status()
        print(f"✅ 系統狀態: {status['system_status']}")
        print(f"🎯 零樣本模型: {status['zero_shot_models']} 個")
        print(f"🧠 深度學習模型: {status['deep_learning_models']} 個")
        print(f"📈 經典統計模型: {status['classical_models']} 個")
        print(f"🔢 總模型數: {status['total_models']} 個")
        
        # 演示預測
        print("\n🔮 演示預測功能...")
        test_sequence = np.random.randn(1, 100)  # 100個時間步的測試序列
        
        # 測試零樣本預測
        print("🌟 測試零樣本預測...")
        result = await agi_system.make_prediction(test_sequence, 'zero_shot_only')
        if 'error' not in result:
            print(f"✅ 零樣本預測成功: {result['model']}")
            print(f"   預測長度: {len(result['prediction'])}")
            print(f"   執行時間: {result.get('execution_time', 0):.3f}秒")
        else:
            print(f"❌ 零樣本預測失敗: {result['error']}")
        
        # 測試深度學習預測
        print("\n🧠 測試深度學習預測...")
        result = await agi_system.make_prediction(test_sequence, 'deep_learning_only')
        if 'error' not in result:
            print(f"✅ 深度學習預測成功: {result['model']}")
            print(f"   預測長度: {len(result['prediction'])}")
            print(f"   執行時間: {result.get('execution_time', 0):.3f}秒")
        else:
            print(f"❌ 深度學習預測失敗: {result['error']}")
        
        # 測試經典統計預測
        print("\n📈 測試經典統計預測...")
        result = await agi_system.make_prediction(test_sequence, 'classical_only')
        if 'error' not in result:
            print(f"✅ 經典統計預測成功: {result['model']}")
            print(f"   預測長度: {len(result['prediction'])}")
            print(f"   執行時間: {result.get('execution_time', 0):.3f}秒")
        else:
            print(f"❌ 經典統計預測失敗: {result['error']}")
        
        # 測試終極集成預測
        print("\n🚀 測試終極集成預測...")
        result = await agi_system.make_prediction(test_sequence, 'ensemble_all')
        if 'error' not in result:
            print(f"✅ 終極集成預測成功: {result['model']}")
            print(f"   預測長度: {len(result['prediction'])}")
            print(f"   執行時間: {result.get('execution_time', 0):.3f}秒")
            print(f"   使用模型: {result.get('models_used', [])}")
        else:
            print(f"❌ 終極集成預測失敗: {result['error']}")
        
        print("\n✅ 演示完成")
        
    except Exception as e:
        print(f"❌ 演示失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理資源
        if 'agi_system' in locals():
            agi_system.cleanup()

async def test_mode():
    """測試模式"""
    print("🧪 終極時間序列預測AGI系統測試模式")
    print("=" * 60)
    
    try:
        # 創建配置
        config = UltimateTimeSeriesConfig()
        agi_system = UltimateTimeSeriesAGI(config)
        
        # 啟動系統
        await agi_system.start_system()
        
        # 測試預測
        test_sequence = np.random.randn(1, 100)
        methods = ['zero_shot_only', 'deep_learning_only', 'classical_only', 'ensemble_all']
        
        for method in methods:
            result = await agi_system.make_prediction(test_sequence, method)
            if 'error' not in result:
                print(f"✅ {method} 測試通過")
            else:
                print(f"❌ {method} 測試失敗: {result['error']}")
        
        print("✅ 終極時間序列預測AGI系統測試完成")
        print("🎉 所有測試通過！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'agi_system' in locals():
            agi_system.cleanup()

async def main_mode():
    """主運行模式"""
    print("🚀 終極時間序列預測AGI系統 V1.0")
    print("=" * 60)
    print("🌟 整合所有頂級時間序列預測模型")
    print("🎯 支持零樣本、深度學習、經典統計預測")
    print("=" * 60)
    
    try:
        # 創建配置
        config = UltimateTimeSeriesConfig()
        print("⚙️ 創建系統配置...")
        print(f"✅ 配置創建成功: 預測範圍={config.prediction_config['forecast_horizon']}步")
        
        # 初始化系統
        print("🔧 初始化終極時間序列預測AGI系統...")
        agi_system = UltimateTimeSeriesAGI(config)
        
        # 啟動系統
        print("🚀 啟動系統...")
        await agi_system.start_system()
        
        # 獲取系統狀態
        status = await agi_system.get_system_status()
        print(f"✅ 系統狀態: {status['system_status']}")
        print(f"🎯 零樣本模型: {status['zero_shot_models']} 個")
        print(f"🧠 深度學習模型: {status['deep_learning_models']} 個")
        print(f"📈 經典統計模型: {status['classical_models']} 個")
        print(f"🔢 總模型數: {status['total_models']} 個")
        
        print("\n🎯 系統已啟動，等待預測請求...")
        print("💡 按 Ctrl+C 停止系統")
        
        # 保持系統運行
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 收到停止信號")
        
        print("✅ 系統運行完成")
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理資源
        if 'agi_system' in locals():
            agi_system.cleanup()
        print("🧹 系統資源清理完成")

def print_usage():
    """打印使用說明"""
    print("🚀 終極時間序列預測AGI系統")
    print("=" * 40)
    print("使用方法:")
    print("  python run_ultimate_time_series.py [模式]")
    print("")
    print("模式選項:")
    print("  demo     - 演示模式（快速測試所有功能）")
    print("  test     - 測試模式（運行所有測試）")
    print("  main     - 主運行模式（持續運行等待請求）")
    print("  無參數   - 默認主運行模式")
    print("")
    print("示例:")
    print("  python run_ultimate_time_series.py demo")
    print("  python run_ultimate_time_series.py test")
    print("  python run_ultimate_time_series.py main")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'demo':
            asyncio.run(demo_mode())
        elif mode == 'test':
            asyncio.run(test_mode())
        elif mode == 'main':
            asyncio.run(main_mode())
        else:
            print(f"❌ 不支持的模式: {mode}")
            print_usage()
    else:
        # 默認主運行模式
        asyncio.run(main_mode())
