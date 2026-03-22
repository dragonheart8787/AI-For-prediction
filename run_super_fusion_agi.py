#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超級融合AGI系統運行腳本
整合所有現有預測模型，持續生成報告，自動儲存數據
"""

import asyncio
import sys
import os
import json
import numpy as np
from datetime import datetime

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from super_fusion_agi import SuperFusionAGI, SuperFusionConfig

async def main():
    """主函數"""
    print("🚀 超級融合AGI系統 V1.0")
    print("=" * 60)
    print("🌟 整合所有預測模型，持續生成報告，自動儲存數據")
    print("=" * 60)
    
    try:
        # 創建配置
        print("⚙️ 創建系統配置...")
        config = SuperFusionConfig()
        print(f"✅ 配置創建成功: 模型路徑={config.models_path}")
        
        # 創建超級融合AGI系統
        print("🔧 初始化超級融合AGI系統...")
        agi_system = SuperFusionAGI(config)
        print("✅ 系統初始化完成")
        
        # 獲取系統狀態
        print("📊 獲取系統狀態...")
        status = await agi_system.get_system_status()
        print(f"📈 系統狀態: {json.dumps(status, ensure_ascii=False, indent=2)}")
        
        # 進行測試預測
        print("\n🔮 進行測試預測...")
        test_input = np.random.randn(1, 10)  # 10維輸入數據
        print(f"📥 測試輸入: {test_input.shape}")
        
        prediction_result = await agi_system.make_prediction(test_input)
        print(f"📊 預測結果: {json.dumps(prediction_result, ensure_ascii=False, indent=2)}")
        
        # 生成所有報告
        print("\n📊 生成系統報告...")
        await agi_system.generate_all_reports()
        print("✅ 所有報告生成完成")
        
        # 獲取最終狀態
        print("\n📈 獲取最終系統狀態...")
        final_status = await agi_system.get_system_status()
        print(f"🎯 最終狀態: {json.dumps(final_status, ensure_ascii=False, indent=2)}")
        
        # 啟動系統（持續運行）
        print("\n🚀 啟動系統持續運行...")
        print("💡 系統將持續生成報告和儲存數據")
        print("💡 按 Ctrl+C 停止系統")
        
        try:
            await agi_system.start_system()
        except KeyboardInterrupt:
            print("\n🛑 收到停止信號，正在關閉系統...")
            await agi_system.stop_system()
        
        print("✅ 系統運行完成")
        return 0
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        try:
            if 'agi_system' in locals():
                agi_system.cleanup()
                print("🧹 系統資源清理完成")
        except Exception as e:
            print(f"⚠️ 資源清理時發生錯誤: {e}")

async def demo_mode():
    """演示模式"""
    print("🎭 超級融合AGI系統演示模式")
    print("=" * 60)
    
    try:
        config = SuperFusionConfig()
        agi_system = SuperFusionAGI(config)
        
        # 快速演示
        print("🔮 快速預測演示...")
        for i in range(3):
            test_input = np.random.randn(1, 10)
            result = await agi_system.make_prediction(test_input)
            print(f"預測 {i+1}: {result['fusion_prediction']:.4f} (置信度: {result['confidence']:.4f})")
        
        # 生成報告
        print("📊 生成演示報告...")
        await agi_system.generate_all_reports()
        
        # 獲取狀態
        status = await agi_system.get_system_status()
        print(f"📈 系統狀態: {status['models_loaded']} 個模型已載入")
        
        agi_system.cleanup()
        print("✅ 演示完成")
        
    except Exception as e:
        print(f"❌ 演示失敗: {e}")

async def test_mode():
    """測試模式"""
    print("🧪 超級融合AGI系統測試模式")
    print("=" * 60)
    
    try:
        from super_fusion_agi import test_super_fusion_agi
        success = await test_super_fusion_agi()
        
        if success:
            print("🎉 所有測試通過！")
        else:
            print("💥 測試失敗！")
            
    except Exception as e:
        print(f"❌ 測試模式失敗: {e}")

if __name__ == "__main__":
    # 檢查命令行參數
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "demo":
            exit_code = asyncio.run(demo_mode())
        elif mode == "test":
            exit_code = asyncio.run(test_mode())
        else:
            print(f"❌ 未知模式: {mode}")
            print("可用模式: demo, test")
            exit_code = 1
    else:
        # 默認運行完整模式
        exit_code = asyncio.run(main())
    
    sys.exit(exit_code)
