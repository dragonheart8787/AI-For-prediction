#!/usr/bin/env python3
"""
終極AGI系統運行腳本
運行最高級的AGI預測系統，包含最先進的訓練方法和持續指標報告
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agi_ultimate_v2 import UltimateConfig, UltimateAGISystem
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    print("請確保 agi_ultimate_v2.py 文件存在")
    sys.exit(1)

async def main():
    """主函數"""
    print("🚀 終極版AGI預測系統 V2.0")
    print("=" * 60)
    print("🌟 具備最先進的訓練方法、持續指標報告、企業級監控")
    print("=" * 60)
    
    try:
        # 創建終極配置
    config = UltimateConfig()
        print(f"⚙️ 配置已載入:")
        print(f"   - 訓練epochs: {config.training_epochs}")
        print(f"   - 批次大小: {config.training_batch_size}")
        print(f"   - 學習率: {config.learning_rate}")
        print(f"   - LSTM隱藏層: {config.lstm_hidden_layers}")
        print(f"   - Transformer層數: {config.transformer_layers}")
        print(f"   - 早停耐心: {config.early_stopping_patience}")
        print()
        
        # 創建終極AGI系統
        print("🔧 初始化終極AGI系統...")
        agi_system = UltimateAGISystem(config)
        print("✅ 系統初始化完成")
        print()
        
        # 訓練所有模型
        print("📚 開始訓練所有高級模型...")
        print("🔄 這可能需要一些時間，請耐心等待...")
        print()
        
        start_time = time.time()
        training_results = await agi_system.train_all_models()
            training_time = time.time() - start_time
            
        if training_results:
            print("✅ 所有模型訓練完成!")
            print(f"⏱️ 總訓練時間: {training_time:.2f} 秒")
            print()
            
            # 進行預測測試
            print("🔮 進行高級預測測試...")
            test_data = np.random.randn(10, 20)  # 更大的測試資料
            
            print("📊 測試LSTM模型...")
            lstm_prediction = await agi_system.make_prediction("financial_lstm", test_data)
            
            print("📊 測試Transformer模型...")
            transformer_prediction = await agi_system.make_prediction("weather_transformer", test_data)
            
            if lstm_prediction:
                print(f"🎯 LSTM預測結果: 置信度 {lstm_prediction['confidence']:.4f}")
                print(f"   - 預測值: {lstm_prediction['prediction'][:3]}...")  # 只顯示前3個
            
            if transformer_prediction:
                print(f"🎯 Transformer預測結果: 置信度 {transformer_prediction['confidence']:.4f}")
                print(f"   - 預測值: {transformer_prediction['prediction'][:3]}...")
            
            print()
            
            # 上傳到雲端
            print("☁️ 上傳模型到雲端...")
            await agi_system.upload_to_cloud("financial_lstm")
            await agi_system.upload_to_cloud("weather_transformer")
            print()
            
            # 獲取系統狀態
            print("📈 獲取系統狀態...")
            status = agi_system.get_system_status()
            print(f"   - 總模型數: {status.get('total_models', 0)}")
            print(f"   - 總預測數: {status.get('total_predictions', 0)}")
            print(f"   - 系統運行: {status.get('system_running', False)}")
            print(f"   - 持續學習: {status.get('continuous_learning_enabled', False)}")
            print()
            
            # 生成報告
            print("📊 生成終極性能報告...")
            await agi_system.generate_comprehensive_report()
            print()
            
            # 啟動持續運行
            print("🔄 啟動持續運行模式...")
            print("💡 系統將持續監控、學習和優化")
            print("🛑 按 Ctrl+C 停止系統")
            print()
            
            try:
                await agi_system.start_continuous_operation()
            except KeyboardInterrupt:
                print("\n🛑 收到停止信號")
                agi_system.stop_continuous_operation()
        
            else:
            print("❌ 模型訓練失敗")
            return 1
        
        print("✅ 終極AGI系統運行完成")
        return 0
        
    except Exception as e:
        print(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            agi_system.cleanup()
        except:
            pass

if __name__ == "__main__":
    # 檢查依賴
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
    except ImportError as e:
        print(f"❌ 缺少依賴: {e}")
        print("請安裝必要的套件:")
        print("pip install numpy matplotlib seaborn scikit-learn")
        sys.exit(1)
    
    # 運行系統
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 