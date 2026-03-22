#!/usr/bin/env python3
"""
持久化AGI系統運行器 - Persistent AGI System Runner
提供完整的持久化AGI預測功能
"""

import asyncio
import argparse
import sys
import os
import numpy as np
from agi_persistent import PersistentConfig, PersistentAGISystem

async def run_full_demo():
    """運行完整演示"""
    print("持久化AGI預測系統 - 完整演示")
    print("=" * 60)
    
    # 創建配置
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        # 1. 訓練所有模型
        print("📚 步驟 1: 訓練所有模型")
        print("-" * 40)
        training_results = await agi_system.train_all_models()
        
        if not training_results:
            print("❌ 模型訓練失敗")
            return
        
        print("模型訓練完成")
        if 'lstm_result' in training_results:
            print(f"   - LSTM模型: {training_results['lstm_result']['final_accuracy']:.4f}")
        if 'transformer_result' in training_results:
            print(f"   - Transformer模型: {training_results['transformer_result']['final_accuracy']:.4f}")
        
        # 2. 進行預測測試
        print("\n步驟 2: 進行預測測試")
        print("-" * 40)
        
        # 生成測試資料
        test_data = np.random.randn(10, 10)
        print(f"測試資料形狀: {test_data.shape}")
        
        # LSTM預測
        lstm_prediction = await agi_system.make_prediction("financial_lstm", test_data)
        if lstm_prediction:
            print(f"📊 LSTM預測結果: {lstm_prediction['prediction'][:3]}...")
            print(f"🎯 置信度: {lstm_prediction['confidence']:.3f}")
        
        # Transformer預測
        transformer_prediction = await agi_system.make_prediction("weather_transformer", test_data)
        if transformer_prediction:
            print(f"📊 Transformer預測結果: {transformer_prediction['prediction'][:3]}...")
            print(f"🎯 置信度: {transformer_prediction['confidence']:.3f}")
        
        # 3. 雲端儲存測試
        print("\n步驟 3: 雲端儲存測試")
        print("-" * 40)
        
        # 上傳模型到雲端
        print("上傳LSTM模型到雲端...")
        lstm_upload = await agi_system.upload_to_cloud("financial_lstm")
        print(f"LSTM上傳結果: {'✅ 成功' if lstm_upload else '❌ 失敗'}")
        
        print("上傳Transformer模型到雲端...")
        transformer_upload = await agi_system.upload_to_cloud("weather_transformer")
        print(f"Transformer上傳結果: {'✅ 成功' if transformer_upload else '❌ 失敗'}")
        
        # 4. 系統狀態檢查
        print("\n步驟 4: 系統狀態檢查")
        print("-" * 40)
        
        status = agi_system.get_system_status()
        print(f"系統運行狀態: {'🟢 運行中' if status.get('system_running', False) else '🔴 已停止'}")
        print(f"總模型數量: {status.get('total_models', 0)}")
        print(f"總預測數量: {status.get('total_predictions', 0)}")
        print(f"持續學習: {'✅ 啟用' if status.get('continuous_learning_enabled', False) else '❌ 停用'}")
        print(f"本地儲存路徑: {status.get('storage_path', 'N/A')}")
        print(f"雲端儲存: {'✅ 啟用' if status.get('cloud_enabled', False) else '❌ 停用'}")
        
        # 5. 持續運行演示
        print("\n步驟 5: 持續運行演示")
        print("-" * 40)
        print("啟動持續運行模式 (按 Ctrl+C 停止)...")
        
        try:
            await agi_system.start_continuous_operation()
        except KeyboardInterrupt:
            print("\n收到停止信號")
            agi_system.stop_continuous_operation()
        
        print("完整演示完成")
        
    except Exception as e:
        print(f"演示過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

async def run_prediction_demo():
    """運行預測演示"""
    print("🔮 持久化AGI預測系統 - 預測演示")
    print("=" * 50)
    
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        # 檢查是否有已訓練的模型
        status = agi_system.get_system_status()
        if status.get('total_models', 0) == 0:
            print("⚠️ 沒有找到已訓練的模型，先進行訓練...")
            await agi_system.train_all_models()
        
        # 進行多個預測
        print("🔮 進行多個預測測試")
        print("-" * 30)
        
        for i in range(5):
            print(f"\n預測 #{i+1}:")
            
            # 生成隨機測試資料
            test_data = np.random.randn(5, 10)
            
            # LSTM預測
            lstm_result = await agi_system.make_prediction("financial_lstm", test_data)
            if lstm_result:
                print(f"  📊 LSTM: {lstm_result['prediction'][0]:.4f} (置信度: {lstm_result['confidence']:.3f})")
            
            # Transformer預測
            transformer_result = await agi_system.make_prediction("weather_transformer", test_data)
            if transformer_result:
                print(f"  📊 Transformer: {transformer_result['prediction'][0]:.4f} (置信度: {transformer_result['confidence']:.3f})")
        
        # 顯示最終統計
        final_status = agi_system.get_system_status()
        print(f"\n📈 預測統計:")
        print(f"   - 總預測數: {final_status.get('total_predictions', 0)}")
        print(f"   - 模型性能: {final_status.get('model_performance', {})}")
        
        print("✅ 預測演示完成")
        
    except Exception as e:
        print(f"❌ 預測演示失敗: {e}")

async def run_training_demo():
    """運行訓練演示"""
    print("📚 持久化AGI預測系統 - 訓練演示")
    print("=" * 50)
    
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        print("🚀 開始訓練所有模型...")
        print("-" * 30)
        
        # 訓練所有模型
        results = await agi_system.train_all_models()
        
        if results:
            print("✅ 訓練完成")
            print(f"   - LSTM最終準確率: {results['lstm_result']['final_accuracy']:.4f}")
            print(f"   - LSTM最終損失: {results['lstm_result']['final_loss']:.4f}")
            print(f"   - Transformer最終準確率: {results['transformer_result']['final_accuracy']:.4f}")
            print(f"   - Transformer最終損失: {results['transformer_result']['final_loss']:.4f}")
            
            # 顯示訓練歷史
            print("\n📊 訓練歷史:")
            lstm_history = results['lstm_result']['training_history']
            transformer_history = results['transformer_result']['training_history']
            
            print(f"   - LSTM訓練輪數: {len(lstm_history['losses'])}")
            print(f"   - Transformer訓練輪數: {len(transformer_history['losses'])}")
            
            # 顯示最佳性能
            lstm_best_acc = max(lstm_history['accuracies'])
            transformer_best_acc = max(transformer_history['accuracies'])
            print(f"   - LSTM最佳準確率: {lstm_best_acc:.4f}")
            print(f"   - Transformer最佳準確率: {transformer_best_acc:.4f}")
        
        # 檢查儲存狀態
        status = agi_system.get_system_status()
        print(f"\n💾 儲存狀態:")
        print(f"   - 總模型數: {status.get('total_models', 0)}")
        print(f"   - 本地儲存路徑: {status.get('storage_path', 'N/A')}")
        
        print("✅ 訓練演示完成")
        
    except Exception as e:
        print(f"❌ 訓練演示失敗: {e}")

async def run_cloud_demo():
    """運行雲端儲存演示"""
    print("☁️ 持久化AGI預測系統 - 雲端儲存演示")
    print("=" * 50)
    
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        # 確保有模型可以上傳
        status = agi_system.get_system_status()
        if status.get('total_models', 0) == 0:
            print("⚠️ 沒有找到模型，先進行訓練...")
            await agi_system.train_all_models()
        
        print("☁️ 上傳模型到雲端...")
        print("-" * 30)
        
        # 上傳LSTM模型
        print("上傳 financial_lstm 模型...")
        lstm_upload = await agi_system.upload_to_cloud("financial_lstm")
        print(f"結果: {'✅ 成功' if lstm_upload else '❌ 失敗'}")
        
        # 上傳Transformer模型
        print("上傳 weather_transformer 模型...")
        transformer_upload = await agi_system.upload_to_cloud("weather_transformer")
        print(f"結果: {'✅ 成功' if transformer_upload else '❌ 失敗'}")
        
        print("\n📥 從雲端下載模型...")
        print("-" * 30)
        
        # 下載模型
        print("下載 financial_lstm 模型...")
        lstm_download = await agi_system.download_from_cloud("financial_lstm")
        print(f"結果: {'✅ 成功' if lstm_download else '❌ 失敗'}")
        
        print("下載 weather_transformer 模型...")
        transformer_download = await agi_system.download_from_cloud("weather_transformer")
        print(f"結果: {'✅ 成功' if transformer_download else '❌ 失敗'}")
        
        print("✅ 雲端儲存演示完成")
        
    except Exception as e:
        print(f"❌ 雲端儲存演示失敗: {e}")

async def run_status_check():
    """運行狀態檢查"""
    print("📊 持久化AGI預測系統 - 狀態檢查")
    print("=" * 50)
    
    config = PersistentConfig()
    agi_system = PersistentAGISystem(config)
    
    try:
        status = agi_system.get_system_status()
        
        print("📈 系統狀態:")
        print("-" * 20)
        print(f"系統運行: {'🟢 是' if status.get('system_running', False) else '🔴 否'}")
        print(f"持續學習: {'✅ 啟用' if status.get('continuous_learning_enabled', False) else '❌ 停用'}")
        print(f"總模型數: {status.get('total_models', 0)}")
        print(f"總預測數: {status.get('total_predictions', 0)}")
        print(f"本地儲存: {status.get('storage_path', 'N/A')}")
        print(f"雲端儲存: {'✅ 啟用' if status.get('cloud_enabled', False) else '❌ 停用'}")
        
        if status.get('last_training_time'):
            print(f"最後訓練: {status.get('last_training_time')}")
        
        if status.get('model_performance'):
            print("\n📊 模型性能:")
            print("-" * 20)
            for model, accuracy in status['model_performance'].items():
                print(f"  {model}: {accuracy:.4f}")
        
        print("✅ 狀態檢查完成")
        
    except Exception as e:
        print(f"❌ 狀態檢查失敗: {e}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="持久化AGI預測系統運行器 - Persistent AGI System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_persistent_agi.py --demo              # 運行完整演示
  python run_persistent_agi.py --prediction        # 運行預測演示
  python run_persistent_agi.py --training          # 運行訓練演示
  python run_persistent_agi.py --cloud             # 運行雲端儲存演示
  python run_persistent_agi.py --status            # 運行狀態檢查
  python run_persistent_agi.py --all               # 運行所有演示
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='運行完整演示')
    parser.add_argument('--prediction', action='store_true',
                       help='運行預測演示')
    parser.add_argument('--training', action='store_true',
                       help='運行訓練演示')
    parser.add_argument('--cloud', action='store_true',
                       help='運行雲端儲存演示')
    parser.add_argument('--status', action='store_true',
                       help='運行狀態檢查')
    parser.add_argument('--all', action='store_true',
                       help='運行所有演示')
    
    args = parser.parse_args()
    
    if not any([args.demo, args.prediction, args.training, args.cloud, args.status, args.all]):
        parser.print_help()
        return
    
    async def run_selected_demos():
        if args.all:
            print("🚀 運行所有演示")
            print("=" * 60)
            
            print("\n1️⃣ 訓練演示")
            await run_training_demo()
            
            print("\n2️⃣ 預測演示")
            await run_prediction_demo()
            
            print("\n3️⃣ 雲端儲存演示")
            await run_cloud_demo()
            
            print("\n4️⃣ 狀態檢查")
            await run_status_check()
            
            print("\n5️⃣ 完整演示")
            await run_full_demo()
            
        else:
            if args.training:
                await run_training_demo()
            if args.prediction:
                await run_prediction_demo()
            if args.cloud:
                await run_cloud_demo()
            if args.status:
                await run_status_check()
            if args.demo:
                await run_full_demo()
    
    try:
        asyncio.run(run_selected_demos())
    except KeyboardInterrupt:
        print("\n🛑 程式被用戶中斷")
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 