#!/usr/bin/env python3
"""
真實AI模型演示腳本
展示數據收集、模型訓練和預測的完整流程
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from real_lstm_model import LSTMTimeSeriesPredictor
from ultimate_time_series_agi import UltimateTimeSeriesConfig, UltimateTimeSeriesAGI

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_data_collection():
    """演示數據收集功能"""
    print("📊 數據收集演示")
    print("=" * 50)
    
    try:
        # 創建配置
        config = UltimateTimeSeriesConfig()
        agi_system = UltimateTimeSeriesAGI(config)
        
        # 啟動系統
        await agi_system.start_system()
        
        # 收集股票數據
        print("📈 收集股票數據...")
        stock_result = await agi_system.collect_training_data('stocks')
        if 'error' not in stock_result:
            print(f"✅ 股票數據收集成功: {len(stock_result['datasets'])} 個數據集")
            for dataset in stock_result['datasets']:
                print(f"   - {dataset}")
        else:
            print(f"❌ 股票數據收集失敗: {stock_result['error']}")
        
        # 收集加密貨幣數據
        print("\n🪙 收集加密貨幣數據...")
        crypto_result = await agi_system.collect_training_data('crypto')
        if 'error' not in crypto_result:
            print(f"✅ 加密貨幣數據收集成功: {len(crypto_result['datasets'])} 個數據集")
            for dataset in crypto_result['datasets']:
                print(f"   - {dataset}")
        else:
            print(f"❌ 加密貨幣數據收集失敗: {crypto_result['error']}")
        
        return agi_system
        
    except Exception as e:
        print(f"❌ 數據收集演示失敗: {e}")
        return None

async def demo_model_training(agi_system):
    """演示模型訓練功能"""
    print("\n🚀 模型訓練演示")
    print("=" * 50)
    
    try:
        # 選擇一個數據集進行訓練
        data_path = Path(agi_system.config.data_path)
        available_datasets = list(data_path.glob("*.csv"))
        
        if not available_datasets:
            print("❌ 沒有可用的數據集")
            return False
        
        # 選擇第一個數據集
        dataset_path = available_datasets[0]
        dataset_name = dataset_path.stem
        print(f"🎯 選擇數據集: {dataset_name}")
        
        # 訓練模型
        print("🧠 開始訓練模型...")
        training_result = await agi_system.train_all_models(dataset_name)
        
        if 'error' not in training_result:
            print("✅ 模型訓練完成")
            print(f"   訓練結果: {len(training_result['training_results'])} 個模型")
            print(f"   評估結果: {len(training_result['evaluation_results'])} 個模型")
            
            # 顯示訓練詳情
            for model_name, result in training_result['training_results'].items():
                if 'error' not in result:
                    print(f"   📊 {model_name}: 訓練時間 {result.get('training_time', 0):.2f}秒")
            
            return True
        else:
            print(f"❌ 模型訓練失敗: {training_result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 模型訓練演示失敗: {e}")
        return False

async def demo_prediction(agi_system):
    """演示預測功能"""
    print("\n🔮 預測功能演示")
    print("=" * 50)
    
    try:
        # 創建測試數據
        sequence_length = agi_system.config.model_config['sequence_length']
        test_sequence = np.random.randn(1, sequence_length)
        print(f"📊 測試序列形狀: {test_sequence.shape}")
        
        # 進行預測
        print("🎯 開始預測...")
        result = await agi_system.make_prediction(test_sequence, 'ensemble')
        
        if 'error' not in result:
            print("✅ 預測成功")
            print(f"   模型: {result['model']}")
            print(f"   預測長度: {len(result['prediction'])}")
            print(f"   執行時間: {result.get('execution_time', 0):.3f}秒")
            print(f"   使用模型: {result.get('models_used', [])}")
            
            # 顯示預測結果
            prediction = result['prediction']
            confidence = result['confidence_interval']
            print(f"   預測值範圍: {min(prediction):.4f} ~ {max(prediction):.4f}")
            print(f"   置信區間範圍: {min(confidence):.4f} ~ {max(confidence):.4f}")
            
            return True
        else:
            print(f"❌ 預測失敗: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 預測演示失敗: {e}")
        return False

def demo_lstm_model():
    """演示LSTM模型"""
    print("\n🧠 LSTM模型演示")
    print("=" * 50)
    
    try:
        # 創建LSTM預測器
        predictor = LSTMTimeSeriesPredictor(
            input_size=5,  # 5個特徵
            hidden_size=64,
            num_layers=2,
            learning_rate=0.001
        )
        
        print(f"✅ LSTM模型創建成功")
        print(f"   設備: {predictor.device}")
        print(f"   輸入維度: {predictor.input_size}")
        print(f"   隱藏層大小: {predictor.hidden_size}")
        print(f"   層數: {predictor.num_layers}")
        
        # 創建模擬數據
        print("\n📊 創建模擬訓練數據...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # 創建時間序列數據
        time_steps = np.arange(n_samples)
        data = pd.DataFrame({
            'Open': 100 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, n_samples),
            'High': 105 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, n_samples),
            'Low': 95 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, n_samples),
            'Close': 100 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, n_samples),
            'Volume': np.random.poisson(1000, n_samples)
        })
        
        print(f"   數據形狀: {data.shape}")
        print(f"   特徵列: {list(data.columns)}")
        
        # 準備訓練數據
        print("\n🔧 準備訓練數據...")
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            data, 
            target_column='Close',
            sequence_length=60,
            test_split=0.2
        )
        
        if X_train is not None:
            print(f"   訓練集: {X_train.shape}")
            print(f"   測試集: {X_test.shape}")
            
            # 訓練模型
            print("\n🚀 開始訓練LSTM模型...")
            training_result = predictor.train(
                X_train, y_train,
                epochs=50,  # 減少epochs以加快演示
                batch_size=32,
                early_stopping_patience=5
            )
            
            if 'error' not in training_result:
                print("✅ LSTM模型訓練完成")
                print(f"   完成epochs: {training_result['epochs_completed']}")
                print(f"   最終訓練損失: {training_result['final_train_loss']:.6f}")
                
                # 評估模型
                print("\n📊 評估模型性能...")
                evaluation_result = predictor.evaluate(X_test, y_test)
                
                if 'error' not in evaluation_result:
                    print("✅ 模型評估完成")
                    print(f"   MSE: {evaluation_result['mse']:.6f}")
                    print(f"   MAE: {evaluation_result['mae']:.6f}")
                    print(f"   RMSE: {evaluation_result['rmse']:.6f}")
                    print(f"   R²: {evaluation_result['r2']:.6f}")
                    
                    # 進行預測
                    print("\n🔮 進行預測...")
                    test_sequence = X_test[:1]  # 取第一個測試序列
                    prediction = predictor.predict(test_sequence)
                    
                    if prediction is not None:
                        print("✅ 預測成功")
                        print(f"   預測值: {prediction[0]:.6f}")
                        print(f"   真實值: {y_test[0]:.6f}")
                        print(f"   誤差: {abs(prediction[0] - y_test[0]):.6f}")
                        
                        # 繪製訓練歷史
                        print("\n📈 繪製訓練歷史...")
                        predictor.plot_training_history('lstm_training_history.png')
                        
                        # 保存模型
                        print("\n💾 保存模型...")
                        predictor.save_model('demo_lstm_model.pth')
                        
                        return True
                    else:
                        print("❌ 預測失敗")
                        return False
                else:
                    print(f"❌ 模型評估失敗: {evaluation_result['error']}")
                    return False
            else:
                print(f"❌ LSTM模型訓練失敗: {training_result['error']}")
                return False
        else:
            print("❌ 數據準備失敗")
            return False
            
    except Exception as e:
        print(f"❌ LSTM模型演示失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函數"""
    print("🚀 真實AI模型演示腳本")
    print("=" * 60)
    print("🌟 展示數據收集、模型訓練和預測的完整流程")
    print("🎯 包含真實的LSTM模型實現")
    print("=" * 60)
    
    try:
        # 1. 數據收集演示
        agi_system = await demo_data_collection()
        if agi_system is None:
            print("❌ 數據收集失敗，無法繼續")
            return
        
        # 2. 模型訓練演示
        training_success = await demo_model_training(agi_system)
        if not training_success:
            print("❌ 模型訓練失敗，無法繼續")
            return
        
        # 3. 預測功能演示
        prediction_success = await demo_prediction(agi_system)
        if not prediction_success:
            print("❌ 預測功能失敗")
        
        # 4. LSTM模型演示
        lstm_success = demo_lstm_model()
        if not lstm_success:
            print("❌ LSTM模型演示失敗")
        
        print("\n" + "=" * 60)
        if training_success and prediction_success and lstm_success:
            print("🎉 所有演示完成！")
        else:
            print("⚠️ 部分演示失敗，請檢查錯誤信息")
        
    except Exception as e:
        print(f"❌ 演示腳本執行失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理資源
        if 'agi_system' in locals():
            agi_system.cleanup()
        print("\n🧹 系統資源清理完成")

if __name__ == "__main__":
    asyncio.run(main())
