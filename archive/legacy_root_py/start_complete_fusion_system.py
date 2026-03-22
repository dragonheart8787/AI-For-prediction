#!/usr/bin/env python3
"""
完整模型融合系統啟動腳本
整合預訓練模型下載、融合訓練和預測功能
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time

# 配置日誌 - 移除emoji並處理Windows編碼問題
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_fusion_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 設置控制台編碼為utf-8 (Windows兼容)
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class CompleteFusionSystem:
    """完整模型融合系統"""
    
    def __init__(self):
        self.model_downloader = None
        self.fusion_trainer = None
        self.advanced_fusion = None
        self.system_status = {
            'models_downloaded': False,
            'fusion_models_created': False,
            'fusion_models_trained': False,
            'system_ready': False
        }
    
    async def initialize_system(self):
        """初始化系統"""
        logger.info("🚀 初始化完整模型融合系統...")
        
        try:
            # 導入必要的模塊
            from model_downloader import ModelDownloader, ModelFusionTrainer
            from advanced_model_fusion import AdvancedModelFusion
            
            # 創建組件
            self.model_downloader = ModelDownloader()
            self.fusion_trainer = ModelFusionTrainer()
            self.advanced_fusion = AdvancedModelFusion()
            
            logger.info("✅ 系統組件初始化完成")
            return True
            
        except ImportError as e:
            logger.error(f"❌ 模塊導入失敗: {e}")
            logger.error("請確保已安裝所有依賴包")
            return False
        except Exception as e:
            logger.error(f"❌ 系統初始化失敗: {e}")
            return False
    
    async def download_pretrained_models(self, force_download: bool = False) -> bool:
        """下載預訓練模型"""
        if not self.model_downloader:
            logger.error("❌ 模型下載器未初始化")
            return False
        
        try:
            logger.info("📥 開始下載預訓練模型...")
            
            # 下載所有模型
            download_result = await self.model_downloader.download_all_models(force_download)
            
            if 'error' in download_result:
                logger.error(f"❌ 下載失敗: {download_result['error']}")
                return False
            
            successful_downloads = download_result.get('successful_downloads', 0)
            total_models = download_result.get('total_models', 0)
            
            if successful_downloads > 0:
                logger.info(f"✅ 模型下載完成: {successful_downloads}/{total_models}")
                self.system_status['models_downloaded'] = True
                return True
            else:
                logger.error("❌ 沒有成功下載的模型")
                return False
                
        except Exception as e:
            logger.error(f"❌ 模型下載過程失敗: {e}")
            return False
    
    async def create_fusion_models(self) -> bool:
        """創建融合模型"""
        if not self.advanced_fusion:
            logger.error("❌ 高級融合系統未初始化")
            return False
        
        try:
            logger.info("🔧 開始創建融合模型...")
            
            # 基礎模型列表
            base_models = ['timesfm', 'chronos', 'tft', 'nbeats', 'lstm_pretrained']
            
            # 創建各種融合策略
            fusion_strategies = list(self.advanced_fusion.fusion_strategies.keys())
            created_count = 0
            
            for strategy in fusion_strategies:
                logger.info(f"📚 創建融合模型: {strategy}")
                result = await self.advanced_fusion.create_fusion_model(strategy, base_models, input_dim=64)
                
                if 'error' not in result:
                    logger.info(f"✅ {strategy} 創建成功")
                    created_count += 1
                else:
                    logger.error(f"❌ {strategy} 創建失敗: {result['error']}")
            
            if created_count > 0:
                logger.info(f"✅ 融合模型創建完成: {created_count}/{len(fusion_strategies)}")
                self.system_status['fusion_models_created'] = True
                return True
            else:
                logger.error("❌ 沒有成功創建的融合模型")
                return False
                
        except Exception as e:
            logger.error(f"❌ 融合模型創建失敗: {e}")
            return False
    
    async def train_fusion_models(self, training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """訓練融合模型"""
        if not self.advanced_fusion:
            logger.error("❌ 高級融合系統未初始化")
            return False
        
        try:
            logger.info("🧠 開始訓練融合模型...")
            
            # 創建模擬訓練數據（如果未提供）
            if training_data is None:
                logger.info("📊 創建模擬訓練數據...")
                np.random.seed(42)
                n_samples = 1000
                n_features = 64
                
                X_train = np.random.randn(n_samples, n_features)
                y_train = np.random.randn(n_samples, 1)
                training_data = (X_train, y_train)
                
                logger.info(f"✅ 訓練數據創建完成: {X_train.shape}")
            
            # 訓練需要訓練的融合模型
            trainable_strategies = ['neural_fusion', 'weighted_average']
            trained_count = 0
            
            for strategy in trainable_strategies:
                if strategy in self.advanced_fusion.fusion_models:
                    logger.info(f"📚 訓練融合模型: {strategy}")
                    training_result = await self.advanced_fusion.train_fusion_model(
                        strategy, training_data, epochs=50
                    )
                    
                    if 'error' not in training_result:
                        logger.info(f"✅ {strategy} 訓練完成")
                        if 'final_loss' in training_result:
                            logger.info(f"   最終損失: {training_result['final_loss']:.6f}")
                        trained_count += 1
                    else:
                        logger.error(f"❌ {strategy} 訓練失敗: {training_result['error']}")
            
            if trained_count > 0:
                logger.info(f"✅ 融合模型訓練完成: {trained_count}/{len(trainable_strategies)}")
                self.system_status['fusion_models_trained'] = True
                return True
            else:
                logger.error("❌ 沒有成功訓練的融合模型")
                return False
                
        except Exception as e:
            logger.error(f"❌ 融合模型訓練失敗: {e}")
            return False
    
    async def test_fusion_predictions(self, test_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """測試融合預測功能"""
        if not self.advanced_fusion:
            return {'error': '高級融合系統未初始化'}
        
        try:
            logger.info("🔮 開始測試融合預測功能...")
            
            # 創建模擬測試數據（如果未提供）
            if test_data is None:
                np.random.seed(42)
                n_samples = 10
                n_features = 64
                test_data = np.random.randn(n_samples, n_features)
                logger.info(f"✅ 測試數據創建完成: {test_data.shape}")
            
            # 測試所有融合策略
            fusion_strategies = list(self.advanced_fusion.fusion_strategies.keys())
            test_results = {}
            
            for strategy in fusion_strategies:
                if strategy in self.advanced_fusion.fusion_models:
                    logger.info(f"🎯 測試融合策略: {strategy}")
                    
                    prediction_result = await self.advanced_fusion.make_ensemble_prediction(
                        test_data, strategy
                    )
                    
                    if 'error' not in prediction_result:
                        logger.info(f"✅ {strategy} 預測成功")
                        test_results[strategy] = {
                            'status': 'success',
                            'execution_time': prediction_result.get('execution_time', 0),
                            'prediction_length': len(prediction_result.get('prediction', [])),
                            'model_type': prediction_result.get('model_type', 'unknown')
                        }
                    else:
                        logger.error(f"❌ {strategy} 預測失敗: {prediction_result['error']}")
                        test_results[strategy] = {
                            'status': 'failed',
                            'error': prediction_result['error']
                        }
            
            # 統計測試結果
            successful_tests = sum(1 for r in test_results.values() if r['status'] == 'success')
            total_tests = len(test_results)
            
            logger.info(f"✅ 融合預測測試完成: {successful_tests}/{total_tests} 成功")
            
            return {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'test_results': test_results
            }
            
        except Exception as e:
            logger.error(f"❌ 融合預測測試失敗: {e}")
            return {'error': str(e)}
    
    async def run_complete_demo(self) -> bool:
        """運行完整演示"""
        logger.info("🎬 開始運行完整演示...")
        
        try:
            # 1. 下載預訓練模型
            logger.info("\n" + "="*60)
            logger.info("📥 步驟1: 下載預訓練模型")
            logger.info("="*60)
            
            if not await self.download_pretrained_models():
                logger.error("❌ 預訓練模型下載失敗，無法繼續")
                return False
            
            # 2. 創建融合模型
            logger.info("\n" + "="*60)
            logger.info("🔧 步驟2: 創建融合模型")
            logger.info("="*60)
            
            if not await self.create_fusion_models():
                logger.error("❌ 融合模型創建失敗，無法繼續")
                return False
            
            # 3. 訓練融合模型
            logger.info("\n" + "="*60)
            logger.info("🧠 步驟3: 訓練融合模型")
            logger.info("="*60)
            
            if not await self.train_fusion_models():
                logger.error("❌ 融合模型訓練失敗，無法繼續")
                return False
            
            # 4. 測試融合預測
            logger.info("\n" + "="*60)
            logger.info("🔮 步驟4: 測試融合預測")
            logger.info("="*60)
            
            test_results = await self.test_fusion_predictions()
            if 'error' in test_results:
                logger.error(f"❌ 融合預測測試失敗: {test_results['error']}")
                return False
            
            # 5. 查看系統性能
            logger.info("\n" + "="*60)
            logger.info("📊 步驟5: 系統性能統計")
            logger.info("="*60)
            
            performance = self.advanced_fusion.get_all_fusion_performance()
            for strategy, stats in performance.items():
                logger.info(f"\n📈 {strategy}:")
                logger.info(f"   最後更新: {stats.get('last_updated', 'N/A')}")
                logger.info(f"   預測次數: {len(stats.get('predictions', []))}")
                if 'execution_times' in stats and stats['execution_times']:
                    avg_time = np.mean(stats['execution_times'])
                    logger.info(f"   平均執行時間: {avg_time:.3f}秒")
            
            # 6. 保存系統狀態
            logger.info("\n" + "="*60)
            logger.info("💾 步驟6: 保存系統狀態")
            logger.info("="*60)
            
            self.advanced_fusion.save_fusion_system("complete_fusion_system.pkl")
            
            # 更新系統狀態
            self.system_status['system_ready'] = True
            
            logger.info("\n🎉 完整演示運行成功！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 完整演示運行失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return {
            'system_status': self.system_status,
            'available_fusion_strategies': list(self.advanced_fusion.fusion_strategies.keys()) if self.advanced_fusion else [],
            'downloaded_models': self.model_downloader.get_downloaded_models() if self.model_downloader else [],
            'fusion_models_created': list(self.advanced_fusion.fusion_models.keys()) if self.advanced_fusion else []
        }
    
    def cleanup(self):
        """清理系統資源"""
        logger.info("🧹 清理系統資源...")
        # 這裡可以添加資源清理邏輯

class InteractiveMenu:
    """交互式菜單"""
    
    def __init__(self, fusion_system: CompleteFusionSystem):
        self.fusion_system = fusion_system
        self.menu_options = {
            '1': ('運行完整演示', self.run_complete_demo),
            '2': ('下載預訓練模型', self.download_models),
            '3': ('創建融合模型', self.create_fusion_models),
            '4': ('訓練融合模型', self.train_fusion_models),
            '5': ('測試融合預測', self.test_fusion_predictions),
            '6': ('查看系統狀態', self.show_system_status),
            '7': ('查看可用融合策略', self.show_fusion_strategies),
            '0': ('退出系統', self.exit_system)
        }
    
    async def run_complete_demo(self):
        """運行完整演示"""
        logger.info("🎬 啟動完整演示...")
        success = await self.fusion_system.run_complete_demo()
        if success:
            logger.info("✅ 完整演示運行成功")
        else:
            logger.error("❌ 完整演示運行失敗")
    
    async def download_models(self):
        """下載預訓練模型"""
        logger.info("📥 啟動模型下載...")
        success = await self.fusion_system.download_pretrained_models()
        if success:
            logger.info("✅ 模型下載成功")
        else:
            logger.error("❌ 模型下載失敗")
    
    async def create_fusion_models(self):
        """創建融合模型"""
        logger.info("🔧 啟動融合模型創建...")
        success = await self.fusion_system.create_fusion_models()
        if success:
            logger.info("✅ 融合模型創建成功")
        else:
            logger.error("❌ 融合模型創建失敗")
    
    async def train_fusion_models(self):
        """訓練融合模型"""
        logger.info("🧠 啟動融合模型訓練...")
        success = await self.fusion_system.train_fusion_models()
        if success:
            logger.info("✅ 融合模型訓練成功")
        else:
            logger.error("❌ 融合模型訓練失敗")
    
    async def test_fusion_predictions(self):
        """測試融合預測"""
        logger.info("🔮 啟動融合預測測試...")
        test_results = await self.fusion_system.test_fusion_predictions()
        if 'error' not in test_results:
            logger.info("✅ 融合預測測試成功")
            logger.info(f"   測試結果: {test_results['successful_tests']}/{test_results['total_tests']} 成功")
        else:
            logger.error(f"❌ 融合預測測試失敗: {test_results['error']}")
    
    def show_system_status(self):
        """顯示系統狀態"""
        logger.info("📊 系統狀態:")
        status = self.fusion_system.get_system_status()
        
        for key, value in status.items():
            if isinstance(value, list):
                logger.info(f"   {key}: {len(value)} 個項目")
                for item in value[:5]:  # 只顯示前5個
                    logger.info(f"     - {item}")
                if len(value) > 5:
                    logger.info(f"     ... 還有 {len(value) - 5} 個項目")
            else:
                logger.info(f"   {key}: {value}")
    
    def show_fusion_strategies(self):
        """顯示可用融合策略"""
        if not self.fusion_system.advanced_fusion:
            logger.error("❌ 高級融合系統未初始化")
            return
        
        logger.info("🔧 可用融合策略:")
        for key, strategy in self.fusion_system.advanced_fusion.fusion_strategies.items():
            logger.info(f"   {key}: {strategy['name']}")
            logger.info(f"     描述: {strategy['description']}")
            logger.info(f"     類型: {strategy['type']}")
            logger.info(f"     自適應: {'是' if strategy['adaptive'] else '否'}")
            logger.info("")
    
    def exit_system(self):
        """退出系統"""
        logger.info("👋 感謝使用完整模型融合系統！")
        self.fusion_system.cleanup()
        sys.exit(0)
    
    def show_menu(self):
        """顯示主菜單"""
        print("\n" + "=" * 60)
        print("🚀 完整模型融合系統")
        print("=" * 60)
        print("請選擇操作:")
        
        for key, (description, _) in self.menu_options.items():
            print(f"  {key}. {description}")
        
        print("=" * 60)
    
    async def run(self):
        """運行交互式菜單"""
        while True:
            self.show_menu()
            
            try:
                choice = input("請輸入選項 (0-7): ").strip()
                
                if choice in self.menu_options:
                    description, func = self.menu_options[choice]
                    logger.info(f"🎯 選擇: {description}")
                    
                    if asyncio.iscoroutinefunction(func):
                        await func()
                    else:
                        func()
                    
                    input("\n按Enter鍵繼續...")
                else:
                    print("❌ 無效選項，請重新選擇")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ 檢測到Ctrl+C，正在退出...")
                break
            except Exception as e:
                logger.error(f"❌ 操作執行失敗: {e}")
                input("\n按Enter鍵繼續...")

async def main():
    """主函數"""
    print("🚀 完整模型融合系統")
    print("=" * 60)
    print("🌟 整合預訓練模型下載、融合訓練和預測功能")
    print("🎯 支持多種融合策略和自適應權重")
    print("=" * 60)
    
    # 創建融合系統
    fusion_system = CompleteFusionSystem()
    
    try:
        # 初始化系統
        if not await fusion_system.initialize_system():
            logger.error("❌ 系統初始化失敗")
            return
        
        # 創建交互式菜單
        menu = InteractiveMenu(fusion_system)
        
        # 運行菜單
        await menu.run()
        
    except Exception as e:
        logger.error(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理資源
        fusion_system.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 系統已退出")
    except Exception as e:
        logger.error(f"❌ 系統運行失敗: {e}")
        import traceback
        traceback.print_exc()
