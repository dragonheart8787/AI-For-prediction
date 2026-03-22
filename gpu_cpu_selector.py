#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU/CPU選擇器 - 簡單的命令行界面
讓用戶可以輕鬆選擇使用GPU或CPU進行計算
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from super_enhanced_ts_system import SuperEnhancedTSSystem, SuperEnhancedTSConfig

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUCPUSelector:
    """GPU/CPU選擇器類別"""
    
    def __init__(self):
        self.system = None
        self.current_device = None
    
    def display_welcome(self):
        """顯示歡迎信息"""
        print("=" * 70)
        print("🚀 超級增強版時間序列預測系統 - GPU/CPU選擇器")
        print("=" * 70)
        print("這個工具可以幫助你選擇使用GPU或CPU進行計算")
        print("GPU可以大幅加速深度學習和大型數據處理任務")
        print("CPU適合輕量級任務和沒有GPU的環境")
        print("=" * 70)
    
    def display_menu(self):
        """顯示主菜單"""
        print("\n📋 請選擇操作:")
        print("1. 🔍 檢測可用的計算設備")
        print("2. 🖥️  使用CPU模式")
        print("3. 🚀 使用GPU模式 (自動選擇)")
        print("4. 🎯 選擇特定GPU後端")
        print("5. 🔄 切換計算設備")
        print("6. 📊 顯示當前狀態")
        print("7. 🧪 運行測試預測")
        print("8. 🚪 退出")
        print("-" * 50)
    
    def get_user_choice(self, prompt="請輸入選項 (1-8): "):
        """獲取用戶輸入"""
        while True:
            try:
                choice = input(prompt).strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    return choice
                else:
                    print("❌ 無效選項，請輸入1-8之間的數字")
            except KeyboardInterrupt:
                print("\n\n👋 再見！")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 再見！")
                sys.exit(0)
    
    def detect_devices(self):
        """檢測可用的計算設備"""
        print("\n🔍 正在檢測可用的計算設備...")
        
        try:
            # 創建臨時系統來檢測設備
            temp_system = SuperEnhancedTSSystem()
            status = temp_system.get_device_status()
            available_backends = temp_system.gpu_config.get_available_backends()
            
            print(f"\n📊 檢測結果:")
            print(f"   當前設備: {status['current_device']}")
            print(f"   後端: {status['backend']}")
            
            print(f"\n🔧 可用的GPU後端:")
            gpu_available = False
            for backend_name, backend_info in available_backends.items():
                status_icon = "✅" if backend_info['available'] else "❌"
                print(f"   {status_icon} {backend_name}: {backend_info['name']}")
                if backend_info['available']:
                    gpu_available = True
            
            if not gpu_available:
                print("\n⚠️  未檢測到可用的GPU後端")
                print("   建議使用CPU模式或安裝GPU驅動")
            else:
                print(f"\n✅ 檢測到 {sum(1 for b in available_backends.values() if b['available'])} 個可用的GPU後端")
                
        except Exception as e:
            print(f"❌ 設備檢測失敗: {e}")
    
    def use_cpu_mode(self):
        """使用CPU模式"""
        print("\n🖥️  正在切換到CPU模式...")
        
        try:
            self.system = SuperEnhancedTSSystem(force_cpu=True)
            status = self.system.get_device_status()
            self.current_device = 'CPU'
            
            print("✅ 已切換到CPU模式")
            print(f"   當前設備: {status['current_device']}")
            print(f"   後端: {status['backend']}")
            
        except Exception as e:
            print(f"❌ 切換到CPU模式失敗: {e}")
    
    def use_gpu_mode(self):
        """使用GPU模式 (自動選擇)"""
        print("\n🚀 正在切換到GPU模式 (自動選擇)...")
        
        try:
            self.system = SuperEnhancedTSSystem(force_gpu=True)
            status = self.system.get_device_status()
            self.current_device = 'GPU'
            
            if status['current_device'] == 'GPU':
                print("✅ 已切換到GPU模式")
                print(f"   當前設備: {status['current_device']}")
                print(f"   後端: {status['backend']}")
                if status['gpu_name']:
                    print(f"   GPU名稱: {status['gpu_name']}")
            else:
                print("⚠️  強制使用GPU但未檢測到GPU，已回退到CPU模式")
                self.current_device = 'CPU'
                
        except Exception as e:
            print(f"❌ 切換到GPU模式失敗: {e}")
    
    def select_specific_gpu_backend(self):
        """選擇特定GPU後端"""
        print("\n🎯 選擇特定GPU後端")
        
        try:
            # 創建臨時系統來檢測可用的後端
            temp_system = SuperEnhancedTSSystem()
            available_backends = temp_system.gpu_config.get_available_backends()
            
            # 顯示可用的後端
            available_options = []
            print("可用的GPU後端:")
            for i, (backend_name, backend_info) in enumerate(available_backends.items()):
                if backend_info['available']:
                    available_options.append(backend_name)
                    print(f"   {len(available_options)}. {backend_name}")
            
            if not available_options:
                print("❌ 沒有可用的GPU後端")
                return
            
            # 獲取用戶選擇
            while True:
                try:
                    choice = input(f"請選擇GPU後端 (1-{len(available_options)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_options):
                        selected_backend = available_options[choice_idx]
                        break
                    else:
                        print(f"❌ 請輸入1-{len(available_options)}之間的數字")
                except ValueError:
                    print("❌ 請輸入有效的數字")
                except KeyboardInterrupt:
                    return
            
            # 創建系統並切換到指定後端
            print(f"\n🚀 正在切換到 {selected_backend}...")
            self.system = SuperEnhancedTSSystem(gpu_preference=selected_backend)
            status = self.system.get_device_status()
            self.current_device = 'GPU'
            
            print("✅ 已切換到指定GPU後端")
            print(f"   當前設備: {status['current_device']}")
            print(f"   後端: {status['backend']}")
            
        except Exception as e:
            print(f"❌ 選擇GPU後端失敗: {e}")
    
    def switch_device(self):
        """切換計算設備"""
        if not self.system:
            print("❌ 請先初始化系統")
            return
        
        print(f"\n🔄 當前設備: {self.current_device}")
        print("請選擇要切換到的設備:")
        print("1. 🖥️  CPU")
        print("2. 🚀 GPU")
        
        choice = input("請選擇 (1-2): ").strip()
        
        try:
            if choice == '1':
                self.system.switch_computation_device('cpu')
                self.current_device = 'CPU'
                print("✅ 已切換到CPU模式")
            elif choice == '2':
                self.system.switch_computation_device('gpu')
                status = self.system.get_device_status()
                self.current_device = status['current_device']
                print(f"✅ 已切換到{self.current_device}模式")
            else:
                print("❌ 無效選項")
        except Exception as e:
            print(f"❌ 切換設備失敗: {e}")
    
    def show_current_status(self):
        """顯示當前狀態"""
        if not self.system:
            print("❌ 系統未初始化")
            return
        
        print("\n📊 當前系統狀態:")
        status = self.system.get_device_status()
        
        print(f"   計算設備: {status['current_device']}")
        print(f"   後端: {status['backend']}")
        print(f"   強制CPU: {status['force_cpu']}")
        print(f"   強制GPU: {status['force_gpu']}")
        print(f"   GPU偏好: {status['gpu_preference']}")
        
        if status['gpu_name']:
            print(f"   GPU名稱: {status['gpu_name']}")
        
        if status['gpu_memory']:
            memory = status['gpu_memory']
            print(f"   GPU記憶體:")
            print(f"     總計: {memory['total']:.2f} GB")
            print(f"     已分配: {memory['allocated']:.2f} GB")
            print(f"     已緩存: {memory['cached']:.2f} GB")
            print(f"     可用: {memory['free']:.2f} GB")
    
    async def run_test_prediction(self):
        """運行測試預測"""
        if not self.system:
            print("❌ 請先初始化系統")
            return
        
        print("\n🧪 正在運行測試預測...")
        
        try:
            # 創建測試數據
            import pandas as pd
            import numpy as np
            
            print("   創建測試數據...")
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            test_data = pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            print("   運行預測任務...")
            results = await self.system.run_super_enhanced_system(['short_term_forecast'])
            
            print("✅ 測試預測完成")
            
            # 顯示結果摘要
            if 'fusion_results' in results:
                for task_type, fusion_result in results['fusion_results'].items():
                    print(f"   任務: {task_type}")
                    print(f"   融合類型: {fusion_result.get('fusion_type', 'N/A')}")
                    print(f"   模型數量: {len(fusion_result.get('model_weights', {}))}")
                    
                    # 顯示模型權重
                    if 'model_weights' in fusion_result:
                        print("   模型權重:")
                        for model, weight in fusion_result['model_weights'].items():
                            print(f"     {model}: {weight:.3f}")
            
        except Exception as e:
            print(f"❌ 測試預測失敗: {e}")
    
    async def run(self):
        """運行選擇器"""
        self.display_welcome()
        
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == '1':
                self.detect_devices()
            elif choice == '2':
                self.use_cpu_mode()
            elif choice == '3':
                self.use_gpu_mode()
            elif choice == '4':
                self.select_specific_gpu_backend()
            elif choice == '5':
                self.switch_device()
            elif choice == '6':
                self.show_current_status()
            elif choice == '7':
                await self.run_test_prediction()
            elif choice == '8':
                print("\n👋 感謝使用！再見！")
                break
            
            input("\n按Enter鍵繼續...")

async def main():
    """主函數"""
    selector = GPUCPUSelector()
    await selector.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 再見！")
    except Exception as e:
        print(f"\n❌ 程序運行錯誤: {e}")
        print("請檢查系統配置和依賴項")
