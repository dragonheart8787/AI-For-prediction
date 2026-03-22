#!/usr/bin/env python3
"""AGI互動式預測系統 - 讓使用者提出具體預測需求"""
import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from agi_enhanced_real_prediction import EnhancedAGISystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveAGIPredictor:
    """互動式AGI預測器"""
    
    def __init__(self):
        self.agi_system = EnhancedAGISystem()
        self.prediction_history = []
        
    async def start_interactive_session(self):
        """開始互動式預測會話"""
        print("🤖 AGI互動式預測系統")
        print("=" * 50)
        print("支援的預測領域:")
        print("1. 金融 (financial) - 股票投資決策、匯率預測")
        print("2. 天氣 (weather) - 天氣預報、極端天氣預警")
        print("3. 醫療 (medical) - 疾病傳播預測、醫療資源需求")
        print("4. 能源 (energy) - 能源需求預測、電網負載預測")
        print("=" * 50)
        
        while True:
            try:
                print("\n請選擇操作:")
                print("1. 進行預測")
                print("2. 查看預測歷史")
                print("3. 查看分析記憶")
                print("4. 系統狀態")
                print("5. 退出")
                
                choice = input("\n請輸入選項 (1-5): ").strip()
                
                if choice == "1":
                    await self.perform_prediction()
                elif choice == "2":
                    self.show_prediction_history()
                elif choice == "3":
                    self.show_analysis_memory()
                elif choice == "4":
                    self.show_system_status()
                elif choice == "5":
                    print("👋 感謝使用AGI互動式預測系統！")
                    break
                else:
                    print("❌ 無效選項，請重新選擇")
                    
            except KeyboardInterrupt:
                print("\n👋 程式被中斷，感謝使用！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
    
    async def perform_prediction(self):
        """執行預測"""
        print("\n📊 預測設定")
        print("-" * 30)
        
        # 選擇領域
        domain = self._select_domain()
        if not domain:
            return
        
        # 選擇預測類型
        request_type = self._select_request_type(domain)
        if not request_type:
            return
        
        # 輸入額外上下文
        additional_context = input("請輸入額外上下文 (可選): ").strip()
        
        print(f"\n🔄 正在進行 {domain} - {request_type} 預測...")
        
        try:
            # 執行預測
            result = await self.agi_system.smart_predict_with_real_data(
                domain, request_type, additional_context
            )
            
            # 顯示結果
            self._display_prediction_result(result)
            
            # 儲存到歷史
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'domain': domain,
                'request_type': request_type,
                'result': result
            })
            
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
    
    def _select_domain(self) -> Optional[str]:
        """選擇預測領域"""
        print("\n請選擇預測領域:")
        print("1. 金融 (financial)")
        print("2. 天氣 (weather)")
        print("3. 醫療 (medical)")
        print("4. 能源 (energy)")
        
        choice = input("請輸入選項 (1-4): ").strip()
        
        domain_mapping = {
            "1": "financial",
            "2": "weather", 
            "3": "medical",
            "4": "energy"
        }
        
        domain = domain_mapping.get(choice)
        if not domain:
            print("❌ 無效選項")
            return None
        
        return domain
    
    def _select_request_type(self, domain: str) -> Optional[str]:
        """選擇預測類型"""
        request_types = {
            "financial": ["股票投資決策", "匯率預測"],
            "weather": ["天氣預報", "極端天氣預警"],
            "medical": ["疾病傳播預測", "醫療資源需求"],
            "energy": ["能源需求預測", "電網負載預測"]
        }
        
        types = request_types.get(domain, [])
        print(f"\n請選擇 {domain} 領域的預測類型:")
        
        for i, req_type in enumerate(types, 1):
            print(f"{i}. {req_type}")
        
        choice = input(f"請輸入選項 (1-{len(types)}): ").strip()
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(types):
                return types[index]
            else:
                print("❌ 無效選項")
                return None
        except ValueError:
            print("❌ 無效輸入")
            return None
    
    def _display_prediction_result(self, result: Dict[str, Any]):
        """顯示預測結果"""
        print("\n" + "=" * 60)
        print("📈 預測結果")
        print("=" * 60)
        
        # 基本資訊
        print(f"領域: {result['domain']}")
        print(f"預測類型: {result['request_type']}")
        print(f"使用模型: {result['model_used']}")
        print(f"時間: {result['timestamp']}")
        
        # 預測值
        prediction = result['prediction'][0][0]
        confidence = result['confidence']
        
        print(f"\n預測值: {prediction:.4f}")
        print(f"信心度: {confidence:.2%}")
        
        # 趨勢分析
        trend = "正面" if prediction > 0 else "負面" if prediction < 0 else "中性"
        confidence_level = "較高" if confidence > 0.8 else "中等" if confidence > 0.6 else "較低"
        
        print(f"趨勢: {trend}")
        print(f"信心度等級: {confidence_level}")
        
        # 模型融合結果
        fusion = result.get('fusion_result', {})
        if fusion.get('model_count', 1) > 1:
            print(f"\n模型融合:")
            print(f"  融合預測: {fusion['fused_prediction'][0][0]:.4f}")
            print(f"  融合信心度: {fusion['confidence']:.2%}")
            print(f"  使用模型數: {fusion['model_count']}")
        
        # 分析筆記
        if 'analysis_notes' in result:
            print(f"\n📝 分析筆記:")
            print(result['analysis_notes'])
        
        # 輸入資料摘要
        input_data = result.get('input_data', {})
        if input_data:
            print(f"\n📊 輸入資料摘要:")
            if isinstance(input_data, dict) and len(input_data) > 0:
                # 如果是多個股票的資料
                if any(isinstance(v, dict) for v in input_data.values()):
                    for symbol, data in list(input_data.items())[:3]:  # 只顯示前3個
                        print(f"  {symbol}: 價格變化 {data.get('price_change', 0):.2%}")
                else:
                    # 一般資料
                    for key, value in list(input_data.items())[:5]:  # 只顯示前5個
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
        
        print("=" * 60)
    
    def show_prediction_history(self):
        """顯示預測歷史"""
        if not self.prediction_history:
            print("\n📝 尚無預測歷史")
            return
        
        print(f"\n📝 預測歷史 (共 {len(self.prediction_history)} 筆)")
        print("-" * 60)
        
        for i, history in enumerate(self.prediction_history[-10:], 1):  # 顯示最近10筆
            result = history['result']
            prediction = result['prediction'][0][0]
            confidence = result['confidence']
            trend = "📈" if prediction > 0 else "📉" if prediction < 0 else "➡️"
            
            print(f"{i:2d}. {trend} {result['domain']} - {result['request_type']}")
            print(f"    預測: {prediction:.4f} | 信心度: {confidence:.2%} | 時間: {history['timestamp'][:19]}")
            print()
    
    def show_analysis_memory(self):
        """顯示分析記憶"""
        history = self.agi_system.get_analysis_history(limit=10)
        
        if not history:
            print("\n💾 尚無分析記憶")
            return
        
        print(f"\n💾 分析記憶 (共 {len(history)} 筆)")
        print("-" * 60)
        
        for i, memory in enumerate(history, 1):
            prediction = memory['prediction_result'][0][0]
            confidence = memory['confidence']
            trend = "📈" if prediction > 0 else "📉" if prediction < 0 else "➡️"
            
            print(f"{i:2d}. {trend} {memory['domain']} - {memory['request_type']}")
            print(f"    預測: {prediction:.4f} | 信心度: {confidence:.2%}")
            print(f"    模型: {memory['model_used']} | 時間: {memory['timestamp'][:19]}")
            print()
    
    def show_system_status(self):
        """顯示系統狀態"""
        print("\n🔧 系統狀態")
        print("-" * 30)
        
        # 載入的模型
        loaded_models = list(self.agi_system.predictor.loaded_models.keys())
        print(f"已載入模型: {len(loaded_models)} 個")
        if loaded_models:
            for model in loaded_models:
                print(f"  - {model}")
        else:
            print("  - 尚無載入的模型")
        
        # 分析記憶
        history = self.agi_system.get_analysis_history(limit=1)
        memory_count = len(history) if history else 0
        print(f"分析記憶條目: {memory_count} 筆")
        
        # 預測歷史
        print(f"本次會話預測: {len(self.prediction_history)} 筆")
        
        # 資料庫路徑
        print(f"資料庫路徑: {self.agi_system.db_path}")
        print(f"模型目錄: {self.agi_system.predictor.model_dir}")

async def main():
    """主函數"""
    predictor = InteractiveAGIPredictor()
    await predictor.start_interactive_session()

if __name__ == "__main__":
    asyncio.run(main()) 