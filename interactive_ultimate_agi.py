#!/usr/bin/env python3
"""互動式終極AGI系統 - 讓使用者與系統互動"""
import asyncio
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from agi_ultimate_system import UltimateAGISystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveUltimateAGI:
    """互動式終極AGI系統"""
    
    def __init__(self):
        self.agi_system = UltimateAGISystem()
        self.session_history = []
        
    async def start_interactive_session(self):
        """開始互動式會話"""
        print("🤖 終極AGI系統 - 互動式會話")
        print("=" * 60)
        print("🎯 系統功能:")
        print("1. 智能預測 - 使用蒙地卡羅模擬和完美預測模型")
        print("2. 自動API選擇 - 智能選擇最佳資料來源")
        print("3. 風險評估 - 基於蒙地卡羅模擬的風險分析")
        print("4. 系統監控 - 實時監控系統狀態和性能")
        print("5. 歷史查詢 - 查看預測歷史和分析結果")
        print("=" * 60)
        
        # 初始化系統
        print("\n🔄 正在初始化系統...")
        init_success = await self.agi_system.initialize_system()
        
        if init_success:
            print("✅ 系統初始化成功！")
        else:
            print("❌ 系統初始化失敗，使用備用模式")
        
        # 主會話循環
        while True:
            try:
                print("\n" + "=" * 60)
                print("請選擇操作:")
                print("1. 🧠 智能預測")
                print("2. 📊 蒙地卡羅模擬")
                print("3. 🔌 API性能測試")
                print("4. 📈 模型性能分析")
                print("5. 📝 預測歷史")
                print("6. 🔍 系統狀態")
                print("7. 🎯 綜合演示")
                print("8. ❌ 退出")
                
                choice = input("\n請輸入選項 (1-8): ").strip()
                
                if choice == "1":
                    await self.perform_intelligent_prediction()
                elif choice == "2":
                    await self.run_monte_carlo_simulations()
                elif choice == "3":
                    await self.test_api_performance()
                elif choice == "4":
                    self.analyze_model_performance()
                elif choice == "5":
                    self.show_prediction_history()
                elif choice == "6":
                    self.show_system_status()
                elif choice == "7":
                    await self.run_comprehensive_demo()
                elif choice == "8":
                    print("👋 感謝使用終極AGI系統！")
                    break
                else:
                    print("❌ 無效選項，請重新選擇")
                    
            except KeyboardInterrupt:
                print("\n👋 程式被中斷，感謝使用！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")
    
    async def perform_intelligent_prediction(self):
        """執行智能預測"""
        print("\n🧠 智能預測")
        print("-" * 40)
        
        # 選擇領域
        domain = self._select_domain()
        if not domain:
            return
        
        # 選擇預測類型
        prediction_type = self._select_prediction_type(domain)
        if not prediction_type:
            return
        
        # 設定需求參數
        requirements = self._set_prediction_requirements(domain)
        
        print(f"\n🔄 正在執行 {domain} - {prediction_type} 智能預測...")
        
        try:
            # 執行預測
            result = await self.agi_system.intelligent_prediction(domain, prediction_type, requirements)
            
            # 顯示結果
            self._display_prediction_result(result)
            
            # 保存到會話歷史
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'intelligent_prediction',
                'domain': domain,
                'prediction_type': prediction_type,
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
        print("5. 自定義")
        
        choice = input("請輸入選項 (1-5): ").strip()
        
        domain_mapping = {
            "1": "financial",
            "2": "weather",
            "3": "medical",
            "4": "energy",
            "5": "custom"
        }
        
        domain = domain_mapping.get(choice)
        if not domain:
            print("❌ 無效選項")
            return None
        
        if domain == "custom":
            domain = input("請輸入自定義領域名稱: ").strip()
            if not domain:
                print("❌ 領域名稱不能為空")
                return None
        
        return domain
    
    def _select_prediction_type(self, domain: str) -> Optional[str]:
        """選擇預測類型"""
        # 預設預測類型
        default_types = {
            "financial": ["股票預測", "匯率預測", "投資組合優化", "風險評估"],
            "weather": ["天氣預報", "極端天氣預警", "氣候趨勢", "農業氣象"],
            "medical": ["疾病傳播", "醫療資源需求", "疫苗接種", "公共衛生"],
            "energy": ["能源需求", "電網負載", "可再生能源", "價格預測"]
        }
        
        types = default_types.get(domain, ["一般預測", "趨勢分析", "風險評估"])
        
        print(f"\n請選擇 {domain} 領域的預測類型:")
        for i, pred_type in enumerate(types, 1):
            print(f"{i}. {pred_type}")
        print(f"{len(types) + 1}. 自定義")
        
        choice = input(f"請輸入選項 (1-{len(types) + 1}): ").strip()
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(types):
                return types[index]
            elif index == len(types):
                custom_type = input("請輸入自定義預測類型: ").strip()
                return custom_type if custom_type else None
            else:
                print("❌ 無效選項")
                return None
        except ValueError:
            print("❌ 無效輸入")
            return None
    
    def _set_prediction_requirements(self, domain: str) -> Dict[str, Any]:
        """設定預測需求參數"""
        requirements = {}
        
        print(f"\n📋 設定 {domain} 領域的預測需求:")
        
        # 時間範圍
        time_range = input("時間範圍 (預設: 1d): ").strip()
        if time_range:
            requirements['time_range'] = time_range
        
        # 地理範圍
        geographic_scope = input("地理範圍 (預設: global): ").strip()
        if geographic_scope:
            requirements['geographic_scope'] = geographic_scope
        
        # 更新頻率
        update_frequency = input("更新頻率 (預設: real_time): ").strip()
        if update_frequency:
            requirements['update_frequency'] = update_frequency
        
        # 準確度要求
        try:
            accuracy = input("準確度要求 0.0-1.0 (預設: 0.8): ").strip()
            if accuracy:
                requirements['accuracy_requirement'] = float(accuracy)
        except ValueError:
            print("❌ 準確度必須是數字")
        
        # 成本約束
        try:
            cost = input("成本約束 (預設: 0.01): ").strip()
            if cost:
                requirements['cost_constraint'] = float(cost)
        except ValueError:
            print("❌ 成本必須是數字")
        
        # 延遲要求
        try:
            latency = input("延遲要求(ms) (預設: 1000): ").strip()
            if latency:
                requirements['latency_requirement'] = float(latency)
        except ValueError:
            print("❌ 延遲必須是數字")
        
        return requirements
    
    def _display_prediction_result(self, result: Dict[str, Any]):
        """顯示預測結果"""
        print("\n" + "=" * 80)
        print("📈 智能預測結果")
        print("=" * 80)
        
        # 基本資訊
        print(f"領域: {result.get('domain', 'N/A')}")
        print(f"預測類型: {result.get('prediction_type', 'N/A')}")
        print(f"時間: {result.get('timestamp', 'N/A')}")
        print(f"API來源: {result.get('api_source', 'N/A')}")
        print(f"API評分: {result.get('api_score', 0):.3f}")
        
        # 預測結果
        prediction = result.get('prediction', [[0.0]])
        confidence = result.get('confidence', 0.0)
        
        print(f"\n預測值: {prediction[0][0]:.6f}")
        print(f"信心度: {confidence:.2%}")
        
        # 趨勢分析
        trend = "📈 正面" if prediction[0][0] > 0 else "📉 負面" if prediction[0][0] < 0 else "➡️ 中性"
        confidence_level = "較高" if confidence > 0.8 else "中等" if confidence > 0.6 else "較低"
        
        print(f"趨勢: {trend}")
        print(f"信心度等級: {confidence_level}")
        
        # 蒙地卡羅洞察
        mc_insights = result.get('monte_carlo_insights', {})
        if mc_insights:
            print(f"\n🔮 蒙地卡羅洞察:")
            for key, value in list(mc_insights.items())[:5]:  # 顯示前5個
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # 風險評估
        risk_assessment = result.get('risk_assessment', {})
        if risk_assessment:
            print(f"\n⚠️ 風險評估:")
            print(f"  風險等級: {risk_assessment.get('risk_level', 'N/A')}")
            print(f"  風險分數: {risk_assessment.get('risk_score', 0):.2f}")
            
            key_risks = risk_assessment.get('key_risks', [])
            if key_risks:
                print(f"  主要風險:")
                for risk in key_risks:
                    print(f"    - {risk}")
            
            mitigation = risk_assessment.get('mitigation_strategies', [])
            if mitigation:
                print(f"  緩解策略:")
                for strategy in mitigation:
                    print(f"    - {strategy}")
        
        # 建議
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 建議:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # API詳情
        api_details = result.get('api_details', {})
        if api_details:
            print(f"\n🔌 API詳情:")
            print(f"  名稱: {api_details.get('name', 'N/A')}")
            print(f"  URL: {api_details.get('url', 'N/A')}")
            print(f"  資料品質: {api_details.get('data_quality', 'N/A')}")
            print(f"  可靠性: {api_details.get('reliability', 0):.2f}")
        
        print("=" * 80)
    
    async def run_monte_carlo_simulations(self):
        """執行蒙地卡羅模擬"""
        print("\n📊 蒙地卡羅模擬")
        print("-" * 40)
        
        print("請選擇模擬類型:")
        print("1. 金融投資組合")
        print("2. 天氣模式")
        print("3. 疾病傳播")
        print("4. 能源需求")
        print("5. 全部執行")
        
        choice = input("請輸入選項 (1-5): ").strip()
        
        try:
            if choice == "1":
                print("🔄 執行金融投資組合模擬...")
                results = self.agi_system.monte_carlo.simulate_financial_portfolio()
                self._display_monte_carlo_results("金融投資組合", results)
            elif choice == "2":
                print("🔄 執行天氣模式模擬...")
                results = self.agi_system.monte_carlo.simulate_weather_patterns()
                self._display_monte_carlo_results("天氣模式", results)
            elif choice == "3":
                print("🔄 執行疾病傳播模擬...")
                results = self.agi_system.monte_carlo.simulate_disease_spread()
                self._display_monte_carlo_results("疾病傳播", results)
            elif choice == "4":
                print("🔄 執行能源需求模擬...")
                results = self.agi_system.monte_carlo.simulate_energy_demand()
                self._display_monte_carlo_results("能源需求", results)
            elif choice == "5":
                print("🔄 執行全部模擬...")
                self.agi_system._run_monte_carlo_simulations()
                print("✅ 全部模擬完成")
            else:
                print("❌ 無效選項")
                return
            
            # 保存到會話歷史
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'monte_carlo_simulation',
                'choice': choice
            })
            
        except Exception as e:
            print(f"❌ 蒙地卡羅模擬失敗: {e}")
    
    def _display_monte_carlo_results(self, sim_type: str, results: Dict[str, Any]):
        """顯示蒙地卡羅模擬結果"""
        print(f"\n📊 {sim_type} 模擬結果:")
        print("-" * 40)
        
        if 'statistics' in results:
            stats = results['statistics']
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                elif isinstance(value, tuple) and len(value) == 2:
                    print(f"{key}: {value[0]:.4f} - {value[1]:.4f}")
                else:
                    print(f"{key}: {value}")
        
        print(f"✅ {sim_type} 模擬完成")
    
    async def test_api_performance(self):
        """測試API性能"""
        print("\n🔌 API性能測試")
        print("-" * 40)
        
        print("🔄 正在測試所有API...")
        
        try:
            results = await self.agi_system.api_selector.batch_test_apis()
            
            print(f"\n📊 API測試結果:")
            print("-" * 40)
            
            success_count = 0
            for api_name, result in results.items():
                status = "✅ 成功" if result['success'] else "❌ 失敗"
                latency = f"{result['latency_ms']:.1f}ms"
                print(f"{api_name}: {status} | 延遲: {latency}")
                
                if result['success']:
                    success_count += 1
            
            print(f"\n📈 測試摘要:")
            print(f"總API數: {len(results)}")
            print(f"成功數: {success_count}")
            print(f"失敗數: {len(results) - success_count}")
            print(f"成功率: {success_count/len(results)*100:.1f}%")
            
            # 保存到會話歷史
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'api_performance_test',
                'total_apis': len(results),
                'success_count': success_count
            })
            
        except Exception as e:
            print(f"❌ API性能測試失敗: {e}")
    
    def analyze_model_performance(self):
        """分析模型性能"""
        print("\n📈 模型性能分析")
        print("-" * 40)
        
        try:
            model_summary = self.agi_system.prediction_model.get_model_summary()
            
            print(f"總模型數: {model_summary['total_models']}")
            print(f"模型類型: {', '.join(model_summary['model_types'])}")
            
            if 'performance' in model_summary and model_summary['performance']:
                print(f"\n📊 性能指標:")
                for model_name, metrics in model_summary['performance'].items():
                    print(f"\n{model_name}:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.6f}")
            
            if 'weights' in model_summary:
                print(f"\n⚖️ 模型權重:")
                for model_name, weight in model_summary['weights'].items():
                    print(f"  {model_name}: {weight:.3f}")
            
            # 保存到會話歷史
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'model_performance_analysis',
                'total_models': model_summary['total_models']
            })
            
        except Exception as e:
            print(f"❌ 模型性能分析失敗: {e}")
    
    def show_prediction_history(self):
        """顯示預測歷史"""
        print("\n📝 預測歷史")
        print("-" * 40)
        
        try:
            # 獲取歷史記錄
            history = self.agi_system.get_prediction_history(limit=20)
            
            if not history:
                print("尚無預測歷史")
                return
            
            print(f"最近 {len(history)} 筆預測記錄:")
            print("-" * 60)
            
            for i, record in enumerate(history, 1):
                domain = record['domain']
                pred_type = record['prediction_type']
                confidence = record['confidence']
                timestamp = record['timestamp'][:19]
                
                # 趨勢圖示
                pred_result = json.loads(record['prediction_result'])
                if pred_result and len(pred_result) > 0 and len(pred_result[0]) > 0:
                    pred_value = pred_result[0][0]
                    trend = "📈" if pred_value > 0 else "📉" if pred_value < 0 else "➡️"
                else:
                    trend = "❓"
                
                print(f"{i:2d}. {trend} {domain} - {pred_type}")
                print(f"    信心度: {confidence:.2%} | 時間: {timestamp}")
                print()
            
        except Exception as e:
            print(f"❌ 獲取預測歷史失敗: {e}")
    
    def show_system_status(self):
        """顯示系統狀態"""
        print("\n🔍 系統狀態")
        print("-" * 40)
        
        try:
            system_summary = self.agi_system.get_system_summary()
            
            print(f"系統狀態: {'✅ 健康' if system_summary['system_status']['initialized'] else '❌ 異常'}")
            print(f"最後更新: {system_summary['last_update']}")
            print(f"可用模型數: {system_summary['available_models']}")
            print(f"可用API數: {system_summary['available_apis']}")
            print(f"蒙地卡羅模擬數: {system_summary['monte_carlo_simulations']}")
            print(f"總預測數: {system_summary['total_predictions']}")
            print(f"資料庫路徑: {system_summary['database_path']}")
            
            # 顯示會話歷史摘要
            if self.session_history:
                print(f"\n📊 本次會話摘要:")
                print(f"會話操作數: {len(self.session_history)}")
                
                # 統計操作類型
                operation_types = {}
                for record in self.session_history:
                    op_type = record['type']
                    operation_types[op_type] = operation_types.get(op_type, 0) + 1
                
                for op_type, count in operation_types.items():
                    print(f"  {op_type}: {count} 次")
            
        except Exception as e:
            print(f"❌ 獲取系統狀態失敗: {e}")
    
    async def run_comprehensive_demo(self):
        """執行綜合演示"""
        print("\n🎯 綜合演示")
        print("-" * 40)
        
        print("🔄 正在執行終極AGI系統綜合演示...")
        print("這可能需要幾分鐘時間，請耐心等待...")
        
        try:
            results = await self.agi_system.run_comprehensive_demo()
            
            print("\n🎉 綜合演示完成！")
            print("=" * 60)
            
            # 顯示結果摘要
            summary = results['summary']
            print(f"📊 總預測數: {summary['total_predictions']}")
            print(f"✅ 成功預測數: {summary['successful_predictions']}")
            print(f"📈 平均信心度: {summary['average_confidence']:.2%}")
            print(f"🔧 系統健康狀態: {summary['system_health']}")
            
            # 保存到會話歷史
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'comprehensive_demo',
                'total_predictions': summary['total_predictions'],
                'successful_predictions': summary['successful_predictions']
            })
            
            # 詢問是否保存報告
            save_choice = input("\n是否保存詳細報告到檔案? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '是']:
                filename = f"comprehensive_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"✅ 報告已保存到 {filename}")
            
        except Exception as e:
            print(f"❌ 綜合演示失敗: {e}")

async def main():
    """主函數"""
    print("🚀 啟動終極AGI系統...")
    
    # 創建互動式系統
    interactive_agi = InteractiveUltimateAGI()
    
    # 開始互動式會話
    await interactive_agi.start_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())
