#!/usr/bin/env python3
"""
AGI Universal Prediction System - 啟動腳本
快速啟動和測試AGI全預測系統

使用方法:
python run_agi.py --demo          # 運行完整演示
python run_agi.py --financial     # 只測試金融預測
python run_agi.py --weather       # 只測試天氣預測
python run_agi.py --medical       # 只測試醫療預測
python run_agi.py --energy        # 只測試能源預測
python run_agi.py --language      # 只測試語言預測
python run_agi.py --fusion        # 測試跨領域融合
python run_agi.py --api           # 啟動API服務器
"""

import argparse
import asyncio
import sys
import numpy as np
from datetime import datetime
from agi_predictor import AGIEngine, PredictionAPI

def print_banner():
    """顯示系統橫幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                🤖 AGI Universal Prediction System 🤖          ║
    ║                     全能預測人工智能系統                        ║
    ║                                                               ║
    ║  🧠 多領域AI模型融合 | 🔄 跨領域推理 | ⚡ 高性能並行處理      ║
    ║                                                               ║
    ║  支持領域: 💰金融 ⚕️醫療 🌤️天氣 ⚡能源 💬語言               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def test_financial_prediction(api: PredictionAPI):
    """測試金融預測功能"""
    print("\n" + "="*60)
    print("💰 金融預測測試")
    print("="*60)
    
    # 生成模擬股價數據
    days = 30
    initial_price = 100
    prices = [initial_price]
    for _ in range(days-1):
        change = np.random.normal(0, 0.02)  # 2%日波動
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    print(f"📈 測試資料: {days}天股價歷史數據")
    print(f"   起始價格: ${initial_price:.2f}")
    print(f"   當前價格: ${prices[-1]:.2f}")
    print(f"   總回報率: {((prices[-1]/prices[0]) - 1)*100:.2f}%")
    
    # 測試不同預測任務
    tasks = [
        ("short_term_forecast", "短期價格預測"),
        ("trend_analysis", "趨勢分析"),
        ("trading_strategy", "交易策略")
    ]
    
    for task_type, description in tasks:
        print(f"\n🔍 {description} ({task_type}):")
        try:
            result = await api.predict_financial(
                asset_type="stocks",
                timeframe="1d",
                historical_data=prices,
                task_type=task_type
            )
            
            print(f"   ✅ 預測成功 (置信度: {result['confidence']:.2%})")
            print(f"   ⏱️ 處理時間: {result['processing_time']:.3f}秒")
            print(f"   🎯 使用模型: {result['model_used']}")
            
            # 顯示關鍵預測結果
            predictions = result['predictions']
            if task_type == "short_term_forecast":
                next_price = predictions.get('next_price')
                if next_price:
                    print(f"   📊 預測明日價格: ${next_price:.2f}")
                    print(f"   📈 預期變化: {predictions.get('price_change', 0):.2f}")
                    
            elif task_type == "trend_analysis":
                trend = predictions.get('trend_direction', 'unknown')
                strength = predictions.get('trend_strength', 0)
                print(f"   📊 趨勢方向: {trend}")
                print(f"   💪 趨勢強度: {strength:.3f}")
                
            elif task_type == "trading_strategy":
                action = predictions.get('recommended_action', 'hold')
                rsi = predictions.get('rsi', 0)
                print(f"   📊 建議操作: {action.upper()}")
                print(f"   📈 RSI指標: {rsi:.1f}")
                
        except Exception as e:
            print(f"   ❌ 預測失敗: {e}")

async def test_weather_prediction(api: PredictionAPI):
    """測試天氣預測功能"""
    print("\n" + "="*60)
    print("🌤️ 天氣預測測試")
    print("="*60)
    
    # 測試不同地點
    locations = [
        (25.0330, 121.5654, "台北"),
        (22.6273, 120.3014, "高雄"),
        (24.1477, 120.6736, "台中")
    ]
    
    for lat, lon, city in locations:
        print(f"\n📍 {city} 天氣預測 (經緯度: {lat:.4f}, {lon:.4f}):")
        
        try:
            result = await api.predict_weather(
                latitude=lat,
                longitude=lon,
                forecast_hours=48
            )
            
            print(f"   ✅ 預測成功 (置信度: {result['confidence']:.2%})")
            print(f"   ⏱️ 處理時間: {result['processing_time']:.3f}秒")
            
            # 顯示當前天氣
            current = result['predictions']['current_conditions']
            print(f"   🌡️ 當前溫度: {current['temperature']:.1f}°C")
            print(f"   💧 濕度: {current['humidity']:.1f}%")
            print(f"   💨 風速: {current['wind_speed']:.1f} km/h")
            print(f"   ☁️ 天氣狀況: {current['conditions']}")
            
            # 顯示預報摘要
            forecast_points = len(result['predictions']['forecast'])
            print(f"   📊 48小時預報點數: {forecast_points}")
            
            # 檢查極端天氣警報
            alerts = result['predictions']['extreme_weather_alerts']
            if alerts:
                print(f"   ⚠️ 極端天氣警報: {len(alerts)} 個")
                for alert in alerts[:2]:  # 只顯示前2個
                    print(f"      - {alert['type']}: {alert['description']}")
            else:
                print(f"   ✅ 無極端天氣警報")
                
        except Exception as e:
            print(f"   ❌ 預測失敗: {e}")

async def test_medical_prediction(api: PredictionAPI):
    """測試醫療預測功能"""
    print("\n" + "="*60)
    print("⚕️ 醫療預測測試")
    print("="*60)
    
    # 測試不同醫療場景
    test_cases = [
        {
            "name": "高風險患者",
            "patient_data": {"age": 75, "gender": "male", "bmi": 28.5},
            "medical_history": ["diabetes", "hypertension", "heart_disease"],
            "task_type": "readmission_risk"
        },
        {
            "name": "中等風險患者", 
            "patient_data": {"age": 45, "gender": "female", "bmi": 24.2},
            "medical_history": ["asthma"],
            "task_type": "readmission_risk"
        },
        {
            "name": "影像診斷測試",
            "patient_data": {"age": 60, "gender": "male"},
            "medical_history": ["smoking"],
            "task_type": "image_diagnosis"
        }
    ]
    
    for case in test_cases:
        print(f"\n🏥 {case['name']} - {case['task_type']}:")
        
        try:
            result = await api.predict_medical(
                patient_data=case["patient_data"],
                medical_history=case["medical_history"],
                task_type=case["task_type"]
            )
            
            print(f"   ✅ 分析完成 (置信度: {result['confidence']:.2%})")
            print(f"   ⏱️ 處理時間: {result['processing_time']:.3f}秒")
            print(f"   🎯 使用模型: {result['model_used']}")
            
            predictions = result['predictions']
            
            if case["task_type"] == "readmission_risk":
                risk_level = predictions.get('risk_level', 'unknown')
                probability = predictions.get('readmission_probability', 0)
                print(f"   📊 再入院風險: {risk_level.upper()}")
                print(f"   📈 風險機率: {probability:.2%}")
                
                interventions = predictions.get('recommended_interventions', [])
                if interventions:
                    print(f"   💡 建議干預措施:")
                    for intervention in interventions[:2]:
                        print(f"      - {intervention}")
                        
            elif case["task_type"] == "image_diagnosis":
                diagnosis = predictions.get('diagnosis', 'unknown')
                severity = predictions.get('severity', 'unknown')
                print(f"   📊 診斷結果: {diagnosis}")
                print(f"   ⚠️ 嚴重程度: {severity}")
                
        except Exception as e:
            print(f"   ❌ 分析失敗: {e}")

async def test_energy_prediction(api: PredictionAPI):
    """測試能源預測功能"""
    print("\n" + "="*60)
    print("⚡ 能源預測測試")
    print("="*60)
    
    # 生成模擬能源數據
    hours = 24
    base_load = 1000  # MW
    consumption_data = []
    
    for h in range(hours):
        # 模擬日負載週期
        daily_factor = 0.8 + 0.4 * np.sin((h - 6) * np.pi / 12)
        noise = np.random.normal(1, 0.1)
        load = base_load * daily_factor * noise
        consumption_data.append(load)
    
    print(f"⚡ 測試資料: {hours}小時電力消耗歷史")
    print(f"   平均負載: {np.mean(consumption_data):.1f} MW")
    print(f"   峰值負載: {np.max(consumption_data):.1f} MW")
    
    # 測試不同能源預測任務
    tasks = [
        ("load_forecast", "electricity", "負載預測"),
        ("renewable_generation", "solar", "太陽能發電預測"),
        ("price_forecast", "electricity", "電價預測")
    ]
    
    for task_type, energy_type, description in tasks:
        print(f"\n🔍 {description} ({task_type}):")
        
        try:
            if task_type == "renewable_generation":
                result = await api.predict_energy(
                    energy_type=energy_type,
                    region="taiwan", 
                    historical_data=[],
                    forecast_hours=24,
                    task_type=task_type
                )
                # 添加再生能源特定數據
                result_data = result.copy()
                result_data['data'] = {
                    "renewable_type": "solar",
                    "installed_capacity": 100
                }
            else:
                result = await api.predict_energy(
                    energy_type=energy_type,
                    region="taiwan",
                    historical_data=consumption_data,
                    forecast_hours=24,
                    task_type=task_type
                )
            
            print(f"   ✅ 預測成功 (置信度: {result['confidence']:.2%})")
            print(f"   ⏱️ 處理時間: {result['processing_time']:.3f}秒")
            
            predictions = result['predictions']
            
            if task_type == "load_forecast":
                summary = predictions.get('summary', {})
                peak_load = summary.get('peak_load_mw', 0)
                avg_load = summary.get('average_load_mw', 0)
                print(f"   📊 預測峰值負載: {peak_load:.1f} MW")
                print(f"   📈 平均負載: {avg_load:.1f} MW")
                
            elif task_type == "renewable_generation":
                summary = predictions.get('summary', {})
                total_gen = summary.get('total_generation_mwh', 0)
                capacity_factor = summary.get('average_capacity_factor', 0)
                print(f"   📊 總發電量: {total_gen:.1f} MWh")
                print(f"   📈 容量因數: {capacity_factor:.2%}")
                
            elif task_type == "price_forecast":
                summary = predictions.get('summary', {})
                avg_price = summary.get('average_price_per_mwh', 0)
                volatility = summary.get('price_volatility', 0)
                print(f"   📊 平均電價: ${avg_price:.2f}/MWh")
                print(f"   📈 價格波動率: {volatility:.3f}")
                
        except Exception as e:
            print(f"   ❌ 預測失敗: {e}")

async def test_language_prediction(api: PredictionAPI):
    """測試語言預測功能"""
    print("\n" + "="*60)
    print("💬 語言預測測試")
    print("="*60)
    
    # 測試不同語言任務
    test_cases = [
        {
            "task": "text_generation",
            "text": "人工智慧的未來發展將會",
            "description": "文本生成"
        },
        {
            "task": "sentiment_analysis", 
            "text": "這個AI系統真的很棒，預測準確度很高！",
            "description": "情感分析"
        },
        {
            "task": "text_classification",
            "text": "股市今日大漲，科技股表現亮眼，投資者信心增強。",
            "description": "文本分類"
        },
        {
            "task": "question_answering",
            "text": "AGI系統有哪些優勢？",
            "description": "問答系統",
            "context": "AGI全預測系統整合了多個領域的AI模型，提供跨領域融合推理能力。"
        },
        {
            "task": "code_generation",
            "text": "創建一個計算斐波那契數列的Python函數",
            "description": "程式碼生成",
            "programming_language": "python"
        }
    ]
    
    for case in test_cases:
        print(f"\n📝 {case['description']} ({case['task']}):")
        print(f"   輸入: {case['text'][:50]}{'...' if len(case['text']) > 50 else ''}")
        
        try:
            kwargs = {
                'text': case['text'],
                'task_type': case['task'],
                'language': 'zh-TW'
            }
            
            # 添加特定任務參數
            if case['task'] == 'question_answering' and 'context' in case:
                kwargs['context'] = case['context']
            elif case['task'] == 'code_generation' and 'programming_language' in case:
                kwargs['programming_language'] = case['programming_language']
            elif case['task'] == 'text_classification':
                kwargs['categories'] = ['科技', '商業', '娛樂', '體育', '政治']
            
            result = await api.predict_language(**kwargs)
            
            print(f"   ✅ 處理成功 (置信度: {result['confidence']:.2%})")
            print(f"   ⏱️ 處理時間: {result['processing_time']:.3f}秒")
            
            predictions = result['predictions']
            
            if case['task'] == 'text_generation':
                generated = predictions.get('generated_text', '')
                print(f"   📄 生成文本: {generated[:100]}{'...' if len(generated) > 100 else ''}")
                
            elif case['task'] == 'sentiment_analysis':
                sentiment = predictions.get('sentiment', 'unknown')
                confidence_score = predictions.get('confidence_score', 0)
                print(f"   😊 情感極性: {sentiment.upper()}")
                print(f"   📊 信心分數: {confidence_score:.3f}")
                
            elif case['task'] == 'text_classification':
                category = predictions.get('predicted_category', 'unknown')
                top_3 = predictions.get('top_3_categories', [])
                print(f"   📊 預測類別: {category}")
                if top_3:
                    print(f"   📈 前三名: {', '.join([f'{cat}({score:.2f})' for cat, score in top_3])}")
                    
            elif case['task'] == 'question_answering':
                answer = predictions.get('answer', 'No answer')
                print(f"   💡 回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
            elif case['task'] == 'code_generation':
                code = predictions.get('generated_code', '')
                lines = code.split('\n')
                print(f"   💻 生成代碼 ({len(lines)} 行):")
                for i, line in enumerate(lines[:3]):  # 只顯示前3行
                    print(f"      {i+1}: {line}")
                if len(lines) > 3:
                    print(f"      ... (共 {len(lines)} 行)")
                    
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")

async def test_fusion_capabilities(api: PredictionAPI):
    """測試跨領域融合功能"""
    print("\n" + "="*60)
    print("🔄 跨領域融合預測測試")
    print("="*60)
    
    # 測試多種融合場景
    scenarios = [
        ("market_analysis", "市場分析場景"),
        ("weather_impact", "天氣影響場景"),
        ("health_monitoring", "健康監測場景")
    ]
    
    for scenario, description in scenarios:
        print(f"\n🎯 {description} ({scenario}):")
        
        try:
            result = await api.predict_multi_domain_scenario(scenario)
            
            if 'error' in result:
                print(f"   ❌ 場景不支持: {result['error']}")
                continue
            
            # 處理結果統計
            summary = result.get('processing_summary', {})
            print(f"   ✅ 融合預測完成")
            print(f"   📊 總請求數: {summary.get('total_requests', 0)}")
            print(f"   ✅ 成功預測: {summary.get('successful_predictions', 0)}")
            print(f"   ❌ 失敗預測: {summary.get('failed_predictions', 0)}")
            print(f"   ⏱️ 總處理時間: {summary.get('total_processing_time', 0):.3f}秒")
            print(f"   🎯 平均置信度: {summary.get('average_confidence', 0):.2%}")
            
            # 顯示跨領域洞察
            fusion_insights = result.get('fusion_insights')
            if fusion_insights:
                patterns = fusion_insights.get('cross_domain_patterns', [])
                synergies = fusion_insights.get('synergistic_effects', [])
                opportunities = fusion_insights.get('optimization_opportunities', [])
                
                print(f"   🔍 跨領域模式: {len(patterns)} 個")
                for pattern in patterns[:2]:
                    print(f"      - {pattern['pattern']}: {pattern['description']}")
                
                print(f"   ⚡ 協同效應: {len(synergies)} 個")
                for synergy in synergies[:1]:
                    print(f"      - {synergy['effect']}: {synergy['description']}")
                
                print(f"   💡 優化機會: {len(opportunities)} 個")
                for opportunity in opportunities[:2]:
                    print(f"      - {opportunity['opportunity']}: {opportunity['description']}")
            else:
                print(f"   ℹ️ 未生成融合洞察")
                
        except Exception as e:
            print(f"   ❌ 融合預測失敗: {e}")

async def run_full_demo():
    """運行完整系統演示"""
    print_banner()
    print(f"🚀 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 創建AGI系統
    agi = AGIEngine(config_path="config.json")
    api = PredictionAPI(agi)
    
    try:
        # 初始化系統
        print("\n⚙️ 正在初始化AGI系統...")
        await api.start_engine()
        
        # 顯示系統狀態
        status = api.get_status()
        print(f"✅ 系統初始化完成!")
        print(f"📡 支持領域: {', '.join(status['supported_domains'])}")
        print(f"🔄 融合功能: {'啟用' if status['fusion_enabled'] else '禁用'}")
        
        # 運行各項測試
        await test_financial_prediction(api)
        await test_weather_prediction(api)
        await test_medical_prediction(api)
        await test_energy_prediction(api)
        await test_language_prediction(api)
        await test_fusion_capabilities(api)
        
        # 顯示最終統計
        print("\n" + "="*60)
        print("📈 系統性能統計")
        print("="*60)
        
        metrics = agi.get_performance_metrics()
        print(f"🎯 總預測次數: {metrics['total_predictions']}")
        print(f"✅ 成功率: {metrics['success_rate']:.2%}")
        print(f"⏱️ 平均處理時間: {metrics['average_processing_time']:.3f}秒")
        print(f"🚀 每分鐘預測數: {metrics['predictions_per_minute']:.1f}")
        print(f"⏰ 系統運行時間: {metrics['uptime_seconds']:.1f}秒")
        
        # 領域使用統計
        if metrics['domain_usage']:
            print(f"\n📊 領域使用統計:")
            for domain, count in metrics['domain_usage'].items():
                percentage = (count / metrics['total_predictions']) * 100
                print(f"   {domain}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n🎉 AGI全預測系統演示完成!")
        print(f"⏱️ 演示時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 系統運行錯誤: {e}")
        return 1
    
    return 0

async def run_single_domain_test(domain: str):
    """運行單一領域測試"""
    print_banner()
    print(f"🎯 單一領域測試: {domain.upper()}")
    
    agi = AGIEngine(config_path="config.json")
    api = PredictionAPI(agi)
    
    try:
        await api.start_engine()
        
        if domain == "financial":
            await test_financial_prediction(api)
        elif domain == "weather":
            await test_weather_prediction(api)
        elif domain == "medical":
            await test_medical_prediction(api)
        elif domain == "energy":
            await test_energy_prediction(api)
        elif domain == "language":
            await test_language_prediction(api)
        elif domain == "fusion":
            await test_fusion_capabilities(api)
        else:
            print(f"❌ 不支持的領域: {domain}")
            return 1
            
        # 顯示性能統計
        metrics = agi.get_performance_metrics()
        print(f"\n📊 測試統計:")
        print(f"   總預測次數: {metrics['total_predictions']}")
        print(f"   成功率: {metrics['success_rate']:.2%}")
        print(f"   平均處理時間: {metrics['average_processing_time']:.3f}秒")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return 1
    
    return 0

def start_api_server():
    """啟動API服務器 (需要額外的Web框架)"""
    print_banner()
    print("🌐 啟動API服務器...")
    print("⚠️ 注意: API服務器功能需要安裝 FastAPI 和 uvicorn")
    print("安裝命令: pip install fastapi uvicorn")
    print("\nAPI服務器啟動後可通過以下方式訪問:")
    print("  - Swagger文檔: http://localhost:8000/docs")
    print("  - 健康檢查: http://localhost:8000/health")
    print("  - 預測接口: http://localhost:8000/predict")
    
    # 這裡可以添加 FastAPI 服務器代碼
    print("\n💡 API服務器功能正在開發中...")

def main():
    parser = argparse.ArgumentParser(
        description="AGI Universal Prediction System - 全能預測人工智能系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_agi.py --demo           # 運行完整演示 (推薦)
  python run_agi.py --financial      # 測試金融預測
  python run_agi.py --weather        # 測試天氣預測
  python run_agi.py --medical        # 測試醫療預測
  python run_agi.py --energy         # 測試能源預測
  python run_agi.py --language       # 測試語言預測
  python run_agi.py --fusion         # 測試跨領域融合
  python run_agi.py --api            # 啟動API服務器
        """
    )
    
    # 添加命令行參數
    parser.add_argument("--demo", action="store_true", help="運行完整系統演示")
    parser.add_argument("--financial", action="store_true", help="測試金融預測功能")
    parser.add_argument("--weather", action="store_true", help="測試天氣預測功能")
    parser.add_argument("--medical", action="store_true", help="測試醫療預測功能")
    parser.add_argument("--energy", action="store_true", help="測試能源預測功能")
    parser.add_argument("--language", action="store_true", help="測試語言預測功能")
    parser.add_argument("--fusion", action="store_true", help="測試跨領域融合功能")
    parser.add_argument("--api", action="store_true", help="啟動API服務器")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路徑")
    
    args = parser.parse_args()
    
    # 如果沒有指定任何參數，顯示幫助信息
    if not any(vars(args).values()):
        parser.print_help()
        return 0
    
    # 執行對應的功能
    try:
        if args.demo:
            return asyncio.run(run_full_demo())
        elif args.api:
            start_api_server()
            return 0
        elif args.financial:
            return asyncio.run(run_single_domain_test("financial"))
        elif args.weather:
            return asyncio.run(run_single_domain_test("weather"))
        elif args.medical:
            return asyncio.run(run_single_domain_test("medical"))
        elif args.energy:
            return asyncio.run(run_single_domain_test("energy"))
        elif args.language:
            return asyncio.run(run_single_domain_test("language"))
        elif args.fusion:
            return asyncio.run(run_single_domain_test("fusion"))
        else:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print("\n\n👋 用戶中斷，系統退出")
        return 0
    except Exception as e:
        print(f"\n❌ 系統錯誤: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 