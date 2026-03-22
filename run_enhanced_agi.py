#!/usr/bin/env python3
"""
Enhanced AGI Deep Learning Prediction System - 運行腳本
超強預測AI系統啟動器

功能:
- 🧠 深度學習模型訓練
- 🔬 預測研究實驗
- 🚀 超強預測功能
- 📊 系統性能分析
- 📚 持續學習
"""

import argparse
import asyncio
import sys
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path

# 抑制警告
warnings.filterwarnings('ignore')

# 導入增強版AGI系統
try:
    from agi_deep_learning import EnhancedAGIPredictor, TrainingConfig
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"❌ 增強版AGI系統導入失敗: {e}")
    print("請安裝必要依賴: pip install torch tensorflow scikit-learn")
    ENHANCED_AVAILABLE = False

def print_banner():
    """顯示系統橫幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║          🧠 Enhanced AGI Deep Learning Prediction System 🧠           ║
    ║                        超強預測人工智能系統                            ║
    ║                                                                       ║
    ║  🚀 深度學習訓練 | 🔬 預測研究 | 📊 AutoML優化 | 📚 持續學習      ║
    ║                                                                       ║
    ║  支持模型: LSTM, Transformer, CNN, GNN, XGBoost, Random Forest      ║
    ║  研究功能: 超參數搜索, 模型比較, 特徵重要性, 不確定性量化           ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def train_all_models():
    """訓練所有領域的模型"""
    print("\n🚀 開始訓練所有領域模型...")
    
    if not ENHANCED_AVAILABLE:
        print("❌ 增強版系統不可用")
        return
    
    agi = EnhancedAGIPredictor()
    
    domains = ['financial', 'weather', 'energy']
    results = {}
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"💡 開始訓練 {domain.upper()} 領域模型")
        print(f"{'='*60}")
        
        try:
            research_results = await agi.train_domain_models(domain)
            results[domain] = research_results
            
            print(f"✅ {domain} 模型訓練完成!")
            print(f"📊 最佳模型: {research_results['best_model'].get('type', 'Unknown')}")
            
            # 顯示研究洞察
            insights = research_results.get('insights', [])
            if insights:
                print(f"🔍 關鍵洞察:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"   {i}. {insight}")
            
            # 顯示改進建議
            recommendations = research_results.get('recommendations', [])
            if recommendations:
                print(f"💡 改進建議:")
                for i, rec in enumerate(recommendations[:2], 1):
                    print(f"   {i}. {rec}")
                    
        except Exception as e:
            print(f"❌ {domain} 模型訓練失敗: {e}")
            results[domain] = {'error': str(e)}
    
    # 顯示訓練總結
    print(f"\n{'='*60}")
    print("📊 訓練總結報告")
    print(f"{'='*60}")
    
    successful_domains = [d for d, r in results.items() if 'error' not in r]
    failed_domains = [d for d, r in results.items() if 'error' in r]
    
    print(f"✅ 成功訓練: {len(successful_domains)} 個領域")
    print(f"❌ 訓練失敗: {len(failed_domains)} 個領域")
    
    if successful_domains:
        print(f"🎯 成功領域: {', '.join(successful_domains)}")
    
    if failed_domains:
        print(f"⚠️ 失敗領域: {', '.join(failed_domains)}")
    
    return results

async def run_super_prediction_demo():
    """運行超強預測演示"""
    print("\n🚀 超強預測功能演示")
    print("="*60)
    
    if not ENHANCED_AVAILABLE:
        print("❌ 增強版系統不可用")
        return
    
    agi = EnhancedAGIPredictor()
    
    # 演示不同領域的超強預測
    demos = [
        {
            'domain': 'financial',
            'task_type': 'short_term_forecast',
            'data': {
                'asset_type': 'stocks',
                'historical_data': list(np.random.uniform(100, 200, 100))
            },
            'description': '股票價格預測'
        },
        {
            'domain': 'weather',
            'task_type': 'weather_forecast',
            'data': {
                'location': {'lat': 25.0330, 'lon': 121.5654},
                'forecast_hours': 48
            },
            'description': '天氣預報'
        },
        {
            'domain': 'energy',
            'task_type': 'load_forecast',
            'data': {
                'energy_type': 'electricity',
                'historical_data': list(np.random.uniform(20000, 30000, 168)),
                'forecast_hours': 24
            },
            'description': '電力負載預測'
        }
    ]
    
    results = []
    
    for demo in demos:
        print(f"\n🎯 {demo['description']} ({demo['domain']})")
        print("-" * 40)
        
        try:
            result = await agi.super_predict(
                domain=demo['domain'],
                task_type=demo['task_type'],
                data=demo['data']
            )
            
            results.append(result)
            
            print(f"✅ 預測成功!")
            print(f"🎯 置信度: {result.confidence:.2%}")
            print(f"⏱️ 處理時間: {result.processing_time:.3f}秒")
            print(f"🔧 使用模型: {result.model_used}")
            
            # 顯示不確定性
            if result.uncertainty_bounds:
                print(f"📊 不確定性範圍: ±{result.uncertainty_bounds[1]:.3f}")
            
            # 顯示關鍵特徵
            if result.feature_importance:
                print("🔍 關鍵特徵 (前3名):")
                sorted_features = sorted(result.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:3]:
                    print(f"   • {feature}: {importance:.2%}")
            
            # 顯示預測結果摘要
            predictions = result.predictions
            if demo['domain'] == 'financial' and 'next_price' in predictions:
                print(f"💰 預測價格: ${predictions['next_price']:.2f}")
                if 'trend_strength' in predictions:
                    print(f"📈 趨勢強度: {predictions['trend_strength']:.3f}")
                    
            elif demo['domain'] == 'weather' and 'forecast' in predictions:
                forecast_points = len(predictions['forecast'])
                print(f"🌤️ 預報點數: {forecast_points}")
                if forecast_points > 0:
                    first_forecast = predictions['forecast'][0]
                    print(f"🌡️ 首個預報溫度: {first_forecast.get('temperature', 'N/A')}°C")
                    
            elif demo['domain'] == 'energy' and 'forecast' in predictions:
                forecast_points = len(predictions['forecast'])
                print(f"⚡ 負載預報點數: {forecast_points}")
                if 'pattern_analysis' in predictions:
                    pattern = predictions['pattern_analysis']
                    print(f"📊 日峰值時段: {pattern.get('daily_peak_hour', 'N/A')}時")
            
        except Exception as e:
            print(f"❌ {demo['description']} 預測失敗: {e}")
    
    # 顯示總體性能
    if results:
        avg_confidence = np.mean([r.confidence for r in results])
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        print(f"\n📈 總體性能統計:")
        print(f"   🎯 平均置信度: {avg_confidence:.2%}")
        print(f"   ⏱️ 平均處理時間: {avg_processing_time:.3f}秒")
        print(f"   ✅ 成功預測數: {len(results)}")

async def run_continuous_learning_demo():
    """運行持續學習演示"""
    print("\n📚 持續學習功能演示")
    print("="*60)
    
    if not ENHANCED_AVAILABLE:
        print("❌ 增強版系統不可用")
        return
    
    agi = EnhancedAGIPredictor()
    
    # 模擬持續學習場景
    scenarios = [
        {
            'domain': 'financial',
            'predicted_value': 150.25,
            'actual_value': 152.10,
            'description': '股價預測vs實際'
        },
        {
            'domain': 'weather',
            'predicted_value': 25.5,
            'actual_value': 24.8,
            'description': '溫度預測vs實際'
        },
        {
            'domain': 'energy',
            'predicted_value': 25000,
            'actual_value': 25500,
            'description': '負載預測vs實際'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔄 {scenario['description']}")
        print("-" * 30)
        
        try:
            learning_result = await agi.continuous_learning(
                domain=scenario['domain'],
                new_data={'prediction': scenario['predicted_value']},
                actual_result=scenario['actual_value']
            )
            
            error_rate = abs(scenario['predicted_value'] - scenario['actual_value']) / scenario['actual_value'] * 100
            
            print(f"📊 預測值: {scenario['predicted_value']}")
            print(f"🎯 實際值: {scenario['actual_value']}")
            print(f"📉 誤差率: {error_rate:.2f}%")
            print(f"🧠 學習狀態: {'已觸發重訓練' if learning_result['learning_applied'] else '持續監控'}")
            
            if learning_result.get('recommendations'):
                print("💡 系統建議:")
                for rec in learning_result['recommendations'][:2]:
                    print(f"   • {rec}")
                    
        except Exception as e:
            print(f"❌ 持續學習失敗: {e}")

async def run_system_analytics():
    """運行系統分析"""
    print("\n📊 系統分析報告")
    print("="*60)
    
    if not ENHANCED_AVAILABLE:
        print("❌ 增強版系統不可用")
        return
    
    agi = EnhancedAGIPredictor()
    
    # 先運行一些預測以生成數據
    print("🔄 生成分析數據...")
    
    try:
        # 快速預測以生成統計數據
        for domain in ['financial', 'weather', 'energy']:
            await agi.super_predict(
                domain=domain,
                task_type='forecast',
                data={'test': 'data'}
            )
    except:
        pass  # 忽略錯誤，繼續分析
    
    # 獲取系統分析
    analytics = agi.get_system_analytics()
    
    print(f"\n🎯 系統概覽:")
    print(f"   已訓練模型數量: {len(analytics['trained_models'])}")
    print(f"   總預測次數: {analytics['total_predictions']}")
    print(f"   平均處理時間: {analytics['average_processing_time']:.3f}秒")
    
    # 領域分佈
    if analytics['domain_distribution']:
        print(f"\n📊 領域使用分佈:")
        for domain, count in analytics['domain_distribution'].items():
            percentage = (count / analytics['total_predictions']) * 100 if analytics['total_predictions'] > 0 else 0
            print(f"   • {domain}: {count} 次 ({percentage:.1f}%)")
    
    # 置信度分析
    if analytics['confidence_distribution']:
        conf_dist = analytics['confidence_distribution']
        print(f"\n🎯 置信度分析:")
        print(f"   平均置信度: {conf_dist.get('mean', 0):.2%}")
        print(f"   置信度標準差: {conf_dist.get('std', 0):.3f}")
        print(f"   最高置信度: {conf_dist.get('max', 0):.2%}")
        print(f"   最低置信度: {conf_dist.get('min', 0):.2%}")
    
    # 最近性能
    if analytics['recent_performance']:
        print(f"\n⚡ 最近性能 (最後100次預測):")
        for domain, perf in analytics['recent_performance'].items():
            print(f"   • {domain}:")
            print(f"     - 預測次數: {perf['predictions']}")
            print(f"     - 平均置信度: {perf['avg_confidence']:.2%}")
            print(f"     - 平均處理時間: {perf['avg_processing_time']:.3f}秒")

async def run_research_experiment(domain: str):
    """運行預測研究實驗"""
    print(f"\n🔬 {domain.upper()} 預測研究實驗")
    print("="*60)
    
    if not ENHANCED_AVAILABLE:
        print("❌ 增強版系統不可用")
        return
    
    agi = EnhancedAGIPredictor()
    
    try:
        print("🔄 開始深入研究實驗...")
        print("   • 收集和預處理數據")
        print("   • 特徵工程和選擇")
        print("   • 模型架構比較")
        print("   • 超參數優化")
        print("   • 性能評估和分析")
        
        research_results = await agi.train_domain_models(domain)
        
        print(f"\n✅ {domain} 研究實驗完成!")
        
        # 顯示實驗結果
        print(f"\n📊 實驗結果摘要:")
        print(f"   數據形狀: {research_results.get('data_shape', 'Unknown')}")
        print(f"   實驗數量: {len(research_results.get('experiments', []))}")
        
        best_model = research_results.get('best_model', {})
        if best_model:
            print(f"   最佳模型: {best_model.get('type', 'Unknown')}")
        
        # 顯示關鍵洞察
        insights = research_results.get('insights', [])
        if insights:
            print(f"\n🔍 研究洞察:")
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        
        # 顯示改進建議
        recommendations = research_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 改進建議:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # 顯示實驗詳情
        experiments = research_results.get('experiments', [])
        if experiments:
            print(f"\n🧪 實驗詳情:")
            for i, exp in enumerate(experiments, 1):
                exp_type = exp.get('type', 'Unknown')
                print(f"   實驗 {i}: {exp_type}")
                
                if 'models' in exp:
                    models = exp['models']
                    best_model_name = min(models.keys(), key=lambda k: models[k].get('mse', float('inf')))
                    best_score = models[best_model_name].get('mse', 'N/A')
                    print(f"      最佳模型: {best_model_name} (MSE: {best_score})")
                
                if 'best_score' in exp:
                    print(f"      最佳分數: {exp['best_score']:.6f}")
                    
    except Exception as e:
        print(f"❌ 研究實驗失敗: {e}")

def check_system_requirements():
    """檢查系統需求"""
    print("🔍 檢查系統需求...")
    
    requirements = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn'
    }
    
    available = {}
    missing = []
    
    for module, name in requirements.items():
        try:
            if module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            available[name] = True
            print(f"   ✅ {name}")
        except ImportError:
            available[name] = False
            missing.append(name)
            print(f"   ❌ {name} (未安裝)")
    
    if missing:
        print(f"\n⚠️ 缺少依賴: {', '.join(missing)}")
        print("安裝命令:")
        if 'PyTorch' in missing:
            print("   pip install torch torchvision")
        if 'TensorFlow' in missing:
            print("   pip install tensorflow")
        if 'Scikit-learn' in missing:
            print("   pip install scikit-learn")
        print("   或運行: pip install -r requirements_deep_learning.txt")
        return False
    
    print("✅ 所有核心依賴已安裝")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced AGI Deep Learning Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_enhanced_agi.py --check              # 檢查系統需求
  python run_enhanced_agi.py --train-all          # 訓練所有模型
  python run_enhanced_agi.py --super-predict      # 超強預測演示
  python run_enhanced_agi.py --continuous         # 持續學習演示
  python run_enhanced_agi.py --analytics          # 系統分析
  python run_enhanced_agi.py --research financial # 研究實驗
        """
    )
    
    parser.add_argument("--check", action="store_true", help="檢查系統需求")
    parser.add_argument("--train-all", action="store_true", help="訓練所有領域模型")
    parser.add_argument("--super-predict", action="store_true", help="運行超強預測演示")
    parser.add_argument("--continuous", action="store_true", help="持續學習演示")
    parser.add_argument("--analytics", action="store_true", help="系統分析報告")
    parser.add_argument("--research", type=str, choices=['financial', 'weather', 'energy'], 
                       help="運行特定領域的研究實驗")
    parser.add_argument("--all", action="store_true", help="運行完整演示")
    
    args = parser.parse_args()
    
    print_banner()
    print(f"🚀 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 如果沒有指定參數，顯示幫助
    if not any(vars(args).values()):
        parser.print_help()
        return 0
    
    try:
        if args.check:
            if not check_system_requirements():
                return 1
        
        if args.train_all or args.all:
            asyncio.run(train_all_models())
        
        if args.super_predict or args.all:
            asyncio.run(run_super_prediction_demo())
        
        if args.continuous or args.all:
            asyncio.run(run_continuous_learning_demo())
        
        if args.analytics or args.all:
            asyncio.run(run_system_analytics())
        
        if args.research:
            asyncio.run(run_research_experiment(args.research))
        
        print(f"\n🎉 Enhanced AGI系統運行完成!")
        print(f"⏱️ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 用戶中斷，系統退出")
        return 0
    except Exception as e:
        print(f"\n❌ 系統錯誤: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 