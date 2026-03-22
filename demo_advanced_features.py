#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 進階功能整合演示
展示所有新增的 AI 能力
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import logging
import time
from datetime import datetime
import json
import os
import sys

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AdvancedFeaturesDemo:
    """進階功能演示器"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def print_banner(self):
        """打印橫幅"""
        print("=" * 80)
        print("🚀 SuperFusionAGI 進階功能演示")
        print("🔬 展示最新的 AI 技術整合")
        print("=" * 80)
        print(f"⏰ 開始時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def demo_automl_optimization(self):
        """演示 AutoML 超參數優化"""
        print("\n🔬 AutoML 超參數優化演示")
        print("-" * 50)
        
        try:
            from automl.hyperparameter_optimization import create_optimizer, PREDEFINED_PARAM_SPACES
            
            # 創建示例數據
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            y = pd.Series(y)
            
            # 創建優化器
            optimizer = create_optimizer("optuna", n_trials=20)
            
            # 優化 Random Forest
            print("🌲 優化 Random Forest...")
            rf_params = PREDEFINED_PARAM_SPACES['random_forest']
            rf_results = optimizer.optimize_sklearn_model(
                RandomForestRegressor, X, y, rf_params, cv=3
            )
            
            print(f"✅ Random Forest 最佳參數: {rf_results['best_params']}")
            print(f"✅ 最佳分數: {rf_results['best_score']:.4f}")
            
            # 優化 XGBoost
            print("\n🚀 優化 XGBoost...")
            xgb_params = PREDEFINED_PARAM_SPACES['xgboost']
            xgb_results = optimizer.optimize_sklearn_model(
                xgb.XGBRegressor, X, y, xgb_params, cv=3
            )
            
            print(f"✅ XGBoost 最佳參數: {xgb_results['best_params']}")
            print(f"✅ 最佳分數: {xgb_results['best_score']:.4f}")
            
            self.results['automl'] = {
                'random_forest': rf_results,
                'xgboost': xgb_results
            }
            
        except Exception as e:
            print(f"❌ AutoML 演示失敗: {e}")
            self.results['automl'] = {'error': str(e)}
    
    def demo_xai_explanations(self):
        """演示解釋型 AI"""
        print("\n🔍 解釋型 AI (XAI) 演示")
        print("-" * 50)
        
        try:
            from xai.explainable_ai import ExplainableAI
            
            # 創建示例數據和模型
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            feature_names = [f'feature_{i}' for i in range(10)]
            
            # 訓練模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 創建 XAI 分析器
            xai = ExplainableAI(model=model, feature_names=feature_names)
            
            # 設置解釋器
            xai.setup_shap_explainer(X, "tree")
            xai.setup_lime_explainer(X, "regression")
            
            # 獲取解釋
            print("📊 獲取 SHAP 解釋...")
            shap_explanations = xai.get_shap_explanations(X[:50])
            print(f"✅ SHAP 前5個重要特徵: {[f[0] for f in shap_explanations['top_features'][:5]]}")
            
            print("\n🍋 獲取 LIME 解釋...")
            lime_explanations = xai.get_lime_explanations(X[:50])
            print(f"✅ LIME 前5個重要特徵: {[f[0] for f in lime_explanations['top_features'][:5]]}")
            
            # 生成解釋報告
            y_pred = model.predict(X[:50])
            report_path = xai.generate_explanation_report(X[:50], y_pred)
            print(f"\n📊 解釋報告已生成: {report_path}")
            
            self.results['xai'] = {
                'shap_features': [f[0] for f in shap_explanations['top_features'][:5]],
                'lime_features': [f[0] for f in lime_explanations['top_features'][:5]],
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"❌ XAI 演示失敗: {e}")
            self.results['xai'] = {'error': str(e)}
    
    async def demo_multi_agent_system(self):
        """演示多代理系統"""
        print("\n🤝 多代理系統演示")
        print("-" * 50)
        
        try:
            from multi_agent.multi_agent_system import MultiAgentSystem, BaseAgent, AgentType
            
            # 創建多代理系統
            mas = MultiAgentSystem()
            await mas.start()
            
            # 創建示例代理
            financial_agent = BaseAgent("financial_001", AgentType.FINANCIAL)
            weather_agent = BaseAgent("weather_001", AgentType.WEATHER)
            medical_agent = BaseAgent("medical_001", AgentType.MEDICAL)
            
            # 添加代理到系統
            mas.add_agent(financial_agent)
            mas.add_agent(weather_agent)
            mas.add_agent(medical_agent)
            
            # 進行預測
            test_data = np.random.randn(10)
            
            print("📊 測試不同共識策略...")
            strategies = ["weighted_average", "confidence_weighted", "expert_opinion"]
            
            strategy_results = {}
            for strategy in strategies:
                print(f"\n🔍 使用 {strategy} 策略:")
                result = await mas.predict(test_data, consensus_strategy=strategy)
                print(f"預測結果: {result['prediction']}")
                print(f"置信度: {result['confidence']:.3f}")
                print(f"推理: {result['reasoning'][:100]}...")
                
                strategy_results[strategy] = {
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                }
            
            # 獲取系統狀態
            status = mas.get_status()
            print(f"\n📈 系統狀態: {status['total_agents']} 個代理，{status['active_agents']} 個活躍")
            
            await mas.stop()
            
            self.results['multi_agent'] = {
                'strategies': strategy_results,
                'system_status': status
            }
            
        except Exception as e:
            print(f"❌ 多代理系統演示失敗: {e}")
            self.results['multi_agent'] = {'error': str(e)}
    
    def demo_cpu_optimization(self):
        """演示 CPU 多核優化"""
        print("\n🖥️ CPU 多核優化演示")
        print("-" * 50)
        
        try:
            from cpu_optimization.cpu_multicore_optimization import CPUOptimizer
            
            # 創建示例數據
            X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
            
            # 創建 CPU 優化器
            optimizer = CPUOptimizer(n_jobs=-1)
            
            # 獲取 CPU 信息
            cpu_info = optimizer.get_cpu_utilization_info()
            print(f"CPU 核心數: {cpu_info['cpu_count']}")
            print(f"配置的並行數: {cpu_info['n_jobs_configured']}")
            
            # 測試 Numba 向量化操作
            print("\n🔢 測試 Numba 向量化操作...")
            numba_result = optimizer.numba_vectorized_operation(X[:, 0])
            print(f"✅ Numba 向量化操作完成，結果形狀: {numba_result.shape}")
            
            # 優化 sklearn 模型
            print("\n🌲 優化 sklearn 模型...")
            sklearn_results = optimizer.optimize_sklearn_models(X, y)
            for model_name, result in sklearn_results.items():
                print(f"{model_name}: 訓練時間 {result['training_time']:.2f}s, 分數 {result['score']:.4f}")
            
            # 優化 XGBoost
            print("\n🚀 優化 XGBoost...")
            xgb_results = optimizer.optimize_xgboost(X, y)
            print(f"XGBoost: 訓練時間 {xgb_results['training_time']:.2f}s, 分數 {xgb_results['score']:.4f}")
            
            # 測試批量推理
            print("\n📦 測試批量推理...")
            batch_predictions = optimizer.batch_inference(sklearn_results['random_forest']['model'], X[:1000], batch_size=64)
            print(f"✅ 批量推理完成，預測形狀: {batch_predictions.shape}")
            
            self.results['cpu_optimization'] = {
                'cpu_info': cpu_info,
                'sklearn_results': {k: {'time': v['training_time'], 'score': v['score']} for k, v in sklearn_results.items()},
                'xgboost_results': {'time': xgb_results['training_time'], 'score': xgb_results['score']},
                'batch_inference_shape': batch_predictions.shape
            }
            
        except Exception as e:
            print(f"❌ CPU 優化演示失敗: {e}")
            self.results['cpu_optimization'] = {'error': str(e)}
    
    def demo_external_data_fusion(self):
        """演示外部數據融合"""
        print("\n📊 外部數據融合演示")
        print("-" * 50)
        
        try:
            from external_indicators.external_data_fusion import ExternalDataFusion
            
            # 創建外部數據融合器
            fusion = ExternalDataFusion()
            
            # 獲取恐懼與貪婪指數
            print("😨 獲取恐懼與貪婪指數...")
            fng_data = fusion.get_fear_greed_index()
            if fng_data:
                print(f"✅ 恐懼與貪婪指數: {fng_data['value']} ({fng_data['classification']})")
            
            # 獲取 Google Trends 數據
            print("\n🔍 獲取 Google Trends 數據...")
            keywords = ['AI', 'machine learning', 'artificial intelligence']
            trends_data = fusion.get_google_trends(keywords)
            if not trends_data.empty:
                print(f"✅ Google Trends 數據獲取成功，共 {len(trends_data)} 條記錄")
                print(f"關鍵詞: {list(trends_data.columns)}")
            
            # 獲取經濟指標
            print("\n💰 獲取經濟指標...")
            economic_data = fusion.get_economic_indicators()
            if economic_data:
                print(f"✅ 經濟指標獲取成功，共 {len(economic_data)} 個指標")
                for indicator, data in list(economic_data.items())[:3]:  # 顯示前3個
                    print(f"  {indicator}: {data.get('value', 'N/A')}")
            
            # 融合所有外部數據
            print("\n🔄 融合所有外部數據...")
            fused_data = fusion.fuse_all_external_data(keywords)
            
            if not fused_data.empty:
                print(f"✅ 融合數據形狀: {fused_data.shape}")
                print(f"特徵列表: {list(fused_data.columns)}")
                
                # 計算特徵重要性
                print("\n🔍 計算特徵重要性...")
                feature_importance = fusion.get_feature_importance(fused_data)
                
                if not feature_importance.empty:
                    print("📊 前5個重要特徵:")
                    print(feature_importance.head())
                
                self.results['external_data_fusion'] = {
                    'fear_greed': fng_data,
                    'trends_data_shape': trends_data.shape if not trends_data.empty else 0,
                    'economic_indicators_count': len(economic_data),
                    'fused_data_shape': fused_data.shape,
                    'top_features': feature_importance.head(5).to_dict() if not feature_importance.empty else {}
                }
            else:
                print("❌ 沒有獲取到融合數據")
                self.results['external_data_fusion'] = {'error': 'No fused data'}
            
        except Exception as e:
            print(f"❌ 外部數據融合演示失敗: {e}")
            self.results['external_data_fusion'] = {'error': str(e)}
    
    def demo_integrated_prediction(self):
        """演示整合預測"""
        print("\n🎯 整合預測演示")
        print("-" * 50)
        
        try:
            # 創建示例數據
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 訓練多個模型
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
            }
            
            predictions = {}
            scores = {}
            
            print("🌲 訓練多個模型...")
            for name, model in models.items():
                print(f"  訓練 {name}...")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                
                predictions[name] = pred
                scores[name] = score
                print(f"    {name} 分數: {score:.4f}")
            
            # 集成預測
            print("\n🔄 集成預測...")
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_score = 1 - np.mean((ensemble_pred - y_test) ** 2) / np.var(y_test)
            
            print(f"✅ 集成預測分數: {ensemble_score:.4f}")
            
            # 計算預測統計
            pred_stats = {
                'individual_scores': scores,
                'ensemble_score': ensemble_score,
                'prediction_mean': np.mean(ensemble_pred),
                'prediction_std': np.std(ensemble_pred),
                'correlation_with_target': np.corrcoef(ensemble_pred, y_test)[0, 1]
            }
            
            print(f"📊 預測統計:")
            print(f"  平均預測值: {pred_stats['prediction_mean']:.4f}")
            print(f"  預測標準差: {pred_stats['prediction_std']:.4f}")
            print(f"  與目標相關性: {pred_stats['correlation_with_target']:.4f}")
            
            self.results['integrated_prediction'] = pred_stats
            
        except Exception as e:
            print(f"❌ 整合預測演示失敗: {e}")
            self.results['integrated_prediction'] = {'error': str(e)}
    
    def generate_demo_report(self):
        """生成演示報告"""
        print("\n📊 生成演示報告...")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'demo_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_features_demoed': len(self.results)
            },
            'results': self.results,
            'summary': {
                'successful_demos': len([r for r in self.results.values() if 'error' not in r]),
                'failed_demos': len([r for r in self.results.values() if 'error' in r]),
                'total_duration': f"{duration:.2f} 秒"
            }
        }
        
        # 保存報告
        report_file = f"advanced_features_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 演示報告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n📈 演示摘要:")
        print(f"  成功演示: {report['summary']['successful_demos']}")
        print(f"  失敗演示: {report['summary']['failed_demos']}")
        print(f"  總耗時: {report['summary']['total_duration']}")
        
        return report

async def main():
    """主函數"""
    demo = AdvancedFeaturesDemo()
    demo.print_banner()
    
    # 運行所有演示
    print("\n🚀 開始進階功能演示...")
    
    # 1. AutoML 超參數優化
    demo.demo_automl_optimization()
    
    # 2. 解釋型 AI
    demo.demo_xai_explanations()
    
    # 3. 多代理系統
    await demo.demo_multi_agent_system()
    
    # 4. CPU 多核優化
    demo.demo_cpu_optimization()
    
    # 5. 外部數據融合
    demo.demo_external_data_fusion()
    
    # 6. 整合預測
    demo.demo_integrated_prediction()
    
    # 生成報告
    report = demo.generate_demo_report()
    
    print("\n🎉 進階功能演示完成！")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
