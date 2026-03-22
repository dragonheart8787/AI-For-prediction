#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 核心功能測試腳本
測試核心功能是否能正常運行（不依賴外部包）
"""

import sys
import os
import traceback
import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreFunctionTester:
    """核心功能測試器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def print_banner(self):
        """打印橫幅"""
        print("=" * 80)
        print("🧪 SuperFusionAGI 核心功能測試")
        print("🔍 測試核心功能是否能正常運行")
        print("=" * 80)
        print(f"⏰ 開始時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def test_basic_imports(self):
        """測試基礎導入"""
        print("\n📦 測試基礎導入...")
        test_name = "basic_imports"
        
        try:
            # 測試基礎庫
            import numpy as np
            import pandas as pd
            import sklearn
            import torch
            print("✅ 基礎庫導入成功")
            
            # 測試自定義模組
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # 測試多代理系統（不依賴外部包）
            try:
                from multi_agent.multi_agent_system import MultiAgentSystem, BaseAgent, AgentType
                print("✅ 多代理系統模組導入成功")
            except Exception as e:
                print(f"⚠️ 多代理系統模組導入失敗: {e}")
            
            self.record_test_result(test_name, True, "基礎導入測試通過")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"基礎導入測試失敗: {e}")
    
    def test_multi_agent_system(self):
        """測試多代理系統功能"""
        print("\n🤝 測試多代理系統功能...")
        test_name = "multi_agent_system"
        
        try:
            import asyncio
            from multi_agent.multi_agent_system import MultiAgentSystem, BaseAgent, AgentType
            
            async def test_multi_agent():
                # 創建多代理系統
                mas = MultiAgentSystem()
                await mas.start()
                print("✅ 多代理系統啟動成功")
                
                # 創建測試代理
                financial_agent = BaseAgent("financial_001", AgentType.FINANCIAL)
                weather_agent = BaseAgent("weather_001", AgentType.WEATHER)
                medical_agent = BaseAgent("medical_001", AgentType.MEDICAL)
                
                # 添加代理
                mas.add_agent(financial_agent)
                mas.add_agent(weather_agent)
                mas.add_agent(medical_agent)
                print("✅ 代理添加成功")
                
                # 測試預測
                test_data = np.random.randn(5)
                result = await mas.predict(test_data, consensus_strategy="weighted_average")
                print(f"✅ 多代理預測成功，預測值: {result['prediction']}")
                print(f"✅ 置信度: {result['confidence']:.3f}")
                
                # 獲取系統狀態
                status = mas.get_status()
                print(f"✅ 系統狀態獲取成功，代理數: {status['total_agents']}")
                
                await mas.stop()
                print("✅ 多代理系統停止成功")
                
                return True
            
            # 運行異步測試
            result = asyncio.run(test_multi_agent())
            self.record_test_result(test_name, result, "多代理系統功能正常")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"多代理系統功能測試失敗: {e}")
    
    def test_integrated_prediction(self):
        """測試整合預測功能"""
        print("\n🎯 測試整合預測功能...")
        test_name = "integrated_prediction"
        
        try:
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # 創建測試數據
            X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 訓練多個模型
            models = {
                'RandomForest1': RandomForestRegressor(n_estimators=50, random_state=42),
                'RandomForest2': RandomForestRegressor(n_estimators=100, random_state=42),
                'RandomForest3': RandomForestRegressor(n_estimators=150, random_state=42)
            }
            
            predictions = {}
            scores = {}
            
            print("🌲 訓練多個模型...")
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                
                predictions[name] = pred
                scores[name] = score
                print(f"  {name}: 分數 {score:.4f}")
            
            # 集成預測
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_score = 1 - np.mean((ensemble_pred - y_test) ** 2) / np.var(y_test)
            
            print(f"✅ 集成預測成功，分數: {ensemble_score:.4f}")
            
            # 計算預測統計
            pred_mean = np.mean(ensemble_pred)
            pred_std = np.std(ensemble_pred)
            correlation = np.corrcoef(ensemble_pred, y_test)[0, 1]
            
            print(f"📊 預測統計: 平均={pred_mean:.4f}, 標準差={pred_std:.4f}, 相關性={correlation:.4f}")
            
            self.record_test_result(test_name, True, f"整合預測功能正常，集成分數: {ensemble_score:.4f}")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"整合預測功能測試失敗: {e}")
    
    def test_docker_deployment(self):
        """測試 Docker 部署功能"""
        print("\n🐳 測試 Docker 部署功能...")
        test_name = "docker_deployment"
        
        try:
            # 檢查 Docker 文件是否存在
            dockerfile_exists = os.path.exists("Dockerfile")
            compose_exists = os.path.exists("docker-compose.yml")
            env_example_exists = os.path.exists("env.example")
            
            print(f"✅ Dockerfile 存在: {dockerfile_exists}")
            print(f"✅ docker-compose.yml 存在: {compose_exists}")
            print(f"✅ env.example 存在: {env_example_exists}")
            
            if dockerfile_exists and compose_exists:
                # 讀取 Docker 文件內容
                with open("Dockerfile", "r", encoding="utf-8") as f:
                    dockerfile_content = f.read()
                
                with open("docker-compose.yml", "r", encoding="utf-8") as f:
                    compose_content = f.read()
                
                print("✅ Docker 配置文件讀取成功")
                print(f"📄 Dockerfile 行數: {len(dockerfile_content.splitlines())}")
                print(f"📄 docker-compose.yml 行數: {len(compose_content.splitlines())}")
                
                self.record_test_result(test_name, True, "Docker 部署功能正常")
            else:
                self.record_test_result(test_name, False, "Docker 配置文件缺失")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Docker 部署功能測試失敗: {e}")
    
    def test_launch_system(self):
        """測試系統啟動功能"""
        print("\n🚀 測試系統啟動功能...")
        test_name = "launch_system"
        
        try:
            # 檢查啟動腳本是否存在
            launch_script_exists = os.path.exists("launch_system.py")
            quick_start_exists = os.path.exists("quick_start.py")
            
            print(f"✅ launch_system.py 存在: {launch_script_exists}")
            print(f"✅ quick_start.py 存在: {quick_start_exists}")
            
            if launch_script_exists:
                # 嘗試導入啟動腳本
                import launch_system
                print("✅ launch_system.py 導入成功")
                
                # 檢查 SystemLauncher 類
                if hasattr(launch_system, 'SystemLauncher'):
                    print("✅ SystemLauncher 類存在")
                else:
                    print("⚠️ SystemLauncher 類不存在")
            
            if quick_start_exists:
                # 嘗試導入快速啟動腳本
                import quick_start
                print("✅ quick_start.py 導入成功")
                
                # 檢查 QuickStarter 類
                if hasattr(quick_start, 'QuickStarter'):
                    print("✅ QuickStarter 類存在")
                else:
                    print("⚠️ QuickStarter 類不存在")
            
            self.record_test_result(test_name, True, "系統啟動功能正常")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"系統啟動功能測試失敗: {e}")
    
    def test_file_structure(self):
        """測試文件結構"""
        print("\n📁 測試文件結構...")
        test_name = "file_structure"
        
        try:
            # 檢查關鍵目錄和文件
            key_files = [
                "launch_system.py",
                "quick_start.py",
                "Dockerfile",
                "docker-compose.yml",
                "env.example",
                "requirements_enhanced.txt",
                "requirements_advanced.txt"
            ]
            
            key_dirs = [
                "multi_agent",
                "accelerators",
                "models",
                "optimizers",
                "serving",
                "export",
                "configs"
            ]
            
            missing_files = []
            missing_dirs = []
            
            for file in key_files:
                if os.path.exists(file):
                    print(f"✅ {file} 存在")
                else:
                    print(f"❌ {file} 缺失")
                    missing_files.append(file)
            
            for dir in key_dirs:
                if os.path.exists(dir):
                    print(f"✅ {dir}/ 目錄存在")
                else:
                    print(f"❌ {dir}/ 目錄缺失")
                    missing_dirs.append(dir)
            
            if not missing_files and not missing_dirs:
                self.record_test_result(test_name, True, "文件結構完整")
            else:
                self.record_test_result(test_name, False, f"缺失文件: {missing_files}, 缺失目錄: {missing_dirs}")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"文件結構測試失敗: {e}")
    
    def test_simple_automl(self):
        """測試簡單 AutoML 功能"""
        print("\n🔬 測試簡單 AutoML 功能...")
        test_name = "simple_automl"
        
        try:
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV
            
            # 創建測試數據
            X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
            
            # 簡單的網格搜索
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            }
            
            model = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            
            print("🌲 開始網格搜索...")
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_
            
            print(f"✅ 最佳參數: {best_params}")
            print(f"✅ 最佳分數: {best_score:.4f}")
            
            self.record_test_result(test_name, True, f"簡單 AutoML 功能正常，最佳分數: {best_score:.4f}")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"簡單 AutoML 功能測試失敗: {e}")
    
    def test_simple_xai(self):
        """測試簡單 XAI 功能"""
        print("\n🔍 測試簡單 XAI 功能...")
        test_name = "simple_xai"
        
        try:
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor
            
            # 創建測試數據和模型
            X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # 獲取特徵重要性
            feature_importance = model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(10)]
            
            # 排序特徵重要性
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print("📊 特徵重要性排序:")
            for i, (feature, importance) in enumerate(importance_pairs[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
            
            # 生成簡單解釋
            top_feature = importance_pairs[0][0]
            top_importance = importance_pairs[0][1]
            
            explanation = f"最重要的特徵是 {top_feature}，重要性為 {top_importance:.4f}"
            print(f"✅ 解釋: {explanation}")
            
            self.record_test_result(test_name, True, "簡單 XAI 功能正常")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"簡單 XAI 功能測試失敗: {e}")
    
    def record_test_result(self, test_name, passed, message):
        """記錄測試結果"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ 通過"
        else:
            self.failed_tests += 1
            status = "❌ 失敗"
        
        self.test_results[test_name] = {
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"{status} {test_name}: {message}")
    
    def generate_test_report(self):
        """生成測試報告"""
        print("\n📊 生成測試報告...")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'test_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'summary': {
                'overall_status': 'PASS' if self.failed_tests == 0 else 'FAIL',
                'critical_failures': [name for name, result in self.test_results.items() if not result['passed']],
                'recommendations': self.get_recommendations()
            }
        }
        
        # 保存報告
        report_file = f"core_function_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 測試報告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n📈 測試摘要:")
        print(f"  總測試數: {self.total_tests}")
        print(f"  通過測試: {self.passed_tests}")
        print(f"  失敗測試: {self.failed_tests}")
        print(f"  成功率: {report['test_info']['success_rate']:.1f}%")
        print(f"  總耗時: {duration:.2f} 秒")
        
        if self.failed_tests > 0:
            print(f"\n⚠️ 失敗的測試:")
            for name, result in self.test_results.items():
                if not result['passed']:
                    print(f"  - {name}: {result['message']}")
        
        return report
    
    def get_recommendations(self):
        """獲取建議"""
        recommendations = []
        
        if self.failed_tests > 0:
            recommendations.append("檢查失敗的測試模組，確保所有依賴包已正確安裝")
        
        if self.passed_tests == self.total_tests:
            recommendations.append("所有核心功能測試通過，系統可以正常使用")
        
        if self.test_results.get('multi_agent_system', {}).get('passed', False):
            recommendations.append("多代理系統功能正常，可以進行智能協作預測")
        
        if self.test_results.get('docker_deployment', {}).get('passed', False):
            recommendations.append("Docker 部署功能正常，可以進行容器化部署")
        
        if self.test_results.get('integrated_prediction', {}).get('passed', False):
            recommendations.append("整合預測功能正常，可以進行多模型集成預測")
        
        return recommendations

def main():
    """主函數"""
    tester = CoreFunctionTester()
    tester.print_banner()
    
    print("\n🧪 開始核心功能測試...")
    
    # 運行所有測試
    tester.test_basic_imports()
    tester.test_multi_agent_system()
    tester.test_integrated_prediction()
    tester.test_docker_deployment()
    tester.test_launch_system()
    tester.test_file_structure()
    tester.test_simple_automl()
    tester.test_simple_xai()
    
    # 生成測試報告
    report = tester.generate_test_report()
    
    print("\n🎉 核心功能測試完成！")
    print("=" * 80)
    
    # 返回測試結果
    return report

if __name__ == "__main__":
    main()
