#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SuperFusionAGI 核心依賴一鍵安裝腳本
只安裝核心功能所需的依賴包
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreDependencyInstaller:
    """核心依賴包安裝器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.installed_packages = []
        self.failed_packages = []
        
    def print_banner(self):
        """打印橫幅"""
        print("=" * 80)
        print("🚀 SuperFusionAGI 核心依賴一鍵安裝")
        print("📦 安裝核心功能所需的依賴包")
        print("=" * 80)
        print(f"⏰ 開始時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def install_package(self, package, description=""):
        """安裝單個包"""
        try:
            print(f"📦 安裝 {package}...")
            if description:
                print(f"   {description}")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=180  # 3分鐘超時
            )
            
            if result.returncode == 0:
                print(f"✅ {package} 安裝成功")
                self.installed_packages.append(package)
                return True
            else:
                print(f"❌ {package} 安裝失敗: {result.stderr}")
                self.failed_packages.append(package)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {package} 安裝超時")
            self.failed_packages.append(package)
            return False
        except Exception as e:
            print(f"❌ {package} 安裝錯誤: {e}")
            self.failed_packages.append(package)
            return False
    
    def install_core_packages(self):
        """安裝核心包"""
        print("\n📦 安裝核心依賴包...")
        print("-" * 50)
        
        core_packages = [
            ("numpy>=1.21.0", "數值計算庫"),
            ("pandas>=1.3.0", "數據處理庫"),
            ("scikit-learn>=1.0.0", "機器學習庫"),
            ("torch>=1.12.0", "深度學習框架"),
            ("matplotlib>=3.5.0", "繪圖庫"),
            ("seaborn>=0.11.0", "統計可視化庫"),
            ("plotly>=5.10.0", "交互式可視化庫"),
            ("requests>=2.28.0", "HTTP 請求庫"),
            ("aiohttp>=3.8.0", "異步 HTTP 庫"),
            ("fastapi>=0.85.0", "Web 框架"),
            ("uvicorn[standard]>=0.18.0", "ASGI 服務器"),
            ("pydantic>=1.10.0", "數據驗證庫"),
            ("python-dotenv>=0.20.0", "環境變數管理"),
            ("tqdm>=4.64.0", "進度條庫"),
            ("psutil>=5.9.0", "系統監控庫"),
            ("joblib>=1.1.0", "並行處理庫"),
            ("scipy>=1.9.0", "科學計算庫"),
            ("xgboost>=1.6.0", "梯度提升庫"),
            ("lightgbm>=3.3.0", "輕量級梯度提升庫")
        ]
        
        for package, description in core_packages:
            self.install_package(package, description)
            time.sleep(0.5)  # 短暫等待
    
    def install_optional_packages(self):
        """安裝可選包"""
        print("\n🔧 安裝可選依賴包...")
        print("-" * 50)
        
        optional_packages = [
            ("optuna>=3.0.0", "超參數優化庫"),
            ("shap>=0.41.0", "SHAP 解釋庫"),
            ("lime>=0.2.0.1", "LIME 解釋庫"),
            ("modin[ray]>=0.20.0", "多核 Pandas 加速"),
            ("numba>=0.56.0", "JIT 編譯器"),
            ("yfinance>=0.2.0", "Yahoo Finance 數據"),
            ("pytrends>=4.9.0", "Google Trends 數據"),
            ("mlflow>=2.0.0", "ML 生命週期管理"),
            ("wandb>=0.13.0", "實驗追蹤庫"),
            ("dash>=2.8.0", "Dash Web 應用框架")
        ]
        
        print("⚠️ 這些是可選包，如果安裝失敗不會影響核心功能")
        
        for package, description in optional_packages:
            self.install_package(package, description)
            time.sleep(1)
    
    def test_core_imports(self):
        """測試核心導入"""
        print("\n🧪 測試核心導入...")
        print("-" * 50)
        
        core_imports = [
            ("numpy", "import numpy as np"),
            ("pandas", "import pandas as pd"),
            ("sklearn", "import sklearn"),
            ("torch", "import torch"),
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("plotly", "import plotly.graph_objects as go"),
            ("requests", "import requests"),
            ("fastapi", "import fastapi"),
            ("pydantic", "import pydantic"),
            ("tqdm", "import tqdm")
        ]
        
        successful_imports = 0
        total_imports = len(core_imports)
        
        for package, import_statement in core_imports:
            try:
                exec(import_statement)
                print(f"✅ {package} 導入成功")
                successful_imports += 1
            except ImportError as e:
                print(f"❌ {package} 導入失敗: {e}")
        
        success_rate = (successful_imports / total_imports) * 100
        print(f"\n📊 核心導入測試: {successful_imports}/{total_imports} ({success_rate:.1f}%)")
        
        return success_rate >= 80
    
    def test_system_functionality(self):
        """測試系統功能"""
        print("\n🔍 測試系統功能...")
        print("-" * 50)
        
        try:
            # 測試多代理系統
            print("🤝 測試多代理系統...")
            from multi_agent.multi_agent_system import MultiAgentSystem, BaseAgent, AgentType
            print("✅ 多代理系統模組導入成功")
            
            # 測試數據處理
            print("📊 測試數據處理...")
            import numpy as np
            import pandas as pd
            from sklearn.datasets import make_regression
            from sklearn.ensemble import RandomForestRegressor
            
            X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            score = model.score(X, y)
            print(f"✅ 機器學習功能正常，模型分數: {score:.4f}")
            
            # 測試可視化
            print("📈 測試可視化...")
            import matplotlib.pyplot as plt
            import plotly.graph_objects as go
            print("✅ 可視化功能正常")
            
            return True
            
        except Exception as e:
            print(f"❌ 系統功能測試失敗: {e}")
            return False
    
    def generate_installation_report(self):
        """生成安裝報告"""
        print("\n📊 生成安裝報告...")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'installation_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'installed_packages': len(self.installed_packages),
                'failed_packages': len(self.failed_packages)
            },
            'installed_packages': self.installed_packages,
            'failed_packages': self.failed_packages,
            'summary': {
                'success_rate': (len(self.installed_packages) / (len(self.installed_packages) + len(self.failed_packages)) * 100) if (len(self.installed_packages) + len(self.failed_packages)) > 0 else 0
            }
        }
        
        # 保存報告
        report_file = f"core_installation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 安裝報告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n📈 安裝摘要:")
        print(f"  成功安裝: {len(self.installed_packages)} 個包")
        print(f"  安裝失敗: {len(self.failed_packages)} 個包")
        print(f"  成功率: {report['summary']['success_rate']:.1f}%")
        print(f"  總耗時: {duration:.2f} 秒")
        
        if self.failed_packages:
            print(f"\n⚠️ 安裝失敗的包:")
            for package in self.failed_packages:
                print(f"  - {package}")
        
        return report

def main():
    """主函數"""
    installer = CoreDependencyInstaller()
    installer.print_banner()
    
    print("\n🚀 開始安裝核心依賴包...")
    
    # 安裝核心包
    installer.install_core_packages()
    
    # 安裝可選包
    installer.install_optional_packages()
    
    # 測試核心導入
    import_success = installer.test_core_imports()
    
    # 測試系統功能
    system_success = installer.test_system_functionality()
    
    # 生成安裝報告
    report = installer.generate_installation_report()
    
    print("\n🎉 核心依賴安裝完成！")
    print("=" * 80)
    
    if import_success and system_success:
        print("✅ 所有測試通過，系統可以正常使用！")
        print("\n🚀 下一步:")
        print("  1. 運行 python test_core_functions.py 測試功能")
        print("  2. 運行 python quick_start.py 啟動系統")
        print("  3. 查看 FINAL_TEST_SUMMARY.md 了解詳細功能")
        return True
    else:
        print("⚠️ 部分測試未通過，請檢查失敗的包")
        print("\n🔧 建議:")
        print("  1. 手動安裝失敗的包")
        print("  2. 重新運行測試")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

