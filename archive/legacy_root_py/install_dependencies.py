#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SuperFusionAGI 一鍵安裝腳本
自動安裝所有必要的依賴包
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

class DependencyInstaller:
    """依賴包安裝器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.installed_packages = []
        self.failed_packages = []
        self.total_packages = 0
        
    def print_banner(self):
        """打印橫幅"""
        print("=" * 80)
        print("🚀 SuperFusionAGI 一鍵安裝腳本")
        print("📦 自動安裝所有必要的依賴包")
        print("=" * 80)
        print(f"⏰ 開始時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def check_pip(self):
        """檢查 pip 是否可用"""
        try:
            import pip
            print("✅ pip 已安裝")
            return True
        except ImportError:
            print("❌ pip 未安裝，正在安裝...")
            try:
                subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
                print("✅ pip 安裝成功")
                return True
            except subprocess.CalledProcessError:
                print("❌ pip 安裝失敗")
                return False
    
    def upgrade_pip(self):
        """升級 pip 到最新版本"""
        try:
            print("🔄 升級 pip...")
            subprocess.check_call([sys.executable, "-m", pip, "install", "--upgrade", "pip"])
            print("✅ pip 升級成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ pip 升級失敗: {e}")
            return False
    
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
                timeout=300  # 5分鐘超時
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
    
    def install_basic_dependencies(self):
        """安裝基礎依賴包"""
        print("\n📦 安裝基礎依賴包...")
        print("-" * 50)
        
        basic_packages = [
            ("numpy>=1.21.0", "數值計算庫"),
            ("pandas>=1.3.0", "數據處理庫"),
            ("scikit-learn>=1.0.0", "機器學習庫"),
            ("torch>=1.12.0", "深度學習框架"),
            ("torchvision>=0.13.0", "計算機視覺庫"),
            ("torchaudio>=0.12.0", "音頻處理庫"),
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
            ("statsmodels>=0.13.0", "統計模型庫")
        ]
        
        for package, description in basic_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)  # 避免安裝過快
    
    def install_ml_dependencies(self):
        """安裝機器學習依賴包"""
        print("\n🤖 安裝機器學習依賴包...")
        print("-" * 50)
        
        ml_packages = [
            ("xgboost>=1.6.0", "梯度提升庫"),
            ("lightgbm>=3.3.0", "輕量級梯度提升庫"),
            ("catboost>=1.1.0", "CatBoost 梯度提升庫"),
            ("transformers>=4.20.0", "Transformer 模型庫"),
            ("accelerate>=0.20.0", "模型加速庫"),
            ("datasets>=2.0.0", "數據集庫"),
            ("tokenizers>=0.13.0", "文本分詞庫"),
            ("sentence-transformers>=2.2.0", "句子嵌入庫"),
            ("nltk>=3.7", "自然語言處理庫"),
            ("spacy>=3.4.0", "高級自然語言處理庫")
        ]
        
        for package, description in ml_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)
    
    def install_automl_dependencies(self):
        """安裝 AutoML 依賴包"""
        print("\n🔬 安裝 AutoML 依賴包...")
        print("-" * 50)
        
        automl_packages = [
            ("optuna>=3.0.0", "超參數優化庫"),
            ("hyperopt>=0.2.7", "貝葉斯優化庫"),
            ("bayesian-optimization>=1.4.0", "貝葉斯優化庫"),
            ("ray[tune]>=2.0.0", "分散式超參數搜索"),
            ("mlflow>=2.0.0", "ML 生命週期管理"),
            ("wandb>=0.13.0", "實驗追蹤庫"),
            ("neptune>=1.0.0", "ML 實驗管理"),
            ("dvc>=2.10.0", "數據版本控制")
        ]
        
        for package, description in automl_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(2)  # AutoML 包通常較大，增加等待時間
    
    def install_xai_dependencies(self):
        """安裝 XAI 依賴包"""
        print("\n🔍 安裝 XAI 依賴包...")
        print("-" * 50)
        
        xai_packages = [
            ("shap>=0.41.0", "SHAP 解釋庫"),
            ("lime>=0.2.0.1", "LIME 解釋庫"),
            ("captum>=0.6.0", "PyTorch 解釋庫"),
            ("alibi>=0.6.0", "模型解釋庫"),
            ("interpret>=0.4.0", "可解釋 AI 庫"),
            ("eli5>=0.13.0", "機器學習解釋庫")
        ]
        
        for package, description in xai_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)
    
    def install_optimization_dependencies(self):
        """安裝優化依賴包"""
        print("\n⚡ 安裝優化依賴包...")
        print("-" * 50)
        
        optimization_packages = [
            ("modin[ray]>=0.20.0", "多核 Pandas 加速"),
            ("numba>=0.56.0", "JIT 編譯器"),
            ("intel-extension-for-pytorch>=2.0.0", "Intel PyTorch 擴展"),
            ("openvino>=2023.0.0", "OpenVINO 推理引擎"),
            ("onnxruntime>=1.15.0", "ONNX 運行時"),
            ("onnx>=1.13.0", "ONNX 模型格式"),
            ("mkl>=2023.0.0", "Intel MKL 數學庫")
        ]
        
        for package, description in optimization_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(2)  # 優化包通常較大
    
    def install_external_data_dependencies(self):
        """安裝外部數據依賴包"""
        print("\n📊 安裝外部數據依賴包...")
        print("-" * 50)
        
        external_packages = [
            ("yfinance>=0.2.0", "Yahoo Finance 數據"),
            ("pytrends>=4.9.0", "Google Trends 數據"),
            ("beautifulsoup4>=4.11.0", "網頁解析庫"),
            ("selenium>=4.8.0", "網頁自動化庫"),
            ("feedparser>=6.0.0", "RSS 解析庫"),
            ("tweepy>=4.12.0", "Twitter API 庫"),
            ("praw>=7.6.0", "Reddit API 庫"),
            ("alpha-vantage>=2.3.0", "Alpha Vantage API"),
            ("quandl>=3.7.0", "Quandl 數據庫"),
            ("fredapi>=0.5.0", "FRED 經濟數據")
        ]
        
        for package, description in external_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)
    
    def install_web_dependencies(self):
        """安裝 Web 相關依賴包"""
        print("\n🌐 安裝 Web 相關依賴包...")
        print("-" * 50)
        
        web_packages = [
            ("dash>=2.8.0", "Dash Web 應用框架"),
            ("dash-bootstrap-components>=1.2.0", "Dash Bootstrap 組件"),
            ("streamlit>=1.15.0", "Streamlit Web 應用"),
            ("flask>=2.2.0", "Flask Web 框架"),
            ("jinja2>=3.1.0", "模板引擎"),
            ("gunicorn>=20.1.0", "WSGI 服務器"),
            ("celery>=5.2.0", "分散式任務隊列"),
            ("redis>=4.3.0", "Redis 客戶端"),
            ("sqlalchemy>=1.4.0", "SQL 工具包"),
            ("alembic>=1.8.0", "數據庫遷移工具")
        ]
        
        for package, description in web_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)
    
    def install_optional_dependencies(self):
        """安裝可選依賴包"""
        print("\n🔧 安裝可選依賴包...")
        print("-" * 50)
        
        optional_packages = [
            ("jupyter>=1.0.0", "Jupyter Notebook"),
            ("notebook>=6.4.0", "Jupyter Notebook 服務器"),
            ("ipython>=8.5.0", "IPython 交互式環境"),
            ("pytest>=7.2.0", "測試框架"),
            ("pytest-asyncio>=0.20.0", "異步測試支持"),
            ("black>=22.0.0", "代碼格式化工具"),
            ("flake8>=5.0.0", "代碼檢查工具"),
            ("mypy>=0.991", "類型檢查工具"),
            ("sphinx>=5.1.0", "文檔生成工具"),
            ("mkdocs>=1.4.0", "文檔網站生成器")
        ]
        
        for package, description in optional_packages:
            self.total_packages += 1
            self.install_package(package, description)
            time.sleep(1)
    
    def install_from_requirements(self, requirements_file):
        """從 requirements 文件安裝"""
        if os.path.exists(requirements_file):
            print(f"\n📄 從 {requirements_file} 安裝...")
            print("-" * 50)
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10分鐘超時
                )
                
                if result.returncode == 0:
                    print(f"✅ {requirements_file} 安裝成功")
                    return True
                else:
                    print(f"❌ {requirements_file} 安裝失敗: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {requirements_file} 安裝超時")
                return False
            except Exception as e:
                print(f"❌ {requirements_file} 安裝錯誤: {e}")
                return False
        else:
            print(f"⚠️ {requirements_file} 文件不存在")
            return False
    
    def test_installation(self):
        """測試安裝結果"""
        print("\n🧪 測試安裝結果...")
        print("-" * 50)
        
        test_imports = [
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
        total_imports = len(test_imports)
        
        for package, import_statement in test_imports:
            try:
                exec(import_statement)
                print(f"✅ {package} 導入成功")
                successful_imports += 1
            except ImportError as e:
                print(f"❌ {package} 導入失敗: {e}")
        
        success_rate = (successful_imports / total_imports) * 100
        print(f"\n📊 導入測試結果: {successful_imports}/{total_imports} ({success_rate:.1f}%)")
        
        return success_rate >= 80  # 80% 以上認為成功
    
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
                'total_packages': self.total_packages,
                'installed_packages': len(self.installed_packages),
                'failed_packages': len(self.failed_packages)
            },
            'installed_packages': self.installed_packages,
            'failed_packages': self.failed_packages,
            'summary': {
                'success_rate': (len(self.installed_packages) / self.total_packages * 100) if self.total_packages > 0 else 0,
                'recommendations': self.get_recommendations()
            }
        }
        
        # 保存報告
        report_file = f"installation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 安裝報告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n📈 安裝摘要:")
        print(f"  總包數: {self.total_packages}")
        print(f"  成功安裝: {len(self.installed_packages)}")
        print(f"  安裝失敗: {len(self.failed_packages)}")
        print(f"  成功率: {report['summary']['success_rate']:.1f}%")
        print(f"  總耗時: {duration:.2f} 秒")
        
        if self.failed_packages:
            print(f"\n⚠️ 安裝失敗的包:")
            for package in self.failed_packages:
                print(f"  - {package}")
        
        return report
    
    def get_recommendations(self):
        """獲取建議"""
        recommendations = []
        
        if len(self.installed_packages) == self.total_packages:
            recommendations.append("所有依賴包安裝成功，系統可以正常使用")
        elif len(self.installed_packages) > self.total_packages * 0.8:
            recommendations.append("大部分依賴包安裝成功，系統基本可用")
        else:
            recommendations.append("部分依賴包安裝失敗，建議手動安裝失敗的包")
        
        if self.failed_packages:
            recommendations.append("可以嘗試手動安裝失敗的包：pip install <package_name>")
        
        recommendations.append("運行 python test_core_functions.py 測試系統功能")
        recommendations.append("運行 python quick_start.py 啟動系統")
        
        return recommendations

def main():
    """主函數"""
    installer = DependencyInstaller()
    installer.print_banner()
    
    # 檢查 pip
    if not installer.check_pip():
        print("❌ pip 檢查失敗，無法繼續安裝")
        return False
    
    # 升級 pip
    installer.upgrade_pip()
    
    print("\n🚀 開始安裝依賴包...")
    
    # 安裝基礎依賴
    installer.install_basic_dependencies()
    
    # 安裝機器學習依賴
    installer.install_ml_dependencies()
    
    # 安裝 AutoML 依賴
    installer.install_automl_dependencies()
    
    # 安裝 XAI 依賴
    installer.install_xai_dependencies()
    
    # 安裝優化依賴
    installer.install_optimization_dependencies()
    
    # 安裝外部數據依賴
    installer.install_external_data_dependencies()
    
    # 安裝 Web 依賴
    installer.install_web_dependencies()
    
    # 安裝可選依賴
    installer.install_optional_dependencies()
    
    # 從 requirements 文件安裝
    installer.install_from_requirements("requirements_enhanced.txt")
    installer.install_from_requirements("requirements_advanced.txt")
    
    # 測試安裝結果
    test_success = installer.test_installation()
    
    # 生成安裝報告
    report = installer.generate_installation_report()
    
    print("\n🎉 依賴包安裝完成！")
    print("=" * 80)
    
    if test_success:
        print("✅ 系統測試通過，可以正常使用！")
        print("\n🚀 下一步:")
        print("  1. 運行 python test_core_functions.py 測試功能")
        print("  2. 運行 python quick_start.py 啟動系統")
        print("  3. 查看 FINAL_TEST_SUMMARY.md 了解詳細功能")
    else:
        print("⚠️ 系統測試未完全通過，請檢查失敗的包")
        print("\n🔧 建議:")
        print("  1. 手動安裝失敗的包")
        print("  2. 重新運行測試")
        print("  3. 查看安裝報告了解詳細信息")
    
    return test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

