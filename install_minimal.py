#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SuperFusionAGI 最小依賴安裝腳本
只安裝系統運行所需的最小依賴包
"""

import subprocess
import sys
import os
from datetime import datetime

def print_banner():
    """打印橫幅"""
    print("=" * 60)
    print("🚀 SuperFusionAGI 最小依賴安裝")
    print("📦 安裝系統運行所需的最小依賴包")
    print("=" * 60)
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def install_package(package):
    """安裝單個包"""
    try:
        print(f"📦 安裝 {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✅ {package} 安裝成功")
            return True
        else:
            print(f"❌ {package} 安裝失敗")
            return False
            
    except Exception as e:
        print(f"❌ {package} 安裝錯誤: {e}")
        return False

def test_imports():
    """測試導入"""
    print("\n🧪 測試導入...")
    
    test_packages = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("sklearn", "import sklearn"),
        ("torch", "import torch"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("requests", "import requests"),
        ("aiohttp", "import aiohttp"),
        ("fastapi", "import fastapi"),
        ("pydantic", "import pydantic")
    ]
    
    success_count = 0
    for package, import_cmd in test_packages:
        try:
            exec(import_cmd)
            print(f"✅ {package} 導入成功")
            success_count += 1
        except ImportError:
            print(f"❌ {package} 導入失敗")
    
    print(f"\n📊 導入測試: {success_count}/{len(test_packages)} 成功")
    return success_count >= len(test_packages) * 0.8

def main():
    """主函數"""
    print_banner()
    
    # 最小依賴包列表
    minimal_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "fastapi>=0.85.0",
        "uvicorn[standard]>=0.18.0",
        "pydantic>=1.10.0",
        "python-dotenv>=0.20.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0",
        "joblib>=1.1.0",
        "scipy>=1.9.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0"
    ]
    
    print(f"\n📦 開始安裝 {len(minimal_packages)} 個最小依賴包...")
    
    success_count = 0
    for package in minimal_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 安裝結果: {success_count}/{len(minimal_packages)} 成功")
    
    # 測試導入
    import_success = test_imports()
    
    print("\n🎉 最小依賴安裝完成！")
    print("=" * 60)
    
    if import_success:
        print("✅ 所有測試通過，系統可以正常使用！")
        print("\n🚀 下一步:")
        print("  1. 運行 python test_core_functions.py 測試功能")
        print("  2. 運行 python quick_start.py 啟動系統")
        return True
    else:
        print("⚠️ 部分測試未通過，但系統基本可用")
        print("🔧 建議手動安裝失敗的包")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

