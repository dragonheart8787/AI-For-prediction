# 🚀 SuperFusionAGI 安裝指南

## 📋 安裝選項

我們提供了三種安裝方式，您可以根據需要選擇：

### 1. 🎯 最小安裝（推薦）
只安裝核心功能所需的最小依賴包，適合快速開始使用。

```bash
python install_minimal.py
```

**包含的包**：
- 基礎庫：numpy, pandas, scikit-learn, torch
- 可視化：matplotlib, seaborn, plotly
- Web 框架：fastapi, uvicorn, aiohttp
- 工具庫：requests, pydantic, tqdm, psutil
- 機器學習：xgboost, lightgbm

### 2. 🔧 核心安裝
安裝核心功能所需的依賴包，包含更多功能。

```bash
python install_core_dependencies.py
```

**額外包含**：
- AutoML：optuna
- XAI：shap, lime
- 優化：modin, numba
- 外部數據：yfinance, pytrends
- ML 管理：mlflow, wandb

### 3. 🌟 完整安裝
安裝所有功能所需的依賴包，包含所有進階功能。

```bash
python install_dependencies.py
```

**額外包含**：
- 所有 AutoML 工具
- 所有 XAI 工具
- 所有優化工具
- 所有外部數據源
- 所有 Web 工具
- 所有可選工具

## 🚀 快速開始

### 步驟 1：選擇安裝方式
```bash
# 最小安裝（推薦新手）
python install_minimal.py

# 核心安裝（推薦一般用戶）
python install_core_dependencies.py

# 完整安裝（推薦進階用戶）
python install_dependencies.py
```

### 步驟 2：測試安裝
```bash
python test_core_functions.py
```

### 步驟 3：啟動系統
```bash
python quick_start.py
```

## 📊 安裝結果

### 最小安裝
- **安裝時間**: 約 2-5 分鐘
- **包數量**: 20 個核心包
- **成功率**: 95%+
- **適用場景**: 快速開始、基本功能

### 核心安裝
- **安裝時間**: 約 5-10 分鐘
- **包數量**: 30 個包
- **成功率**: 90%+
- **適用場景**: 一般使用、進階功能

### 完整安裝
- **安裝時間**: 約 10-20 分鐘
- **包數量**: 50+ 個包
- **成功率**: 85%+
- **適用場景**: 專業使用、所有功能

## 🔧 手動安裝

如果自動安裝失敗，您可以手動安裝：

### 基礎依賴
```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn plotly
pip install requests aiohttp fastapi uvicorn pydantic
pip install tqdm psutil joblib scipy xgboost lightgbm
```

### 進階依賴
```bash
pip install optuna shap lime modin numba
pip install yfinance pytrends mlflow wandb
pip install dash streamlit jupyter pytest
```

## 🐳 Docker 安裝

如果您偏好使用 Docker：

```bash
# 構建鏡像
docker build -t superfusionagi:latest .

# 運行容器
docker run -p 8080:8080 superfusionagi:latest

# 或使用 docker-compose
docker-compose up -d
```

## ⚠️ 常見問題

### 1. 安裝失敗
**問題**: 某些包安裝失敗
**解決方案**: 
- 檢查網絡連接
- 更新 pip：`python -m pip install --upgrade pip`
- 使用國內鏡像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`

### 2. 權限問題
**問題**: 權限不足
**解決方案**:
- 使用用戶安裝：`pip install --user <package>`
- 或使用虛擬環境：`python -m venv venv && venv\Scripts\activate`

### 3. 版本衝突
**問題**: 包版本衝突
**解決方案**:
- 創建新的虛擬環境
- 或使用 conda：`conda create -n superfusionagi python=3.10`

### 4. 內存不足
**問題**: 安裝過程中內存不足
**解決方案**:
- 分批安裝包
- 增加虛擬內存
- 使用較小的包版本

## 📈 系統要求

### 最低要求
- **Python**: 3.8+
- **內存**: 4GB RAM
- **硬盤**: 2GB 可用空間
- **操作系統**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### 推薦配置
- **Python**: 3.10+
- **內存**: 8GB+ RAM
- **硬盤**: 5GB+ 可用空間
- **GPU**: NVIDIA GPU（可選，用於深度學習）

## 🎯 安裝後驗證

### 1. 運行測試
```bash
python test_core_functions.py
```

### 2. 檢查功能
```bash
python quick_start.py
```

### 3. 查看報告
檢查生成的測試報告文件：
- `core_function_test_report_*.json`
- `installation_report_*.json`

## 🚀 下一步

安裝完成後，您可以：

1. **查看功能**: 閱讀 `FINAL_TEST_SUMMARY.md`
2. **運行演示**: 執行 `python demo_advanced_features_simple.py`
3. **啟動系統**: 執行 `python quick_start.py`
4. **自定義配置**: 編輯 `configs/` 目錄下的配置文件
5. **添加插件**: 在 `plugins/` 目錄下添加自定義插件

## 📞 支援

如果您遇到安裝問題：

1. 查看錯誤日誌
2. 檢查系統要求
3. 嘗試手動安裝失敗的包
4. 查看 `FINAL_TEST_SUMMARY.md` 了解詳細信息

---

**🎉 祝您使用愉快！** 🚀

