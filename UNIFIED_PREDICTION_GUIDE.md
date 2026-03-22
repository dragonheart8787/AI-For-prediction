## 統一預測介面使用指南（CPU 友善）

### 安裝
- 基礎：`pip install numpy scikit-learn pyyaml`
- 可選：`pip install xgboost lightgbm onnxruntime skl2onnx`
- **AutoML + 深度模型**：`pip install -r requirements-automl.txt`（Optuna + PyTorch）

### 快速開始（單/多地平線）
```python
import numpy as np
from unified_predict import UnifiedPredictor

X = np.random.randn(1024, 16)
w = np.random.randn(16)
# 多地平線標籤 [N, H]
y = np.stack([
    X @ w + np.random.randn(1024)*0.1,
    X @ (w*0.5) + np.random.randn(1024)*0.15,
], axis=1)

p = UnifiedPredictor()
p.fit(X[:512], y[:512], model="linear")  # 可選: xgboost / lightgbm / ensemble / automl / mlp_torch / lstm / transformer
print(p.predict(X[:2], domain="financial"))
print(p.predict_many(X[:128], domain="financial", batch_size=64))
```

### ONNX Runtime 推論
```python
# 需先安裝 onnxruntime 與 skl2onnx
p.export_onnx("models/linear.onnx", input_dim=16)
p.load_onnx("models/linear.onnx", intra_threads=0, inter_threads=0)
print(p.predict(X[:4], domain="financial"))
```

### 自動 ONNX（auto_onnx）
- 自 vX.X 起，`UnifiedPredictor(auto_onnx=True)` 預設開啟：
  - 在 `fit()` 成功後，若環境已安裝 `onnxruntime` 與 `skl2onnx`，系統會自動：
    1) 匯出 ONNX 到 `models/auto_<model>.onnx`
    2) 自動切換為 ONNX Runtime（CPUExecutionProvider）執行推論
- 關閉方式：`UnifiedPredictor(auto_onnx=False)`
- 注意：若匯出或載入失敗，系統會安靜回退至原生模型推論，不會影響功能。

### 效能建議
- 使用 `predict_many(..., batch_size=1024)` 進行微批推論
- XGBoost/LightGBM + ONNX Runtime（CPU）：建議大批量 4096~8192，能顯著提升吞吐
- Sklearn 線性/樹模型：建議批量 1024~4096（依特徵數與 CPU 核心調整）
- ONNX Runtime：依 CPU 調整 `intra_threads`/`inter_threads`
- 多地平線：模型訓練使用 [N, H] 目標，推論輸出形狀為 [batch, H]

### 回測示例（跨領域、含自動 ONNX）
```bash
# 預設自動模型選擇與回退（xgboost -> lightgbm -> linear），Linux/Windows 通用
python demo_backtest_all.py

# 指定模型與批量大小（跨平台）：
python demo_backtest_all.py --model auto --batch 4096
python demo_backtest_all.py --model xgboost --batch 8192    # 建議高吞吐測試
python demo_backtest_all.py --model lightgbm --batch 4096
python demo_backtest_all.py --model linear --batch 2048
```

命令列參數：
- `--model {auto,xgboost,lightgbm,linear}`：預設 `auto`，會依序嘗試並在失敗時回退
- `--batch <int>`：`predict_many` 的批量大小，預設 4096

範例重點：
- 透過 `YahooFinanceConnector`、`EIAConnector`、`NewsAPIConnector` 取得金融/能源/新聞三源資料（有網路即線上、無網路則離線後備）。
- 自動推斷時間鍵並轉為特徵矩陣；建立合成連續標籤以便快速驗證。
- 使用：
```python
from unified_predict import UnifiedPredictor

p = UnifiedPredictor()           # 預設 auto_onnx=True
try:
    p.fit(X, y, model="xgboost")  # 環境具備時會自動匯出/切換 ONNX
except Exception:
    p.fit(X, y, model="linear")   # 回退策略
out = p.predict_many(X, domain="custom", batch_size=4096)
```

輸出鍵包含 `model`（實際使用模型，若為 ONNX 會顯示 onnx/xgboost_onnx 等）、`prediction`（[batch, H]）。大批量（4096~8192）在 ONNX Runtime CPU 上通常可得到最高吞吐；在 Linux 上可直接使用相同命令與批量配置。

輸出還包含：
- `batch`：實際推論批量（若指定批量大於樣本數，將自動截斷為樣本數）
- `onnx`：是否已切換為 ONNX Runtime 推論（true/false）

### 快取與限流
- 內建 LRU 快取（以模型名/領域/地平線鍵與輸入哈希組成）
- 內建簡易令牌桶限流，預設每秒補充 30，容量 60（可視需要修改原始碼）

### 回傳格式
- `domain`: 使用的領域 ID
- `horizons`: 對應領域的預設地平線（可覆寫）
- `model`: 實際使用的模型名稱（含 onnx）
- `prediction`: 二維陣列 [batch, H]
- `confidence`: 0~1 的簡易信心值（對每一維 std 取平均後裁切）

### AutoML、Torch 與近似不確定性

```python
# Optuna 自動選模組（需 optuna）；由爬蟲 CLI: --model automl --automl-trials 40
p.fit(X, y, model="automl", automl_trials=20, automl_include_deep=False)

# 手動 LSTM（需 torch）；訓練後 p._torch_feature_attr 為梯度近似特徵重要性（合 1）
p.fit(X, y, model="lstm", seq_len=12, torch_epochs=80, torch_lr=1e-3)

# 以訓練集 RMSE 當 σ 的粗估區間（非嚴格統計區間）
band = p.predict_interval_naive(X_test, z_score=1.96)
# band["lower"], band["upper"], band["prediction"]

# 選用：將 Torch 子網匯出 ONNX（可能因算子／版本失敗，回傳 bool）
# ok = p.export_torch_model_onnx("models/torch_task.onnx")
```

爬蟲管線對應參數：`--seq-len`、`--torch-epochs`、`--torch-hidden` 等；walk-forward 與深度模型預設不相容，需加 `--wf-allow-deep`。
