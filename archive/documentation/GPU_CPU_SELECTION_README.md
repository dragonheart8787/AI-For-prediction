# 🚀 GPU/CPU選擇功能使用說明

## 📖 概述

超級增強版時間序列預測系統現在支持**GPU加速**和**CPU/GPU選擇**功能！你可以根據自己的需求和硬件配置，靈活選擇使用GPU或CPU進行計算。

## ✨ 主要特性

- 🔍 **自動檢測**: 自動檢測可用的GPU後端
- 🎯 **靈活選擇**: 支持PyTorch、TensorFlow、CuPy、Numba等多種GPU後端
- 🔄 **動態切換**: 運行時可以動態切換計算設備
- 📊 **實時監控**: 監控GPU記憶體使用情況
- ⚡ **性能優化**: 自動優化GPU設置和記憶體管理
- 🛡️ **智能回退**: 當GPU不可用時自動回退到CPU

## 🎮 使用方法

### 方法1: 使用圖形化選擇器 (推薦)

運行GPU/CPU選擇器：

```bash
python gpu_cpu_selector.py
```

選擇器提供以下功能：

1. **🔍 檢測可用的計算設備** - 查看系統支持的GPU後端
2. **🖥️ 使用CPU模式** - 強制使用CPU進行計算
3. **🚀 使用GPU模式** - 自動選擇最佳GPU後端
4. **🎯 選擇特定GPU後端** - 手動選擇特定的GPU後端
5. **🔄 切換計算設備** - 運行時切換設備
6. **📊 顯示當前狀態** - 查看系統狀態和GPU記憶體
7. **🧪 運行測試預測** - 測試當前配置的性能

### 方法2: 程序化配置

在Python代碼中直接配置：

```python
from super_enhanced_ts_system import SuperEnhancedTSSystem

# 強制使用CPU
system = SuperEnhancedTSSystem(force_cpu=True)

# 強制使用GPU (自動選擇)
system = SuperEnhancedTSSystem(force_gpu=True)

# 指定GPU後端偏好
system = SuperEnhancedTSSystem(gpu_preference='pytorch')

# 動態切換設備
system.switch_computation_device('cpu')  # 切換到CPU
system.switch_computation_device('gpu')  # 切換到GPU
system.switch_computation_device('gpu', 'tensorflow')  # 切換到特定後端
```

### 方法3: 配置文件

編輯 `gpu_cpu_config.json` 文件來預設配置：

```json
{
    "computation_device": {
        "default_mode": "auto",
        "force_cpu": false,
        "force_gpu": false,
        "gpu_preference": "pytorch"
    }
}
```

## 🔧 支持的GPU後端

### 1. PyTorch GPU
- **優勢**: 深度學習最佳支持，CUDA優化
- **適用**: 神經網絡、Transformer模型
- **安裝**: `pip install torch torchvision torchaudio`

### 2. TensorFlow GPU
- **優勢**: 生產環境穩定，多GPU支持
- **適用**: 大型模型訓練、部署
- **安裝**: `pip install tensorflow`

### 3. CuPy GPU
- **優勢**: NumPy兼容，科學計算快速
- **適用**: 數值計算、特徵工程
- **安裝**: `pip install cupy-cuda11x` (根據CUDA版本)

### 4. Numba CUDA
- **優勢**: Python代碼直接編譯，易於使用
- **適用**: 自定義算法、性能關鍵代碼
- **安裝**: `pip install numba`

## 📊 性能對比

| 任務類型 | CPU | GPU | 加速比 |
|---------|-----|-----|--------|
| 短期預測 | 1x | 3-5x | 3-5x |
| 中期預測 | 1x | 5-10x | 5-10x |
| 長期預測 | 1x | 8-15x | 8-15x |
| 高頻交易 | 1x | 10-20x | 10-20x |
| 風險管理 | 1x | 5-8x | 5-8x |

*註: 實際加速比取決於數據大小、模型複雜度和硬件配置*

## 🚀 最佳實踐

### 1. 選擇合適的計算設備

- **小數據集 (< 10K樣本)**: CPU足夠
- **中等數據集 (10K-100K樣本)**: GPU推薦
- **大數據集 (> 100K樣本)**: GPU必需
- **實時預測**: GPU優先考慮

### 2. GPU後端選擇建議

- **深度學習**: PyTorch > TensorFlow
- **數值計算**: CuPy > Numba
- **通用用途**: PyTorch (最穩定)
- **生產環境**: TensorFlow (最成熟)

### 3. 記憶體管理

```python
# 清理GPU記憶體
system.gpu_config.clear_memory()

# 監控記憶體使用
memory_info = system.gpu_config.get_memory_info()
print(f"GPU記憶體使用: {memory_info['allocated']:.2f} GB")
```

### 4. 錯誤處理

```python
try:
    system = SuperEnhancedTSSystem(force_gpu=True)
except Exception as e:
    print(f"GPU初始化失敗: {e}")
    # 自動回退到CPU
    system = SuperEnhancedTSSystem(force_cpu=True)
```

## 🔍 故障排除

### 常見問題

1. **GPU不可用**
   - 檢查CUDA驅動是否安裝
   - 確認GPU驅動版本
   - 檢查CUDA版本兼容性

2. **記憶體不足**
   - 減少batch_size
   - 啟用記憶體增長
   - 使用混合精度訓練

3. **性能不理想**
   - 檢查GPU利用率
   - 優化數據加載
   - 調整模型參數

### 調試命令

```python
# 檢查設備狀態
status = system.get_device_status()
print(status)

# 檢查可用後端
backends = system.gpu_config.get_available_backends()
print(backends)

# 運行性能測試
await system.run_super_enhanced_system(['short_term_forecast'])
```

## 📈 性能監控

系統自動記錄以下指標：

- GPU記憶體使用情況
- 計算時間對比
- 模型訓練速度
- 預測準確性

查看日誌文件 `super_enhanced_ts.log` 獲取詳細信息。

## 🎯 進階配置

### 環境變量

```bash
# 限制GPU記憶體使用
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# TensorFlow GPU設置
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 自定義CUDA路徑

```json
{
    "advanced": {
        "custom_cuda_path": "/usr/local/cuda-11.8"
    }
}
```

## 🤝 貢獻

如果你發現問題或有改進建議，請：

1. 檢查現有issues
2. 創建新的issue
3. 提交pull request

## 📞 支持

- 📧 郵箱: support@example.com
- 📖 文檔: [完整文檔](https://example.com/docs)
- 🐛 問題報告: [GitHub Issues](https://github.com/example/issues)

---

**�� 享受GPU加速帶來的性能提升！**
