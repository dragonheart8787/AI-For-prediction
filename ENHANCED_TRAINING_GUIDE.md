# 🚀 增強訓練系統使用指南

本指南介紹如何使用整合了編譯器、VLM、優化器、分散式訓練等功能的增強訓練系統。

## 📋 系統概述

增強訓練系統包含以下核心功能：

- 🧱 **編譯器加速**: torch.compile、OpenXLA、TVM 支援
- 🖼️ **VLM 支援**: 視覺語言模型訓練
- 🔁 **增強優化器**: SGD/SGDM 等優化器及調度策略
- 🌐 **分散式訓練**: DDP/FSDP 支援
- ⚡ **推論服務**: vLLM/SGLang/TensorRT-LLM 整合
- 🖥️ **CPU 最佳化**: Intel IPEX、OpenVINO、ONNX Runtime

## 🛠️ 安裝依賴

```bash
# 安裝基本依賴
pip install -r requirements_enhanced.txt

# 可選：Intel 最佳化（需要 Intel 硬體）
pip install intel-extension-for-pytorch

# 可選：OpenVINO 最佳化
pip install openvino optimum[openvino]

# 可選：推論服務
pip install vllm sglang
```

## 🎯 快速開始

### 1. 基本訓練

```bash
# 使用預設配置訓練
python train_enhanced.py --config configs/train.yaml

# 指定訓練輪數
python train_enhanced.py --config configs/train.yaml --epochs 50

# 訓練後匯出模型
python train_enhanced.py --config configs/train.yaml --export
```

### 2. 查看訓練信息

```bash
# 顯示系統配置和最佳化信息
python train_enhanced.py --config configs/train.yaml --info
```

### 3. 分散式訓練

```bash
# 單機多卡
torchrun --standalone --nproc_per_node=4 train_enhanced.py --config configs/train.yaml

# 多機多卡
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train_enhanced.py --config configs/train.yaml
```

## ⚙️ 配置說明

### 編譯器設定

```yaml
compiler:
  enable_torch_compile: true
  torch_compile_backend: "inductor"  # inductor, aot_eager, nvfuser
  torch_compile_mode: "max-autotune"  # default, reduce-overhead, max-autotune
  torch_compile_fullgraph: true
  use_channels_last: true
```

### AMP 設定

```yaml
amp:
  enable_amp: true
  dtype: "auto"  # auto, bf16, fp16, fp32
  prefer_bf16: true
  use_autocast: true
```

### 優化器設定

```yaml
optimizer:
  optimizer_type: "adamw"  # adamw, sgd, sgd_momentum, adagrad, rmsprop
  learning_rate: 2e-4
  weight_decay: 1e-4
  momentum: 0.9

scheduler:
  scheduler_type: "warmup_cosine"  # cosine, step, exponential, plateau, warmup_cosine
  warmup_epochs: 5
  total_epochs: 100
```

### 分散式訓練設定

```yaml
distributed:
  enable_ddp: false
  enable_fsdp: false
  backend: "nccl"
  fsdp_min_num_params: 1000000
  fsdp_sharding_strategy: "FULL_SHARD"
```

### VLM 設定

```yaml
vlm:
  vision_encoder_type: "resnet"  # resnet, vit, clip
  text_encoder_type: "transformer"  # transformer, bert
  fusion_type: "cross_attention"  # cross_attention, concat, mlp
  vision_dim: 2048
  text_dim: 768
  hidden_dim: 1024
```

## 🔧 模組化使用

### 編譯器模組

```python
from accelerators.compile import CompilerManager, create_compiler_config

# 創建編譯器配置
config = create_compiler_config(
    enable_torch_compile=True,
    backend="inductor",
    mode="max-autotune"
)

# 編譯模型
compiler_manager = CompilerManager(config)
optimized_model = compiler_manager.compile_model(model, "my_model")
```

### AMP 模組

```python
from accelerators.amp import AMPManager, create_amp_config

# 創建 AMP 配置
config = create_amp_config(
    enable_amp=True,
    dtype="auto",
    prefer_bf16=True
)

# 使用 AMP
amp_manager = AMPManager(config)

with amp_manager.autocast():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

scaled_loss = amp_manager.scale_loss(loss)
scaled_loss.backward()
amp_manager.step_optimizer(optimizer)
```

### 分散式訓練模組

```python
from accelerators.ddp import DistributedManager, create_distributed_config

# 創建分散式配置
config = create_distributed_config(
    enable_ddp=True,
    backend="nccl"
)

# 初始化分散式環境
distributed_manager = DistributedManager(config)
distributed_manager.initialize()

# 包裝模型
model = distributed_manager.wrap_model(model)
```

### VLM 模組

```python
from models.vlm_models import VLM, create_vlm_config

# 創建 VLM 配置
config = create_vlm_config(
    vision_encoder_type="resnet",
    text_encoder_type="transformer",
    fusion_type="cross_attention"
)

# 創建 VLM 模型
model = VLM(config)
```

### 優化器模組

```python
from optimizers.enhanced_optimizers import OptimizerFactory

# 創建優化器
optimizer = OptimizerFactory.create_optimizer(
    model,
    optimizer_type="sgd_momentum",
    learning_rate=1e-2,
    scheduler_type="warmup_cosine"
)
```

## 🚀 推論服務

### vLLM 服務

```bash
# 啟動 vLLM 服務
bash serving/run_vllm.sh

# 或手動啟動
python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/my_model \
    --port 8000
```

### SGLang 服務

```bash
# 啟動 SGLang 服務
bash serving/run_sglang.sh

# 或手動啟動
python -m sglang.launch_server \
    --model ./checkpoints/my_model \
    --port 30000
```

### TensorRT-LLM 服務

```bash
# 啟動 TensorRT-LLM 服務
bash serving/run_trtllm.sh
```

## 🖥️ CPU 最佳化

### Intel IPEX 最佳化

```python
from accelerators.cpu_optimization import CPUOptimizer, create_cpu_optimization_config

# 創建 CPU 最佳化配置
config = create_cpu_optimization_config(
    enable_cpu_optimization=True,
    enable_ipex=True,
    num_threads=16
)

# 最佳化模型
optimizer = CPUOptimizer(config)
optimized_model = optimizer.optimize_model(model)
```

### OpenVINO 最佳化

```python
from accelerators.cpu_optimization import OpenVINOManager

# 轉換模型為 OpenVINO 格式
ov_manager = OpenVINOManager(precision="INT8")
ov_manager.convert_model("./checkpoints/my_model", "./ov_model")

# 載入 OpenVINO 模型
model, tokenizer = ov_manager.load_model("./ov_model")
```

## 📊 監控和日誌

### TensorBoard 監控

```bash
# 啟動 TensorBoard
tensorboard --logdir ./runs
```

### 訓練日誌

訓練日誌會自動保存到 `./logs/training.log`，包含：
- 訓練進度
- 損失和指標
- 學習率變化
- 系統狀態

## 🔍 故障排除

### 常見問題

1. **編譯器錯誤**
   - 確保 PyTorch 版本 >= 2.0
   - 檢查 CUDA 版本兼容性

2. **分散式訓練問題**
   - 檢查 NCCL 環境變數
   - 確保網路連接正常

3. **記憶體不足**
   - 啟用 FSDP
   - 減少批次大小
   - 使用梯度累積

4. **CPU 最佳化問題**
   - 檢查 Intel 硬體支援
   - 確認 IPEX 安裝正確

### 性能調優

1. **編譯器調優**
   - 嘗試不同的 backend 和 mode
   - 調整 fullgraph 設定

2. **AMP 調優**
   - 根據硬體選擇最佳精度
   - 調整梯度縮放參數

3. **分散式調優**
   - 選擇合適的分片策略
   - 調整批次大小和執行緒數

## 📚 進階用法

### 自定義模型

```python
from train_enhanced import EnhancedTrainer

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的模型定義
    
    def forward(self, x):
        # 你的前向傳播
        return output

# 在 EnhancedTrainer 中替換 _create_standard_model 方法
```

### 自定義數據加載器

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 在訓練腳本中使用
```

## 🎉 總結

增強訓練系統提供了完整的 AI 模型訓練解決方案，整合了最新的編譯器技術、多模態模型支援、分散式訓練和推論服務。通過模組化設計，您可以根據需求靈活配置和使用各個組件。

開始使用增強訓練系統，體驗更高效、更靈活的 AI 模型訓練！
