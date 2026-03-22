# 🔧 編譯器、VLM 與高效部署實務指南

本指南彙整 AI 編譯器、生產級推論引擎、分散式部署與 CPU/GPU 最佳化的實用做法，並提供可直接貼用的程式與命令。

---

## 🧱 Compiler（編譯器，AI 相關）

- **PyTorch 2.x `torch.compile` / TorchInductor**：最小程式碼改動做 JIT/AOT 編譯，對 NVIDIA/AMD GPU 以 OpenAI Triton 產生高效 kernel。適合訓練與推論通吃的圖層融合/動態形狀優化。
  - 參考：PyTorch Documentation（自行搜尋官方文件）
- **OpenXLA / XLA（含 StableHLO / MLIR）**：跨框架（JAX、TF、PyTorch）與跨硬體的 ML 編譯器；以 MLIR 方言層層 Lowering。適合需要多硬體後端與生態整合的團隊。
  - 參考：OpenXLA Project（官方入口）
- **Apache TVM**：模型輸入→IR→各硬體後端產碼的開源編譯框架，擅長端邊緣與異質硬體部署。
  - 參考：tvm.apache.org

何時用？
- 想要一鍵提速：先試 `torch.compile`
- 跨框架/自訂後端或端邊緣：看 XLA/TVM
- 需要極限優化熱點：用 Triton 自寫 kernel

---

## 🖼️ VLM（Vision-Language Model）

能同時理解視覺與文字（如 LLaVA、GPT-4V、IDEFICS、Qwen-VL），支援圖文問答、看圖指令等。2025 綜述涵蓋主流架構、訓練目標、效能與安全議題。

何時用？
- 需要看圖下指令、看截圖做操作、或多模態檢索/理解場景時。

---

## 🔁 SG（多半指 SGD/Stochastic Gradient 一系）

常見優化器（SGD/SGDM），用隨機子批次近似真梯度，效率高、記憶體友善。

何時用？
- 需要穩健、簡潔、易調參的基線優化器；搭配動量/餘弦退火等排程常有不錯表現。

---

## 🌐 LAN（區域網路，對部署很關鍵）

在機房/學校/公司內把多台 GPU 伺服器串起來做分散式推論與伸縮。理解 OSI/TCP/IP 分層有助排查延遲、MTU、DNS、L4/L7 負載均衡等。

何時用？
- 多機多 GPU、K8s 叢集、或要把前端服務與後端推論節點拆開時（例：SGLang/vLLM 分散式部署的連線設定）。

---

## ⬛ Tensor（張量）

DL 的基本資料結構（N-D 陣列）。編譯器/執行期會盡量把 Tensor 運算融合為更少、更大的 GPU kernel 以提升吞吐（如算子融合、記憶體規劃）。

---

## ⚡ TensorRT-LLM（TRT-LLM）

NVIDIA 的 LLM/VLM 高效推論工具鏈，建立在 CUDA + TensorRT 上，支援量化（如 INT8/FP8）、高效注意力（Paged Attention 等）、KV 快取與多 GPU。

何時用？
- 在 NVIDIA GPU 上追求極致吞吐/低延遲（特別是 H100/Blackwell），能接受較窄模型支援與較深的 NVIDIA 綁定。

---

## 🚀 vLLM & SGLang（開源 LLM/VLM 服務引擎）

- **vLLM**：以 PagedAttention 等技巧實現高吞吐與記憶體效率，社群/模型支援度高；已進入 PyTorch Foundation 生態。
- **SGLang**：後端執行期 + 前端語言共設計，支持推測解碼、MoE/EP 並行，強調低延遲與可控互動；多後端探索活躍。

何時用？
- 最快上、模型支援廣：先試 vLLM
- 追求極低延遲/特殊佈局或 AMD/NPU 等多後端探索：試 SGLang
- 極限 NVIDIA 單卡/多卡吞吐：考慮 TensorRT-LLM

---

## 🛠️ 一次接上「訓練→加速→匯出→服務」範例

### A. 在原訓練程式最小改動就能快很多

```python
# train.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def build_model():
    model = YourModel()
    model = model.to(memory_format=torch.channels_last)
    return model

def main(args):
    use_ddp = args.ddp
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    # 1) 編譯加速
    model = torch.compile(model, backend="inductor", mode="max-autotune", fullgraph=True)

    # 2) AMP（自動混合精度）
    bf16_ok = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16))

    # 3) 優化器與資料
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataset = YourDataset(...)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=os.cpu_count()//2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if use_ddp:
        model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)

    model.train()
    for epoch in range(args.epochs):
        if use_ddp and sampler: sampler.set_epoch(epoch)
        for batch in loader:
            batch = to_device(batch, device, memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                loss = model_forward_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype==torch.float16:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

    # 匯出（HF 風格）
    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(args.out_dir)
    # 如有 tokenizer：tokenizer.save_pretrained(args.out_dir)

# 其餘 util 省略（與您提供內容一致）
```

啟動（單機多卡）

```bash
torchrun --standalone --nproc_per_node=8 train.py --ddp --bs 8
```

跨機（LAN 兩台各 8 卡，node0 做 rendezvous）

```bash
# node0
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train.py --ddp
# node1
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 train.py --ddp
```

建議 NCCL 環境（依網卡/IB 調整）：
- `NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_IB_TIMEOUT=22`
- `NCCL_DEBUG=INFO`（除錯時開）

### B. 超大模型：FSDP 記憶體切分

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

auto_wrap = size_based_auto_wrap_policy(min_num_params=1_000_000)
model = FSDP(
    model.to(device),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap,
    use_orig_params=True,
)
# 儲存全量權重（便於 vLLM / TRT-LLM）
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig()):
    torch.save(model.state_dict(), f"{args.out_dir}/full_state_dict.pt")
```

### C. 直接把「訓練好的模型」接到推論服務

- **vLLM（最快上線）**
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model ./checkpoints/my_model --port 8000
```

- **SGLang（追極低延遲/可控互動）**
```bash
pip install "sglang[all]"
python -m sglang.launch_server --model ./checkpoints/my_model --port 30000
```

- **TensorRT-LLM（NVIDIA 極致效能）**
```bash
# HF 權重 → 轉圖與建 engine（示意）
python tools/export_hf.py --src ./checkpoints/my_model --dst ./trtllm_workspace
python tools/build.py --workspace ./trtllm_workspace --dtype fp16
python -m tensorrt_llm.runtime --engine_dir ./trtllm_workspace/build --port 9000
```

---

## 🖥️ 只有 CPU（訓練或推論）怎麼跑

- 建議組合：`PyTorch CPU` + `OpenVINO(Optimum-Intel)` 或 `ONNX Runtime` 或 `llama.cpp (GGUF)`。
- Intel 平台：`IPEX` + `oneDNN`（BF16/AMX）可顯著提速。

最小可行範例（CPU 訓練/推論開關）

```python
import torch
# 設定執行緒
torch.set_num_threads(16)
torch.set_num_interop_threads(2)

model = YourModel()
try:
    import intel_extension_for_pytorch as ipex
    model.eval()
    model = ipex.optimize(model, dtype=torch.bfloat16 if torch.cpu.is_available() else torch.float32)
except Exception:
    pass
```

OpenVINO（最快把 HF 模型搬到 CPU 推論）

```python
pip install "optimum[openvino,nncf]" openvino
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("你的HF模型")
ov_model = OVModelForCausalLM.from_pretrained(
    "你的HF模型", export=True, compile=True,
    ov_config={"INFERENCE_PRECISION": "INT8"}
)
out = ov_model.generate(**tok("hello", return_tensors="pt"))
print(tok.decode(out[0]))
```

ONNX Runtime（通用 CPU）

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 16
so.inter_op_num_threads = 2
sess = ort.InferenceSession("model.onnx", sess_options=so, providers=["CPUExecutionProvider"])
pred = sess.run(None, {"input_ids": ids, "attention_mask": mask})
```

llama.cpp（LLM 純 CPU）

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -S . && cmake --build build -j
./build/bin/llama-quantize model-f32.gguf model-q5_k_m.gguf Q5_K_M
./build/bin/llama-server -m model-q5_k_m.gguf -t 16 -c 4096
```

CPU 跑更快的小訣竅：
- 精度：優先 BF16（支援時）或 INT8（PTQ）
- 執行緒/親和性：合理設定 intra/inter-op 與必要時的 NUMA/affinity
- 動態量化（PyTorch）對 Linear 層很有效
- I/O：多進程 DataLoader、pin memory、裁切序列長度

---

## 📁 建議目錄骨架（可插拔模組）

```
your_ai_project/
├─ train.py                         # 上面範例：compile + AMP + (D)DP
├─ models/                          # 你的模型/VLM/LLM
├─ data/
├─ accelerators/
│   ├─ compile.py                   # 封裝 torch.compile 開關
│   ├─ amp.py                       # AMP / BF16 控制
│   ├─ ddp.py                       # DDP/FSDP 初始化、儲存/載入
├─ export/
│   ├─ to_hf.py                     # 匯出為 HF 目錄
│   ├─ to_trtllm.py                 # 產 TRT-LLM workspace/engine
├─ serving/
│   ├─ run_vllm.sh                  # vLLM 服務腳本
│   ├─ run_sglang.sh                # SGLang 服務腳本
│   └─ run_trtllm.sh                # TRT-LLM 服務腳本
└─ configs/
    └─ train.yaml                   # lr/bs/精度/是否 compile/DDP/FSDP/導出選項
```
