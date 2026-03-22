#!/bin/bash
# TensorRT-LLM 服務啟動腳本

# 設置基本參數
MODEL_PATH=${MODEL_PATH:-"./checkpoints/my_model"}
TRTLLM_WORKSPACE=${TRTLLM_WORKSPACE:-"./trtllm_workspace"}
PORT=${PORT:-9000}
HOST=${HOST:-"0.0.0.0"}
DTYPE=${DTYPE:-"fp16"}  # fp16, fp8, int8

# 檢查模型路徑
if [ ! -d "$MODEL_PATH" ]; then
    echo "錯誤: 模型路徑不存在: $MODEL_PATH"
    echo "請設置正確的 MODEL_PATH 環境變數"
    exit 1
fi

# 檢查 TensorRT-LLM 是否安裝
if ! python -c "import tensorrt_llm" 2>/dev/null; then
    echo "錯誤: TensorRT-LLM 未安裝"
    echo "請參考官方文檔安裝 TensorRT-LLM"
    exit 1
fi

echo "開始 TensorRT-LLM 工作流程..."

# 步驟 1: 匯出 HF 模型到 TRT-LLM 格式
echo "步驟 1: 匯出模型..."
python tools/export_hf.py \
    --src "$MODEL_PATH" \
    --dst "$TRTLLM_WORKSPACE" \
    --dtype "$DTYPE"

if [ $? -ne 0 ]; then
    echo "錯誤: 模型匯出失敗"
    exit 1
fi

# 步驟 2: 建構 TRT-LLM engine
echo "步驟 2: 建構 engine..."
python tools/build.py \
    --workspace "$TRTLLM_WORKSPACE" \
    --dtype "$DTYPE" \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 1024

if [ $? -ne 0 ]; then
    echo "錯誤: Engine 建構失敗"
    exit 1
fi

# 步驟 3: 啟動 TRT-LLM 服務
echo "步驟 3: 啟動服務..."
python -m tensorrt_llm.runtime \
    --engine_dir "$TRTLLM_WORKSPACE/build" \
    --port "$PORT" \
    --host "$HOST" \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 1024

echo "TensorRT-LLM 服務已啟動"
echo "API 端點: http://$HOST:$PORT"
