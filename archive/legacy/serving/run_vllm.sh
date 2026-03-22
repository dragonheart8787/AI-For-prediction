#!/bin/bash
# vLLM 服務啟動腳本

# 設置基本參數
MODEL_PATH=${MODEL_PATH:-"./checkpoints/my_model"}
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-true}

# 檢查模型路徑
if [ ! -d "$MODEL_PATH" ]; then
    echo "錯誤: 模型路徑不存在: $MODEL_PATH"
    echo "請設置正確的 MODEL_PATH 環境變數"
    exit 1
fi

# 檢查 vLLM 是否安裝
if ! python -c "import vllm" 2>/dev/null; then
    echo "正在安裝 vLLM..."
    pip install vllm
fi

echo "啟動 vLLM 服務..."
echo "模型路徑: $MODEL_PATH"
echo "端口: $PORT"
echo "主機: $HOST"

# 啟動 vLLM 服務
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code "$TRUST_REMOTE_CODE" \
    --served-model-name "$(basename "$MODEL_PATH")" \
    --api-key "your-api-key-here" \
    --disable-log-requests

echo "vLLM 服務已啟動"
echo "API 端點: http://$HOST:$PORT"
echo "文檔: http://$HOST:$PORT/docs"
