#!/bin/bash
# SGLang 服務啟動腳本

# 設置基本參數
MODEL_PATH=${MODEL_PATH:-"./checkpoints/my_model"}
PORT=${PORT:-30000}
HOST=${HOST:-"0.0.0.0"}
MEMORY_FRACTION=${MEMORY_FRACTION:-0.9}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# 檢查模型路徑
if [ ! -d "$MODEL_PATH" ]; then
    echo "錯誤: 模型路徑不存在: $MODEL_PATH"
    echo "請設置正確的 MODEL_PATH 環境變數"
    exit 1
fi

# 檢查 SGLang 是否安裝
if ! python -c "import sglang" 2>/dev/null; then
    echo "正在安裝 SGLang..."
    pip install "sglang[all]"
fi

echo "啟動 SGLang 服務..."
echo "模型路徑: $MODEL_PATH"
echo "端口: $PORT"
echo "主機: $HOST"

# 啟動 SGLang 服務
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --mem-fraction-static "$MEMORY_FRACTION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-log-stats

echo "SGLang 服務已啟動"
echo "API 端點: http://$HOST:$PORT"
