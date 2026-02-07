#!/bin/bash
#
# vllm_serve_llama.sh - 启动 vLLM 生成模型服务
#
# 环境变量:
#   ULTRARAG_ROOT  - UltraRAG 项目根目录 (必须)
#   GENERATION_MODEL - 生成模型路径 (可选)
#   VLLM_PORT - vLLM 服务端口 (可选，默认 65504)
#

if [ -z "$ULTRARAG_ROOT" ]; then
    echo "[ERROR] ULTRARAG_ROOT 环境变量未设置！"
    exit 1
fi

# 激活 UltraRAG 虚拟环境
if [ -f "$ULTRARAG_ROOT/.venv/bin/activate" ]; then
    source "$ULTRARAG_ROOT/.venv/bin/activate"
else
    echo "[ERROR] UltraRAG 虚拟环境不存在: $ULTRARAG_ROOT/.venv"
    exit 1
fi

# 配置参数
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3-3b-instruct}"
GENERATION_MODEL="${GENERATION_MODEL:-/home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct}"
VLLM_PORT="${VLLM_PORT:-65504}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

echo "[INFO] Starting vLLM service..."
echo "  Model: $GENERATION_MODEL"
echo "  Host: $VLLM_HOST"
echo "  Port: $VLLM_PORT"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --served-model-name "$SERVED_MODEL_NAME" \
    --model "$GENERATION_MODEL" \
    --trust-remote-code \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --enforce-eager
