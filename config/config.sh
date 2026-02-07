#!/bin/bash
#
# UltraRAG Service Configuration
# Source this file to set up environment variables for the service package
#

# ============================================================================
# UltraRAG 项目根目录 (必须设置)
# ============================================================================
export ULTRARAG_ROOT="/home/NagaiYoru/LLM_tuning/UltraRAG"

# ============================================================================
# 模型路径配置
# ============================================================================
# Embedding 模型 (用于索引构建和检索)
export EMBEDDING_MODEL="/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B"

# Reranker 模型 (用于重排序)
export RERANKER_MODEL="/home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light"

# 生成模型 (用于 QA 生成)
export GENERATION_MODEL="/home/NagaiYoru/LLM_model/Llama-3.2-3B-Instruct"

# ============================================================================
# vLLM 服务配置
# ============================================================================
export VLLM_BASE_URL="http://127.0.0.1:65504/v1"
export VLLM_PORT="65504"
export VLLM_HOST="127.0.0.1"
export SERVED_MODEL_NAME="llama3-3b-instruct"

# ============================================================================
# 索引配置
# ============================================================================
export INDEX_BACKEND="faiss"  # 可选: faiss, milvus
export COLLECTION_NAME="wiki_auto"

# ============================================================================
# 系统提示词 (用于 QA 生成)
# ============================================================================
export SYSTEM_PROMPT="你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业"

# ============================================================================
# 使用方法:
# ============================================================================
# 1. 修改上述配置为您实际的路径
# 2. 加载配置: source config/config.sh
# 3. 运行服务:
#    - ./scripts/start_index_watch.sh data/raw_docs output/kb_index
#    - ./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index
# ============================================================================
