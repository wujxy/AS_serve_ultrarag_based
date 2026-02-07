#!/bin/bash
#
# start_qa_batch.sh - 启动 QA 批处理运行器
#
# 功能: 启动 vLLM 服务，对问题文件执行在线 RAG QA 批处理
#
# 使用方法:
#   ./scripts/start_qa_batch.sh [问题文件] [知识库目录] [选项]
#   ULTRARAG_ROOT=/path/to/ultrarag ./scripts/start_qa_batch.sh ...
#
# 示例:
#   ./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index
#   ./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index --keep_server
#   ULTRARAG_ROOT=/path/to/ultrarag ./scripts/start_qa_batch.sh /path/to/questions.jsonl /path/to/kb_dir
#

set -e

source config/config.sh

# 获取服务包根目录
SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# UltraRAG路径 - 必须通过环境变量指定
if [ -z "$ULTRARAG_ROOT" ]; then
    echo "[ERROR] ULTRARAG_ROOT 环境变量未设置！"
    echo "[INFO] 请设置 UltraRAG 项目根目录，例如："
    echo "  export ULTRARAG_ROOT=/path/to/UltraRAG"
    echo "  或: ULTRARAG_ROOT=/path/to/UltraRAG ./scripts/start_qa_batch.sh ..."
    exit 1
fi

# 转换为绝对路径
ULTRARAG_ROOT="$(cd "$ULTRARAG_ROOT" && pwd)"

# 工作目录（用于日志和状态文件）
WORK_DIR="${WORK_DIR:-$SERVICE_ROOT}"

# 基础参数
QUESTIONS_FILE="${1:-$SERVICE_ROOT/data/questions/test.jsonl}"  # 问题文件
KB_DIR="${2:-$SERVICE_ROOT/output/kb_index}" # 知识库目录
KEEP_SERVER=""

# 转换为绝对路径
if [[ ! "$QUESTIONS_FILE" = /* ]]; then
    QUESTIONS_FILE="$(pwd)/$QUESTIONS_FILE"
fi
if [[ ! "$KB_DIR" = /* ]]; then
    KB_DIR="$(pwd)/$KB_DIR"
fi

# 模型和服务配置
# 生成模型相关：
VLLM_SCRIPT="$SERVICE_ROOT/scripts/vllm_serve_llama.sh"
GENERATION_MODEL="${GENERATION_MODEL:-/home/NagaiYoru/LLM_model/Llama-3.2-3B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:65504/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3-3b-instruct}"

# 其他模型：(走sentence-transformers的模型需要指定路径)
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B}"
RERANKER_MODEL="${RERANKER_MODEL:-/home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light}"

# 默认配置
PIPELINE="pipelines/online_rag_qa_batch.yaml"  # QA 批处理流水线配置
PARAMETER_TEMPLATE="templates/online_rag_qa_batch_parameter.yaml.template" # 参数模板
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业}"

# 激活UltraRAG虚拟环境
if [ -f "$ULTRARAG_ROOT/.venv/bin/activate" ]; then
    source "$ULTRARAG_ROOT/.venv/bin/activate"
    echo "[INFO] Virtual environment activated"
fi

# 检查问题文件是否存在
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "[ERROR] 问题文件不存在: $QUESTIONS_FILE"
    echo "[INFO] 可以在 $SERVICE_ROOT/data/questions/ 目录下创建问题文件"
    exit 1
fi

# 检查知识库目录是否存在
if [ ! -d "$KB_DIR" ]; then
    echo "[ERROR] 知识库目录不存在: $KB_DIR"
    echo "[INFO] 请先运行 start_index_watch.sh 生成知识库"
    exit 1
fi

# 检查知识库文件
if [ ! -f "$KB_DIR/chunks.jsonl" ]; then
    echo "[ERROR] 知识库文件缺失: $KB_DIR/chunks.jsonl"
    exit 1
fi

if [ ! -f "$KB_DIR/index.index" ]; then
    echo "[ERROR] 知识库文件缺失: $KB_DIR/index.index"
    exit 1
fi

# 检查是否保持生成模型服务运行
if [[ "$3" == "--keep_server" ]]; then
    KEEP_SERVER="--keep_server"
fi

# 显示配置信息
echo "=========================================="
echo "  QA Batch Runner - 批处理问答服务"
echo "=========================================="
echo "[配置]"
echo "  服务包根目录:     $SERVICE_ROOT"
echo "  UltraRAG 根目录:   $ULTRARAG_ROOT"
echo "  工作目录:         $WORK_DIR"
echo "  问题文件:         $QUESTIONS_FILE"
echo "  知识库目录:       $KB_DIR/"
echo "    ├── chunks.jsonl"
echo "    └── index.index"
echo "  vLLM 脚本:        $VLLM_SCRIPT"
echo "  vLLM 地址:        $VLLM_BASE_URL"
if [ -n "$KEEP_SERVER" ]; then
    echo "  服务策略:         完成后保持 vLLM 运行"
else
    echo "  服务策略:         完成后自动关闭 vLLM"
fi
echo "=========================================="
echo ""

# 启动服务
python "$SERVICE_ROOT/auto_scripts/run_qa_batch.py" \
    --service_root "$SERVICE_ROOT" \
    --ultrarag_root "$ULTRARAG_ROOT" \
    --work_dir "$WORK_DIR" \
    --questions "$QUESTIONS_FILE" \
    --kb_dir "$KB_DIR" \
    --pipeline "$PIPELINE" \
    --parameter_template "$PARAMETER_TEMPLATE" \
    --vllm_script "$VLLM_SCRIPT" \
    --embedding_model "$EMBEDDING_MODEL" \
    --reranker_model "$RERANKER_MODEL" \
    --generation_model "$GENERATION_MODEL" \
    --vllm_base_url "$VLLM_BASE_URL" \
    --served_model_name "$SERVED_MODEL_NAME" \
    --system_prompt "$SYSTEM_PROMPT" \
    $KEEP_SERVER

echo ""
echo "=========================================="
echo "  QA 批处理完成！"
echo "=========================================="
