#!/bin/bash
#
# start_index_watch.sh - 启动 Index 自动更新服务
#
# 功能: 监控指定目录下的新 PDF 文件，自动触发索引构建
#
# 使用方法:
#   ./scripts/start_index_watch.sh [PDF目录] [输出目录] [延迟时间]
#   ULTRARAG_ROOT=/path/to/ultrarag ./scripts/start_index_watch.sh ...
#
# 示例:
#   ./scripts/start_index_watch.sh data/raw_docs                    # 使用默认输出目录
#   ./scripts/start_index_watch.sh data/raw_docs output/kb_index     # 指定输出目录
#   ./scripts/start_index_watch.sh data/raw_docs output/kb_index 10m # 10分钟后开始
#   ULTRARAG_ROOT=/path/to/ultrarag ./scripts/start_index_watch.sh /path/to/pdfs /path/to/output
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
    echo "  或: ULTRARAG_ROOT=/path/to/UltraRAG ./scripts/start_index_watch.sh ..."
    exit 1
fi

# 转换为绝对路径
ULTRARAG_ROOT="$(cd "$ULTRARAG_ROOT" && pwd)"

# 解析参数
RAW_PDF_DIR="${1:-$SERVICE_ROOT/data/raw_docs}"
OUTPUT_DIR="${2:-$SERVICE_ROOT/output/kb_index}"
START_AFTER="${3:-}"

# 转换为绝对路径
if [[ ! "$RAW_PDF_DIR" = /* ]]; then
    RAW_PDF_DIR="$(pwd)/$RAW_PDF_DIR"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

# 工作目录（用于日志和状态文件）
WORK_DIR="${WORK_DIR:-$SERVICE_ROOT}"

# 其他默认配置
PIPELINE="pipelines/offline_build_index_watch.yaml"
PARAMETER_TEMPLATE="templates/offline_build_index_watch_parameter.yaml.template"
SCAN_INTERVAL=60
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B}"
INDEX_BACKEND="${INDEX_BACKEND:-faiss}"
COLLECTION_NAME="${COLLECTION_NAME:-wiki_auto}"

# 激活UltraRAG虚拟环境
if [ -f "$ULTRARAG_ROOT/.venv/bin/activate" ]; then
    source "$ULTRARAG_ROOT/.venv/bin/activate"
    echo "[INFO] Virtual environment activated"
else
    echo "[WARNING] Virtual environment not found at $ULTRARAG_ROOT/.venv. Continuing without activation."
fi

echo "=========================================="
echo "  Index Watch Service - 自动索引更新服务"
echo "=========================================="
echo "[配置]"
echo "  服务包根目录:     $SERVICE_ROOT"
echo "  UltraRAG 根目录:   $ULTRARAG_ROOT"
echo "  工作目录:         $WORK_DIR"
echo "  PDF 监控目录:     $RAW_PDF_DIR"
echo "  输出目录:         $OUTPUT_DIR/"
echo "    ├── chunks.jsonl              # QA 批处理需要"
echo "    ├── index.index               # QA 批处理需要 (FAISS)"
echo "    └── .intermediate/            # 中间文件"
echo "        ├── corpus.jsonl"
echo "        └── embeddings.npy"
if [ -n "$START_AFTER" ]; then
    echo "  启动延迟:         $START_AFTER"
fi
echo "  扫描间隔:         ${SCAN_INTERVAL}秒"
echo "  索引后端:         $INDEX_BACKEND"
echo "=========================================="

# 检查 PDF 目录是否存在
if [ ! -d "$RAW_PDF_DIR" ]; then
    echo "[INFO] PDF 目录不存在，正在创建: $RAW_PDF_DIR"
    mkdir -p "$RAW_PDF_DIR"
fi

# 构建命令
CMD="python \"$SERVICE_ROOT/auto_scripts/index_watch_service.py\" \
    --service_root \"$SERVICE_ROOT\" \
    --ultrarag_root \"$ULTRARAG_ROOT\" \
    --work_dir \"$WORK_DIR\" \
    --raw_pdf_dir \"$RAW_PDF_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --pipeline \"$PIPELINE\" \
    --parameter_template \"$PARAMETER_TEMPLATE\" \
    --scan_interval $SCAN_INTERVAL \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --index_backend \"$INDEX_BACKEND\" \
    --collection_name \"$COLLECTION_NAME\""

if [ -n "$START_AFTER" ]; then
    CMD="$CMD --start_after \"$START_AFTER\""
fi

# 执行命令
eval $CMD
