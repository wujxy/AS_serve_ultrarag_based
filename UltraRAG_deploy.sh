#!/bin/bash
#
# UltraRAG 外置服务部署脚本
#
# 功能: 自动部署 UltraRAG 及其外置服务包
#
# 使用方法:
#   bash UltraRAG_deploy.sh
#
# 环境要求:
#   - Python 3.8+
#   - Git
#   - 足够的磁盘空间 (建议 20GB+)
#   - GPU (推荐，用于模型加速)
#

set -e

# ============================================================================
# 配置区域 (用户可根据需要修改)
# ============================================================================

# 安装目录
INSTALL_DIR="${HOME}/LLM_tuning"
ULTRARAG_REPO="https://github.com/OpenBMB/UltraRAG.git"

# 模型目录
MODEL_DIR="${HOME}/LLM_model"

# 服务包目录 (当前目录)
SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 工具函数
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# ============================================================================
# 检查系统环境
# ============================================================================

check_environment() {
    print_header "检查系统环境"

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装 Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python 版本: $PYTHON_VERSION"

    # 检查 Git
    if ! command -v git &> /dev/null; then
        print_error "Git 未安装，请先安装 Git"
        exit 1
    fi
    print_success "Git 已安装"

    # 检查 CUDA (可选)
    if command -v nvidia-smi &> /dev/null; then
        print_success "检测到 NVIDIA GPU"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
            print_info "  GPU: $line"
        done
    else
        print_warning "未检测到 GPU，将使用 CPU 运行 (速度较慢)"
    fi

    echo ""
}

# ============================================================================
# 安装依赖工具
# ============================================================================

install_dependencies() {
    print_header "安装依赖工具"

    # 检查 uv 是否已安装
    if ! command -v uv &> /dev/null; then
        print_info "安装 uv 包管理器..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        print_success "uv 安装完成"
    else
        print_success "uv 已安装"
    fi

    echo ""
}

# ============================================================================
# 安装 UltraRAG
# ============================================================================

install_ultrarag() {
    print_header "安装 UltraRAG"

    # 创建安装目录
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    ULTRARAG_DIR="$INSTALL_DIR/UltraRAG"

    # 检查是否已安装
    if [ -d "$ULTRARAG_DIR" ]; then
        print_warning "UltraRAG 目录已存在: $ULTRARAG_DIR"
        read -p "是否重新安装? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "删除旧版本..."
            rm -rf "$ULTRARAG_DIR"
        else
            print_info "跳过安装，使用现有版本"
            cd "$ULTRARAG_DIR"
            return 0
        fi
    fi

    # 克隆仓库
    print_info "克隆 UltraRAG 仓库..."
    git clone "$ULTRARAG_REPO" --depth 1
    cd "$ULTRARAG_DIR"

    # 同步依赖
    print_info "安装 Python 依赖..."
    uv sync --all-extras

    print_success "UltraRAG 安装完成: $ULTRARAG_DIR"
    echo ""
}

# ============================================================================
# 配置服务包
# ============================================================================

configure_service() {
    print_header "配置外置服务包"

    # 创建必要目录
    print_info "创建目录结构..."
    mkdir -p "$SERVICE_ROOT/data/raw_docs"
    mkdir -p "$SERVICE_ROOT/data/questions"
    mkdir -p "$SERVICE_ROOT/output/kb_index"
    mkdir -p "$SERVICE_ROOT/logs"
    mkdir -p "$SERVICE_ROOT/state"
    mkdir -p "$SERVICE_ROOT/pipelines/parameter"

    print_success "目录结构创建完成"

    # 更新配置文件
    print_info "更新配置文件..."

    CONFIG_FILE="$SERVICE_ROOT/config/config.sh"
    CONFIG_TEMPLATE="$SERVICE_ROOT/config/config.sh.template"

    # 如果配置文件不存在或需要更新
    cat > "$CONFIG_FILE" << EOF
#!/bin/bash
#
# UltraRAG Service Configuration
# 自动生成的配置文件
#

# ============================================================================
# UltraRAG 项目根目录 (自动设置)
# ============================================================================
export ULTRARAG_ROOT="$INSTALL_DIR/UltraRAG"

# ============================================================================
# 模型路径配置 (请根据实际情况修改)
# ============================================================================
# Embedding 模型 (用于索引构建和检索)
export EMBEDDING_MODEL="$MODEL_DIR/Qwen3-Embedding-0.6B"

# Reranker 模型 (用于重排序)
export RERANKER_MODEL="$MODEL_DIR/MiniCPM-Reranker-Light"

# 生成模型 (用于 QA 生成)
export GENERATION_MODEL="$MODEL_DIR/Llama-3.2-3B-Instruct"

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
EOF

    print_success "配置文件已生成: $CONFIG_FILE"
    print_warning "请根据实际情况修改模型路径!"

    echo ""
}

# ============================================================================
# 显示使用说明
# ============================================================================

show_usage() {
    print_header "使用说明"

    cat << EOF
${GREEN}UltraRAG 外置服务部署完成!${NC}

${YELLOW}1. 配置模型路径${NC}
   编辑配置文件，设置您的模型路径:
   vim $SERVICE_ROOT/config/config.sh

${YELLOW}2. 准备数据${NC}
   # 放置 PDF 文档
   cp /path/to/your/pdfs/*.pdf $SERVICE_ROOT/data/raw_docs/

   # 创建问题文件 (data/questions/test.jsonl)
   echo '{"question": "什么是 UltraRAG?"}' > $SERVICE_ROOT/data/questions/test.jsonl

${YELLOW}3. 运行索引监控服务${NC}
   cd $SERVICE_ROOT
   ./scripts/start_index_watch.sh data/raw_docs output/kb_index

${YELLOW}4. 运行 QA 批处理服务${NC}
   cd $SERVICE_ROOT
   ./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index

${YELLOW}目录结构:${NC}
   $SERVICE_ROOT/
   ├── data/           # 输入数据
   ├── output/         # 输出结果
   ├── logs/           # 日志文件
   └── state/          # 状态文件

${YELLOW}详细文档:${NC}
   cat $SERVICE_ROOT/README.md

${BLUE}========================================${NC}
EOF
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    print_header "UltraRAG 外置服务部署脚本"
    echo ""

    # 检查环境
    check_environment

    # 安装依赖
    install_dependencies

    # 安装 UltraRAG
    install_ultrarag

    # 配置服务
    configure_service

    # 显示使用说明
    show_usage

    print_success "部署完成!"
}

# 运行主流程
main "$@"
