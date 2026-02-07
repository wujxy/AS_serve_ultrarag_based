# UltraRAG 独立服务包

这是一个独立的 UltraRAG 服务包，包含所有必要的脚本和配置文件，可以在任何目录下运行。

## 目录结构

```
ultrarag_service/
├── scripts/                    # 启动脚本
│   ├── start_index_watch.sh   # 启动索引监控服务
│   ├── start_qa_batch.sh      # 启动 QA 批处理服务
│   └── vllm_serve_llama.sh    # vLLM 生成模型服务脚本
├── auto_scripts/              # Python 服务脚本
│   ├── index_watch_service.py # 索引监控服务
│   └── run_qa_batch.py        # QA 批处理服务
├── templates/                  # 参数模板
│   ├── offline_build_index_watch_parameter.yaml.template
│   └── online_rag_qa_batch_parameter.yaml.template
├── config/                    # 配置文件
│   └── config.sh              # 环境变量配置
├── data/                      # 数据目录
│   ├── raw_docs/              # 原始 PDF 文件
│   └── questions/             # 问题文件 (jsonl 格式)
├── output/                    # 服务输出目录
│   ├── kb_index/              # 知识库输出 (chunks.jsonl + index.index)
│   ├── memory_*.json          # UltraRAG 原始输出 (从 UltraRAG/output 复制)
│   └── memory_*_answers.jsonl # 提取的答案文件
├── logs/                      # 日志目录
├── state/                     # 状态文件目录
└── README.md                  # 本文件
```

## 快速开始

### 1. 配置环境

编辑 `config/config.sh`，设置您的 UltraRAG 路径和模型路径：

```bash
# 编辑配置文件
vim config/config.sh

# 必须设置: UltraRAG 项目根目录
export ULTRARAG_ROOT="/path/to/your/UltraRAG"

# 可选设置: 模型路径（如不设置将使用默认值）
export EMBEDDING_MODEL="/path/to/embedding/model"
export RERANKER_MODEL="/path/to/reranker/model"
export GENERATION_MODEL="/path/to/generation/model"
```

或在命令行中设置环境变量：

```bash
export ULTRARAG_ROOT=/path/to/your/UltraRAG
```

### 2. 准备数据

将 PDF 文件放入 `data/raw_docs/` 目录：

```bash
cp /path/to/your/pdfs/*.pdf data/raw_docs/
```

创建问题文件 `data/questions/test.jsonl`：

```json
{"question": "什么是 UltraRAG？"}
{"question": "如何使用 UltraRAG 构建知识库？"}
```

### 3. 运行服务

#### 服务 A: 索引监控服务

监控 PDF 目录并自动构建/更新知识库索引：

```bash
# 使用默认路径
./scripts/start_index_watch.sh data/raw_docs output/kb_index

# 指定延迟启动 (例如 10 分钟后)
./scripts/start_index_watch.sh data/raw_docs output/kb_index 10m

# 在任意目录运行
ULTRARAG_ROOT=/path/to/UltraRAG /path/to/ultrarag_service/scripts/start_index_watch.sh /path/to/pdfs /path/to/output
```

#### 服务 B: QA 批处理服务

对问题文件执行批量 RAG 问答：

```bash
# 使用默认路径
./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index

# 完成后保持 vLLM 服务运行
./scripts/start_qa_batch.sh data/questions/test.jsonl output/kb_index --keep_server
```

## 环境变量

| 环境变量 | 说明 | 必需 |
|---------|------|------|
| `ULTRARAG_ROOT` | UltraRAG 项目根目录 | 是 |
| `EMBEDDING_MODEL` | Embedding 模型路径 | 否 |
| `RERANKER_MODEL` | Reranker 模型路径 | 否 |
| `GENERATION_MODEL` | 生成模型路径 | 否 |
| `VLLM_BASE_URL` | vLLM 服务地址 | 否 |
| `SYSTEM_PROMPT` | 系统提示词 | 否 |

## 输出文件

### 索引监控服务输出

```
output/kb_index/
├── chunks.jsonl              # 文本块 (QA 批处理需要)
├── index.index               # FAISS 索引 (QA 批处理需要)
└── .intermediate/
    ├── corpus.jsonl          # 语料库
    └── embeddings.npy        # 向量嵌入
```

### QA 批处理服务输出

```
output/
├── memory_nq_online_rag_qa_batch_<timestamp>.json        # UltraRAG 原始输出 (从 UltraRAG/output 复制)
└── memory_nq_online_rag_qa_batch_<timestamp>_answers.jsonl  # 提取的答案 (id + question + answer)
```

**注意**: UltraRAG 的原始输出文件仍保留在 `$ULTRARAG_ROOT/output/` 目录中，服务会自动复制到 `ultrarag_service/output/` 并提取答案。

## 日志和状态

- **日志文件**: `logs/index_watch.log`, `logs/qa_batch.log`
- **状态文件**: `state/index_watch_state.json` (记录已处理的文件)

## 故障排除

### 错误: ULTRAGAG_ROOT 环境变量未设置

```bash
# 设置环境变量
export ULTRARAG_ROOT=/path/to/your/UltraRAG

# 或使用配置文件
source config/config.sh
```

### 错误: 虚拟环境不存在

确保 UltraRAG 项目已正确安装虚拟环境：

```bash
cd $ULTRARAG_ROOT
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 错误: 模型文件不存在

检查 `config/config.sh` 中的模型路径是否正确。

## 从任意位置运行

服务包可以在任意目录下运行，只需指定 `ULTRARAG_ROOT`：

```bash
# 在任意目录下
ULTRARAG_ROOT=/path/to/UltraRAG /absolute/path/to/ultrarag_service/scripts/start_index_watch.sh /path/to/pdfs /path/to/output
```

## 许可证

本服务包遵循 UltraRAG 项目的许可证。
