#!/usr/bin/env python3
"""
QA Batch Runner - UltraRAG External Script B

Starts vLLM generation service and runs online RAG QA pipeline on a batch of questions.

Usage:
    python scripts/run_qa_batch.py --questions data/questions/test.jsonl --index_path index/index.index
    python scripts/run_qa_batch.py --questions data/questions/test.jsonl --index_backend milvus --collection_name wiki
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Global variables (will be set in main())
LOG_DIR = None
LOG_FILE = None
logger = None


def setup_logging(work_dir: Path):
    """Setup logging directory based on work_dir."""
    global LOG_DIR, LOG_FILE, logger

    # Setup logging
    LOG_DIR = work_dir / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "qa_batch.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)


def find_ultrarag_root() -> Path:
    """Find the UltraRAG root directory."""
    script_dir = Path(__file__).parent.absolute()
    if script_dir.name == "auto_scripts":
        return script_dir.parent
    return Path.cwd()


def wait_for_service(
    health_url: str,
    timeout: int = 300,
    interval: int = 5
) -> bool:
    """Wait for vLLM service to be ready."""
    logger.info(f"Waiting for service to be ready at {health_url}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if HAS_REQUESTS:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Service is ready!")
                    return True
            else:
                # Fallback: use urllib
                import urllib.request
                import urllib.error
                try:
                    with urllib.request.urlopen(health_url, timeout=5) as response:
                        if response.status == 200:
                            logger.info("Service is ready!")
                            return True
                except urllib.error.URLError:
                    pass
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        logger.info(f"Service not ready yet... waiting {interval}s (elapsed: {int(time.time() - start_time)}s)")
        time.sleep(interval)

    logger.error(f"Timeout waiting for service at {health_url}")
    return False


def start_vllm_service(
    script_path: Path,
    ultrarag_root: Path,
    log_file: Path = None
) -> subprocess.Popen:
    """Start vLLM service using the provided bash script."""
    logger.info(f"Starting vLLM service using script: {script_path}")

    if not script_path.exists():
        logger.error(f"vLLM script not found: {script_path}")
        sys.exit(1)

    # Prepare log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"vllm_service_{timestamp}.log"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Start the service
    with open(log_file, "w") as log_handle:
        process = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=ultrarag_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # Create new process group for easier cleanup
        )

    logger.info(f"vLLM service started with PID {process.pid}")
    logger.info(f"Service logs: {log_file}")
    return process


def stop_vllm_service(process: subprocess.Popen):
    """Stop vLLM service."""
    logger.info("Stopping vLLM service...")
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
        logger.info("vLLM service stopped")
    except subprocess.TimeoutExpired:
        logger.warning("Service did not stop gracefully, forcing...")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        logger.info("vLLM service force stopped")
    except Exception as e:
        logger.error(f"Error stopping service: {e}")


def create_runtime_parameter(
    template_path: Path,
    output_path: Path,
    replacements: dict
):
    """Create a runtime parameter file from template."""
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    for key, value in replacements.items():
        placeholder = f"__{key}__"
        content = content.replace(placeholder, str(value))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Created runtime parameter: {output_path}")


def run_ultrarag_pipeline(
    pipeline_yaml: Path,
    ultrarag_root: Path
) -> bool:
    """Run UltraRAG pipeline.

    Args:
        pipeline_yaml: Path to the pipeline YAML file
        ultrarag_root: UltraRAG project root directory
    """
    logger.info(f"Running UltraRAG pipeline: {pipeline_yaml}")

    # Find ultrag command (prefer venv)
    venv_ultrarag = ultrarag_root / ".venv" / "bin" / "ultrarag"
    venv_bin = ultrarag_root / ".venv" / "bin"

    if venv_ultrarag.exists():
        ultrarag_cmd = str(venv_ultrarag)
        logger.debug(f"Using venv ultrarag: {ultrarag_cmd}")
        # Set up environment for venv
        env = os.environ.copy()
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(ultrarag_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["VIRTUAL_ENV"] = str(ultrarag_root / ".venv")
    else:
        ultrarag_cmd = "ultrarag"
        logger.debug(f"Using system ultrarag")
        env = None

    try:
        # Run from ultrarag_root so servers/ paths resolve correctly
        result = subprocess.run(
            [ultrarag_cmd, "run", str(pipeline_yaml)],
            cwd=ultrarag_root,
            capture_output=True,
            text=True,
            check=False,
            env=env
        )

        if result.stdout:
            logger.info(f"ultrarag run stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"ultrarag run stderr:\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"ultrarag run failed with code {result.returncode}")
            return False

        logger.info("UltraRAG pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running UltraRAG pipeline: {e}")
        return False


def find_output_file(output_dir: Path, pattern: str = "*online_rag_qa*.json") -> Path:
    """Find the most recent output file."""
    files = list(output_dir.glob(pattern))
    if not files:
        logger.error(f"No output files found in {output_dir}")
        return None

    # Return the most recently modified file
    return max(files, key=lambda f: f.stat().st_mtime)


def extract_answers(output_file: Path, questions_file: Path) -> Path:
    """Extract answers from the output JSON file.

    UltraRAG outputs a JSON array of step records. The final answer is in
    the "custom.output_extract_from_boxed" step's memory.memory_pred_ls field.
    """
    logger.info(f"Extracting answers from: {output_file}")

    output_answers_path = output_file.parent / f"{output_file.stem}_answers.jsonl"

    # Read the entire output file (it's a JSON array, not JSONL)
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Parse as JSON array
    try:
        steps_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse output file as JSON: {e}")
        # Try parsing as JSONL (fallback)
        logger.info("Trying to parse as JSONL format...")
        steps_data = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        steps_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Find the final answers from "custom.output_extract_from_boxed" step
    pred_ls = None
    q_ls = None
    gt_ls = None

    for step_record in steps_data:
        if isinstance(step_record, dict):
            step_name = step_record.get("step", "")
            memory = step_record.get("memory", {})

            # Get questions from benchmark.get_data step
            if step_name == "benchmark.get_data":
                q_ls = memory.get("memory_q_ls", [])
                gt_ls = memory.get("memory_gt_ls", [])

            # Get final answers from custom.output_extract_from_boxed step
            elif step_name == "custom.output_extract_from_boxed":
                pred_ls = memory.get("memory_pred_ls", [])

    if pred_ls is None:
        logger.warning("No 'memory_pred_ls' found in output. Looking for alternative fields...")
        # Fallback: check generation.generate step for memory_ans_ls
        for step_record in steps_data:
            if isinstance(step_record, dict):
                step_name = step_record.get("step", "")
                memory = step_record.get("memory", {})
                if step_name == "generation.generate":
                    pred_ls = memory.get("memory_ans_ls", [])
                    break

    if pred_ls is None:
        logger.error("Could not find answers in output file")
        return None

    # Read questions to get IDs (if available)
    question_ids = {}
    if questions_file.exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if "id" in data:
                        question_ids[idx] = data["id"]
                except json.JSONDecodeError:
                    pass

    # Build answer records
    answers = []
    for idx, answer in enumerate(pred_ls):
        record = {
            "id": question_ids.get(idx, idx),
            "question": q_ls[idx] if q_ls and idx < len(q_ls) else "",
            "answer": answer
        }

        # Include golden answers if available
        if gt_ls and idx < len(gt_ls):
            record["golden_answers"] = gt_ls[idx]

        answers.append(record)

    # Write answers as JSONL
    with open(output_answers_path, "w", encoding="utf-8") as f:
        for record in answers:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Extracted {len(answers)} answers to: {output_answers_path}")
    return output_answers_path


def main():
    parser = argparse.ArgumentParser(
        description="QA Batch Runner - Run online RAG QA on a batch of questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With FAISS index
  python scripts/run_qa_batch.py --questions data/questions/test.jsonl --index_path index/index.index

  # With Milvus
  python scripts/run_qa_batch.py --questions data/questions/test.jsonl --index_backend milvus --collection_name wiki

  # Keep vLLM service running after completion
  python scripts/run_qa_batch.py --questions data/questions/test.jsonl --index_path index/index.index --keep_server
        """
    )

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions JSONL file"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipelines/online_rag_qa_batch.yaml",
        help="Path to the pipeline YAML file"
    )
    parser.add_argument(
        "--parameter_template",
        type=str,
        default="parameter/online_rag_qa_batch_parameter.yaml.template",
        help="Path to the parameter template file"
    )
    parser.add_argument(
        "--vllm_script",
        type=str,
        default="vllm_serve_llama.sh",
        help="Path to vLLM service startup script"
    )
    parser.add_argument(
        "--service_root",
        type=str,
        default=None,
        help="Standalone service package root directory (default: auto-detect from script location)"
    )
    parser.add_argument(
        "--ultrarag_root",
        type=str,
        default=None,
        help="UltraRAG project root directory (default: auto-detect from script location)"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for logs (default: current directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for QA results (default: <ultrarag_root>/output)"
    )

    # Knowledge base configuration - Use --kb_dir for unified directory (from index_watch)
    parser.add_argument(
        "--kb_dir",
        type=str,
        default=None,
        help="Knowledge base directory from index_watch (auto-detects chunks.jsonl and index.index)"
    )
    parser.add_argument(
        "--index_backend",
        type=str,
        default="faiss",
        choices=["faiss", "milvus"],
        help="Index backend type"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to FAISS index file (overridden by --kb_dir)"
    )
    parser.add_argument(
        "--milvus_uri",
        type=str,
        default="tcp://127.0.0.1:29901",
        help="Milvus URI"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="wiki",
        help="Milvus collection name"
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default=None,
        help="Path to chunks JSONL file (overridden by --kb_dir)"
    )

    # Model configuration
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B",
        help="Path to embedding model"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="/home/NagaiYoru/LLM_model/MiniCPM-Reranker-Light",
        help="Path to reranker model"
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="/home/NagaiYoru/LLM_model/Llama-3.2-3B",
        help="Path to generation model"
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://127.0.0.1:65504/v1",
        help="vLLM service base URL"
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default="llama3-3b-instruct",
        help="Name of the served model"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是一个专业的UltraRAG问答助手。请一定记住使用中文回答问题,且足够专业",
        help="System prompt for generation"
    )

    # Service control
    parser.add_argument(
        "--keep_server",
        action="store_true",
        help="Keep vLLM service running after completion"
    )
    parser.add_argument(
        "--skip_service_start",
        action="store_true",
        help="Skip starting vLLM service (if already running)"
    )
    parser.add_argument(
        "--health_url",
        type=str,
        default=None,
        help="Health check URL (default: <vllm_base_url>/models)"
    )
    parser.add_argument(
        "--health_timeout",
        type=int,
        default=300,
        help="Health check timeout in seconds"
    )
    parser.add_argument(
        "--extract_answers",
        action="store_true",
        default=True,
        help="Extract answers to separate file"
    )

    args = parser.parse_args()

    # Setup logging (work_dir for logs)
    work_dir = Path(args.work_dir) if args.work_dir else Path.cwd()
    work_dir = work_dir.absolute()
    setup_logging(work_dir)

    # Determine service root (standalone service package)
    if args.service_root:
        service_root = Path(args.service_root).absolute()
    else:
        # Try to find service root by looking for the script location
        script_dir = Path(__file__).parent.absolute()
        if script_dir.name == "auto_scripts":
            service_root = script_dir.parent
        else:
            service_root = Path.cwd()

    # Determine UltraRAG root
    if args.ultrarag_root:
        ultrarag_root = Path(args.ultrarag_root).absolute()
    else:
        ultrarag_root = find_ultrarag_root()

    logger.info(f"Service root: {service_root}")
    logger.info(f"UltraRAG root: {ultrarag_root}")
    logger.info(f"Work directory: {work_dir}")

    # Handle unified knowledge base directory (from index_watch)
    if args.kb_dir:
        kb_dir = Path(args.kb_dir).absolute()
        if not kb_dir.exists():
            logger.error(f"Knowledge base directory not found: {kb_dir}")
            sys.exit(1)

        # Auto-detect chunks and index paths
        chunks_path = kb_dir / "chunks.jsonl"
        if args.index_backend == "faiss":
            index_path = kb_dir / "index.index"
        else:  # milvus
            index_path = kb_dir  # Not used for Milvus

        if not chunks_path.exists():
            logger.error(f"Chunks file not found in KB directory: {chunks_path}")
            sys.exit(1)
        if args.index_backend == "faiss" and not index_path.exists():
            logger.error(f"Index file not found in KB directory: {index_path}")
            sys.exit(1)

        logger.info(f"Using KB directory: {kb_dir}")
        logger.info(f"  Chunks: {chunks_path}")
        logger.info(f"  Index: {index_path if args.index_backend == 'faiss' else 'Milvus'}")
    else:
        # Use individual paths
        chunks_path = Path(args.chunks_path) if args.chunks_path else ultrarag_root / "data/chunks/chunks.jsonl"
        index_path = Path(args.index_path) if args.index_path else ultrarag_root / "index/index.index"

    # Validate paths
    questions_file = Path(args.questions).absolute()
    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    vllm_script = service_root / args.vllm_script
    pipeline_yaml = ultrarag_root / args.pipeline
    parameter_template = service_root / args.parameter_template

    if not pipeline_yaml.exists():
        logger.error(f"Pipeline file not found: {pipeline_yaml}")
        sys.exit(1)
    if not parameter_template.exists():
        logger.error(f"Parameter template not found: {parameter_template}")
        sys.exit(1)

    # Health check URL
    if args.health_url is None:
        # Derive from vllm_base_url
        base = args.vllm_base_url.rstrip("/").replace("/v1", "")
        args.health_url = urljoin(base + "/", "v1/models")

    vllm_process = None

    try:
        # Start vLLM service if needed
        if not args.skip_service_start:
            vllm_process = start_vllm_service(vllm_script, ultrarag_root)

            # Wait for service to be ready
            if not wait_for_service(args.health_url, args.health_timeout):
                logger.error("vLLM service failed to start in time")
                if vllm_process:
                    stop_vllm_service(vllm_process)
                sys.exit(1)
        else:
            logger.info("Skipping vLLM service start (--skip_service_start)")
            # Still check if service is ready
            if not wait_for_service(args.health_url, 30):
                logger.warning("Service may not be ready, but proceeding anyway...")

        # Create runtime parameter file - UltraRAG looks for {pipeline_name}_parameter.yaml
        # When running pipelines/online_rag_qa_batch.yaml, it will find
        # parameter/_runtime/online_rag_qa_batch_parameter.yaml
        runtime_param = ultrarag_root / "parameter" / "_runtime" / "online_rag_qa_batch_parameter.yaml"
        replacements = {
            "QUESTIONS_PATH": str(questions_file.absolute()),
            "CHUNKS_PATH": str(chunks_path.absolute()),
            "EMBEDDING_MODEL_PATH": args.embedding_model,
            "RERANKER_MODEL_PATH": args.reranker_model,
            "GENERATION_MODEL_PATH": args.generation_model,
            "VLLM_BASE_URL": args.vllm_base_url,
            "SERVED_MODEL_NAME": args.served_model_name,
            "SYSTEM_PROMPT": args.system_prompt,
            "INDEX_BACKEND": args.index_backend,
            "INDEX_PATH": str(index_path.absolute()) if args.index_backend == "faiss" else "",
            "MILVUS_URI": args.milvus_uri,
            "COLLECTION_NAME": args.collection_name,
        }

        create_runtime_parameter(parameter_template, runtime_param, replacements)

        # Also copy to the location where UltraRAG expects it
        target_param = ultrarag_root / "pipelines" / "parameter" / "online_rag_qa_batch_parameter.yaml"
        target_param.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(runtime_param, target_param)
        logger.info(f"Copied parameter file to: {target_param}")

        # Use the original pipeline file from pipelines/ (relative server paths work there)
        runtime_pipeline = pipeline_yaml

        # Run UltraRAG pipeline (UltraRAG will find the parameter file automatically)
        success = run_ultrarag_pipeline(runtime_pipeline, ultrarag_root)

        if not success:
            logger.error("QA pipeline failed")
            sys.exit(1)

        # UltraRAG outputs to ultrarag_root/output/
        # We read from there and extract answers to service_root/output/
        ultrarag_output_dir = ultrarag_root / "output"
        service_output_dir = service_root / "output"
        service_output_dir.mkdir(parents=True, exist_ok=True)

        if ultrarag_output_dir.exists():
            output_file = find_output_file(ultrarag_output_dir)
            if output_file:
                logger.info(f"UltraRAG output file: {output_file}")

                # Extract answers if requested
                if args.extract_answers:
                    # Copy to service output dir first, then extract
                    import shutil
                    copied_file = service_output_dir / output_file.name
                    shutil.copy2(output_file, copied_file)
                    logger.info(f"Copied output to: {copied_file}")

                    answers_file = extract_answers(copied_file, questions_file)
                    logger.info(f"Answers extracted to: {answers_file}")

        logger.info("QA batch processing completed successfully")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Stop vLLM service if needed
        if vllm_process and not args.keep_server:
            stop_vllm_service(vllm_process)
        elif vllm_process and args.keep_server:
            logger.info(f"vLLM service kept running (PID {vllm_process.pid})")
            logger.info(f"To stop it later: kill -TERM {vllm_process.pid}")


if __name__ == "__main__":
    main()
