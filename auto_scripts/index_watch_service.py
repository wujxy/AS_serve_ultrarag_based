#!/usr/bin/env python3
"""
Index Watch Service - UltraRAG External Script A

Monitors a directory for new PDF files and automatically triggers
index building via UltraRAG build/run pipeline.

Usage:
    python scripts/index_watch_service.py --raw_pdf_dir data/pdfs --start_after 10m
    python scripts/index_watch_service.py --raw_pdf_dir data/pdfs --start_at "2026-02-06 02:00:00"
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# Global variables (will be set in main())
LOG_DIR = None
LOG_FILE = None
STATE_DIR = None
STATE_FILE = None
LOCK_FILE = None
logger = None


def setup_logging_and_state(work_dir: Path):
    """Setup logging and state directories based on work_dir."""
    global LOG_DIR, LOG_FILE, STATE_DIR, STATE_FILE, LOCK_FILE, logger

    # Setup logging
    LOG_DIR = work_dir / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "index_watch.log"

    # State management
    STATE_DIR = work_dir / "state"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE = STATE_DIR / "index_watch_state.json"
    LOCK_FILE = STATE_DIR / "index_watch.lock"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)


class FileState:
    """Represents the state of a single file."""
    def __init__(self, path: str, size: int, mtime: float, checksum: str = None):
        self.path = path
        self.size = size
        self.mtime = mtime
        self.checksum = checksum

    @classmethod
    def from_path(cls, path: Path, use_checksum: bool = False):
        """Create FileState from a file path."""
        stat = path.stat()
        checksum = None
        if use_checksum:
            checksum = compute_file_hash(path)
        return cls(
            path=str(path.absolute()),
            size=stat.st_size,
            mtime=stat.st_mtime,
            checksum=checksum
        )

    def to_dict(self):
        return {
            "path": self.path,
            "size": self.size,
            "mtime": self.mtime,
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            path=data["path"],
            size=data["size"],
            mtime=data["mtime"],
            checksum=data.get("checksum")
        )


def compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha1.update(chunk)
    return sha1.hexdigest()


def load_state() -> dict:
    """Load the state file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    path: FileState.from_dict(state_data)
                    for path, state_data in data.get("files", {}).items()
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load state file: {e}. Starting with empty state.")
    return {}


def save_state(state: dict):
    """Save the state file."""
    data = {
        "files": {
            path: file_state.to_dict()
            for path, file_state in state.items()
        },
        "last_updated": datetime.now().isoformat()
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_new_file(file_path: Path, state: dict, use_checksum: bool = False) -> bool:
    """Check if a file is new or has been modified."""
    path_str = str(file_path.absolute())
    if path_str not in state:
        return True

    old_state = state[path_str]
    current_stat = file_path.stat()

    # Check size first (fast)
    if current_stat.st_size != old_state.size:
        return True

    # Check mtime
    if current_stat.st_mtime != old_state.mtime:
        # If using checksum, verify content hasn't changed
        if use_checksum:
            current_checksum = compute_file_hash(file_path)
            return current_checksum != old_state.checksum
        return True

    return False


def get_pdf_files(directory: Path) -> list[Path]:
    """Get all PDF files in the directory."""
    extensions = [".pdf", ".PDF"]
    return [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix in extensions
    ]


def acquire_lock() -> bool:
    """Acquire a lock to prevent concurrent runs."""
    try:
        if LOCK_FILE.exists():
            # Check if the lock is stale (older than 4 hours)
            lock_time = datetime.fromtimestamp(LOCK_FILE.stat().st_mtime)
            if datetime.now() - lock_time > timedelta(hours=4):
                logger.warning("Found stale lock file, removing it.")
                LOCK_FILE.unlink()
            else:
                logger.info("Another build is already in progress (lock file exists).")
                return False
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.touch()
        return True
    except Exception as e:
        logger.error(f"Failed to acquire lock: {e}")
        return False


def release_lock():
    """Release the lock."""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as e:
        logger.error(f"Failed to release lock: {e}")


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
    """Run UltraRAG run command."""
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
        # Run `ultrarag run`
        logger.info("Running: ultrarag run ...")
        result_run = subprocess.run(
            [ultrarag_cmd, "run", str(pipeline_yaml)],
            cwd=ultrarag_root,
            capture_output=True,
            text=True,
            check=False,
            env=env
        )

        if result_run.stdout:
            logger.info(f"ultrarag run stdout:\n{result_run.stdout}")
        if result_run.stderr:
            logger.warning(f"ultrarag run stderr:\n{result_run.stderr}")

        if result_run.returncode != 0:
            logger.error(f"ultrarag run failed with code {result_run.returncode}")
            return False

        logger.info("UltraRAG pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running UltraRAG pipeline: {e}")
        return False


def parse_start_time(start_after: str = None, start_at: str = None) -> datetime:
    """Parse the start time from arguments."""
    now = datetime.now()

    if start_at:
        try:
            return datetime.strptime(start_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.error(f"Invalid start_at format: {start_at}. Use 'YYYY-MM-DD HH:MM:SS'")
            sys.exit(1)

    if start_after:
        # Parse like "10m", "1h", "30s"
        unit = start_after[-1]
        try:
            value = int(start_after[:-1])
        except ValueError:
            logger.error(f"Invalid start_after format: {start_after}. Use like '10m', '1h', '30s'")
            sys.exit(1)

        if unit == "s":
            return now + timedelta(seconds=value)
        elif unit == "m":
            return now + timedelta(minutes=value)
        elif unit == "h":
            return now + timedelta(hours=value)
        else:
            logger.error(f"Invalid unit in start_after: {unit}. Use 's', 'm', or 'h'")
            sys.exit(1)

    # Default: start immediately
    return now


def main():
    parser = argparse.ArgumentParser(
        description="Index Watch Service - Monitor directory and auto-update UltraRAG index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start after 10 minutes
  python scripts/index_watch_service.py --raw_pdf_dir data/pdfs --start_after 10m

  # Start at specific time
  python scripts/index_watch_service.py --raw_pdf_dir data/pdfs --start_at "2026-02-06 02:00:00"

  # Run immediately
  python scripts/index_watch_service.py --raw_pdf_dir data/pdfs
        """
    )

    parser.add_argument(
        "--raw_pdf_dir",
        type=str,
        required=True,
        help="Directory to monitor for new PDF files"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipelines/offline_build_index_watch.yaml",
        help="Path to the pipeline YAML file"
    )
    parser.add_argument(
        "--parameter_template",
        type=str,
        default="parameter/offline_build_index_watch_parameter.yaml.template",
        help="Path to the parameter template file"
    )
    parser.add_argument(
        "--start_after",
        type=str,
        default=None,
        help="Start after specified delay (e.g., '10m', '1h', '30s')"
    )
    parser.add_argument(
        "--start_at",
        type=str,
        default=None,
        help="Start at specific time (format: 'YYYY-MM-DD HH:MM:SS')"
    )
    parser.add_argument(
        "--scan_interval",
        type=int,
        default=60,
        help="Directory scan interval in seconds (default: 60)"
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
        help="Working directory for logs and state files (default: current directory)"
    )
    parser.add_argument(
        "--use_checksum",
        action="store_true",
        help="Use file checksum for change detection (slower but more reliable)"
    )

    # Unified output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="index_watch_output",
        help="Output directory for all generated files (chunks, index, etc.)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/home/NagaiYoru/LLM_model/Qwen3-Embedding-0.6B",
        help="Path to embedding model"
    )
    parser.add_argument(
        "--index_backend",
        type=str,
        default="faiss",
        choices=["faiss", "milvus"],
        help="Index backend type"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="wiki_auto",
        help="Collection name for Milvus"
    )
    parser.add_argument(
        "--index_use_gpu",
        type=str,
        default="true",
        help="Use GPU for FAISS index (true/false)"
    )
    parser.add_argument(
        "--milvus_uri",
        type=str,
        default="tcp://127.0.0.1:29901",
        help="Milvus URI (if using Milvus backend)"
    )

    args = parser.parse_args()

    # Setup logging and state directories
    work_dir = Path(args.work_dir) if args.work_dir else Path.cwd()
    work_dir = work_dir.absolute()
    setup_logging_and_state(work_dir)

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
        # Try to find UltraRAG root by looking for the script location
        script_dir = Path(__file__).parent.absolute()
        if script_dir.name == "auto_scripts":
            ultrarag_root = script_dir.parent
        else:
            ultrarag_root = Path.cwd()

    logger.info(f"Service root: {service_root}")
    logger.info(f"UltraRAG root: {ultrarag_root}")
    logger.info(f"Work directory: {work_dir}")

    # Setup unified output directory (can be absolute or relative to current dir)
    output_dir = Path(args.output_dir).absolute()
    intermediate_dir = output_dir / ".intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    corpus_output = intermediate_dir / "corpus.jsonl"
    chunks_output = output_dir / "chunks.jsonl"
    embedding_output = intermediate_dir / "embeddings.npy"

    if args.index_backend == "faiss":
        index_output = output_dir / "index.index"
    else:  # milvus
        index_output = output_dir  # Milvus doesn't use file path

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  Chunks: {chunks_output}")
    logger.info(f"  Index: {index_output if args.index_backend == 'faiss' else 'Milvus'}")

    # Validate paths
    raw_pdf_dir = Path(args.raw_pdf_dir)
    if not raw_pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {raw_pdf_dir}")
        sys.exit(1)

    # Pipeline is in UltraRAG root, parameter template is in service root
    pipeline_yaml = ultrarag_root / args.pipeline
    parameter_template = service_root / args.parameter_template

    if not pipeline_yaml.exists():
        logger.error(f"Pipeline file not found: {pipeline_yaml}")
        sys.exit(1)
    if not parameter_template.exists():
        logger.error(f"Parameter template not found: {parameter_template}")
        sys.exit(1)

    # Parse start time
    start_time = parse_start_time(args.start_after, args.start_at)
    now = datetime.now()
    if start_time > now:
        wait_seconds = (start_time - now).total_seconds()
        logger.info(f"Waiting until {start_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds:.0f} seconds)")
        time.sleep(wait_seconds)

    logger.info("Index Watch Service started")
    logger.info(f"Monitoring directory: {raw_pdf_dir}")
    logger.info(f"Scan interval: {args.scan_interval} seconds")

    # Load state
    state = load_state()

    # Main monitoring loop
    try:
        while True:
            logger.info("Scanning for new PDF files...")

            current_files = get_pdf_files(raw_pdf_dir)
            new_files = [
                f for f in current_files
                if is_new_file(f, state, args.use_checksum)
            ]

            if new_files:
                logger.info(f"Found {len(new_files)} new/modified files:")
                for f in new_files:
                    logger.info(f"  - {f}")

                # Try to acquire lock
                if acquire_lock():
                    try:
                        # Create runtime parameter file - UltraRAG looks for {pipeline_name}_parameter.yaml
                        # When running pipelines/offline_build_index_watch.yaml, it will find
                        # parameter/_runtime/offline_build_index_watch_parameter.yaml
                        runtime_param = ultrarag_root / "parameter" / "_runtime" / "offline_build_index_watch_parameter.yaml"
                        replacements = {
                            "RAW_PDF_DIR": str(raw_pdf_dir.absolute()),
                            "CORPUS_OUTPUT_PATH": str(corpus_output.absolute()),
                            "CHUNKS_OUTPUT_PATH": str(chunks_output.absolute()),
                            "EMBEDDING_OUTPUT_PATH": str(embedding_output.absolute()),
                            "INDEX_OUTPUT_PATH": str(index_output.absolute()),
                            "EMBEDDING_MODEL_PATH": args.embedding_model,
                            "INDEX_USE_GPU": args.index_use_gpu.lower(),
                            "COLLECTION_NAME": args.collection_name,
                            "MILVUS_URI": args.milvus_uri,
                        }

                        create_runtime_parameter(parameter_template, runtime_param, replacements)

                        # Also copy to the location where UltraRAG expects it
                        # This is needed because --param doesn't seem to work correctly
                        target_param = ultrarag_root / "pipelines" / "parameter" / "offline_build_index_watch_parameter.yaml"
                        target_param.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(runtime_param, target_param)
                        logger.info(f"Copied parameter file to: {target_param}")

                        # Use the original pipeline file from pipelines/ (relative server paths work there)
                        runtime_pipeline = pipeline_yaml

                        # Run UltraRAG pipeline (UltraRAG will find the parameter file automatically)
                        success = run_ultrarag_pipeline(runtime_pipeline, ultrarag_root)

                        if success:
                            # Update state with current files
                            for f in current_files:
                                file_state = FileState.from_path(f, args.use_checksum)
                                state[file_state.path] = file_state
                            save_state(state)
                            logger.info("State updated. Index build completed successfully.")
                        else:
                            logger.error("Index build failed. Will retry on next scan.")

                    finally:
                        release_lock()
                else:
                    logger.info("Could not acquire lock. Another build is in progress. Will retry later.")
            else:
                logger.debug("No new files found.")

            logger.info(f"Next scan in {args.scan_interval} seconds...")
            time.sleep(args.scan_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
