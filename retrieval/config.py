import os
import datetime
import logging
import json
from typing import List, Tuple
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Define internal hosts from environment or use defaults
internal_hosts = os.environ.get("NO_PROXY", "localhost,127.0.0.1")

# specific helper to update a proxy var safely
def update_proxy_var(var_name, new_hosts):
    current = os.environ.get(var_name, "")
    if current:
        # Append to existing, avoiding duplicates if you run this multiple times
        if new_hosts not in current: 
            os.environ[var_name] = f"{current},{new_hosts}"
    else:
        os.environ[var_name] = new_hosts

# Update both Lowercase and Uppercase
update_proxy_var("no_proxy", internal_hosts)
update_proxy_var("NO_PROXY", internal_hosts)

class BenchmarkConfig:
    """Benchmark configuration shared across retrieval workflows."""

    # ---- Retrieving Stage Configuration ----

    # Static Configs
    LLM_MODEL_NAME: str = 'Meta-Llama-3-8B-Instruct'
    LLM_BASE_URL: str = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1')
    EMBEDDING_MODEL_NAME: str = 'baai-bge-m3'
    EMBEDDING_BASE_URL: str = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:8001/v1')
    
    LLM_MAX_WORKERS: int = 128  # Number of parallel LLM inference workers
    EMBEDDING_MAX_WORKERS: int = 64  # Number of parallel workers for embedding-heavy retrieval tasks

    # Batch sizes for different operations
    # LLM_BATCH_SIZE: int = 8  # Reserved for future LLM batch processing
    EMBEDDING_BATCH_SIZE: int = 16  # Batch size for embedding API calls (increase for high-end GPUs)

    # ---- Graph building Extraction Stage Configuration ----
    ENTITY_EXTRACT_MAX_TOKENS: int = 1024
    
    # ---- QRAG Question Generation Stage Configuration ----
    QUESTION_GEN_MAX_TOKENS: int = 1024  # Safety cap to prevent runaway generation

    # ---- Retriever Save dir ----
    SAVE_DIR_ROOT: str = 'retriever_ckpts'

    # ---- Input for bench_run.py. The retrievers to be benchmarked ----
    RETRIEVER_LIST: List[str] = [
        'GRAG-naive-semantic',
        'QRAG-bge-m3',
        'BM25',
        'RAG-bge-m3',
        'hippoRAG',
    ]

    NUM_TO_RETRIEVE: int = 800  # Top K documents to retrieve during bench_run.py

    # -- Input for bench_run.py. List of datasets to process. 
    # Each tuple must follow the format: (corpus_path, qa_path, benchmark_name)
    DATASET_LIST: List[Tuple[str, str, str]] = [
        ('data/hpqa/hotpotqa_corpus.json', 'data/hpqa/hotpotqa.json', 'HotpotQA'),
    ]
    
    # Set to > 0 to limit the number of documents used for index building. Set to -1 to use the entire corpus.
    SUBSAMPLE_CORPUS_SIZE: int = 1000
    # Set to > 0 to limit the number of questions to process. Set to -1 to use all queries.
    SUBSAMPLE_QA_SIZE: int = 10
    
    
    # ---- Output For bench_run.py, Input for bench_post_run.py ----
    RESULTS_FILE: str = 'results/result.jsonl'
    
    # ---- Post Benchmark Organize Stage Configs ---- 
    # post_bench_run.py will generate a new file named RESULTS_FILE_POST_ORG
    RESULTS_FILE_POST_ORG: str = 'results/result_post.jsonl'
    
    # ---- Mutual Info Compute Stage Configs ----
    # post_bench_run.py will convert aligned scores into a folder under {config.MI_LAYER_INPUTS}
    MI_LAYER_INPUTS = 'out'

    post_topK_threshold = 100   # top K to be aligned during post_bench_run.py. Must be <= NUM_TO_RETRIEVE
    gamma = 5   # controling the scale where ground truth label will be applied on top of LLM cross entropy distributions
    anchor_retriever = 'hippoRAG'   # anchor to compute MI based on who's top-K retrieval results
    # anchor_retriever = 'ALL'   # anchor to compute MI based on who's top-K retrieval results
    
    # Choose one way to normalize scores across different types of retrievers
    # Options: "shift_0_5_softmax", "softmax_tau_max", "max", "softmax_tau_5max"
    SCORE_NORM_MODE = "shift_0_5_softmax"
    
    # ---- Logging Configuration ----
    LOG_DIR: str = 'logs'
    
    def __init__(self) -> None:
        print(
            "\n\n\n-----\n"
            f"{self.RETRIEVER_LIST}\n"
            f"{self.DATASET_LIST}\n"
            f"{(self.SUBSAMPLE_CORPUS_SIZE, self.SUBSAMPLE_QA_SIZE)}\n"
            "------\n\n\n"
        )
        # Ensure directories exist
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.RESULTS_FILE), exist_ok=True)

    def ckpt_path(self, retriever_name: str, dataset_name: str) -> str:
        """Return the checkpoint directory for a retriever/dataset pair."""
        cut_ckpt_folder = os.path.join(self.SAVE_DIR_ROOT, f'{retriever_name}@{dataset_name}')
        os.makedirs(cut_ckpt_folder, exist_ok=True)
        return cut_ckpt_folder
    
    def get_logger(self, logger_name: str) -> logging.Logger:
        """Create or return a logger with timestamped file output in the logs directory."""
        logger = logging.getLogger(logger_name)
        
        # FIX: Only add handlers if none exist (prevents duplicates in serial execution)
        if not logger.handlers:
            # Create timestamp: YYYY-MM-DD_HH-MM-SS
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # New format: logs/LoggerName__Timestamp.log
            filename = f"{logger_name}__{timestamp}.log"
            log_path = os.path.join(self.LOG_DIR, filename)

            # Use 'a' (append) mode for safety, though timestamp prevents collision
            handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger

    def dump_snapshot(self) -> str:
        """Saves a JSON snapshot of the current configuration for reproducibility."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"config_snapshot__{timestamp}.json"
        output_path = os.path.join(self.LOG_DIR, filename)
        
        # Merge class defaults with instance overrides
        data = {k: v for k, v in vars(BenchmarkConfig).items() if not k.startswith('_') and not callable(v)}
        data.update({k: v for k, v in vars(self).items() if not k.startswith('_') and not callable(v)})
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        return output_path