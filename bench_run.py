import os
import gc
import jsonlines
from typing import List, Dict, Any

# Try to import torch for GPU cleanup (optional - won't crash if unavailable)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure tqdm for clean output in serial execution
from tqdm import tqdm
tqdm.monitor_interval = 0  # Prevents threading issues with progress bars

from retrieval.data_loader import load_and_prepare_dataset


def retriever_result_exists(file_path: str, current_dataset: str, retriever_name: str) -> bool:
    """Checks if a [retriever_name] record already exists for the current dataset."""
    if not os.path.exists(file_path):
        return False
    with jsonlines.open(file_path, mode='r') as reader:
        for obj in reader:
            if obj.get('retriever_name') == retriever_name and obj.get('dataset_name') == current_dataset:
                return True
    return False


def run_retrieval_benchmark(retriever_name, dataset_triple):
    from retrieval.retrievers import config, setup_retriever
    from hipporag.evaluation.retrieval_eval import RetrievalRecall
    from hipporag.utils.config_utils import BaseConfig
    
    corpus_path, qa_path, dataset_name = dataset_triple
    logger = config.get_logger(f"{retriever_name}_{dataset_name}")

    logger.info("=======================================================")
    logger.info("DATASET: %s | RETRIEVER: %s", dataset_name, retriever_name)
    logger.info("Results -> %s", config.RESULTS_FILE)
    logger.info("=======================================================")

    if retriever_result_exists(config.RESULTS_FILE, dataset_name, retriever_name):
        logger.info("%s already exists, skipping.", retriever_name)
        return None  # Return None to indicate no cleanup needed

    current_retriever = setup_retriever(retriever_name, dataset_name, **{**vars(config), **vars(config.__class__)})
    
    # Unpack the new cid_to_title mapping from the data loader
    docs_with_cid, queries, gold_cids_map, qa_data, cid_to_title = load_and_prepare_dataset(
        corpus_path=corpus_path,
        qa_path=qa_path,
        subsample_corpus_size=config.SUBSAMPLE_CORPUS_SIZE,
        subsample_qa_size=config.SUBSAMPLE_QA_SIZE,
    )

    logger.info("Indexing %d documents...", len(docs_with_cid))
    current_retriever.index(docs=docs_with_cid)
    logger.info("Running retrieval for %d queries...", len(queries))
    
    # Convert gold standard integer IDs into string titles for matching
    gold_titles = [
        [cid_to_title.get(cid, str(cid)) for cid in gold_cids_map.get(qa_data[i]['qid'], [])]
        for i in range(len(queries))
    ]
    
    # 1. Run retrieval and ignore the internal placeholder metrics
    retrieval_results, _ = current_retriever.retrieve(
        queries=queries,
        gold_docs=gold_titles,
        num_to_retrieve=config.NUM_TO_RETRIEVE,
    )
    
    # Translate retrieved integer IDs to string titles for the evaluator
    for sol in retrieval_results:
        sol.docs = [cid_to_title.get(cid, str(cid)) for cid in (sol.docIDs or [])]
    
    # 3. Calculate the ACTUAL metrics using the evaluator
    eval_config = BaseConfig(
        llm_name=config.LLM_MODEL_NAME,
        llm_base_url=config.LLM_BASE_URL,
        embedding_base_url=config.EMBEDDING_BASE_URL,
    )
    k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
    evaluator = RetrievalRecall(global_config=eval_config)
    summary_metrics, _ = evaluator.calculate_metric_scores(
        gold_docs=gold_titles,
        retrieved_docs=[r.docs for r in retrieval_results],
        k_list=k_list,
    )
    
    logger.info("Retrieval complete.")

    retriever_results_list: List[Dict[str, Any]] = []
    for i, solution in enumerate(retrieval_results):
        qid = qa_data[i].get("qid", f"NO_QID_{i}")
        ranked_cids_with_scores = [
            [cid, float(score)]
            for cid, score in zip(solution.docIDs or [], solution.doc_scores if solution.doc_scores is not None else [])
        ]
        retriever_results_list.append({"qid": qid, "ranked_cids_with_scores": ranked_cids_with_scores})

    # 4. Use bench_run.py summary_metrics in the final record
    final_retriever_record = {
        "retriever_name": retriever_name,
        "dataset_name": dataset_name,
        "summary_metrics": summary_metrics,
        "results": retriever_results_list,
    }
    current_retriever.save_ckpt(config.ckpt_path(retriever_name, dataset_name))

    gt_results_list: List[Dict[str, Any]] = []
    for i, qa_entry in enumerate(qa_data):
        qid = qa_entry.get("qid", f"NO_QID_{i}")
        gold_cids = gold_cids_map.get(qid, [])
        gt_results_list.append({"qid": qid, "ranked_cids_with_scores": [[cid, 1.0] for cid in gold_cids], "summary_metrics": {}})
    final_gt_record = {"retriever_name": "gt", "dataset_name": dataset_name, "results": gt_results_list}

    with jsonlines.open(config.RESULTS_FILE, mode='a') as writer:
        if not retriever_result_exists(config.RESULTS_FILE, dataset_name, 'gt'):
            writer.write(final_gt_record)
            logger.info("Saved GT record for %s.", dataset_name)
        writer.write(final_retriever_record)
        logger.info("Appended results for %s on %s. Summary: %s", retriever_name, dataset_name, summary_metrics)

    logger.info("--- Benchmark complete ---")
    
    # Return retriever instance for cleanup in main()
    return current_retriever


def main():
    from retrieval.retrievers import config
    
    snapshot_path = config.dump_snapshot()
    print(f"[Main] Configuration snapshot saved to: {snapshot_path}")
    
    tasks = [(r_name, d_triple) for r_name in config.RETRIEVER_LIST for d_triple in config.DATASET_LIST]
    
    print(f"--- Starting Benchmark for {len(tasks)} tasks sequentially ---")
    
    # Serial execution: one task at a time
    for i, (r_name, d_triple) in enumerate(tasks):
        dataset_name = d_triple[2]
        print(f"\n[Task {i+1}/{len(tasks)}] Running {r_name} on {dataset_name}...")
        
        try:
            # Call the function directly - no Process spawning
            retriever_instance = run_retrieval_benchmark(r_name, d_triple)
            
            # --- CLEANUP: Force memory release after each task ---
            if retriever_instance is not None:
                del retriever_instance
            
            # Force Python garbage collection (clears CPU memory)
            gc.collect()
            
            # Clear GPU cache if PyTorch is available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                print(f"[Cleanup] GPU cache cleared. Allocated: {allocated_gb:.2f} GB")
            
        except Exception as e:
            # Catch errors so one failure doesn't stop the entire benchmark
            print(f"!!! CRITICAL FAILURE on {r_name}/{dataset_name} !!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next task instead of crashing
    
    print("\n--- All tasks complete. ---")


if __name__ == '__main__':
    main()