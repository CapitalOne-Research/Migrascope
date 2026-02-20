from typing import List, Union, Set, Dict, Any, Optional
from copy import deepcopy
from tqdm import tqdm
import json
import os
import shutil
from pathlib import Path
import jsonlines
import concurrent.futures
from retrieval.compute_MI import calculate_part2_MI, use_a_to_sort_b
from retrieval.retrievers import config, setup_retriever
import numpy as np
import multiprocessing

JSONLine = Dict[str, Any]


def read_jsonl_file(path: Union[str, Path]) -> List[JSONLine]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_file(path: Union[str, Path], rows: List[JSONLine]) -> None:
    path = Path(path)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')


def extract_cids(jsonl_path: str, dataset_name: str, retriever_name: str, retriever_list_to_merge: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """
    Extracts a qid -> [cid_list] map.
    
    If retriever_list_to_merge is None:
        Extracts for a single retriever (retriever_name). Returns immediately upon finding the row.
        
    If retriever_list_to_merge is a List[str]:
        Ignores 'retriever_name' argument.
        Finds all rows matching dataset_name and any retriever in the list.
        For each qid, calculates the INTERSECTION of cids across these retrievers.
    """
    qid_to_cidList = {}
    
    # === Branch 1: Multiple Retriever Merge Logic (Intersection) ===
    if retriever_list_to_merge is not None:
        target_retrievers = set(retriever_list_to_merge)
        # tmp store: qid -> [set(cids_from_retriever_A), set(cids_from_retriever_B), ...]
        qid_to_sets_buffer: Dict[str, List[set]] = {} 
        
        found_retrievers_count = 0
        
        with jsonlines.open(jsonl_path, mode='r') as reader:
            for obj in reader:

                r_name = obj.get('retriever_name')
                if obj.get('dataset_name') == dataset_name and r_name in target_retrievers:
                    
                    found_retrievers_count += 1
                    
                    for result_entry in obj.get('results', []):
                        qid = result_entry['qid']

                        cids = {cid for cid, score in result_entry.get('ranked_cids_with_scores', [])}
                        
                        if qid not in qid_to_sets_buffer:
                            qid_to_sets_buffer[qid] = []
                        qid_to_sets_buffer[qid].append(cids)
        
        # Calculate the intersection
        # Note: If a QID does not exist in a Retriever, logically it should not exist in the intersection either.
        # Here we only take the intersection of the "found" results.

        for qid, list_of_cid_sets in qid_to_sets_buffer.items():
            if list_of_cid_sets:
                intersection_set = set.intersection(*list_of_cid_sets)
                qid_to_cidList[qid] = list(intersection_set)
        
        if found_retrievers_count == 0:
            print(f"Warning: No entries found for dataset '{dataset_name}' with retrievers in {retriever_list_to_merge}.")

        # make sure no empty cid list
        for k in deepcopy(qid_to_cidList):
            if len(qid_to_cidList[k])==0:
                qid_to_cidList.pop(k)

        return qid_to_cidList

    # === Branch 2: Existing Single Retriever Logic ===
    else:
        with jsonlines.open(jsonl_path, mode='r') as reader:
            for obj in reader:
                if obj.get('dataset_name') == dataset_name and obj.get('retriever_name') == retriever_name:
                    # Found the target row
                    for result_entry in obj.get('results', []):
                        qid = result_entry['qid']
                        # Extract just the cids, preserving order
                        cids = [cid for cid, score in result_entry.get('ranked_cids_with_scores', [])]
                        qid_to_cidList[qid] = cids
                    
                    return qid_to_cidList # Found it, break and return

        # If we get here, no matching row was found
        print(f"Warning: No entry found for dataset '{dataset_name}' and retriever '{retriever_name}'.")
        return qid_to_cidList


def cal_ppl_based_probs(qid: str, cid_list: List[int], answer: str, 
                          cid_to_text: Dict[int, str], 
                          qid_to_query_text: Dict[str, str]) -> List[float]:
    """
    (Placeholder) Calculates a score (e.g., PPL) for a list of CIDs based on 
    their ability to answer a query.
    [OPTIMIZED]: Uses ThreadPoolExecutor to parallelize LLM API calls.
    """
    query = qid_to_query_text[qid]
    chunk_text_list = [cid_to_text[cid] for cid in cid_list]
    
    # Helper for mapping
    def _score_single_chunk(chunk_text):
        return calculate_part2_MI(f"Question: {query}? Context: {chunk_text}\nAnswer: ", answer)

    # Use LLM_MAX_WORKERS from config
    max_workers = getattr(config, 'LLM_MAX_WORKERS', 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        scores = list(tqdm(
            executor.map(_score_single_chunk, chunk_text_list),
            total=len(chunk_text_list),
            desc=f"Scoring chunks for {qid}",
            leave=False
        ))
        
    return scores


def reinforce(cid_list: List[int], 
              labels: List[int], 
              scores: List[Union[int, float]], 
              amplifier: Union[int, float] = 5) -> List[Union[int, float]]:
    """
    Reinforces scores for items that are present in the labels list.

    This function iterates through the cid_list and scores simultaneously.
    If a content ID (cid) from cid_list is found within the labels,
    the corresponding score at the same index is multiplied by the amplifier.
    If the cid is not in labels, the score remains unchanged.

    Args:
        cid_list: A list of integer content IDs.
        labels: A list of integer "ground truth" or "target" IDs.
        scores: A list of numerical scores, corresponding 1-to-1 with cid_list.
        amplifier: The multiplicative factor to apply to scores of
                   labeled items. Defaults to 5.

    Returns:
        A new list containing the reinforced scores.
        
    Raises:
        ValueError: If cid_list and scores have different lengths.
    """
    if len(cid_list) != len(scores):
        raise ValueError("cid_list and scores must have the same length.")
        
    # Convert labels to a set for efficient O(1) average-time lookups.
    # This is much faster than checking "cid in list" inside a loop.
    label_set: Set[int] = set(labels)
    
    # Use a list comprehension to build the new list.
    # zip() efficiently pairs each cid with its corresponding score.
    reinforced_scores = [
        score * np.exp(amplifier) if cid in label_set else score
        for cid, score in zip(cid_list, scores)
    ]
    
    return reinforced_scores

def embody_gt_score(jsonl_path: str, 
                    jsonl_path_organized: str, 
                    dataset_name: str, 
                    tar_qid_to_cidList: Dict[str, List[int]],
                    cid_to_text: Dict[int, str],
                    post_topK_threshold: int,
                    qa_data: List[Dict[str, Any]]):
    """
    This function runs for a single dataset and, only 1 retriever (GT).
    Finds the 'gt' row for the specified dataset in the JSONL file and adds a aligned data
    
    This new key contains results where the CIDs are taken from tar_qid_to_cidList 
    and the scores are calculated by cal_ppl_based_probs.
    
    NOTE: This function creates a new file, which is only adding two more lines compared with original jsonl_path: gt_MI and gt_MI_reinforce
    """

    if not os.path.exists(jsonl_path_organized):
        shutil.copy2(jsonl_path, jsonl_path_organized)

    # 1. Create lookup maps from the original qa_data

    qid_to_query_text = {entry['qid']: entry['question'] for entry in qa_data}
    qid_to_answer = {entry['qid']: entry['answer'] for entry in qa_data}

    # 2. Read all data from the file
    with jsonlines.open(jsonl_path_organized, mode='r') as reader:
        for i, obj in enumerate(reader):
            if obj.get('dataset_name') == dataset_name and obj.get('retriever_name') == 'gt_MI':
                print(f"gt_MI already exists (then gt_MI_reinforce should also exist), quit to not re-make a new one at this time.")
                return
    

    gt_row_MI = None
    gt_row_MI_reinforce = None
    with jsonlines.open(jsonl_path_organized, mode='r') as reader:
        for i, obj in enumerate(reader):
            if obj.get('dataset_name') == dataset_name and obj.get('retriever_name') == 'gt':
                gt_row_MI = deepcopy(obj)
                gt_row_MI_reinforce = deepcopy(obj)

    if gt_row_MI is None:
        print(f"Error: Could not find 'gt' row for dataset '{dataset_name}'. No modifications made.")
        return

    gt_row_MI['retriever_name'] = 'gt_MI'
    gt_row_MI_reinforce['retriever_name'] = 'gt_MI_reinforce'

    qid_to_gt_sparse_labels = {}
    for dic in gt_row_MI['results']:
        qid_to_gt_sparse_labels[dic['qid']] = [x[0] for x in dic['ranked_cids_with_scores']]
        
    
    results_MI = []
    results_MI_reinforce = []
    for qid, cid_list_ext in tqdm(tar_qid_to_cidList.items()):
        if qid not in qid_to_query_text or qid not in qid_to_answer:
            raise ValueError(f"Fatal: Missing metadata for QID {qid} in qa_data.")
        
        answer = qid_to_answer[qid]

        if post_topK_threshold>0:
            cid_list_ext = cid_list_ext[:post_topK_threshold]
        
        # 4. Call the PPL function to get scores
        scores_MI = cal_ppl_based_probs( # This score is without label reinforcement.
            qid=qid,
            cid_list=cid_list_ext,
            answer=answer,
            cid_to_text=cid_to_text,
            qid_to_query_text=qid_to_query_text
        )

        labels = qid_to_gt_sparse_labels[qid]
        scores_MI_reinforce_unordered = reinforce(cid_list_ext, labels, scores_MI, amplifier = config.gamma)

        # CRITICAL: before this point, all are aligned by the order of `cid_list_ext`, e.g., `scores_MI` is unordered.
        scores_MI_reinforce, cid_list_reinforce = use_a_to_sort_b(scores_MI_reinforce_unordered, cid_list_ext, reverse=True) # after this, all are aligned by `scores_MI_reinforce_unordered`
        scores_MI_reinforce, scores_MI = use_a_to_sort_b(scores_MI_reinforce_unordered, scores_MI, reverse=True) # after this, all are aligned by `scores_MI_reinforce_unordered`
        
        # 5. Format the entry
        ranked_cids_MI = [[cid, score] for cid, score in zip(cid_list_reinforce, scores_MI)]
        ranked_cids_MI_reinforce = [[cid, score] for cid, score in zip(cid_list_reinforce, scores_MI_reinforce)]
        
        results_MI.append({
            "qid": qid,
            "ranked_cids_with_scores": ranked_cids_MI,
            "summary_metrics": {}
        })
    
        
        results_MI_reinforce.append({
            "qid": qid,
            "ranked_cids_with_scores": ranked_cids_MI_reinforce,
            "summary_metrics": {}
        })
    
    # 6. Add the new key to the 'gt' row in memory
    gt_row_MI['results'] = results_MI
    gt_row_MI_reinforce['results'] = results_MI_reinforce

    # 7. Overwrite the entire jsonl file
    all_data = []
    with jsonlines.open(jsonl_path_organized, mode='r') as reader:
        for i, obj in enumerate(reader):
            all_data.append(obj)
    all_data.append(gt_row_MI)
    all_data.append(gt_row_MI_reinforce)
    with jsonlines.open(jsonl_path_organized, mode='w') as writer:
        writer.write_all(all_data)
    print(f"Successfully added embodied results as new 'gt_MI' and 'gt_MI_reinforce' rows for {dataset_name} in {jsonl_path_organized}.")


def convert_to_MI_layer(input_jsonl_path: str, MI_output_folder: str, tar_dataset: str, retrievers_to_convert: List):
    """
    Converts a JSONL file of retriever results into the MI-layer format,
    filtering for a specific dataset_name (tar_dataset).
    """
    
    # 1. Setup Output Directory
    output_path = Path(MI_output_folder)
    retrievers_path = output_path / "retrievers"

    print(f"Targeting dataset: {tar_dataset}")
    print(f"Setting up output directory: {output_path}")
    if output_path.exists():
        print("Output folder exists. Clearing it first.")
        shutil.rmtree(output_path)

    retrievers_path.mkdir(parents=True, exist_ok=False)
    print("Created output folders.")

    # 2. Process Input File
    print(f"Reading from input file: {input_jsonl_path}")
    with open(input_jsonl_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)

            # 3. Apply the dataset filter
            if data.get('dataset_name') != tar_dataset:
                continue

            retriever_name = data.get('retriever_name')
            results_list = data.get('results', [])

            if retriever_name not in retrievers_to_convert:
                continue
            
            # 4. (Inlined) Transform results
            output_lines = []
            for item in results_list:
                qid = item.get('qid', 'unknown_qid')
                ranked_pairs = item.get('ranked_cids_with_scores', [])
                
                # Unzip pairs, handling the empty case
                chunk_ids, scores = zip(*ranked_pairs) if ranked_pairs else ([], [])
                
                # # Normalize the scores.
                scores = normalize(scores, config.SCORE_NORM_MODE)
                    
                transformed_data = {
                    "qid": qid,
                    "chunk_ids": list(chunk_ids), # convert tuples
                    "scores": list(scores)      # convert tuples
                }
                output_lines.append(json.dumps(transformed_data))
            
            # 5. Write to the correct file
            output_content = "\n".join(output_lines)
            
            if retriever_name == 'gt_MI_reinforce':
                output_file_path = output_path / "pseudo_gt.jsonl"
            else:
                output_file_path = retrievers_path / f"{retriever_name}.jsonl"
            
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                f_out.write(output_content)
                if output_content:
                    f_out.write('\n')
                        
            print(f"Successfully wrote {len(output_lines)} lines for '{retriever_name}' to {output_file_path}")

    print("\nConversion complete.")

def get_cid_to_text(ds_path):
    with open(ds_path, 'r', encoding='utf-8') as file:
        ds = json.load(file)
    cid_to_text = {}
    for dic in ds:
        cid_to_text[dic['cid']] = dic['text']
    return cid_to_text


def run_rescore(jsonl_path: str,
                dataset_name: str,
                retriever_list: List[str],
                qa_data: List[Dict[str, Any]]):

    """
    This function runs for a single dataset and ALL retrievers.
    Loads retriever results, aligns them to a target CID list by
    rescoring missing chunks, and saves the updated file.
    """
    qid_to_query_text = {entry['qid']: entry['question'] for entry in qa_data}
    all_lines_data = read_jsonl_file(jsonl_path)

    tar_qid_to_cidList = {}
    for line_data in all_lines_data:
        current_retriever_name = line_data.get('retriever_name')
        current_dataset_name = line_data.get('dataset_name')
        if (current_dataset_name == dataset_name and current_retriever_name =='gt_MI'):
            for dic in line_data['results']:
                tar_qid_to_cidList[dic['qid']] = [x[0] for x in dic['ranked_cids_with_scores']]
            break
    if not tar_qid_to_cidList:
        raise ValueError(f'gt_MI not found in {jsonl_path} under {dataset_name}.')
            

    modified_data = {}
    retrievers_found = {name: False for name in retriever_list}
    
    # [Added for thread-safe logging]
    logger = config.get_logger(f"Rescore_{dataset_name}")

    # 2. Iterate through each line (each retriever's full result)
    for line_data in all_lines_data:
        current_retriever_name = line_data.get('retriever_name')
        current_dataset_name = line_data.get('dataset_name')

        # 3. Check if this is a line we need to modify
        if (current_dataset_name == dataset_name and 
            current_retriever_name in retriever_list):
            
            retrievers_found[current_retriever_name] = True
            
            num_chunks_existing = len(line_data['results'][0]['ranked_cids_with_scores'])
            if num_chunks_existing==config.post_topK_threshold:
                print(f'skipping rescore for {current_dataset_name} + {current_retriever_name}, as in jsonl it already have be completed with expected num of top K chunks.')
                continue
            
            print(f"\n--- Processing {current_retriever_name} for {dataset_name} ---")
            
            # 4. Get the *instance* of the retriever and load its index
            retriever = setup_retriever(current_retriever_name, dataset_name, reload=True, **{**vars(config), **vars(config.__class__)})
            
            # Build a quick lookup map for the retriever's *original* results
            existing_results_map: Dict[str, Dict[int, float]] = {}
            for q_result in line_data.get('results', []):
                qid = q_result['qid']
                scores_map = {cid: score for cid, score in q_result.get('ranked_cids_with_scores', [])}
                existing_results_map[qid] = scores_map
                
            # --- PARALLEL WORKER FUNCTION (Extracted for Threading) ---
            def _process_single_query_rescore(args):
                qid, target_cids = args
                
                # Validation checks from original loop
                if qid not in qid_to_query_text: 
                    return None
                    
                query_text = qid_to_query_text[qid]
                aligned_cid_scores: List[List[Any]] = [] 
                cids_to_rescore: List[int] = []
                
                current_existing_scores = existing_results_map.get(qid, {})
                
                # 6. Find what's present and what's missing
                for cid in target_cids:
                    if cid in current_existing_scores:
                        # We have it! Add the existing score.
                        aligned_cid_scores.append([cid, current_existing_scores[cid]])
                    else:
                        # We are missing it. Add to the list to rescore.
                        cids_to_rescore.append(cid)
                
                # 7. Run rescore on *only* the missing cids
                if cids_to_rescore:
                    try:
                        # Note: This is the blocking API call that benefits from threading
                        new_scores = retriever.rescore(query_text, cids_to_rescore)
                        for cid, score in zip(cids_to_rescore, new_scores):
                            aligned_cid_scores.append([cid, score])
                    except Exception as e:
                        # [LOGGING ADDED] Use logger instead of print for thread safety
                        logger.error(f"Error rescoring qid {qid}: {e}")
                        # Fallback: append 0.0 to avoid crashing the pipeline
                        for cid in cids_to_rescore:
                            aligned_cid_scores.append([cid, 0.0])
                            
                # 8. Sort the final aligned list by score (descending)
                aligned_cid_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 9. Return the formatted entry (instead of appending directly)
                return {
                    "qid": qid,
                    "ranked_cids_with_scores": aligned_cid_scores,
                    "summary_metrics": {} 
                }
            # --- END WORKER FUNCTION ---

            # Execute parallel rescoring
            # Use EMBEDDING_MAX_WORKERS from config
            max_workers = getattr(config, 'EMBEDDING_MAX_WORKERS', 32)
            new_results_aligned = []
            work_items = list(tar_qid_to_cidList.items())
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # map maintains order, so results align with work_items if needed, 
                # but we just need the list of results here.
                results = list(tqdm(
                    executor.map(_process_single_query_rescore, work_items), 
                    total=len(work_items),
                    desc=f"Rescoring {current_retriever_name}"
                ))
                
                # Filter out failures (None)
                new_results_aligned = [r for r in results if r is not None]

            # 10. Add the new field to the line_data
            # line_data['results_pre_alignment'] = deepcopy(line_data['results'])
            line_data['results'] = new_results_aligned
            modified_data[(current_dataset_name, current_retriever_name)] = line_data
            print(f"--- Finished processing {current_retriever_name} ---")
        
        else:
            # This line is not one we're processing, add it as-is
            continue

    # 11. Check if we found all requested retrievers
    for name, found in retrievers_found.items():
        if not found:
            # This is a critical error, as requested
            raise ValueError(f"Retriever '{name}' for dataset '{dataset_name}' was not found in {jsonl_path}")

    # 12. Write all data (original and modified) back to the file. Need to read again in case of async.
    all_lines_data = read_jsonl_file(jsonl_path)
    for idx, line_data in enumerate(all_lines_data):
        current_retriever_name = line_data.get('retriever_name')
        current_dataset_name = line_data.get('dataset_name')
        if (current_dataset_name, current_retriever_name) in modified_data:
            all_lines_data[idx] = modified_data[(current_dataset_name, current_retriever_name)]
    write_jsonl_file(jsonl_path, all_lines_data)
    print(f"\nRescore and alignment complete. File updated: {jsonl_path}")


def normalize(vec, which: str):
    """
    vec: 1D list/np.ndarray of non-negative scores
    which:
        - "max"
        - "softmax_tau_max"
        - "softmax_tau_5max"
        - "shift_0_5_softmax"
    """

    x = np.asarray(vec, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input vec must be 1D.")

    n = x.size
    if n == 0:
        return list(x)

    if which == "max":
        # Max normalization: x_i' = x_i / max(x)
        m = x.max()
        if m == 0:
            # All zeros -> return zeros
            return np.zeros_like(x)
        return list(x / m)

    elif which == "softmax_tau_max":
        # Softmax with tau = max(x_i)
        m = x.max()
        if m <= 0:
            # All zeros or negative (degenerate) -> uniform distribution
            return list(np.ones_like(x) / n)
        tau = m
        # Stable softmax: exp((x/tau) - max(x/tau))
        z = x / tau
        z = z - z.max()
        e = np.exp(z)
        return list(e / e.sum())

    elif which == "softmax_tau_5max":
        # Softmax with tau = 5 * max(x_i)
        m = x.max()
        if m <= 0:
            # Degenerate -> uniform
            return list(np.ones_like(x) / n)
        tau = 5.0 * m
        z = x / tau
        z = z - z.max()
        e = np.exp(z)
        return list(e / e.sum())

    elif which == "shift_0_5_softmax":
        # 1) Shift+scale to [0,5]: x'_i = 5 * (x_i - min)/(max-min)
        # 2) Softmax on x'
        xmin = x.min()
        xmax = x.max()
        if xmax == xmin:
            # All equal -> softmax of constant -> uniform
            return np.ones_like(x) / n
        x_scaled = 5.0 * (x - xmin) / (xmax - xmin)
        z = x_scaled - x_scaled.max()
        e = np.exp(z)
        return list(e / e.sum())

    else:
        raise ValueError(f"Unknown mode: {which}")

def align_all_retriever_chunk_order_to_gt(jsonl_path, dataset_name, retrievers_to_align):
    all_lines_data = read_jsonl_file(jsonl_path)
    tar_qid_to_cidList = {}
    gt_line = None
    ids = []
    gt_idx = None
    for il, line_data in enumerate(all_lines_data):
        if line_data.get('dataset_name') != dataset_name:
            continue
        if line_data.get('retriever_name')=='gt_MI_reinforce':
            gt_line = line_data
            gt_idx = il
        elif line_data.get('retriever_name') in retrievers_to_align:
            ids.append(il)
        
    if gt_line is None:
        raise ValueError(f'gt_MI_reinforce not in {dataset_name} for {jsonl_path}')
    
    qids_in_order = []
    cids_in_order = []
    discarded_qids = []  # some questions have all 0 due to embody error; discard or re-run previous steps.
    new_gt_results = []
    for dic in gt_line['results']:
        if any([x[1]==0 for x in dic['ranked_cids_with_scores']]):
            discarded_qids.append(dic['qid'])
            continue
        else:

            new_gt_results.append(dic)
            qids_in_order.append(dic['qid'])
            cids_in_order.append([x[0] for x in dic['ranked_cids_with_scores']])

    all_lines_data[gt_idx]['results'] = new_gt_results


    for idx in ids:

        def cids_reordered_aligned_(res, qids_in_order, cids_in_order):
            res_reformat_dic = {x['qid']: x['ranked_cids_with_scores'] for x in res}
            new_res = []
            for i in range(len(qids_in_order)):  # each qid

                cur_qid = qids_in_order.pop(0)
                cur_cids = cids_in_order.pop(0)

                new_res.append({'qid': cur_qid, 'ranked_cids_with_scores': res_reformat_dic[cur_qid]})

                assert new_res[-1]['qid']==cur_qid
                def sort_a2_by_b1_order(a2, b1):
                    lookup_map = {item[0]: item for item in a2}
                    sorted_a2 = [lookup_map[key] for key in b1]
                    return sorted_a2
                new_res[-1]['ranked_cids_with_scores'] = sort_a2_by_b1_order(new_res[-1]['ranked_cids_with_scores'], cur_cids)
            assert len(qids_in_order)==len(cids_in_order)==0
            return new_res
        all_lines_data[idx]['results'] = cids_reordered_aligned_(all_lines_data[idx]['results'], deepcopy(qids_in_order), deepcopy(cids_in_order))
        
    write_jsonl_file(jsonl_path, all_lines_data)
    print(f"\nAll lines of dataset {dataset_name} aligned to gt_MI_reinforce, File updated: {jsonl_path}")

    return

def post_organize_rerun_pipeline(dataset_triple):
    '''
    After all retrievers under current dataset has finished running:
        1. Unify HippoRAG and LightRAG chunks, set as ck_supports
        2. Embody GT on ck_supports, save to same jsonl
        3(optional*). go back and run other RAG approachs (for missing chunks give them scores), save to same jsonl
        4. Load HippoRAG, LightRAG, GT, other RAG jsonl, convert to expected format for MI layer testing
    '''

    ds_corpus_path, ds_qa_path, tar_dataset = dataset_triple
    
    logger = config.get_logger(f"PostRun_{tar_dataset}")
    logger.info(f"Starting post-run organization for {tar_dataset}")

    input_jsonl_path = config.RESULTS_FILE
    jsonl_path_organized = config.RESULTS_FILE_POST_ORG

    tar_qid_to_cidList = extract_cids(jsonl_path = input_jsonl_path, dataset_name = tar_dataset, retriever_name=config.anchor_retriever, retriever_list_to_merge=config.RETRIEVER_LIST if config.anchor_retriever=='ALL' else None)
    
    cid_to_text = get_cid_to_text(ds_corpus_path)

    with open(ds_qa_path, 'r', encoding='utf-8') as qa_file:
        qa_data = json.load(qa_file)

    logger.info("Embodying GT scores...")
    embody_gt_score(jsonl_path = input_jsonl_path,
                    jsonl_path_organized = jsonl_path_organized,
                    dataset_name = tar_dataset,
                    tar_qid_to_cidList = tar_qid_to_cidList,
                    cid_to_text = cid_to_text,
                    post_topK_threshold = config.post_topK_threshold,
                    qa_data = qa_data)
    
    if config.anchor_retriever!='ALL':
        logger.info("Running rescore for retrievers...")
        run_rescore(jsonl_path = jsonl_path_organized,
                    dataset_name = tar_dataset,
                    retriever_list=config.RETRIEVER_LIST,
                    qa_data = qa_data)

    logger.info("Aligning retriever chunk orders to GT...")
    align_all_retriever_chunk_order_to_gt(jsonl_path = jsonl_path_organized,
                                            dataset_name = tar_dataset,
                                            retrievers_to_align = config.RETRIEVER_LIST+['gt_MI', 'gt_MI_reinforce'])

    logger.info("Converting to MI layer format...")
    convert_to_MI_layer(input_jsonl_path=jsonl_path_organized, 
                        MI_output_folder=f'{config.MI_LAYER_INPUTS}/{tar_dataset}', 
                        tar_dataset=tar_dataset,
                        retrievers_to_convert=config.RETRIEVER_LIST+['gt_MI', 'gt_MI_reinforce'])
    
    logger.info(f"Post-run organization complete for {tar_dataset}")
    return


def main():
    for dataset_triple in config.DATASET_LIST:
        p = multiprocessing.Process(target=post_organize_rerun_pipeline, args=(dataset_triple,))
        p.start()
        print(f"[Main] Launched PID {p.pid} -> {dataset_triple[-1]}")
    print("--- All tasks dispatched. Main script exiting. ---")
    print("Check the individual .log files for progress.")
    print("Use 'top' or 'htop' to monitor the PIDs printed above.")


if __name__ == '__main__':
    main()