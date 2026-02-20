import json
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

def load_and_prepare_dataset(corpus_path: str, qa_path: str, subsample_corpus_size: int = -1, subsample_qa_size: int = -1) -> Tuple[List[Tuple[int, str]], List[str], Dict[str, List[int]], List[Dict[str, Any]], Dict[int, str]]:
    """
    Loads corpus and QA data, extracts docs (with cID), queries, and gold cIDs.
    
    Returns: (docs_with_cid, queries, gold_cids_map, qa_data, cid_to_title)
    """
    
    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, 'r') as f:
        corpus_data = json.load(f)
    
    print(f"Loading QA set from: {qa_path}")
    with open(qa_path, 'r') as f:
        qa_data_full = json.load(f)
    
    # 1. Prepare Corpus (docs for indexing)
    
    if subsample_corpus_size > 0:
        corpus_data = corpus_data[:subsample_corpus_size]
        print(f"Corpus subsampled to {len(corpus_data)} chunks.")

    # Docs must be a list of (cID, text) tuples for the extended interface
    docs_with_cid: List[Tuple[int, str]] = [(chunk['cid'], chunk['text']) for chunk in tqdm(corpus_data, desc="Extracting Docs with cID")]
    cid_ttl_chunk: List[Tuple[int, str, str]] = [(chunk['cid'], chunk['title'], chunk['text']) for chunk in tqdm(corpus_data, desc="Extracting Docs with cID")]
    
    # Build mapping from integer cid to string title for evaluation
    cid_to_title: Dict[int, str] = {chunk['cid']: chunk['title'] for chunk in corpus_data}
    
    # 2. Prepare QA data (queries and gold cIDs)
    if subsample_qa_size > 0:
        qa_data = qa_data_full[:subsample_qa_size]
        print(f"QA subsampled to {len(qa_data)} queries.")
    else:
        qa_data = qa_data_full
        
    queries: List[str] = []
    gold_cids_map: Dict[str, List[int]] = {}
    
    for qa_entry in tqdm(qa_data, desc="Extracting Queries & Gold cIDs"):
        qid = qa_entry.get('qid', f'NO_QID_{len(queries)}')
        queries.append(qa_entry['question'])
        
        # Flatten the list of all supporting sentences/chunks' text
        gold_text_list: List[str] = []
        gold_ttl_list: List[str] = []
        for ttl, lclIdx, all_sents in qa_entry.get('gt_title_locolSentIdx_allSents', []):
            gold_text_list.extend(all_sents)
            gold_ttl_list.append(ttl)
        
        # Determine the unique cIDs that contain these gold sentences/text
        gold_cids: List[int] = []
        # for gt_text in gold_text_list:
        for gt_ttl in gold_ttl_list:

            # for corpus_cid, corpus_text in docs_with_cid:
            for corpus_cid, title, corpus_text in cid_ttl_chunk:
                # if gt_text in corpus_text and corpus_cid not in gold_cids:
                if gt_ttl==title and corpus_cid not in gold_cids:
                    gold_cids.append(corpus_cid)
                    # break
                    
        gold_cids_map[qid] = gold_cids

    # Return the new mapping as the fifth element
    return docs_with_cid, queries, gold_cids_map, qa_data, cid_to_title