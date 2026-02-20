import json
from typing import Dict, List, Tuple

def read_jsonl(path: str):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, records: List[dict], append: bool=True):
    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def load_pseudo_gt(jsonl_path: str) -> Dict[str, Tuple[List[str], List[float]]]:
    store = {}
    for obj in read_jsonl(jsonl_path):
        store[obj['qid']] = (obj['chunk_ids'], obj['scores'])
    return store

def load_retriever(jsonl_path: str) -> Dict[str, Tuple[List[str], List[float]]]:
    store = {}
    for obj in read_jsonl(jsonl_path):
        store[obj['qid']] = (obj['chunk_ids'], obj['scores'])
    return store
