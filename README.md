### Overview
Lightweight tooling for the paper “Revisiting RAG Retrievers: An Information Theoretic Benchmark,” focusing on analyzing retriever synergy, redundancy, and contribution.

### Abstract
Retrieval-Augmented Generation (RAG) systems rely critically on the retriever module to surface relevant context for large language models. Although numerous retrievers have recently been proposed, each built on different ranking principles such as lexical matching, dense embeddings, or graph citations, there remains a lack of systematic understanding of how these mechanisms differ and overlap. Existing benchmarks primarily compare entire RAG pipelines or introduce new datasets, providing little guidance on selecting or combining retrievers themselves. Those that do compare retrievers directly use a limited set of evaluation tools which fail to capture complementary and overlapping strengths. This work presents MIGRASCOPE, a Mutual Information based RAG Retriever Analysis Scope. We revisit state-of-the-art retrievers and introduce principled metrics grounded in information and statistical estimation theory to quantify retrieval quality, redundancy, synergy, and marginal contribution. We further show that if chosen carefully, an ensemble of retrievers outperforms any single retriever. We leverage the developed tools over major RAG corpora to provide unique insights on contribution levels of the state-of-the-art retrievers. Our findings provide a fresh perspective on the structure of modern retrieval techniques and actionable guidance for designing robust and efficient RAG systems.

### Capabilities
- Execute bundled SOTA retrievers and persist checkpoints (`bench_run.py`).
- Align retriever outputs, estimate pseudo ground truth, and compute MI-based divergence (`bench_post_run.py`).
- Attribute marginal contributions and interactions (`run_attribution.py`).
- Search fusion weights for ensembles (`run_fusion_search.py`).

### Installation
```bash
conda create -y -n migraenv python=3.10
conda activate migraenv
pip install -r requirements.txt
```

### Configuration (Required Before Run)
Before initiating the benchmark, you must configure your environment settings in `retrieval/config.py`:
- **API Endpoints:** Set `LLM_BASE_URL` and `EMBEDDING_BASE_URL` to point to your active LLM and embedding service endpoints (e.g., a local vLLM instance or remote provider).[^1]
- **Datasets:** Update the `DATASET_LIST` tuples to point to the correct local paths for your downloaded corpus and QA JSON files.

### Benchmark Workflow

#### 1. Benchmark Retrieval
```bash
python bench_run.py
```
- **Inputs:** Configure paths in `retrieval/config.py` (e.g., dataset locations).
- **Outputs:** Top-K chunk IDs/scores per retriever saved to `RESULTS_FILE`.

#### 2. Post-process & Estimate Pseudo-GT
```bash
python bench_post_run.py
```
- **Inputs:** `RESULTS_FILE` from step 1.
- **Configuration:** Ensure LLM_BASE_URL in retrieval/config.py is active (required for scoring) and anchor_retriever is set (default: 'ALL').
- **Outputs:** `RESULTS_FILE_POST_ORG` plus aligned retriever/ground-truth JSONL files under `MI_LAYER_INPUTS` (default: `out/`).

#### 3. Attribution Analysis

Example:

```bash
python run_attribution.py \
  --dataset <DATASET> \
  --retrievers GRAG-naive-semantic QRAG-bge-m3 BM25 RAG-bge-m3 hippoRAG \
  --gaussian false \
  --datadir out \
  --outdir results/MI
```
- Consumes `out/<DATASET>/{pseudo_gt.jsonl,retrievers/*.jsonl}` and saves results to `results/MI/<DATASET>/attribution.jsonl`.
- You will notice GRAG-naive-semantic and RAG-bge-m3 are highly redundant here, thus the former will be omitted in the next step!

#### 4. Fusion Search

Example:

```bash
python run_fusion_search.py \
  --dataset <DATASET> \
  --retrievers QRAG-bge-m3 BM25 RAG-bge-m3 hippoRAG \
  --datadir out \
  --outdir results/fusion \
  --seed 42 \
  --tau 1.0 \
  --num_weights_to_search 30
```
- Splits queries into train/dev/test, tunes fusion weights, and saves evaluations to `results/fusion/<DATASET>/fusion_eval.jsonl`.

### Supported Retrievers
The initial open-source release supports the following verified retriever configurations:
* `BM25` (Lexical)
* `RAG-bge-m3` (Dense)
* `QRAG-bge-m3` (Question-generation Dense)
* `GRAG-naive-semantic` (Naive Co-occurrence Graph)
* `HippoRAG`

**Roadmap:** The academic manuscript evaluates additional experimental graph construction techniques (e.g., `LightRAG`, `GRAG-window`, `GRAG-global`). The code for these extended methodologies is planned for inclusion in future releases.

### Datasets
This benchmark evaluates retrievers across several multi-hop QA datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue, and TriviaQA.

For local testing and verification, a sample dataset is already included in this repository under the `data/` directory:
```
data/
└── hpqa/
    ├── hotpotqa_corpus.json
    └── hotpotqa.json
```

To run the pipeline on full datasets, you must download them and format them into the following JSON schemas expected by our `data_loader.py`:

#### 1. Corpus JSON Schema (`<dataset>_corpus.json`)
A list of dictionaries representing the document chunks:
```json
[
  {
    "cid": 0,
    "title": "Document Title",
    "text": "The actual text content of the chunk."
  }
]
```

#### 2. QA JSON Schema (`<dataset>.json`)
A list of dictionaries containing the queries and ground-truth supporting evidence:
```json
[
  {
    "qid": "unique_query_id",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "gt_title_locolSentIdx_allSents": [
      ["Document Title", 0, ["The actual text content of the chunk."]]
    ]
  }
]
```

### Citation
If you find this code or our framework useful in your research, please consider citing our paper (currently under review):
```bibtex
@article{migrascope2026,
  title={Revisiting RAG Retrievers: An Information Theoretic Benchmark},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---
[^1]: **CRITICAL LLM REQUIREMENT:** The pseudo-ground-truth estimation in `bench_post_run.py` relies on extracting prompt-level token log probabilities. Therefore, your LLM serving engine (e.g., a local `vLLM` instance) **must** support the legacy `/v1/completions` endpoint and the `echo=True` parameter. Standard `/v1/chat/completions` endpoints or some OpenAI models (like `gpt-4o`) that drop support for `echo=True` will crash during the post-run phase.

