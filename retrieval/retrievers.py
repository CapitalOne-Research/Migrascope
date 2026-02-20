import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# Configure tqdm for clean output in serial execution
tqdm.monitor_interval = 0

# ==================================================================
# ARCHITECTURE FIX: Apply patches BEFORE importing HippoRAG
# ==================================================================
from retrieval.patches import apply_hipporag_patches
apply_hipporag_patches()
# ==================================================================

from hipporag import HippoRAG
from openai import OpenAI
from retrieval.config import BenchmarkConfig
from retrieval.graph_retrievers import GraphRAG, QuerySolution

# Set up configuration
config = BenchmarkConfig()


class DummyRetriever:
    """A placeholder retriever that returns fixed, unsorted cIDs for testing."""
    def __init__(self, **kwargs):
        pass

    def index(self, docs: List[Tuple[int, str]]):
        # Docs here is a list of (cID, text) tuples
        self.doc_cids = [cid for cid, text in docs]

    def retrieve(self, queries: List[str], gold_docs: List[List[str]], num_to_retrieve: int) -> Tuple[List[Any], Dict[str, float]]:
        
        # Simulate retrieval results: return the first 3 cIDs unsorted for every query
        retrieval_results = []
        for i, query in enumerate(queries):
            # Simulate QuerySolution structure
            solution = QuerySolution(
                question=query, 
                # Simulate docIDs being a list of cIDs
                docIDs=self.doc_cids[:min(3, len(self.doc_cids))], 
                doc_scores=np.array([0.9, 0.8, 0.7][:min(3, len(self.doc_cids))]),
                docs=[f"Simulated doc for {cid}" for cid in self.doc_cids[:min(3, len(self.doc_cids))]],
                answer=None, gold_answers=None, gold_docs=gold_docs[i]
            )
            retrieval_results.append(solution)

        # Simulate placeholder metrics
        summary_metrics = {'Recall@1': 0.0, 'Recall@5': 0.0}
        return retrieval_results, summary_metrics
    

class RetrieverWrapper:
    """
    Wrapper for retrievers that don't support external docID input during indexing,
    mapping internal results back to external cIDs.
    """
    def __init__(self, base_retriever_cls, **kwargs):
        self.base_retriever = base_retriever_cls(**kwargs)
        self.cid_map: Dict[str, int] = {} # Map: chunk text -> external cID
        self.cid_list: List[int] = [] # Ordered list of cIDs for algorithms that use internal indexing
        
    def index(self, docs: List[Tuple[int, str]]):
        """
        Indexes documents and builds the text-to-cID map.
        The base retriever is indexed only with raw text strings.
        """
        raw_texts = []
        self.cid_map = {}
        self.cid_list = []
        
        for cid, text in docs:
            raw_texts.append(text)
            self.cid_map[text] = cid
            self.cid_list.append(cid)
            
        # The base retriever indexes raw text strings
        self.base_retriever.index(raw_texts) 

    def retrieve(self, queries: List[str], gold_docs: List[List[str]], num_to_retrieve: int) -> Tuple[List[QuerySolution], Dict[str, float]]:
        """
        Calls base retriever's retrieve method and maps retrieved text back to cIDs.
        """
        # Call base retriever but ignore its internally calculated metrics
        retrieval_results, _ = self.base_retriever.retrieve(
            queries=queries, 
            gold_docs=gold_docs, 
            num_to_retrieve=num_to_retrieve
        )
        
        # Post-process: map text back to cID
        processed_results = []
        for solution in retrieval_results:
            retrieved_cids = []
            
            # For each retrieved text, find the original cID
            for doc_text in solution.docs:
                cid = self.cid_map.get(doc_text)
                if cid is not None:
                    retrieved_cids.append(cid)
                # Note: If a doc_text is not found, we skip its cID.
                
            # Create a new QuerySolution with docIDs as cIDs
            processed_solution = QuerySolution(
                question=solution.question,
                docs=solution.docs,
                docIDs=retrieved_cids,
                doc_scores=solution.doc_scores,
                gold_docs=solution.gold_docs,
                answer=getattr(solution, 'answer', None),
                gold_answers=getattr(solution, 'gold_answers', None)
            )
            processed_results.append(processed_solution)
        
        # Return placeholder metrics to satisfy bench_run.py signature
        return processed_results, {'Recall@K_placeholder': 0.0}
    def rescore(self, query: str, chunk_IDs: List[int]) -> List[float]:
        """
        Delegates rescoring to the base retriever.
        If not implemented, warns user about configuration restrictions.
        
        Args:
            query: The query string to score against
            chunk_IDs: List of chunk IDs to score
            
        Returns:
            List of scores (one per chunk)
            
        Raises:
            NotImplementedError: If base retriever lacks rescore capability
        """
        if hasattr(self.base_retriever, 'rescore'):
            return self.base_retriever.rescore(query, chunk_IDs)

        # Fallback for retrievers like hippoRAG that lack an easy rescore mechanism
        retriever_type = type(self.base_retriever).__name__
        retriever_name = getattr(self.base_retriever, 'retriever_name', retriever_type)
        
        logger = logging.getLogger(__name__)
        logger.critical("RESCORE NOT IMPLEMENTED: %s", retriever_type)
        logger.critical("Set anchor_retriever = '%s' in config.py", retriever_name)

        raise NotImplementedError(f"{retriever_type} does not support rescoring. Set it as the anchor_retriever.")

    def save_ckpt(self, *w):
        return
    def load_ckpt(self, *w):
        return



class BM25:
    """
    A simple BM25 retriever using the 'rank-bm25' library.
    """
    def __init__(self, k1=1.2, b=0.75, **kwargs):
        """
        Initializes the BM25 model.
        :param k1: BM25's k1 hyperparameter (controls term frequency saturation)
        :param b: BM25's b hyperparameter (controls document length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        
        # We need to store cIDs and their text to return them in retrieve()
        # self.cid_map: maps internal index (0, 1, 2...) -> your cID
        # self.doc_store: maps your cID -> original text string
        self.cid_map: Dict[int, int] = {} 
        self.doc_store: Dict[int, str] = {}
        self.cid_to_idx: Dict[int, int] = {}  # Reverse map: cID -> internal index

    def index(self, docs: List[Tuple[int, str]]):
        """
        Indexes the documents for BM25.
        :param docs: A list of (cID, text) tuples.
        """
        print(f"Indexing {len(docs)} documents for BM25...")
        
        tokenized_corpus = []
        for i, (cid, text) in enumerate(docs):
            self.cid_map[i] = cid      # Map internal index i to your external cID
            self.cid_to_idx[cid] = i   # Reverse map: cID to internal index
            self.doc_store[cid] = text # Store original text for retrieval
            
            # Simple tokenization (BM25 expects a list of words)
            words = text.lower().split()
            tokenized_corpus.append(words)

        # Create the BM25 index from the tokenized corpus
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print("Indexing complete.")

    def retrieve(self, queries: List[str], gold_docs: List[List[str]], num_to_retrieve: int) -> Tuple[List[QuerySolution], Dict[str, float]]:
        """
        Retrieves documents for a list of queries.
        """
        if self.bm25 is None:
            raise ValueError("You must call .index() before .retrieve()")

        retrieval_results = []
        
        for i, query in enumerate(queries):
            # 1. Tokenize the query
            tokenized_query = query.lower().split()
            
            # 2. Get scores for all documents in the index
            #    This is an array of N scores, where N = total docs in index
            all_doc_scores = self.bm25.get_scores(tokenized_query)
            
            # 3. Get the top-N *internal indices*
            #    np.argsort is ascending, so [::-1] reverses it for descending scores
            top_n_indices = np.argsort(all_doc_scores)[::-1][:num_to_retrieve]
            
            # 4. Extract the scores for these top-N
            top_n_scores = all_doc_scores[top_n_indices]
            
            # 5. Map the internal indices (top_n_indices) back to your cIDs
            top_n_cids = [self.cid_map[idx] for idx in top_n_indices]
            
            # 6. Get the original document text using the cIDs
            top_n_docs = [self.doc_store[cid] for cid in top_n_cids]

            # 7. Build the QuerySolution
            solution = QuerySolution(
                question=query,
                docIDs=top_n_cids,
                doc_scores=top_n_scores,
                docs=top_n_docs,
                gold_docs=gold_docs[i]
            )
            retrieval_results.append(solution)

        summary_metrics = {'Recall@K_placeholder': 0.0}
        return retrieval_results, summary_metrics

    def rescore(self, query: str, chunk_IDs: List[int]) -> List[float]:
        """
        Scores a list of new/unseen chunk texts against a single query.
        This uses the corpus IDF statistics from the .index() call.
        
        :param query: The query string.
        :param chunk_IDs: A list of chunk IDs to score.
        :return: A list of BM25 scores, corresponding to each chunk_ID.
        """
        if self.bm25 is None:
            raise ValueError("You must call .index() before .rescore()")
        
        # 1. Tokenize the query
        tokenized_query = query.lower().split()
        
        # 2. Get scores for all documents in the index
        all_doc_scores = self.bm25.get_scores(tokenized_query)

        # 3. Use the reverse map for safe lookups
        scores = [all_doc_scores[self.cid_to_idx[cid]] for cid in chunk_IDs]
        
        return list(scores)

    def save_ckpt(self, save_dir: str):
        """Saves the indexed BM25 data to a directory."""
        
        if self.bm25 is None:
            print("BM25 Warning: Attempting to save an un-indexed retriever.")
            
        # Define file paths
        doc_map_file = os.path.join(save_dir, 'bm25.doc_map.jbl')
        cid_map_file = os.path.join(save_dir, 'bm25.cid_map.jbl')
        idx_map_file = os.path.join(save_dir, 'bm25.idx_map.jbl')
        bm25_model_file = os.path.join(save_dir, 'bm25.model.jbl')
        
        # Save components
        joblib.dump(self.doc_store, doc_map_file)
        joblib.dump(self.cid_map, cid_map_file)
        joblib.dump(self.cid_to_idx, idx_map_file)
        joblib.dump(self.bm25, bm25_model_file)
        
        print(f"BM25: Checkpoint successfully saved to {save_dir}")

    def load_ckpt(self, save_dir: str):
        """Loads an indexed BM25 checkpoint from a directory."""

        # Define file paths
        doc_map_file = os.path.join(save_dir, 'bm25.doc_map.jbl')
        cid_map_file = os.path.join(save_dir, 'bm25.cid_map.jbl')
        idx_map_file = os.path.join(save_dir, 'bm25.idx_map.jbl')
        bm25_model_file = os.path.join(save_dir, 'bm25.model.jbl')

        if not all(os.path.exists(f) for f in [doc_map_file, cid_map_file, idx_map_file, bm25_model_file]):
            raise FileNotFoundError(f"BM25: Checkpoint not found or incomplete in {save_dir}")
            
        # Load components
        self.doc_store = joblib.load(doc_map_file)
        self.cid_map = joblib.load(cid_map_file)
        self.cid_to_idx = joblib.load(idx_map_file)
        self.bm25 = joblib.load(bm25_model_file)
        
        print(f"BM25: Checkpoint successfully loaded from {save_dir}")
        print(f"BM25: Loaded {len(self.cid_map)} chunk mappings and model.")

def batch_embed_texts(texts, url, model, batch_size=8, max_retries=5, timeout=90):
    """Embeds texts in batches using a remote API endpoint."""
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Embedding batches") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for attempt in range(max_retries):
                try:
                    payload = {"model": model, "input": batch}
                    response = requests.post(f"{url}/embeddings", json=payload, timeout=timeout)
                    response.raise_for_status()
                    embeddings = [item["embedding"] for item in response.json()["data"]]
                    all_embeddings.extend(embeddings)
                    break # Success
                except Exception as e:
                    print(f"Batch {i//batch_size+1}: Embedding request failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise # Max retries reached
            time.sleep(0.2) # Small delay to avoid hammering
            pbar.update(1)
    return np.array(all_embeddings)

class RAG:
    """
    A RAG retriever that implements index and retrieve methods.
    It can operate in two modes based on __init__ kwargs['mode']:
    1. 'api': Uses a remote embedding URL and model.
    2. 'local': Uses a local SentenceTransformer model.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the retriever.
        
        Kwargs:
            embedding_url (str, optional): URL for the embedding API.
            embedding_model (str, optional): Model name for the API.
            batch_size (int, optional): Batch size for embedding. Defaults to 256.
        
        If 'embedding_url' and 'embedding_model' are provided,
        mode is set to 'api'. Otherwise, it defaults to 'local'.
        """
        self.embedding_url = kwargs.get('EMBEDDING_BASE_URL', None)
        self.embedding_model_name = kwargs.get('EMBEDDING_MODEL_NAME', None)
        
        # --- Configurable batch size ---
        self.batch_size = kwargs.get('batch_size', 256)
        
        # --- Storage for indexed data ---
        self.doc_map: Dict[int, str] = {}
        self.chunk_ids: List[int] = []
        self.normalized_chunk_embeddings: Optional[np.ndarray] = None
        self.name = kwargs['retriever_name']
        
        # --- Mode selection ---
        if 'bge' in kwargs['retriever_name']:
            self.mode = 'api'
            print(f"SimpleRAGRetriever initialized in 'api' mode.")
            print(f"URL: {self.embedding_url}, Model: {self.embedding_model_name}")
        else:
            self.mode = 'local'
            self.local_model_name = 'all-mpnet-base-v2'
            
            from sentence_transformers import SentenceTransformer
            
            if SentenceTransformer is None:
                raise ImportError(
                    "SentenceTransformers library is required for 'local' mode"
                    " but is not installed. Please run: pip install sentence-transformers"
                )
            print(f"SimpleRAGRetriever initialized in 'local' mode.")
            print(f"Loading model: {self.local_model_name}")
            self.local_model = SentenceTransformer(self.local_model_name)
            print("Model loaded.")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Internal helper to embed texts based on the current mode."""
        if self.mode == 'api':
            if not self.embedding_url or not self.embedding_model_name:
                raise ValueError("API mode selected but 'embedding_url' or 'embedding_model' is missing.")
            return batch_embed_texts(
                texts,
                self.embedding_url,
                self.embedding_model_name,
                batch_size=self.batch_size,
            )
        else:
            # Assumes self.mode == 'local'
            return self.local_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True
            )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizes embeddings for efficient cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Add epsilon to prevent division by zero
        return embeddings / (norms + 1e-8)

    def index(self, docs: List[Tuple[int, str]]):
        """
        Indexes a list of documents.
        
        Args:
            docs: A list of (cID, text) tuples.
        """
        if not docs:
            print("Warning: No documents provided to index.")
            return

        print(f"Indexing {len(docs)} documents...")
        self.doc_map = {}
        self.chunk_ids = []
        doc_texts = []

        for cid, text in docs:
            self.doc_map[cid] = text
            self.chunk_ids.append(cid)
            doc_texts.append(text)
            
        # Embed and normalize all documents
        chunk_embeddings = self._embed(doc_texts)
        self.normalized_chunk_embeddings = self._normalize(chunk_embeddings)
        
        print(f"Indexing complete. {len(self.chunk_ids)} embeddings stored and normalized.")

    def retrieve(self, queries: List[str], gold_docs: List[List[str]], num_to_retrieve: int) -> Tuple[List[QuerySolution], Dict[str, float]]:
        """
        Retrieves the top 'num_to_retrieve' documents for each query.
        
        Args:
            queries: A list of query strings.
            gold_docs: A list of lists, where each inner list contains
                       gold standard document strings for the corresponding query.
            num_to_retrieve: The number of documents to retrieve per query.
            
        Returns:
            A tuple containing:
            1. A list of QuerySolution objects, one for each query.
            2. A dictionary of summary metrics (placeholder).
        """
        if self.normalized_chunk_embeddings is None:
            raise Exception("Retriever has not been indexed. Call .index() first.")

        print(f"Retrieving for {len(queries)} queries...")
        
        # --- 1. Embed all queries ---
        query_embeddings = self._embed(queries)
        normalized_query_embeddings = self._normalize(query_embeddings)
        
        # --- 2. Calculate all-pairs cosine similarity ---
        # This is a fast matrix multiplication:
        # (num_queries, embed_dim) @ (embed_dim, num_docs) = (num_queries, num_docs)
        all_sims = normalized_query_embeddings @ self.normalized_chunk_embeddings.T
        
        # --- 3. Process results for each query ---
        retrieval_results = []
        for i, query in enumerate(queries):
            # Get the similarity scores for this specific query
            chunk_sims = all_sims[i]
            
            # Get the indices of the top-k highest scores
            # np.argsort returns smallest to largest, so we reverse it
            top_k_indices = np.argsort(chunk_sims)[::-1][:num_to_retrieve]
            
            # Map indices back to cIDs, scores, and document text
            retrieved_cids = [self.chunk_ids[idx] for idx in top_k_indices]
            retrieved_scores = chunk_sims[top_k_indices]
            retrieved_docs = [self.doc_map[cid] for cid in retrieved_cids]
            
            # Get corresponding gold docs if available
            current_gold_docs = gold_docs[i] if gold_docs and i < len(gold_docs) else []

            # Construct the QuerySolution object
            solution = QuerySolution(
                question=query,
                docIDs=retrieved_cids,
                doc_scores=retrieved_scores,
                docs=retrieved_docs,
                gold_docs=current_gold_docs,
                answer=None,
                gold_answers=None
            )
            retrieval_results.append(solution)

        # Return placeholder metrics (evaluation now handled in bench_run.py)
        summary_metrics = {'Recall@K_placeholder': 0.0}
        print("Retrieval complete.")
        
        return retrieval_results, summary_metrics

    def rescore(self, query: str, chunk_IDs: List[int]) -> List[float]:
        """
        Calculates the cosine similarity score between a query and a list of
        new chunk texts. This method does not depend on the indexed documents,
        only on the embedding model.
        
        Args:
            query: The query string.
            chunk_texts: A list of new document/chunk strings to score.
            
        Returns:
            A list of cosine similarity scores, one for each chunk_text.
        """
        if not chunk_IDs:
            return []
            
        print(f"Rescoring {len(chunk_IDs)} chunks against query '{query[:50]}...'")
        
        chunk_texts = [self.doc_map[i] for i in chunk_IDs]

        # --- 1. Embed query and new chunks ---
        # _embed already returns a 2D numpy array (e.g., shape (1, 768))
        query_embedding = self._embed([query]) 
        
        # This returns shape (num_chunks, 768)
        chunk_embeddings = self._embed(chunk_texts)
        
        # --- 2. Normalize all embeddings for cosine similarity ---
        normalized_query_embedding = self._normalize(query_embedding)
        normalized_chunk_embeddings = self._normalize(chunk_embeddings)
        
        # --- 3. Calculate cosine similarity (dot product) ---
        # Matrix multiplication:
        # (1, embed_dim) @ (embed_dim, num_chunks) = (1, num_chunks)
        all_sims = normalized_query_embedding @ normalized_chunk_embeddings.T
        
        # --- 4. Return as a 1D list ---
        # all_sims[0] extracts the 1D array of scores from the (1, N) matrix
        return list(all_sims[0])
    
    def save_ckpt(self, save_dir: str):
        
        if self.normalized_chunk_embeddings is None:
            print("RAG Warning: Attempting to save an un-indexed retriever.")
            
        # Define file paths
        doc_map_file = os.path.join(save_dir, 'rag.doc_map.jbl')
        chunk_ids_file = os.path.join(save_dir, 'rag.chunk_ids.jbl')
        embeddings_file = os.path.join(save_dir, 'rag.embeddings.jbl')
        
        # Save components
        joblib.dump(self.doc_map, doc_map_file)
        joblib.dump(self.chunk_ids, chunk_ids_file)
        joblib.dump(self.normalized_chunk_embeddings, embeddings_file)
        
        print(f"RAG: Checkpoint successfully saved to {save_dir}")

    def load_ckpt(self, save_dir: str):
            
        # Define file paths
        doc_map_file = os.path.join(save_dir, 'rag.doc_map.jbl')
        chunk_ids_file = os.path.join(save_dir, 'rag.chunk_ids.jbl')
        embeddings_file = os.path.join(save_dir, 'rag.embeddings.jbl')

        if not all(os.path.exists(f) for f in [doc_map_file, chunk_ids_file, embeddings_file]):
            raise FileNotFoundError(f"RAG: Checkpoint not found or incomplete in {save_dir}")
            
        # Load components
        self.doc_map = joblib.load(doc_map_file)
        self.chunk_ids = joblib.load(chunk_ids_file)
        self.normalized_chunk_embeddings = joblib.load(embeddings_file)
        
        print(f"RAG: Checkpoint successfully loaded from {save_dir}")
        print(f"RAG: Loaded {len(self.chunk_ids)} chunk embeddings and {len(self.doc_map)} docs.")

class QRAG(RAG):
    """
    A "Question" RAG (QRAG) retriever.
    
    This retriever indexes documents by first generating a set of
    "simulated questions" for each chunk, then embedding those questions.
    """
    
    # This prompt is now split into a system and user message
    # in _call_llm_to_generate_questions
    QUESTION_GENERATION_PROMPT_TEMPLATE = """
Below is a paragraph. Your task is to generate a list of questions that a 
user might ask about this paragraph.

Key Requirements:
1. The questions must be answerable *only* using the information in the paragraph.
2. The questions should cover the main topics and key details of the paragraph.
3. Do not ask questions that are too broad or require external knowledge.
4. Output *only* a numbered list of questions, e.g., "1. What is...\\n2. How does...".
5. Do not add any other text before or after the list.

Paragraph:
---
{chunk}
---
Questions:
"""

    def __init__(self, **kwargs):
        """
        Initializes the QRAG retriever.
        
        Inherits 'embedding_url' and 'embedding_model' from RAG.
        
        Kwargs:
            llm_base_url (str): The base URL for the OpenAI-compatible LLM API.
            llm_model_name (str): The model name for the LLM.
            llm_api_key (str): The API key (can be a dummy key for local servers).
            llm_max_retries (int): Num of retries for the LLM on failure.
            retrieval_oversampling_factor (int): How many questions to
                fetch to find 'num_to_retrieve' unique chunks.
        """
        # Initialize the parent RAG class (for _embed, _normalize, etc.)
        super().__init__(**kwargs)
        
        # --- NEW: LLM Client Configuration ---
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed. Please run: pip install openai")
            
        self.llm_base_url = kwargs.get('LLM_BASE_URL')
        self.llm_model_name = kwargs.get('LLM_MODEL_NAME')
        self.llm_api_key = kwargs.get('llm_api_key', 'DUMMY_API_KEY')
        
        self.llm_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
            
        print(f"QRAG: LLM client initialized for model {self.llm_model_name} at {self.llm_base_url}")
        
        # --- Other QRAG config ---
        self.llm_max_retries = kwargs.get('llm_max_retries', 3)
        self.retrieval_oversampling_factor = kwargs.get('retrieval_oversampling_factor', 5)
        
        # --- NEW: Capture safety limit for question generation ---
        self.question_gen_max_tokens = kwargs.get('QUESTION_GEN_MAX_TOKENS', 512)
        
        # --- QRAG-specific storage ---
        # self.doc_map is inherited from RAG and stores cID -> text
        self.simulated_questions: List[str] = []
        self.question_to_cid_map: List[int] = []
        self.normalized_question_embeddings: Optional[np.ndarray] = None
        
        # --- NEW: Reverse map for rescore ---
        # Maps cID -> list of question indices in self.simulated_questions
        self.cid_to_question_indices: Dict[int, List[int]] = {}
        
        print(f"QRAG: Initialized.")


    def _call_llm_to_generate_questions(self, chunk_text: str) -> List[str]:
        """Calls the configured LLM API to simulate questions."""
        user_prompt = self.QUESTION_GENERATION_PROMPT_TEMPLATE.format(chunk=chunk_text)
        
        for attempt in range(self.llm_max_retries):
            try:
                # --- NEW: Use the OpenAI client ---
                completion = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=False,
                    temperature=0.0,
                    max_tokens=self.question_gen_max_tokens,  # Hard cap on output length
                )
                
                response_content = completion.choices[0].message.content
                # --- END: New client code ---

                # Parse the output
                parsed_questions = self._parse_llm_output(response_content)
                if parsed_questions:
                    return parsed_questions
                else:
                    # LLM gave an empty or malformed response
                    raise ValueError(f"LLM returned unparseable output: {response_content[:100]}...")

            except Exception as e:
                print(f"QRAG: LLM call failed for chunk (Attempt {attempt + 1}/{self.llm_max_retries}). Error: {e}")
                time.sleep(1.0 * (attempt + 1)) # Exponential backoff
        
        print(f"QRAG Warning: Failed to generate questions for chunk after {self.llm_max_retries} retries.")
        return []

    def _parse_llm_output(self, llm_response: str) -> List[str]:
        questions = []
        for line in llm_response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('.', 1)
            if len(parts) == 2 and parts[0].isdigit():
                questions.append(parts[1].strip())
            elif len(line) > 5:
                questions.append(line)
                
        return questions

    def index(self, docs: List[Tuple[int, str]]):
        """
        Indexes documents by generating and embedding questions for each chunk.
        Parallelized using LLM_MAX_WORKERS.
        
        [MODIFIED] Also builds the cid_to_question_indices map.
        """
        if not docs:
            print("QRAG Warning: No documents provided to index.")
            return

        print(f"QRAG: Indexing {len(docs)} documents...")
        
        # Reset storage
        self.doc_map = {}
        self.simulated_questions = []
        self.question_to_cid_map = []
        self.cid_to_question_indices = {}
        
        # 1. Pre-fill doc_map
        for cid, text in docs:
            self.doc_map[cid] = text
            self.cid_to_question_indices[cid] = []
        
        total_questions = 0
        
        # Use the global config directly - max_workers is the standard Python variable name
        max_workers = BenchmarkConfig.LLM_MAX_WORKERS
        
        # 3. Helper function for threading
        def generate_for_doc(cid, text):
            # Calls the synchronous LLM method
            qs = self._call_llm_to_generate_questions(text)
            return cid, qs
        
        # 4. Parallel Execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_cid = {
                executor.submit(generate_for_doc, cid, text): cid 
                for cid, text in docs
            }
            
            # Process as they complete
            for future in tqdm(as_completed(future_to_cid), total=len(docs), desc=f"QRAG Gen ({max_workers} workers)"):
                try:
                    cid, generated_questions = future.result()
                    
                    if generated_questions:
                        # Synchronized storage (main thread)
                        start_idx = len(self.simulated_questions)
                        self.simulated_questions.extend(generated_questions)
                        
                        # Map new indices
                        new_indices = list(range(start_idx, start_idx + len(generated_questions)))
                        self.cid_to_question_indices[cid] = new_indices
                        
                        # Reverse map
                        self.question_to_cid_map.extend([cid] * len(generated_questions))
                        total_questions += len(generated_questions)
                        
                except Exception as e:
                    print(f"QRAG Error processing chunk {future_to_cid[future]}: {e}")

        if not self.simulated_questions:
            raise Exception("QRAG: Indexing failed. No questions were generated from any documents.")
            
        print(f"QRAG: Generated {total_questions} questions from {len(docs)} chunks.")
        
        # 5. Embed and normalize ALL simulated questions
        print(f"QRAG: Embedding {len(self.simulated_questions)} simulated questions...")
        question_embeddings = self._embed(self.simulated_questions)
        self.normalized_question_embeddings = self._normalize(question_embeddings)
        
        print(f"QRAG: Indexing complete.")

    def retrieve(self, queries: List[str], gold_docs: List[List[str]], num_to_retrieve: int) -> Tuple[List[QuerySolution], Dict[str, float]]:
        """
        Retrieves the top 'num_to_retrieve' documents for each query.
        """
        if self.normalized_question_embeddings is None:
            raise Exception("QRAG: Retriever has not been indexed. Call .index() first.")

        print(f"QRAG: Retrieving for {len(queries)} queries...")
        
        query_embeddings = self._embed(queries)
        normalized_query_embeddings = self._normalize(query_embeddings)
        
        all_sims = normalized_query_embeddings @ self.normalized_question_embeddings.T
        
        retrieval_results = []
        k_candidate_questions = num_to_retrieve * self.retrieval_oversampling_factor
        k_candidate_questions = min(k_candidate_questions, len(self.simulated_questions))
        
        for i, query in enumerate(queries):
            question_sims = all_sims[i]
            top_k_question_indices = np.argsort(question_sims)[::-1][:k_candidate_questions]
            
            retrieved_cids_ordered = []
            retrieved_scores_map = {}
            seen_cids = set()

            for idx in top_k_question_indices:
                cid = self.question_to_cid_map[idx]
                
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    retrieved_cids_ordered.append(cid)
                    retrieved_scores_map[cid] = question_sims[idx] 
                
                if len(retrieved_cids_ordered) == num_to_retrieve:
                    break
            
            retrieved_docs = [self.doc_map[cid] for cid in retrieved_cids_ordered]
            retrieved_scores = np.array([retrieved_scores_map[cid] for cid in retrieved_cids_ordered])
            current_gold_docs = gold_docs[i] if gold_docs and i < len(gold_docs) else []

            solution = QuerySolution(
                question=query, docIDs=retrieved_cids_ordered, doc_scores=retrieved_scores,
                docs=retrieved_docs, gold_docs=current_gold_docs
            )
            retrieval_results.append(solution)

        summary_metrics = {'Recall@K_placeholder': 0.0}
        print("QRAG: Retrieval complete.")
        return retrieval_results, summary_metrics

    def rescore(self, query: str, chunk_IDs: List[int]) -> List[float]:
        """
        Re-scores a list of *already indexed* chunk IDs 
        against a query.
        
        This method is now fast and does *not* call the LLM. It uses
        the indexed questions.
        
        Args:
            query: The query string.
            chunk_IDs: A list of chunk cIDs *that are already in the index*.
            
        Returns:
            A list of scores, one for each chunk_ID. The score is the
            *maximum* similarity between the query and any of
            the chunk's *indexed* questions.
        """
        if self.normalized_question_embeddings is None:
            raise Exception("QRAG: Retriever has not been indexed. Call .index() first.")

        print(f"QRAG: Re-scoring {len(chunk_IDs)} indexed chunks...")

        # --- 1. Embed the query (once) ---
        query_embedding = self._embed([query])
        normalized_query_embedding = self._normalize(query_embedding)[0] # Shape (embed_dim,)
        
        final_scores = []

        # --- 2. Iterate over each chunk ID ---
        for cid in chunk_IDs:
            # 2a. Get the *indexed* question indices for this chunk
            question_indices = self.cid_to_question_indices.get(cid, [])
            
            if not question_indices:
                # This chunk had no questions generated, or cID is invalid
                final_scores.append(0.0)
                continue
            
            # 2b. Get the pre-computed, normalized embeddings for these questions
            # Shape: (num_questions_for_this_chunk, embed_dim)
            relevant_q_embeddings = self.normalized_question_embeddings[question_indices]
            
            # 2c. Calculate similarity (dot product)
            # (embed_dim,) @ (embed_dim, num_q) = (num_q,)
            sims = relevant_q_embeddings @ normalized_query_embedding
            
            # 2d. The score for the chunk is its *best* question's score
            max_score = np.max(sims) if sims.size > 0 else 0.0
            final_scores.append(max_score)

        print("QRAG: Rescoring complete.")
        return final_scores

    # --- NEW CHECKPOINT METHODS ---
    
    def save_ckpt(self, save_dir: str):
        """Saves the indexed QRAG data to a directory."""
        
        if self.normalized_question_embeddings is None:
            print("QRAG Warning: Attempting to save an un-indexed retriever.")
            
        # Define file paths
        doc_map_file = os.path.join(save_dir, 'qrag.doc_map.jbl')
        questions_file = os.path.join(save_dir, 'qrag.questions.jbl')
        q_to_cid_file = os.path.join(save_dir, 'qrag.q_to_cid.jbl')
        cid_to_q_file = os.path.join(save_dir, 'qrag.cid_to_q.jbl')
        q_embed_file = os.path.join(save_dir, 'qrag.q_embeddings.jbl')
        
        # Save components
        joblib.dump(self.doc_map, doc_map_file)
        joblib.dump(self.simulated_questions, questions_file)
        joblib.dump(self.question_to_cid_map, q_to_cid_file)
        joblib.dump(self.cid_to_question_indices, cid_to_q_file)
        joblib.dump(self.normalized_question_embeddings, q_embed_file)
        
        print(f"QRAG: Checkpoint successfully saved to {save_dir}")

    def load_ckpt(self, save_dir: str):

        # Define file paths
        doc_map_file = os.path.join(save_dir, 'qrag.doc_map.jbl')
        questions_file = os.path.join(save_dir, 'qrag.questions.jbl')
        q_to_cid_file = os.path.join(save_dir, 'qrag.q_to_cid.jbl')
        cid_to_q_file = os.path.join(save_dir, 'qrag.cid_to_q.jbl')
        q_embed_file = os.path.join(save_dir, 'qrag.q_embeddings.jbl')

        if not all(os.path.exists(f) for f in [doc_map_file, questions_file, q_to_cid_file, cid_to_q_file, q_embed_file]):
            raise FileNotFoundError(f"QRAG: Checkpoint not found or incomplete in {save_dir}")
            
        # Load components
        self.doc_map = joblib.load(doc_map_file)
        self.simulated_questions = joblib.load(questions_file)
        self.question_to_cid_map = joblib.load(q_to_cid_file)
        self.cid_to_question_indices = joblib.load(cid_to_q_file)
        self.normalized_question_embeddings = joblib.load(q_embed_file)
        
        print(f"QRAG: Checkpoint successfully loaded from {save_dir}")
        print(f"QRAG: Loaded {len(self.simulated_questions)} questions and {len(self.doc_map)} docs.")

RETRIEVER_CLASS_MAP = {
    'hippoRAG': HippoRAG, # Will be wrapped in the main script
    'RAG-bge-m3': RAG,
    'RAG-allmpnet': RAG,
    'BM25': BM25,
    'GRAG': GraphRAG,
    'QRAG': QRAG,
}


def setup_retriever(retriever_name: str, dataset_name: str, reload = False, **kwargs) -> Any:
    """
    Initializes the specific retriever class, applying the wrapper if necessary.
    """
    retriever_cls = RETRIEVER_CLASS_MAP.get(retriever_name)
    kwargs['retriever_name'] = retriever_name
    
    # Pass embedding batch size from config to retrievers
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = config.EMBEDDING_BATCH_SIZE
    
    if retriever_cls is None:
        if retriever_name.startswith('GRAG'):
            retriever_cls = RETRIEVER_CLASS_MAP['GRAG']
        elif retriever_name.startswith('QRAG'):
            retriever_cls = RETRIEVER_CLASS_MAP['QRAG']
        else:
            raise ValueError(f"Retriever '{retriever_name}' not defined in RETRIEVER_CLASS_MAP.")

    # Determine if the class needs the cID mapping wrapper
    if retriever_name == 'hippoRAG':
        print(f"Wrapping {retriever_name} with RetrieverWrapper for cID mapping.")
        base_instance = retriever_cls(
            save_dir=config.ckpt_path(retriever_name, dataset_name),
            llm_model_name=config.LLM_MODEL_NAME,
            llm_base_url=config.LLM_BASE_URL,
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            embedding_base_url=config.EMBEDDING_BASE_URL
        )
        retriever = RetrieverWrapper(base_retriever_cls=lambda **kw: base_instance, **kwargs)
    else:
        retriever = retriever_cls(**kwargs)
    if reload:
        retriever.load_ckpt(os.path.join(kwargs['SAVE_DIR_ROOT'], f'{retriever_name}@{dataset_name}'))
    return retriever

def cal_jsd(list1, list2):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between two lists of floats.

    Args:
        list1 (list of float): First probability distribution.
        list2 (list of float): Second probability distribution.

    Returns:
        float: The Jensen-Shannon Divergence between the two distributions.
    """
    if len(list1)!=len(list2):
        return float('inf') # invalid

    # Convert lists to numpy arrays
    p = np.array(list1, dtype=np.float64)
    q = np.array(list2, dtype=np.float64)
    
    # Normalize the distributions to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the Jensen-Shannon Divergence
    jsd = jensenshannon(p, q, base=2) ** 2  # JSD is the square of the JS distance
    
    return jsd

def calculate_retriever_metrics(jsonl_path: str, 
                                k_values: List[int] = None
                               ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Loads a .jsonl file, groups by 'dataset_name', compares each retriever's
    results against its dataset's 'gt', and returns a nested dictionary of metrics.

    Args:
        jsonl_path: Path to the .jsonl results file.
        k_values: A list of k values for which to calculate Recall@k.
                  Defaults to [1, 2, 5, 10, 20, 30, 50, 100, 150, 200].

    Returns:
        A dictionary structured as:
        {
            "dataset_name_A": {
                "retriever_name_1": {
                    "mrr": 0.85,
                    "num_missing_gt": 10,
                    "total_evaluable_questions": 100,
                    "recall@1": 0.80,
                    ...
                },
                "retriever_name_2": { ... }
            },
            "dataset_name_B": { ... }
        }
    """
    
    # Set default k_list if not provided
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
    
    # This will be the final nested dictionary we return
    # Structure: dataset -> retriever -> metrics
    all_metrics_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # --- 1. Load and Validate File ---
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found: {jsonl_path}")
        return all_metrics_results

    all_data_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                all_data_lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {i+1}.")

    # --- 2. Separate data by Dataset ---
    # Store GT and retriever data grouped by dataset name
    gt_data_by_dataset: Dict[str, Dict[str, Any]] = {}
    gt_MIR_data_by_dataset: Dict[str, Dict[str, Any]] = {}
    retrievers_by_dataset: Dict[str, List[Dict[str, Any]]] = {}

    for item in all_data_lines:
        retriever_name = item.get('retriever_name')
        # Use a default if dataset_name is missing
        dataset_name = item.get('dataset_name', 'unknown_dataset') 

        if retriever_name == 'gt':
            if dataset_name in gt_data_by_dataset:
                print(f"Warning: Multiple 'gt' entries found for dataset '{dataset_name}'. Using the last one.")
            gt_data_by_dataset[dataset_name] = item
        elif retriever_name == 'gt_MI_reinforce':
            if dataset_name in gt_MIR_data_by_dataset:
                print(f"Warning: Multiple 'gt' entries found for dataset '{dataset_name}'. Using the last one.")
            gt_MIR_data_by_dataset[dataset_name] = item
            
        elif 'results' in item and 'retriever_name' in item:
            # Initialize list for this dataset if it's the first time
            if dataset_name not in retrievers_by_dataset:
                retrievers_by_dataset[dataset_name] = []
            retrievers_by_dataset[dataset_name].append(item)

    
    # --- 3. Create Ground Truth Lookups for Each Dataset ---
    gt_lookups_by_dataset: Dict[str, Dict[str, Set[int]]] = {}
    for dataset_name, gt_data in gt_data_by_dataset.items():
        gt_lookup: Dict[str, Set[int]] = {}
        gt_results = gt_data.get('results', [])

        for result in gt_results:
            qid = result.get('qid')
            if not qid: continue
            gt_cids = {cid for cid, score in result.get('ranked_cids_with_scores', [])}
            gt_lookup[qid] = gt_cids
            
        gt_lookups_by_dataset[dataset_name] = gt_lookup
        print(f"--- Processed Ground Truth for dataset: '{dataset_name}' ({len(gt_lookup)} questions) ---")

    
    gt_MIR_lookups_by_dataset: Dict[str, Dict[str, List[float]]] = {}
    for dataset_name, gt_MIR_data in gt_MIR_data_by_dataset.items():
        gt_MIR_lookup: Dict[str, List[float]] = {}
        gt_MIR_results = gt_MIR_data.get('results', [])

        for result in gt_MIR_results:
            qid = result.get('qid')
            if not qid: continue
            gt_MIR_scores = [score for cid, score in result.get('ranked_cids_with_scores', [])]
            gt_MIR_lookup[qid] = gt_MIR_scores
            
        gt_MIR_lookups_by_dataset[dataset_name] = gt_MIR_lookup
        print(f"--- Processed Ground Truth MI reinforce for dataset: '{dataset_name}' ({len(gt_MIR_lookup)} questions) ---")


    # --- 4. Evaluate Each Retriever, Grouped by Dataset ---
    for dataset_name, retriever_list in retrievers_by_dataset.items():
        # Check if we have GT data for this dataset
        if dataset_name not in gt_lookups_by_dataset:
            print(f"Warning: Skipping dataset '{dataset_name}'. No 'gt' data found for it.")
            continue
            
        gt_lookup = gt_lookups_by_dataset[dataset_name]
        gt_MIR_lookup = gt_MIR_lookups_by_dataset.get(dataset_name, {})
        
        # Initialize the results dict for this dataset
        all_metrics_results[dataset_name] = {}
        print(f"--- Analyzing retrievers for dataset: '{dataset_name}' ---")

        # This inner loop is mostly the same as the old function
        for retriever in retriever_list:
            retriever_name = retriever.get('retriever_name', 'Unknown Retriever')

            divs: List[float] = []
            reciprocal_ranks: List[float] = []
            missed_questions_count: int = 0
            total_evaluable_questions: int = 0
            hits_at_k: Dict[int, int] = {k: 0 for k in k_values}

            for result in retriever.get('results', []):

                qid = result.get('qid')
                
                # Check if this qid is in this dataset's GT lookup
                if not qid or qid not in gt_lookup:
                    continue
                gt_cids_set = gt_lookup[qid]
                if not gt_cids_set:
                    continue # Skip questions with no GT labels
                
                total_evaluable_questions += 1

                transpose = lambda matrix: [[row[i] for row in matrix] for i in range(len(matrix[0]))]
                retrieved_cids_list, retrieved_scores_list = transpose(result.get('ranked_cids_with_scores', []))

                divs.append(cal_jsd(retrieved_scores_list, gt_MIR_lookup.get(qid, [0]*len(retrieved_scores_list))))

                # a) Calculate Reciprocal Rank (MRR) and find first_hit_rank
                rr = 0.0
                first_hit_rank = float('inf')
                
                for i, retrieved_cid in enumerate(retrieved_cids_list):
                    if retrieved_cid in gt_cids_set:
                        rr = 1.0 / (i + 1)
                        first_hit_rank = i + 1
                        break 
                reciprocal_ranks.append(rr)

                # b) Calculate Recall@k
                for k in k_values:
                    if first_hit_rank <= k:
                        hits_at_k[k] += 1

                # c) Check for missed labels
                retrieved_cids_set = set(retrieved_cids_list)
                if not gt_cids_set.issubset(retrieved_cids_set):
                    missed_questions_count += 1
            
            # --- 5. Calculate Final Metrics for this retriever ---
            retriever_metrics: Dict[str, Any] = {}

            if total_evaluable_questions > 0:
                retriever_metrics['mrr'] = sum(reciprocal_ranks) / total_evaluable_questions
                retriever_metrics['div'] = np.mean(divs)
                for k in k_values:
                    recall_key = f"recall@{k}"
                    retriever_metrics[recall_key] = hits_at_k[k] / total_evaluable_questions
            else:
                retriever_metrics['mrr'] = 0.0
                for k in k_values:
                    retriever_metrics[f"recall@{k}"] = 0.0

            retriever_metrics['num_missing_gt'] = missed_questions_count
            retriever_metrics['total_evaluable_questions'] = total_evaluable_questions

            # Add this retriever's metrics to the main results dictionary
            all_metrics_results[dataset_name][retriever_name] = retriever_metrics

    print(f"--- Analysis Complete. Returning metrics. ---")
    return all_metrics_results

