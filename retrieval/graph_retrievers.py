from dataclasses import dataclass
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from retrieval.config import BenchmarkConfig
from retrieval.graph_building import build_graph

# ==================================================================
# Patches are now applied in retrieval/patches.py
# which is imported before HippoRAG in retrieval/retrievers.py
# ==================================================================

@dataclass
class QuerySolution:
    """Stores retrieval outputs for a single query."""
    question: str
    docIDs: List[Any]
    doc_scores: np.ndarray
    docs: List[str]
    gold_docs: List[str]
    answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None


def robust_llm_rerank_request(payload, llm_url, max_retries=5, timeout=60):
    """Submit a rerank request to an LLM endpoint with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(llm_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as exc:
            print(f"HTTPError during LLM rerank (attempt {attempt+1}/{max_retries}): {exc}")
        except Exception as exc:
            print(f"Error during LLM rerank (attempt {attempt+1}/{max_retries}): {exc}")
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print("Max retries reached. Returning empty rerank result.")
            return ""


class GraphRAG:
    """
    Implements the standard retriever interface for a graph-based RAG model.
    
    This class acts as a wrapper around the externally defined functions
    (build_graph, gather_context, etc.)
    """
    
    def __init__(self, 
                 retriever_name: str, 
                 LLM_BASE_URL: str,
                 LLM_MODEL_NAME: str,
                 EMBEDDING_MODEL_NAME: str,
                 EMBEDDING_BASE_URL: str,
                 NUM_TO_RETRIEVE: int,
                 max_chunks_index: int = -1, # -1 for all
                 batch_size: int = 64,
                 **kwargs):
        """
        Initializes the GraphRAG retriever.
        
        Args:
            embed_fn: A function that takes List[str] and returns embeddings.
            llm_url: The URL for the graph-building LLM.
            llm_model: The model name for the graph-building LLM.
            ranker_option: The ranking strategy (e.g., 'simple', 'bm25').
            mode: The graph build mode (e.g., 'full').
            max_chunks_index: Max chunks to process during .index().
            max_entities_retrieve: Max entities to use during .retrieve().
            NUM_TO_RETRIEVE: Max chunks to return during .retrieve().
        """
        print("Initializing GraphRAG...")
        
        # Parse retriever_name to extract mode and ranker_option
        # Expected format: "GRAG-{mode}-{ranker}" e.g., "GRAG-window-semantic"
        self.mode = "naive"
        self.ranker_option = "semantic"
        try:
            parts = retriever_name.split('-')
            if len(parts) >= 3:
                _, self.mode, self.ranker_option = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                _, self.mode = parts[0], parts[1]
        except (ValueError, IndexError):
            print(f"Warning: Could not parse retriever_name '{retriever_name}'. Using defaults: mode='naive', ranker='semantic'")

        # Store configuration as instance attributes
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.EMBEDDING_BASE_URL = EMBEDDING_BASE_URL
        self.embedding_max_workers = kwargs.get('EMBEDDING_MAX_WORKERS', 16)  # NEW: Configurable concurrency
        self.llm_url = f'{LLM_BASE_URL}/chat/completions'
        self.llm_model = LLM_MODEL_NAME
        self.max_chunks_index = max_chunks_index
        self.max_entities_retrieve = NUM_TO_RETRIEVE
        self.NUM_TO_RETRIEVE = NUM_TO_RETRIEVE
        self.batch_size = batch_size

        # Capture the new config value for entity extraction safety
        self.entity_extract_max_tokens = kwargs.get('ENTITY_EXTRACT_MAX_TOKENS', 1024)

        # Initialize graph data attributes
        self.entity_names: Optional[np.ndarray] = None
        self.entity_name_embeddings: Optional[np.ndarray] = None
        self.entity_to_chunks: Optional[Dict[str, List[int]]] = None
        self.chunk_store: Optional[Dict[int, str]] = None # Maps cID -> text
        
        self.ckpt_file_name = f"{retriever_name}_checkpoint.jbl"

    def _batch_embed_texts(self, texts: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a batch of texts using the configured embedding service."""
        all_embeddings = []
        bar = tqdm(range(0, len(texts), self.batch_size)) if verbose else range(0, len(texts), self.batch_size)
        
        for i in bar:
            batch = texts[i:i + self.batch_size]
            for attempt in range(5):  # max_retries
                try:
                    payload = {"model": self.EMBEDDING_MODEL_NAME, "input": batch}
                    response = requests.post(
                        f"{self.EMBEDDING_BASE_URL}/embeddings",
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    embeddings = [item["embedding"] for item in response.json()["data"]]
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    print(f"Batch {i // self.batch_size + 1}: Embedding request failed (attempt {attempt + 1}/5): {e}")
                    if attempt < 4:
                        sleep_time = 2 ** attempt
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise
            time.sleep(0.2)
        
        return np.array(all_embeddings)

    def embed_fn(self, texts: List[str]) -> np.ndarray:
        """Public embedding function to pass to build_graph."""
        return self._batch_embed_texts(texts)

    def _embed_question(self, question: str) -> np.ndarray:
        """Embed a single question."""
        return self._batch_embed_texts([question])[0]

    def _collect_candidate_chunk_ids(
        self,
        question_embedding: np.ndarray,
        max_entities: int,
    ) -> Tuple[List[str], List[Any]]:
        """Return entities and chunk IDs that best match the question embedding."""
        q_norm = np.linalg.norm(question_embedding) + 1e-8
        entity_norms = np.linalg.norm(self.entity_name_embeddings, axis=1) + 1e-8
        sims = (self.entity_name_embeddings @ question_embedding) / (entity_norms * q_norm)
        top_idx = np.argsort(sims)[::-1][:max_entities]
        selected_entities = [self.entity_names[i] for i in top_idx]

        candidate_chunk_ids: List[Any] = []
        for ent in selected_entities:
            candidate_chunk_ids.extend(list(self.entity_to_chunks.get(ent, [])))
        return selected_entities, list(set(candidate_chunk_ids))

    def _gather_context_semantic_ranked(
        self,
        question: str,
        max_entities: int = 50,
        max_chunks: int = 50,
        return_ids: bool = True
    ):
        """Retrieve context via semantic similarity followed by chunk ranking."""
        q_emb = self._embed_question(question)
        _, candidate_chunk_ids = self._collect_candidate_chunk_ids(q_emb, max_entities)

        if not candidate_chunk_ids:
            if return_ids:
                return "", []
            return ""

        candidate_texts = [self.chunk_store[cid] for cid in candidate_chunk_ids if cid in self.chunk_store]
        chunk_embeddings = self._batch_embed_texts(candidate_texts)
        chunk_sims = np.dot(chunk_embeddings, q_emb) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        top_chunk_idx = np.argsort(chunk_sims)[::-1][:max_chunks]
        selected_chunk_ids = [candidate_chunk_ids[i] for i in top_chunk_idx]

        context = "\n".join(self.chunk_store[cid] for cid in selected_chunk_ids if cid in self.chunk_store)
        if return_ids:
            return context, selected_chunk_ids
        return context

    def _gather_context_llm_ranked(
        self,
        question: str,
        max_entities: int = 50,
        max_chunks: int = 50,
        return_ids: bool = False
    ):
        """Retrieve context via semantic entity search followed by LLM reranking."""
        q_emb = self._embed_question(question)
        _, candidate_chunk_ids = self._collect_candidate_chunk_ids(q_emb, max_entities)

        if not candidate_chunk_ids or not self.llm_url or not self.llm_model:
            if return_ids:
                return "", [], []
            return ""

        prompt_lines = [
            "Given the following question:\n\n",
            question,
            "\n\nAnd these candidate chunks:\n",
        ]
        for i, cid in enumerate(candidate_chunk_ids, start=1):
            prompt_lines.append(f"\nChunk {i} (ID: {cid}):\n{self.chunk_store[cid]}\n")
        prompt_lines.append(
            "\nPlease rank the chunks from most to least relevant to the question. "
            "Return a list of chunk IDs in order of relevance."
        )
        prompt = "".join(prompt_lines)

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}]
        }
        llm_reply = robust_llm_rerank_request(payload, self.llm_url)
        match = re.findall(r'chunk-[a-f0-9]+', llm_reply)
        reranked_chunk_ids = [cid for cid in match if cid in candidate_chunk_ids] or candidate_chunk_ids
        selected_chunk_ids = reranked_chunk_ids[:max_chunks]

        context = "\n".join(self.chunk_store[cid] for cid in selected_chunk_ids if cid in self.chunk_store)
        if return_ids:
            return context, selected_chunk_ids
        return context

    def _gather_context(
        self,
        question: str,
        ranker_option: str = "semantic",
        max_entities: int = 15,
        max_chunks: int = 15,
        return_ids: bool = True
    ):
        """Dispatch to the appropriate context gathering strategy."""
        if ranker_option == "semantic":
            return self._gather_context_semantic_ranked(
                question, max_entities=max_entities, max_chunks=max_chunks, return_ids=return_ids
            )
        elif ranker_option == "llm":
            return self._gather_context_llm_ranked(
                question, max_entities=max_entities, max_chunks=max_chunks, return_ids=return_ids
            )
        elif ranker_option in ["none", "orig"]:
            # No re-ranking: just collect all candidate chunks
            candidate_chunk_ids = []
            entity_list = list(self.entity_names)[:max_entities]
            for ent in entity_list:
                candidate_chunk_ids.extend(list(self.entity_to_chunks.get(ent, [])))
            candidate_chunk_ids = list(set(candidate_chunk_ids))[:max_chunks]
            context = "\n".join(self.chunk_store[cid] for cid in candidate_chunk_ids if cid in self.chunk_store)
            if return_ids:
                return context, candidate_chunk_ids
            return context
        else:
            raise ValueError(f"Unknown ranker_option: {ranker_option}")

    def index(self, docs: List[Tuple[int, str]]):
        """
        Builds the graph index from a list of (cID, text) documents.
        
        This maps to your 'build_graph' flow.
        """
        if not docs:
            print("GraphRAG Warning: No documents provided to index.")
            return

        print(f"GraphRAG: Indexing {len(docs)} documents...")
        
        # 1. Convert docs list to the required chunk_store dict (int -> str)
        self.chunk_store = dict(docs)
        
        # 2. Call the external build_graph function
        # We pass self.chunk_store (int -> str)
        graph_data = build_graph(
            self.chunk_store,
            self.embed_fn,
            self.llm_url,
            self.llm_model,
            max_chunks=self.max_chunks_index if self.max_chunks_index > 0 else None,
            mode=self.mode,
            max_tokens=self.entity_extract_max_tokens  # NEW: Pass token limit
        )
        
        # 3. Store the built graph components
        self.entity_names = graph_data["entity_names"]
        self.entity_name_embeddings = graph_data["entity_name_embeddings"]
        self.entity_to_chunks = graph_data["entity_to_chunks"]
        
        print(f"GraphRAG: Indexing complete. Found {len(self.entity_names)} entities.")

    def retrieve(
        self,
        queries: List[str],
        gold_docs: List[List[str]],
        num_to_retrieve: int
    ) -> Tuple[List[QuerySolution], Dict[str, float]]:
        """
        Retrieves documents for a list of queries using the graph.
        Parallelized using EMBEDDING_MAX_WORKERS.
        """
        if self.chunk_store is None or self.entity_names is None:
            raise Exception("GraphRAG: Retriever has not been indexed. Call .index() first.")

        print(f"GraphRAG: Retrieving for {len(queries)} queries...")
        retrieval_results = [None] * len(queries)
        
        # Note: We use self.NUM_TO_RETRIEVE if num_to_retrieve is not set,
        # otherwise, the user's request overrides it.
        k_to_retrieve = num_to_retrieve if num_to_retrieve > 0 else self.NUM_TO_RETRIEVE
        
        # Define worker function
        def _process_grag_query(idx, q):
            try:
                # Heavy I/O operation
                _context_str, chunk_ids = self._gather_context(
                    question=q,
                    ranker_option=self.ranker_option,
                    max_entities=self.max_entities_retrieve,
                    max_chunks=k_to_retrieve,
                    return_ids=True
                )
                
                # Format results
                retrieved_cids = list(chunk_ids)
                retrieved_docs = [self.chunk_store.get(cid, "") for cid in retrieved_cids]
                placeholder_scores = np.linspace(1.0, 0.0, len(retrieved_cids), dtype=float) if retrieved_cids else np.array([])
                current_gold_docs = gold_docs[idx] if gold_docs and idx < len(gold_docs) else []
                
                return idx, QuerySolution(
                    question=q,
                    docIDs=retrieved_cids,
                    doc_scores=placeholder_scores,
                    docs=retrieved_docs,
                    gold_docs=current_gold_docs
                )
            except Exception as e:
                print(f"Error processing query {idx}: {e}")
                return idx, QuerySolution(q, [], np.array([]), [], [])

        # Execute in Parallel
        print(f"GraphRAG: Retrieving for {len(queries)} queries with {self.embedding_max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.embedding_max_workers) as executor:
            future_to_idx = {executor.submit(_process_grag_query, i, q): i for i, q in enumerate(queries)}
            
            for future in tqdm(as_completed(future_to_idx), total=len(queries), desc="GraphRAG Querying"):
                idx, solution = future.result()
                retrieval_results[idx] = solution

        summary_metrics = {'Recall@K_placeholder': 0.0}
        print(f"GraphRAG: Retrieval complete. Summary: {summary_metrics}")
        return retrieval_results, summary_metrics

    def rescore(self, query: str, chunk_IDs: List[int]) -> List[float]:
        """
        Calculates similarity scores for specific chunks against the query.
        Used by bench_post_run.py to fill in missing scores for alignment.
        
        Args:
            query: The query string to score against
            chunk_IDs: List of chunk IDs to score
            
        Returns:
            List of cosine similarity scores (one per chunk)
        """
        if not chunk_IDs:
            return []

        # 1. Retrieve the text for the requested chunks
        chunk_texts = [self.chunk_store.get(cid, "") for cid in chunk_IDs]

        # 2. Embed the query (shape: [1, dim])
        query_emb = self.embed_fn([query])[0]

        # 3. Embed the chunks (shape: [n, dim])
        chunk_embs = self.embed_fn(chunk_texts)

        # 4. Compute Cosine Similarity
        q_norm = np.linalg.norm(query_emb) + 1e-8
        c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-8
        sims = (chunk_embs @ query_emb) / (c_norms * q_norm)

        return list(sims)

    def save_ckpt(self, save_dir: str):
        """
        Saves the indexed GraphRAG data to a directory.
        
        This maps to your 'save_graph_checkpoint' function.
        """
        if self.chunk_store is None or self.entity_names is None:
            print("GraphRAG Warning: Attempting to save an un-indexed retriever.")

        checkpoint_path = os.path.join(save_dir, self.ckpt_file_name)
        
        # 1. Bundle all graph data
        graph_data = {
            "entity_names": self.entity_names,
            "entity_name_embeddings": self.entity_name_embeddings,
            "entity_to_chunks": self.entity_to_chunks,
        }
        
        # 2. Call the external save_graph_checkpoint function
        data_to_save = {
            "graph_data": graph_data,
            "chunk_store": self.chunk_store
        }
        joblib.dump(data_to_save, checkpoint_path)
        print(f"GraphRAG: Checkpoint successfully saved to {checkpoint_path}")

    def load_ckpt(self, save_dir: str):
        """
        Loads an indexed GraphRAG checkpoint from a directory.
        
        This maps to your 'load_graph_checkpoint' function.
        """
        checkpoint_path = os.path.join(save_dir, self.ckpt_file_name)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GraphRAG: Checkpoint not found at {checkpoint_path}")
            
        data = joblib.load(checkpoint_path)

        # 2. Un-bundle the data into class attributes
        graph_data = data.get("graph_data", {})
        self.entity_names = graph_data.get("entity_names")
        self.entity_name_embeddings = graph_data.get("entity_name_embeddings")
        self.entity_to_chunks = graph_data.get("entity_to_chunks")
        self.chunk_store = data.get("chunk_store")
        
        if self.chunk_store is None or self.entity_names is None:
            print("GraphRAG Warning: Loaded checkpoint is missing data.")
        
        print(f"GraphRAG: Checkpoint successfully loaded from {checkpoint_path}")
        print(f"GraphRAG: Loaded {len(self.chunk_store)} chunks and {len(self.entity_names)} entities.")
