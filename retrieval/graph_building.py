import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx
import requests
from collections import Counter
import re
import pandas as pd
import lz4.frame
import os
from io import BytesIO
import numpy as np
import json
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from retrieval.config import BenchmarkConfig

def normalize_entity(ent):
    # Example normalization: lowercase, strip, remove punctuation
    ent = ent.lower().strip()
    ent = re.sub(r'[^\w\s]', '', ent)
    return ent

def clean_entity(entity: str) -> str:
    # Remove leading numbers and whitespace, e.g. "1 Capital" -> "Capital"
    return re.sub(r"^\s*\d+\s*", "", entity).strip()

def extract_entities_llm(session, chunk_id, text, llm_url, llm_model, max_tokens=1024):
    """Makes an LLM call to extract entities using a requests.Session object."""
    prompt = (
        "Extract all named entities from the following text. "
        "Return only a numbered list, one entity per line, no explanations:\n\n"
        f"{text}\n\nEntities:"
    )
    
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,  
    }
    try:
        # Use the passed session object instead of requests.post
        response = session.post(llm_url, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        entities_str = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        entities = []
        for line in entities_str.splitlines():
            m = re.match(r"\s*\d+\s+(.+?)(?:\s+\d+\s*instances?)?$", line.strip(), re.IGNORECASE)
            if m:
                entity = m.group(1)
                entities.append(normalize_entity(entity))
        if not entities:
            entities = [normalize_entity(line) for line in entities_str.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error extracting entities for chunk {chunk_id}: {e}")
        entities = []
            
    entities = [clean_entity(e) for e in entities if e.strip()]

    return chunk_id, entities

def build_graph(
    chunk_store,
    embed_fn,
    llm_url,
    llm_model,
    max_chunks=None,
    mode="naive",  # "naive"
    window_size=2,
    min_cooc=2,
    max_tokens=1024,
):

    chunk_store = pd.DataFrame([{
        'chunk-id': chunk_id,
        'doc-id': chunk_id,
        'chunk_content': content
        } for chunk_id, content in chunk_store.items()])

    assert isinstance(chunk_store, pd.DataFrame), "Expected chunk_store to be Pandas Dataframe."

    chunk_items = [(row["chunk-id"], row) for _, row in chunk_store.iterrows()]
    if max_chunks is not None:
        chunk_items = chunk_items[:max_chunks]

    # This outer section defines variables and prepares the session.
    chunk_entities = {}
    nodes = set()
    edges = []
    entity_to_chunks = {}

    # Configure a retry strategy for handling transient network errors.
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # The 'with' block now wraps all code that makes API calls.
    with requests.Session() as session:
        session.mount("http://", adapter)

        # --- Stage 1: Entity Extraction (runs for all modes) ---
        print("Extracting entities from chunks...")
        with ThreadPoolExecutor(max_workers=BenchmarkConfig.LLM_MAX_WORKERS) as executor:
            # Pass the 'session' object to the worker function.
            futures = [
                executor.submit(
                    extract_entities_llm,
                    session, # <-- Pass the session here
                    chunk_id,
                    chunk["chunk_content"],
                    llm_url,
                    llm_model,
                    max_tokens  # NEW: Pass token limit parameter
                )
                for chunk_id, chunk in chunk_items
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting entities"):
                chunk_id, entities = future.result()
                chunk_entities[chunk_id] = entities

        # --- Stage 2: Graph Construction (logic depends on mode) ---
        if mode == "naive":
            for chunk_id, entities in chunk_entities.items():
                nodes.update(entities)
                for ent in entities:
                    entity_to_chunks.setdefault(ent, set()).add(chunk_id)
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        edges.append((entities[i], entities[j], {"chunk_id": chunk_id}))
        else:
            raise ValueError(f"Unknown mode: {mode}. Only 'naive' is supported in this release.")

    # --- Final Graph Assembly and Embedding ---
    # This part happens after the session is closed.
    print(f"Total unique nodes identified: {len(nodes)}")
    print(f"Total edges identified: {len(edges)}")
    
    is_directed = False
    G = nx.Graph()
    
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Create embeddings for all nodes in the final graph.
    entity_names = list(G.nodes())    
    entity_name_embeddings = embed_fn(entity_names)

    return {
        "G": G,
        "chunk_entities": chunk_entities,
        "entity_to_chunks": entity_to_chunks,
        "entity_names": np.array(entity_names, dtype=object),
        "entity_name_embeddings": entity_name_embeddings
    }

def save_graph_checkpoint(graph_data, chunk_store, checkpoint_path):
    """
    Saves a checkpoint by pickling, compressing, and writing to disk with a progress bar.
    """
    print("Serializing and compressing data in memory...")
    data_to_save = {**graph_data, "chunk_store": chunk_store}

    # Serialize and compress data in-memory first.
    pickled_data = pickle.dumps(data_to_save, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = lz4.frame.compress(pickled_data)

    # Prepare for writing with a progress bar.
    total_size = len(compressed_data)
    chunk_size = 1024 * 1024  # 1MB chunks
    data_stream = BytesIO(compressed_data) # Treat the compressed data as a file in memory

    print(f"Writing {total_size / (1024*1024):.2f} MB to disk...")
    with open(checkpoint_path, "wb") as f, tqdm(
        total=total_size,
        desc=f"Saving {os.path.basename(checkpoint_path)}",
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        # Write to disk in chunks to update the progress bar.
        while True:
            chunk = data_stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Checkpoint saved to {checkpoint_path}")

def load_graph_checkpoint(checkpoint_path):
    """
    Loads a checkpoint from disk with a progress bar, then decompresses and unpickles.
    """
    total_size = os.path.getsize(checkpoint_path)
    chunk_size = 1024 * 1024 # 1MB chunks
    buffer = BytesIO()

    print(f"Reading {total_size / (1024*1024):.2f} MB from disk...")
    # Read the compressed file from disk with a progress bar.
    with open(checkpoint_path, "rb") as f, tqdm(
        total=total_size,
        desc=f"Loading {os.path.basename(checkpoint_path)}",
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            pbar.update(len(chunk))
    
    compressed_data = buffer.getvalue()
    
    # Decompress and unpickle the data.
    print("Decompressing and unpickling data...")
    pickled_data = lz4.frame.decompress(compressed_data)
    data = pickle.loads(pickled_data)
    print("✅ Load complete.")
    return data