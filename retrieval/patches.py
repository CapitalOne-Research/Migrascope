import logging
import traceback
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from retrieval.config import BenchmarkConfig

def apply_hipporag_patches():
    """
    Applies patches to HippoRAG using Code Injection (__code__ swap).
    """
    print("Applying HippoRAG patches (Code Injection Mode)...")
    
    try:
        # =========================================================
        # 1. EMBEDDING PATCH (Code Injection)
        # =========================================================
        import hipporag.embedding_model as emb_pkg
        
        original_func = emb_pkg._get_embedding_model_class
        
        def _patched_implementation(embedding_model_name: str = "nvidia/NV-Embed-v2"):
            # Local Imports
            from retrieval.config import BenchmarkConfig
            from hipporag.embedding_model import (
                OpenAIEmbeddingModel,
                GritLMEmbeddingModel,
                NVEmbedV2EmbeddingModel,
                ContrieverModel
            )

            if embedding_model_name == BenchmarkConfig.EMBEDDING_MODEL_NAME:
                print(f"   Patch Hit: Redirecting '{embedding_model_name}' to OpenAIEmbeddingModel")
                return OpenAIEmbeddingModel

            if "GritLM" in embedding_model_name:
                return GritLMEmbeddingModel
            elif "NV-Embed-v2" in embedding_model_name:
                return NVEmbedV2EmbeddingModel
            elif "contriever" in embedding_model_name:
                return ContrieverModel
            
            print(f"   Unknown model '{embedding_model_name}' detected. Defaulting to OpenAIEmbeddingModel.")
            return OpenAIEmbeddingModel

        original_func.__code__ = _patched_implementation.__code__
        print(f"   Embedding factory patched via code injection.")


        # =========================================================
        # 2. CONCURRENCY PATCH (Robust, Debugging, Safe Limits)
        # =========================================================
        import hipporag.information_extraction.openie_openai as openie_module
        
        def _patched_batch_openie(self, chunks: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            Monkey-patched version of OpenIE.batch_openie.
            """
            # 1. Sanitize Input
            chunk_passages = {}
            for k, c in chunks.items():
                if isinstance(c, dict):
                    chunk_passages[k] = c.get("content", "")
                else:
                    chunk_passages[k] = str(c)

            # 2. Conservative Concurrency Limit
            # 512 is too high for many local servers/caches. We cap at 32 for stability.
            config_workers = getattr(BenchmarkConfig, 'LLM_MAX_WORKERS', 16)
            max_workers = min(config_workers, 64) 
            
            print(f"   Starting OpenIE extraction for {len(chunk_passages)} chunks with {max_workers} workers...")
            
            # Helper to catch tracebacks inside threads
            def safe_ner_wrapper(k, text):
                try:
                    # Force key to string to avoid hash issues in caches
                    return self.ner(str(k), text)
                except Exception as e:
                    return e

            # A. NER Extraction
            ner_results_list = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {}
                for key, text in chunk_passages.items():
                    if not text: continue
                    # Submit wrapper instead of direct call
                    f = executor.submit(safe_ner_wrapper, key, text)
                    future_to_key[f] = key

                # Process results
                for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="NER"):
                    key = future_to_key[future]
                    result = future.result()
                    
                    if isinstance(result, Exception):
                        # Print full traceback only for the first few errors to avoid spam
                        if len(ner_results_list) < 3: 
                            print(f"\n❌ NER Error for chunk {key}: {result}")
                            # traceback.print_exception(type(result), result, result.__traceback__)
                        continue
                        
                    ner_results_list.append(result)

            # B. Triple Extraction
            triple_results_list = []
            
            # Helper for triples
            def safe_triple_wrapper(cid, txt, ents):
                try:
                    return self.triple_extraction(str(cid), txt, ents)
                except Exception as e:
                    return e

            if ner_results_list:
                print(f"   Starting Triple Extraction for {len(ner_results_list)} NER results...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_id = {}
                    for res in ner_results_list:
                        if res.chunk_id not in chunk_passages: continue
                        
                        text = chunk_passages[res.chunk_id]
                        f = executor.submit(safe_triple_wrapper, res.chunk_id, text, res.unique_entities)
                        future_to_id[f] = res.chunk_id

                    for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Triples"):
                        cid = future_to_id[future]
                        result = future.result()
                        
                        if isinstance(result, Exception):
                             if len(triple_results_list) < 3:
                                print(f"\n❌ Triple Error for chunk {cid}: {result}")
                             continue

                        triple_results_list.append(result)
            else:
                print("   ⚠️ No NER results found. Skipping Triple Extraction.")

            return {res.chunk_id: res for res in ner_results_list}, {res.chunk_id: res for res in triple_results_list}

        # Apply the method patch
        openie_module.OpenIE.batch_openie = _patched_batch_openie
        print(f"   Concurrency patched (Capped Workers: {BenchmarkConfig.LLM_MAX_WORKERS} -> capped internally).")

    except Exception as e:
        print(f"   Error applying HippoRAG patches: {e}")
        traceback.print_exc()