from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from openai.types import Completion
from retrieval.config import BenchmarkConfig

client = OpenAI(
    base_url=BenchmarkConfig.LLM_BASE_URL, 
    api_key=os.environ.get("LLM_API_KEY", "sk-dummy-key")
)

def _map_tokens_to_spans_by_find(prompt: str, tokens: List[str]) -> List[Optional[Tuple[int, int]]]:
    """
    Use prompt.find(token, cursor) to match sequentially, mapping each token to [start, end].
    If the token cannot be found, a conservative fallback is performed: set start to the current cursor and set cursor += len(token).
    For special tokens like <|begin_of_text|> that do not appear in the original text, logprob=None, which is usually sufficient.
    """
    spans: List[Optional[Tuple[int,int]]] = []
    cursor = 0
    for tok in tokens:
        if tok.startswith("<|") and tok.endswith("|>"):
            spans.append(None)  # Special mark, not in the original text
            continue
        j = prompt.find(tok, cursor)
        if j == -1:
            # Degenerate matching: Try to keep the monotony moving forward to avoid miscounting the previous characters into part2
            j = cursor
        start = j
        end = start + len(tok)
        spans.append((start, end))
        cursor = end
    return spans

def _avg_nll_ppl(logprobs: List[float]) -> Tuple[float, float]:
    n = len(logprobs)
    avg_nll = -sum(logprobs)/n
    return avg_nll, math.exp(avg_nll)


def calculate_part2_MI(
    part1: str,
    part2: str,
    model: str = "Meta-Llama-3-8B-Instruct",
    verbose: bool = False,
) -> Optional[float]:
    """Compute the average NLL of tokens belonging to part2 using prompt logprobs."""
    prompt = part1 + part2

    resp: Completion = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=0,
        logprobs=1,
        echo=True,
    )

    ch = resp.choices[0]
    lp = ch.logprobs
    if not lp or not lp.tokens or not lp.token_logprobs:
        raise RuntimeError("No prompt-token logprobs returned. echo=True may be unsupported.")

    tokens: List[str] = lp.tokens
    token_logprobs: List[Optional[float]] = lp.token_logprobs
    
    # 1) Map tokens back to their exact character spans in the prompt
    spans = _map_tokens_to_spans_by_find(prompt, tokens)
    
    # 2) Define the character boundary for part2 [p2_start, p2_end)
    p2_start = len(part1)
    p2_end = p2_start + len(part2)
    
    # 3) Select all tokens that *start* within the part2 span
    selected_idx: List[int] = []
    for i, sp in enumerate(spans):
        if sp is None:    # Skip special tokens (e.g., <|endoftext|>)
            continue
        s, _ = sp
        # Token must start within the part2 boundary and have a valid logprob
        if s >= p2_start and s < p2_end and token_logprobs[i] is not None:
            selected_idx.append(i)

    # 4. Check if we actually found any tokens
    if not selected_idx:
        # print(f"** WARNING: No tokens found for part2 in prompt: {prompt}")
        return 0.0 # Return None as avg_nll cannot be computed

    # 5. Extract logprobs for part2
    toks_p2 = [tokens[i] for i in selected_idx]
    lps_p2 = [float(token_logprobs[i]) for i in selected_idx]

    # 6. Calculate avg_nll
    # (Assuming _avg_nll_ppl is defined elsewhere and returns (avg_nll, ppl))
    avg_nll, ppl = _avg_nll_ppl(lps_p2)
    if verbose:
        print(f'** Prompt:   {prompt}')
        print(f'** Part 2 Tokens:   {toks_p2}')
        print(f'** Avg NLL:         {avg_nll}\n\n')
        print(f'** Avg ppl:         {ppl}\n\n')

    # 7. Return *only* the avg_nll
    return avg_nll


def use_a_to_sort_b(
    a_score: List[float],
    b_data: List[Any],
    reverse: bool = False,
) -> Tuple[List[float], List[Any]]:
    """Sort b_data according to a_score while returning the reordered scores."""
    rows_with_sim = list(zip(b_data, a_score))
    sorted_rows = sorted(rows_with_sim, key=lambda x: x[1], reverse=reverse)
    transpose = lambda matrix: [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    sorted_data, sorted_score = transpose(sorted_rows)
    return sorted_score, sorted_data

