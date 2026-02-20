import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.random import Generator

from migrascope.evaluation import avg_jsd_per_query
from migrascope.fusion import (
    fusion_borda,
    fusion_linear_z,
    fusion_logit_pool,
    fusion_noisy_or,
    fusion_prob_logop,
    fusion_rrf,
    rank_centrality,
    robust_rank_aggregation,
)
from migrascope.io import load_pseudo_gt, load_retriever, write_jsonl
from migrascope.utils import softmax

PseudoGT = Dict[str, Tuple[List[str], List[float]]]
RetrieverOutputs = Dict[str, Dict[str, Tuple[List[str], List[float]]]]
ScoreByQid = Dict[str, np.ndarray]
ProbByQid = Dict[str, np.ndarray]

WEIGHTLESS_METHODS = {"rrf", "rra", "rank_centrality"}
ALL_METHODS = [
    "linear_z",
    "prob_logop",
    "logit_pool",
    "noisy_or",
    "rrf",
    "borda",
    "rra",
    "rank_centrality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search fusion methods across retrievers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrievers", nargs="+", required=True)
    parser.add_argument("--datadir", default="data/samples")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--num_weights_to_search",
        type=int,
        default=20,
        help="Number of randomly initialized weight vectors (minimum 1).",
    )
    return parser.parse_args()


def load_dataset(dataset: str, datadir: str, retriever_names: Sequence[str]) -> Tuple[PseudoGT, RetrieverOutputs, str]:
    ds_dir = os.path.join(datadir, dataset)
    ps_path = os.path.join(ds_dir, "pseudo_gt.jsonl")
    ps = load_pseudo_gt(ps_path)
    retrievers = {
        name: load_retriever(os.path.join(ds_dir, "retrievers", f"{name}.jsonl"))
        for name in retriever_names
    }
    return ps, retrievers, ds_dir


def build_query_mats(
    ps: PseudoGT,
    retrievers: RetrieverOutputs,
    retriever_names: Sequence[str],
) -> Tuple[ScoreByQid, ProbByQid]:
    S_by_qid: ScoreByQid = {}
    CP_by_qid: ProbByQid = {}
    for qid, (chunks, cp) in ps.items():
        cols = []
        ok = True
        for name in retriever_names:
            r_chunks, r_scores = retrievers[name].get(qid, ([], []))
            if r_chunks != chunks or len(r_scores) != len(cp):
                ok = False
                break
            cols.append(np.asarray(r_scores, dtype=float).reshape(-1, 1))
        if ok and cols:
            S_by_qid[qid] = np.hstack(cols)
            CP_by_qid[qid] = np.asarray(cp, dtype=float)
    return S_by_qid, CP_by_qid


def split_qids(
    ps: PseudoGT,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Tuple[List[str], List[str], List[str]]:
    qids = list(ps.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n = len(qids)
    n_tr = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    tr = qids[:n_tr]
    dv = qids[n_tr : n_tr + n_dev]
    te = qids[n_tr + n_dev :]
    return tr, dv, te


def subset_dict(data: Dict[str, np.ndarray], keys: Iterable[str]) -> Dict[str, np.ndarray]:
    return {k: data[k] for k in keys if k in data}


def generate_weight_candidates(num_retrievers: int, rng: Generator, count: int) -> List[np.ndarray]:
    if num_retrievers <= 0:
        return []
    base = np.ones(num_retrievers, dtype=float) / num_retrievers
    if count <= 1:
        return [base]
    candidates = [base]
    for _ in range(count - 1):
        candidates.append(rng.dirichlet(np.ones(num_retrievers, dtype=float)))
    return candidates


def eval_method(
    name: str,
    S_by_qid: ScoreByQid,
    CP_by_qid: ProbByQid,
    retr_names: Sequence[str],
    weights: np.ndarray | None = None,
    tau: float = 1.0,
) -> Tuple[float, Dict[str, np.ndarray]]:
    default_weights = np.ones(len(retr_names), dtype=float) / len(retr_names)
    weight_vec = weights if weights is not None else default_weights
    fused: Dict[str, np.ndarray] = {}
    for qid, S in S_by_qid.items():
        if name == "linear_z":
            fused[qid] = fusion_linear_z(S, weight_vec)
        elif name == "prob_logop":
            fused[qid] = fusion_prob_logop(S, weight_vec, tau=tau)
        elif name == "logit_pool":
            P = np.stack([softmax(S[:, j], tau=tau) for j in range(S.shape[1])], axis=1)
            fused[qid] = fusion_logit_pool(P, weight_vec)
        elif name == "noisy_or":
            P = np.stack([softmax(S[:, j], tau=tau) for j in range(S.shape[1])], axis=1)
            fused[qid] = fusion_noisy_or(P, weight_vec)
        elif name == "rrf":
            fused[qid] = fusion_rrf(S, k=60.0)
        elif name == "borda":
            fused[qid] = fusion_borda(S, weight_vec)
        elif name == "rra":
            fused[qid] = robust_rank_aggregation(S)
        elif name == "rank_centrality":
            fused[qid] = rank_centrality(S, iters=100, lr=0.85)
        else:
            raise ValueError(f"Unknown method {name}")
    return avg_jsd_per_query(fused, CP_by_qid, tau=tau), fused


def select_best_method(
    methods: Sequence[str],
    retriever_names: Sequence[str],
    weight_candidates: List[np.ndarray],
    S_dev: ScoreByQid,
    CP_dev: ProbByQid,
    tau: float,
) -> Dict[str, object]:
    best = {"name": None, "jsd": float("inf"), "weights": None}
    for method in methods:
        candidate_weights = weight_candidates if method not in WEIGHTLESS_METHODS else [None]
        for weights in candidate_weights:
            jsd_dev, _ = eval_method(method, S_dev, CP_dev, retriever_names, weights=weights, tau=tau)
            if jsd_dev < best["jsd"]:
                best["name"] = method
                best["jsd"] = jsd_dev
                best["weights"] = None if weights is None else weights.copy()
    return best


def main() -> None:
    args = parse_args()
    ps, retrievers, ds_dir = load_dataset(args.dataset, args.datadir, args.retrievers)
    S_by_qid, CP_by_qid = build_query_mats(ps, retrievers, args.retrievers)
    _, dev_qids, test_qids = split_qids(ps, seed=args.seed)
    S_dev = subset_dict(S_by_qid, dev_qids)
    CP_dev = subset_dict(CP_by_qid, dev_qids)
    S_test = subset_dict(S_by_qid, test_qids)
    CP_test = subset_dict(CP_by_qid, test_qids)
    rng = np.random.default_rng(args.seed)
    weight_candidates = generate_weight_candidates(len(args.retrievers), rng, args.num_weights_to_search)
    best = select_best_method(ALL_METHODS, args.retrievers, weight_candidates, S_dev, CP_dev, args.tau)
    best_weight_array = None if best["weights"] is None else np.asarray(best["weights"])
    jsd_test, _ = eval_method(
        best["name"],
        S_test,
        CP_test,
        args.retrievers,
        weights=best_weight_array,
        tau=args.tau,
    )
    out_dir = os.path.join(args.outdir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fusion_eval.jsonl")
    rec = {
        "dataset": args.dataset,
        "retrievers": args.retrievers,
        "seed": args.seed,
        "tau": args.tau,
        "best_method_dev": best["name"],
        "best_weights_dev": None if best["weights"] is None else best["weights"].tolist(),
        "dev_jsd": best["jsd"],
        "test_jsd": jsd_test,
    }
    write_jsonl(out_path, [rec], append=False)
    print(f"Fusion search saved to {out_path}")


if __name__ == "__main__":
    main()

