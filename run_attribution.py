from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List

from migrascope.attribution import AttributionEngine
from migrascope.estimators import EstimatorConfig
from migrascope.io import load_pseudo_gt, load_retriever, write_jsonl


def str_to_bool(value: str) -> bool:
    """Return True when the string represents a truthy value."""
    return value.lower() == 'true'


def parse_args() -> argparse.Namespace:
    """Configure and return the CLI arguments."""
    parser = argparse.ArgumentParser(description="Run attribution over retriever outputs.")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--retrievers', nargs='+', required=True)
    parser.add_argument('--gaussian', type=str, default='true')
    parser.add_argument('--datadir', default='input_for_MI')
    parser.add_argument('--outdir', default='results_MI')
    return parser.parse_args()


def load_retrievers_by_id(retriever_ids: List[str], ret_dir: Path) -> Dict[str, Any]:
    """Load retriever outputs from disk keyed by retriever identifier."""
    retrievers: Dict[str, Any] = {}
    for retriever_id in retriever_ids:
        retriever_path = ret_dir / f"{retriever_id}.jsonl"
        retrievers[retriever_id] = load_retriever(str(retriever_path))
    return retrievers


def build_record(args: argparse.Namespace, gaussian: bool, results: Any) -> Dict[str, Any]:
    """Assemble the attribution record that will be persisted."""
    return {
        "dataset": args.dataset,
        "retrievers": args.retrievers,
        "gaussian": gaussian,
        "F_all": results.F_all,
        "marginals": results.marginals,
        "shapley": results.shapley,
        "interaction_matrix": results.interaction_matrix,
        "total_mi": results.total_mi,
    }


def main() -> None:
    """Entrypoint that orchestrates loading data, running attribution, and persisting results."""
    args = parse_args()
    gaussian = str_to_bool(args.gaussian)

    dataset_dir = Path(args.datadir) / args.dataset
    pseudo_gt_path = dataset_dir / 'pseudo_gt.jsonl'
    retriever_dir = dataset_dir / 'retrievers'

    pseudo_gt = load_pseudo_gt(str(pseudo_gt_path))
    retrievers = load_retrievers_by_id(args.retrievers, retriever_dir)

    engine = AttributionEngine(EstimatorConfig(gaussian=gaussian))
    results = engine.compute(pseudo_gt, retrievers)

    out_dir = Path(args.outdir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'attribution.jsonl'

    record = build_record(args, gaussian, results)
    write_jsonl(str(out_path), [record], append=False)
    print(f"Attribution results saved to {out_path}")


if __name__ == '__main__':
    main()
