"""
Run eProduct-style retrieval evaluation and optionally log to MLflow.
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.splitter import load_split_ids
from src.evaluation.evaluator import RetrievalEvaluator
from src.utils.config import (
    QUERY_IDS_PATH,
    METADATA_PATH,
    EVAL_RESULTS_PATH,
    ARTIFACTS_DIR,
    ABO_IMAGES_DIR,
    LISTINGS_PARQUET_PATH,
)

import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def run(
    query_ids_path: Path | None = None,
    metadata_path: Path | None = None,
    index_path: Path | None = None,
    fusion_alpha: float = 0.7,
    use_fusion: bool = True,
    top_k: int = 10,
    log_mlflow: bool = False,
    by_category: bool = True,
    max_queries: int = 500,
) -> dict:
    """
    Run retrieval evaluation and save eval_results.json.
    by_category=True (default): relevant = same category in index; gives meaningful metrics
    when query set is disjoint from index. by_category=False: relevant = query product (only
    meaningful if query products are in the index).
    """
    query_ids_path = query_ids_path or QUERY_IDS_PATH
    metadata_path = metadata_path or METADATA_PATH
    if not query_ids_path.exists():
        raise FileNotFoundError(f"Query IDs not found: {query_ids_path}. Run build_index first to create splits.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run build_index first.")

    _, query_ids, _ = load_split_ids()
    if not query_ids:
        with open(query_ids_path) as f:
            query_ids = [line.strip() for line in f if line.strip()]

    index_metadata = pd.read_parquet(metadata_path)
    evaluator = RetrievalEvaluator(fusion_alpha=fusion_alpha)
    evaluator.load(images_dir=ABO_IMAGES_DIR)

    if by_category:
        if not LISTINGS_PARQUET_PATH.exists():
            raise FileNotFoundError(
                f"Full listings needed for category eval: {LISTINGS_PARQUET_PATH}. Run build_index first."
            )
        query_metadata = pd.read_parquet(LISTINGS_PARQUET_PATH)
        # Only keep query_ids that appear in full listings (have image path etc.)
        query_ids_sub = [q for q in query_ids[:max_queries] if q in query_metadata["item_id"].astype(str).values]
        if not query_ids_sub:
            raise ValueError("No query IDs found in listings.parquet. Check data.")
        results = evaluator.evaluate_by_category(
            query_ids=query_ids_sub,
            query_metadata_df=query_metadata,
            index_metadata_df=index_metadata,
            images_dir=ABO_IMAGES_DIR,
            top_k=top_k,
            use_fusion=use_fusion,
        )
    else:
        results = evaluator.evaluate(
            query_ids=query_ids[:max_queries],
            metadata_df=index_metadata,
            images_dir=ABO_IMAGES_DIR,
            top_k=top_k,
            use_fusion=use_fusion,
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Eval results:", results)

    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("visual-product-recommender")
        with mlflow.start_run():
            mlflow.log_metrics({
                "recall_at_1": results["recall_at_1"],
                "recall_at_5": results["recall_at_5"],
                "recall_at_10": results["recall_at_10"],
                "map_at_10": results["map_at_10"],
            })
            mlflow.log_artifact(str(EVAL_RESULTS_PATH))

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fusion", action="store_true", help="Evaluate image-only")
    parser.add_argument("--fusion-alpha", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument("--by-category", action="store_true", default=True, help="Relevant = same category (default)")
    parser.add_argument("--by-product", action="store_true", help="Relevant = same product (0 when query not in index)")
    parser.add_argument("--max-queries", type=int, default=500, help="Max number of queries to run")
    args = parser.parse_args()
    by_category = not args.by_product
    run(
        use_fusion=not args.no_fusion,
        fusion_alpha=args.fusion_alpha,
        top_k=args.top_k,
        log_mlflow=args.mlflow,
        by_category=by_category,
        max_queries=args.max_queries,
    )
