"""
Train / query / index split generator for eProduct-style evaluation.

Uses fixed random seed for reproducibility. Splits product IDs so that:
- index_ids: used to build the FAISS index (train set)
- query_ids: held-out set for evaluation (embed query, search index, compute Recall@K)
"""

import random
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.config import (
    RANDOM_SEED,
    DEFAULT_QUERY_RATIO,
    DATA_PROCESSED,
    TRAIN_IDS_PATH,
    QUERY_IDS_PATH,
    INDEX_IDS_PATH,
    LISTINGS_PARQUET_PATH,
)


def create_splits(
    listings_path: Path | None = None,
    query_ratio: float = DEFAULT_QUERY_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: Path | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Create train (index) and query splits from listings.
    Returns (train_ids, query_ids, index_ids). Here index_ids == train_ids (same set used for index).
    """
    path = listings_path or LISTINGS_PARQUET_PATH
    output_dir = output_dir or DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"Listings not found: {path}. Run loader and save_listings_parquet first.")

    df = pd.read_parquet(path)
    ids = df["item_id"].astype(str).unique().tolist()
    random.Random(seed).shuffle(ids)

    n_query = max(1, int(len(ids) * query_ratio))
    query_ids = ids[:n_query]
    train_ids = ids[n_query:]
    index_ids = train_ids  # index is built from train set

    TRAIN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_IDS_PATH, "w") as f:
        f.write("\n".join(train_ids))
    with open(QUERY_IDS_PATH, "w") as f:
        f.write("\n".join(query_ids))
    with open(INDEX_IDS_PATH, "w") as f:
        f.write("\n".join(index_ids))

    return train_ids, query_ids, index_ids


def load_split_ids(
    train_path: Path | None = None,
    query_path: Path | None = None,
    index_path: Path | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """Load train, query, index ID lists from disk."""
    train_path = train_path or TRAIN_IDS_PATH
    query_path = query_path or QUERY_IDS_PATH
    index_path = index_path or INDEX_IDS_PATH

    def _load(p: Path) -> list[str]:
        if not p.exists():
            return []
        with open(p) as f:
            return [line.strip() for line in f if line.strip()]

    return _load(train_path), _load(query_path), _load(index_path)
