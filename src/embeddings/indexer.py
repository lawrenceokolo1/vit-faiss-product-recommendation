"""
FAISS index builder and loader.

Uses IndexFlatIP (inner product) for exact search on L2-normalized vectors (= cosine similarity).
"""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.utils.config import ARTIFACTS_DIR, INDEX_PATH


class FAISSIndexer:
    """Build and query a FAISS IndexFlatIP index over 768-d vectors."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index: faiss.IndexFlatIP | None = None
        self.id_list: List[str] = []

    def build(self, vectors: np.ndarray, id_list: List[str]) -> None:
        """Build index from (N, dim) float32 vectors and ordered list of product IDs."""
        assert vectors.shape[0] == len(id_list), "vectors and id_list length mismatch"
        assert vectors.shape[1] == self.dim
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.id_list = list(id_list)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Query vector (1, dim) or (N, dim). Returns (scores, indices) where indices refer to id_list."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        query = np.ascontiguousarray(query.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self.index.search(query, min(k, len(self.id_list)))
        return scores, indices

    def search_ids(
        self, query: np.ndarray, k: int = 10
    ) -> List[Tuple[List[str], List[float]]]:
        """Same as search but returns list of (id_list, score_list) per query row."""
        scores, indices = self.search(query, k)
        out = []
        for i in range(scores.shape[0]):
            ids = [
                self.id_list[int(j)]
                for j in indices[i]
                if 0 <= int(j) < len(self.id_list)
            ]
            sc = scores[i].tolist()[: len(ids)]
            out.append((ids, sc))
        return out

    def save(
        self, index_path: Path | None = None, id_list_path: Path | None = None
    ) -> None:
        """Save FAISS index and ID list (indexer loads both)."""
        if self.index is None:
            raise RuntimeError("No index to save")
        index_path = index_path or INDEX_PATH
        id_list_path = id_list_path or (ARTIFACTS_DIR / "product_index_ids.txt")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(id_list_path, "w") as f:
            f.write("\n".join(self.id_list))

    def load(
        self, index_path: Path | None = None, id_list_path: Path | None = None
    ) -> None:
        """Load FAISS index and ID list."""
        index_path = index_path or INDEX_PATH
        id_list_path = id_list_path or (ARTIFACTS_DIR / "product_index_ids.txt")
        self.index = faiss.read_index(str(index_path))
        self.dim = self.index.d
        with open(id_list_path) as f:
            self.id_list = [line.strip() for line in f if line.strip()]
