"""
Run eProduct-style retrieval evaluation: embed query set, search index, compute Recall@1/5/10 and mAP@10.
"""

from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from src.embeddings.extractor import ViTExtractor
from src.embeddings.text_encoder import TextEncoder
from src.embeddings.fusion import fuse_embeddings
from src.embeddings.indexer import FAISSIndexer
from src.evaluation.metrics import (
    recall_at_k,
    mean_average_precision_at_k,
    recall_at_k_set,
    mean_average_precision_at_k_set,
)
from src.utils.config import (
    INDEX_PATH,
    METADATA_PATH,
    ARTIFACTS_DIR,
    DEFAULT_FUSION_ALPHA,
    EMBED_BATCH_SIZE,
)
from src.data.loader import get_image_path
from src.utils.config import ABO_IMAGES_DIR


class RetrievalEvaluator:
    """Load index and metadata, run evaluation on query IDs."""

    def __init__(
        self,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        fusion_alpha: float = DEFAULT_FUSION_ALPHA,
    ):
        self.index_path = index_path or INDEX_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self.fusion_alpha = fusion_alpha
        self.indexer: FAISSIndexer | None = None
        self.metadata: pd.DataFrame | None = None
        self.extractor: ViTExtractor | None = None
        self.text_encoder: TextEncoder | None = None

    def load(self, images_dir: Path | None = None) -> None:
        """Load FAISS index and product metadata."""
        self.indexer = FAISSIndexer()
        self.indexer.load(
            self.index_path,
            id_list_path=ARTIFACTS_DIR / "product_index_ids.txt",
        )
        if self.metadata_path.exists():
            self.metadata = pd.read_parquet(self.metadata_path)
        self.extractor = ViTExtractor()
        self.text_encoder = TextEncoder()
        self._images_dir = images_dir

    def evaluate(
        self,
        query_ids: List[str],
        metadata_df: pd.DataFrame,
        images_dir: Path | None = None,
        top_k: int = 10,
        use_fusion: bool = True,
    ) -> Dict[str, Any]:
        """
        For each query_id: get its image/text, embed (optionally fuse), search index.
        Ground truth: query_id itself is relevant (we exclude it from index in standard eval;
        here we assume index contains all and we measure if query ranks itself or similar).
        For proper eval, query set should be disjoint from index; we compute Recall@K where
        relevant = same product (or same category). Here we use same product as relevant for simplicity.
        """
        if self.indexer is None or self.metadata is None:
            self.load(images_dir)

        list_retrieved = []
        list_relevant = []
        images_dir = images_dir or self._images_dir

        for qid in query_ids:
            row = metadata_df[metadata_df["item_id"].astype(str) == str(qid)]
            if row.empty:
                list_retrieved.append([])
                list_relevant.append(qid)
                continue
            row = row.iloc[0]
            img_path = get_image_path(
                str(row["item_id"]),
                str(row.get("main_image_id", "")),
                images_dir or ABO_IMAGES_DIR,
                image_path_rel=row.get("image_path") if "image_path" in row and pd.notna(row.get("image_path")) else None,
            )
            if not img_path.exists():
                list_retrieved.append([])
                list_relevant.append(qid)
                continue
            img_emb = self.extractor.encode_image(str(img_path))
            text_emb = self.text_encoder.encode_listing(
                str(row.get("title", "")),
                str(row.get("product_type", "")),
                str(row.get("color", "")),
                str(row.get("material", "")),
            )
            if use_fusion:
                q_emb = fuse_embeddings(img_emb, text_emb, self.fusion_alpha)
            else:
                q_emb = img_emb
            result = self.indexer.search_ids(q_emb.reshape(1, -1), k=top_k)
            list_retrieved.append(result[0][0] if result else [])
            list_relevant.append(qid)

        r1 = sum(recall_at_k(ret, rel, 1) for ret, rel in zip(list_retrieved, list_relevant)) / max(len(list_retrieved), 1)
        r5 = sum(recall_at_k(ret, rel, 5) for ret, rel in zip(list_retrieved, list_relevant)) / max(len(list_retrieved), 1)
        r10 = sum(recall_at_k(ret, rel, 10) for ret, rel in zip(list_retrieved, list_relevant)) / max(len(list_retrieved), 1)
        map10 = mean_average_precision_at_k(list_retrieved, list_relevant, 10)

        return {
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "map_at_10": map10,
            "n_queries": len(query_ids),
        }

    def evaluate_by_category(
        self,
        query_ids: List[str],
        query_metadata_df: pd.DataFrame,
        index_metadata_df: pd.DataFrame,
        images_dir: Path | None = None,
        top_k: int = 10,
        use_fusion: bool = True,
        category_col: str = "category",
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval when relevant = same category (products in index with same category as query).
        Use query_metadata_df (full listings) to get query product image path and category; use
        index_metadata_df to define which index products are in each category. Gives meaningful
        metrics when query set is disjoint from index.
        """
        if self.indexer is None or self.metadata is None:
            self.load(images_dir)
        images_dir = images_dir or self._images_dir

        # Build category -> set of index product IDs (from index metadata only)
        if category_col not in index_metadata_df.columns:
            index_metadata_df = index_metadata_df.copy()
            index_metadata_df[category_col] = index_metadata_df.get("product_type", "") or ""
        cat_to_ids: Dict[str, set] = {}
        for _, row in index_metadata_df.iterrows():
            c = str(row.get(category_col, "") or "").strip()
            if c not in cat_to_ids:
                cat_to_ids[c] = set()
            cat_to_ids[c].add(str(row["item_id"]))

        list_retrieved = []
        list_relevant_sets = []

        for qid in query_ids:
            row = query_metadata_df[query_metadata_df["item_id"].astype(str) == str(qid)]
            if row.empty:
                list_retrieved.append([])
                list_relevant_sets.append(set())
                continue
            row = row.iloc[0]
            img_path = get_image_path(
                str(row["item_id"]),
                str(row.get("main_image_id", "")),
                images_dir or ABO_IMAGES_DIR,
                image_path_rel=row.get("image_path") if "image_path" in row and pd.notna(row.get("image_path")) else None,
            )
            if not img_path.exists():
                list_retrieved.append([])
                list_relevant_sets.append(set())
                continue
            cat = str(row.get(category_col, "") or "").strip()
            relevant_set = cat_to_ids.get(cat, set()).copy()
            # Exclude query if it's in index (e.g. when using index IDs as queries)
            relevant_set.discard(str(qid))

            img_emb = self.extractor.encode_image(str(img_path))
            text_emb = self.text_encoder.encode_listing(
                str(row.get("title", "")),
                str(row.get("product_type", "")),
                str(row.get("color", "")),
                str(row.get("material", "")),
            )
            if use_fusion:
                q_emb = fuse_embeddings(img_emb, text_emb, self.fusion_alpha)
            else:
                q_emb = img_emb
            result = self.indexer.search_ids(q_emb.reshape(1, -1), k=top_k)
            retrieved = result[0][0] if result else []
            list_retrieved.append(retrieved)
            list_relevant_sets.append(relevant_set)

        r1 = sum(recall_at_k_set(ret, rel, 1) for ret, rel in zip(list_retrieved, list_relevant_sets)) / max(len(list_retrieved), 1)
        r5 = sum(recall_at_k_set(ret, rel, 5) for ret, rel in zip(list_retrieved, list_relevant_sets)) / max(len(list_retrieved), 1)
        r10 = sum(recall_at_k_set(ret, rel, 10) for ret, rel in zip(list_retrieved, list_relevant_sets)) / max(len(list_retrieved), 1)
        map10 = mean_average_precision_at_k_set(list_retrieved, list_relevant_sets, 10)

        return {
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "map_at_10": map10,
            "n_queries": len(query_ids),
            "relevance": "same_category",
        }
