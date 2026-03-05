"""
Text embedding for product metadata (title, product_type, color, material).

Uses sentence-transformers/all-MiniLM-L6-v2 (384-d) and projects to 768-d
to match ViT for late fusion.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import TEXT_MODEL_NAME, TEXT_EMBEDDING_DIM, EMBEDDING_DIM


class TextEncoder:
    """Encode product text to 384-d, then project to 768-d for fusion with ViT."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Simple linear projection 384 -> 768 (learned could be added later)
        self.projection = np.eye(TEXT_EMBEDDING_DIM, EMBEDDING_DIM, dtype=np.float32)
        # Pad with zeros to get 768-d (384 + 384 zeros, then normalize)
        self._proj_matrix = np.zeros((TEXT_EMBEDDING_DIM, EMBEDDING_DIM), dtype=np.float32)
        self._proj_matrix[:, :TEXT_EMBEDDING_DIM] = np.eye(TEXT_EMBEDDING_DIM)

    def _text_from_listing(self, title: str, product_type: str, color: str, material: str) -> str:
        parts = [str(title), str(product_type), str(color), str(material)]
        return " ".join(p for p in parts if p).strip() or "unknown"

    def encode_text(self, text: str) -> np.ndarray:
        """Single text -> 768-d L2-normalized (384-d from model, padded and normalized)."""
        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        # Project to 768: pad with zeros and re-normalize
        emb_768 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_768[:TEXT_EMBEDDING_DIM] = emb
        n = np.linalg.norm(emb_768)
        if n > 1e-8:
            emb_768 /= n
        return emb_768

    def encode_listing(self, title: str, product_type: str = "", color: str = "", material: str = "") -> np.ndarray:
        """Build text from listing fields and encode to 768-d."""
        text = self._text_from_listing(title, product_type, color, material)
        return self.encode_text(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts to (N, 768) L2-normalized."""
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        out = np.zeros((len(embs), EMBEDDING_DIM), dtype=np.float32)
        out[:, :TEXT_EMBEDDING_DIM] = embs
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        out /= norms
        return out
