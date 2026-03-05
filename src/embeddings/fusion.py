"""
Late fusion: alpha * image_embedding + (1 - alpha) * text_embedding, then L2-normalize.
"""

from typing import Union

import numpy as np


def fuse_embeddings(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Fuse image and text embeddings. alpha=1 -> image only, alpha=0 -> text only.
    Output is L2-normalized for use with inner-product index (cosine similarity).
    """
    fused = alpha * image_emb + (1.0 - alpha) * text_emb
    n = np.linalg.norm(fused)
    if n < 1e-8:
        return fused
    return (fused / n).astype(np.float32)


def fuse_batch(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Batch fusion: (N, D), (N, D) -> (N, D) L2-normalized."""
    fused = alpha * image_embs + (1.0 - alpha) * text_embs
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (fused / norms).astype(np.float32)
