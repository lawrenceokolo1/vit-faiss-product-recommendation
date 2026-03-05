"""Tests for FAISS indexer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.embeddings.indexer import FAISSIndexer
from src.utils.config import EMBEDDING_DIM


@pytest.fixture
def indexer():
    return FAISSIndexer(dim=EMBEDDING_DIM)


def test_build_and_search(indexer):
    n = 10
    vectors = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    for i in range(n):
        vectors[i] /= np.linalg.norm(vectors[i])
    ids = [f"id_{i}" for i in range(n)]
    indexer.build(vectors, ids)
    query = vectors[0:1]
    scores, indices = indexer.search(query, k=3)
    assert scores.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert indices[0][0] == 0


def test_search_ids(indexer):
    n = 5
    vectors = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    for i in range(n):
        vectors[i] /= np.linalg.norm(vectors[i])
    ids = ["a", "b", "c", "d", "e"]
    indexer.build(vectors, ids)
    id_scores = indexer.search_ids(vectors[0:1], k=2)
    assert len(id_scores) == 1
    assert len(id_scores[0][0]) == 2
    assert len(id_scores[0][1]) == 2


def test_save_load(indexer):
    n = 4
    vectors = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    for i in range(n):
        vectors[i] /= np.linalg.norm(vectors[i])
    ids = ["x", "y", "z", "w"]
    indexer.build(vectors, ids)
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        indexer.save(dp / "index.faiss", dp / "ids.txt")
        other = FAISSIndexer(dim=EMBEDDING_DIM)
        other.load(dp / "index.faiss", dp / "ids.txt")
        scores, _ = other.search(vectors[0:1], k=2)
        assert scores.shape == (1, 2)
        assert other.id_list == ids
