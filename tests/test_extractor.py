"""Tests for ViT extractor."""

import numpy as np
import pytest
from PIL import Image

from src.embeddings.extractor import ViTExtractor
from src.utils.config import EMBEDDING_DIM


@pytest.fixture
def extractor():
    return ViTExtractor()


def test_encode_image_returns_768d(extractor):
    """Output shape is (768,) and L2-normalized."""
    img = Image.new("RGB", (224, 224), color="red")
    emb = extractor.encode_image(img)
    assert emb.shape == (EMBEDDING_DIM,)
    assert emb.dtype == np.float32
    np.testing.assert_almost_equal(np.linalg.norm(emb), 1.0, decimal=4)


def test_encode_batch(extractor):
    """Batch encode returns (N, 768)."""
    imgs = [Image.new("RGB", (224, 224), color=(i % 256, 0, 0)) for i in range(3)]
    embs = extractor.encode_batch(imgs, batch_size=2)
    assert embs.shape == (3, EMBEDDING_DIM)
    for i in range(3):
        np.testing.assert_almost_equal(np.linalg.norm(embs[i]), 1.0, decimal=4)
