"""Tests for late fusion."""

import numpy as np
import pytest

from src.embeddings.fusion import fuse_embeddings, fuse_batch


def test_fuse_embeddings_normalized():
    a = np.random.randn(768).astype(np.float32)
    b = np.random.randn(768).astype(np.float32)
    fused = fuse_embeddings(a, b, alpha=0.7)
    assert fused.shape == (768,)
    np.testing.assert_almost_equal(np.linalg.norm(fused), 1.0, decimal=5)


def test_fuse_alpha_one_is_image():
    a = np.random.randn(768).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = np.random.randn(768).astype(np.float32)
    fused = fuse_embeddings(a, b, alpha=1.0)
    np.testing.assert_array_almost_equal(fused, a)


def test_fuse_batch():
    n = 4
    img = np.random.randn(n, 768).astype(np.float32)
    txt = np.random.randn(n, 768).astype(np.float32)
    out = fuse_batch(img, txt, alpha=0.7)
    assert out.shape == (n, 768)
    for i in range(n):
        np.testing.assert_almost_equal(np.linalg.norm(out[i]), 1.0, decimal=5)
