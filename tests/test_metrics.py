"""Tests for retrieval metrics."""

import pytest

from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    average_precision_at_k,
    mean_average_precision_at_k,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], "a", 1) == 1.0
    assert recall_at_k(["a", "b", "c"], "b", 2) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert recall_at_k(["a", "b", "c"], "c", 3) == 1.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], "a", 1) == 1.0
    assert precision_at_k(["a", "b", "c"], "b", 2) == 0.5


def test_average_precision_at_k():
    assert average_precision_at_k(["a", "b", "c"], "a", 3) == 1.0
    assert average_precision_at_k(["x", "a", "y"], "a", 3) == 1.0 / 2


def test_mean_average_precision_at_k():
    list_retrieved = [["a", "b"], ["x", "a"], ["a", "b", "c"]]
    list_relevant = ["a", "a", "a"]
    m = mean_average_precision_at_k(list_retrieved, list_relevant, k=3)
    assert 0 <= m <= 1
