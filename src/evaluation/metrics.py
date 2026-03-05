"""
Retrieval metrics: Recall@K, Precision@K, mAP@K.
"""

from typing import List


def recall_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    """1 if relevant_id in retrieved_ids[:k], else 0."""
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    """1/k if relevant_id in retrieved_ids[:k], else 0."""
    if relevant_id not in retrieved_ids[:k]:
        return 0.0
    return 1.0 / k


def average_precision_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    """AP@k: average of precision@i for i where retrieved_ids[i] == relevant_id."""
    hits = [i + 1 for i, rid in enumerate(retrieved_ids[:k]) if rid == relevant_id]
    if not hits:
        return 0.0
    return sum(1.0 / h for h in hits) / len(hits)


def mean_average_precision_at_k(
    list_retrieved: List[List[str]],
    list_relevant: List[str],
    k: int,
) -> float:
    """mAP@k over a list of (retrieved_ids, relevant_id) pairs."""
    if not list_retrieved or not list_relevant:
        return 0.0
    assert len(list_retrieved) == len(list_relevant)
    aps = [average_precision_at_k(ret, rel, k) for ret, rel in zip(list_retrieved, list_relevant)]
    return sum(aps) / len(aps)


# --- Set-based relevance (e.g. relevant = same category) ---


def recall_at_k_set(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """1.0 if any of retrieved_ids[:k] is in relevant_ids, else 0.0."""
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def average_precision_at_k_set(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """AP@k when relevant = set of IDs. AP = (1/|relevant|) * sum over relevant of (precision at rank)."""
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_ids) if relevant_ids else 0.0


def mean_average_precision_at_k_set(
    list_retrieved: List[List[str]],
    list_relevant: List[set],
    k: int,
) -> float:
    """mAP@k when each query has a set of relevant IDs."""
    if not list_retrieved or not list_relevant:
        return 0.0
    assert len(list_retrieved) == len(list_relevant)
    aps = [
        average_precision_at_k_set(ret, rel, k)
        for ret, rel in zip(list_retrieved, list_relevant)
    ]
    return sum(aps) / len(aps)
