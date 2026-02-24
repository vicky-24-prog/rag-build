"""
RETRIEVAL METRICS  (Upgrade 1 / 3 support module)

Pure-function implementations of standard Information Retrieval metrics:

  - Recall@k      : fraction of relevant docs found in top-k results
  - MRR           : Mean Reciprocal Rank across a query set
  - NDCG@k        : Normalised Discounted Cumulative Gain

All functions operate on plain Python objects so they can be used in any
context (offline batch, unit tests, notebooks) without importing the full
RAG stack.

Definitions
-----------
retrieved_ids   : ordered list of doc/product IDs returned by the retriever,
                  best match first (index 0).
relevant_ids    : set (or list) of IDs that are considered relevant for this
                  query (ground truth).
relevance_scores: dict {doc_id: gain_value} for graded relevance in NDCG.
                  If omitted, binary relevance is assumed (1 if in relevant_ids).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Union


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
    k: int,
) -> float:
    """
    Recall@k = |{relevant} ∩ {top-k retrieved}| / |{relevant}|

    Parameters
    ----------
    retrieved_ids : ordered list returned by the retriever (best first).
    relevant_ids  : ground-truth relevant document IDs.
    k             : cutoff rank.

    Returns
    -------
    float in [0, 1]. Returns 0.0 if relevant_ids is empty.

    Examples
    --------
    >>> compute_recall_at_k(["a","b","c","d"], {"a","c","e"}, k=3)
    0.6666666666666666
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def compute_recall_at_k_batch(
    retrieved_list: List[List[str]],
    relevant_list: List[Union[Set[str], List[str]]],
    k: int,
) -> float:
    """
    Macro-average Recall@k across a list of queries.

    Parameters
    ----------
    retrieved_list : one ordered list per query.
    relevant_list  : one relevant-set per query (same order).
    k              : cutoff rank.

    Returns
    -------
    Mean Recall@k (float).
    """
    if len(retrieved_list) != len(relevant_list):
        raise ValueError("retrieved_list and relevant_list must have the same length.")
    if not retrieved_list:
        return 0.0
    scores = [
        compute_recall_at_k(ret, rel, k)
        for ret, rel in zip(retrieved_list, relevant_list)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# MRR  (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------

def compute_rr(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_result  (0.0 if not found)

    Parameters
    ----------
    retrieved_ids : ordered ranked list (best first).
    relevant_ids  : ground-truth relevant IDs.

    Returns
    -------
    float in (0, 1] or 0.0.

    Examples
    --------
    >>> compute_rr(["x","a","b"], {"a","c"})
    0.5
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_mrr(
    retrieved_list: List[List[str]],
    relevant_list: List[Union[Set[str], List[str]]],
) -> float:
    """
    Mean Reciprocal Rank across a query set.

    MRR = (1/|Q|) * Σ RR(q)

    Parameters
    ----------
    retrieved_list : one ranked list per query.
    relevant_list  : one relevant-set per query.

    Returns
    -------
    float in [0, 1].

    Examples
    --------
    >>> compute_mrr([["a","b"],["x","a"]], [{"a"},{"a"}])
    0.75
    """
    if not retrieved_list:
        return 0.0
    rr_scores = [
        compute_rr(ret, rel)
        for ret, rel in zip(retrieved_list, relevant_list)
    ]
    return sum(rr_scores) / len(rr_scores)


# ---------------------------------------------------------------------------
# NDCG@k  (Normalised Discounted Cumulative Gain)
# ---------------------------------------------------------------------------

def _dcg_at_k(gains: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    DCG@k = Σ_{i=1}^{k}  gain_i / log2(i + 1)
    """
    return sum(
        gain / math.log2(rank + 2)         # rank is 0-indexed, so +2
        for rank, gain in enumerate(gains[:k])
    )


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    NDCG@k for a single query.

    NDCG@k = DCG@k / IDCG@k

    Parameters
    ----------
    retrieved_ids    : ordered ranked list from retriever (best first).
    relevant_ids     : ground-truth relevant IDs (for binary relevance).
    k                : cutoff rank.
    relevance_scores : optional {doc_id: gain} for graded relevance.
                       If None, binary relevance (1/0) is used.

    Returns
    -------
    float in [0, 1].

    Examples
    --------
    >>> compute_ndcg_at_k(["a","b","c"], {"a","c"}, k=3)
    0.9306765580733931
    """
    relevant_set = set(relevant_ids)

    def gain(doc_id: str) -> float:
        if relevance_scores:
            return float(relevance_scores.get(doc_id, 0.0))
        return 1.0 if doc_id in relevant_set else 0.0

    actual_gains = [gain(doc_id) for doc_id in retrieved_ids[:k]]

    # Ideal ranking: put the highest possible gains in the top-k positions.
    # For binary relevance this is min(|relevant|, k) ones followed by zeros.
    # We must NOT look at retrieved_ids here — the ideal is built from the
    # known relevant set only (documents we *could* have retrieved).
    if relevance_scores:
        all_possible_gains = sorted(relevance_scores.values(), reverse=True)
    else:
        all_possible_gains = [1.0] * len(relevant_set)
    ideal_gains = (all_possible_gains + [0.0] * k)[:k]

    # If ideal is all zeros, NDCG is undefined → return 0
    idcg = _dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return _dcg_at_k(actual_gains, k) / idcg


def compute_ndcg_at_k_batch(
    retrieved_list: List[List[str]],
    relevant_list: List[Union[Set[str], List[str]]],
    k: int,
    relevance_scores_list: Optional[List[Optional[Dict[str, float]]]] = None,
) -> float:
    """
    Macro-average NDCG@k across a query set.

    Parameters
    ----------
    retrieved_list        : one ranked list per query.
    relevant_list         : one relevant-set per query.
    k                     : cutoff rank.
    relevance_scores_list : optional per-query graded relevance dicts.

    Returns
    -------
    Mean NDCG@k (float).
    """
    if not retrieved_list:
        return 0.0
    n = len(retrieved_list)
    rs_list = relevance_scores_list or [None] * n
    scores = [
        compute_ndcg_at_k(ret, rel, k, rs)
        for ret, rel, rs in zip(retrieved_list, relevant_list, rs_list)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Convenience: compute all standard k-values at once
# ---------------------------------------------------------------------------

def compute_all_retrieval_metrics(
    retrieved_list: List[List[str]],
    relevant_list: List[Union[Set[str], List[str]]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Compute Recall@k, NDCG@k for each k, plus MRR, for a batch of queries.

    Returns
    -------
    Dict with keys: "recall@1", "recall@3", ..., "ndcg@1", ..., "mrr".

    Examples
    --------
    >>> results = compute_all_retrieval_metrics(
    ...     [["a","b","c"],["x","a"]],
    ...     [{"a","c"}, {"a"}],
    ... )
    >>> results["mrr"]
    0.75
    """
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = compute_recall_at_k_batch(
            retrieved_list, relevant_list, k
        )
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k_batch(
            retrieved_list, relevant_list, k
        )
    metrics["mrr"] = compute_mrr(retrieved_list, relevant_list)
    return metrics
