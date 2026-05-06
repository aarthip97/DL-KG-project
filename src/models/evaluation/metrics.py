"""Top-K recommendation evaluation metrics.

Definitions
-----------
For a user :math:`u` with ground-truth set :math:`G_u` and a top-``K``
recommendation list :math:`R_u = (r_1, \\ldots, r_K)`:

* **Recall@K**       :math:`= \\dfrac{|R_u \\cap G_u|}{\\min(|G_u|, K)}`
* **Precision@K**    :math:`= \\dfrac{|R_u \\cap G_u|}{K}`
* **HitRate@K**      :math:`= \\mathbb{1}[R_u \\cap G_u \\ne \\emptyset]`
* **MRR@K**          :math:`= \\dfrac{1}{\\text{rank of first hit}}` (0 if none)
* **DCG@K**          :math:`= \\sum_{i=1}^{K} \\dfrac{\\text{hit}_i}{\\log_2(i+1)}`
* **NDCG@K**         :math:`= \\dfrac{\\text{DCG@K}}{\\text{IDCG@K}}`
* **Coverage**       :math:`= \\dfrac{|\\bigcup_u R_u|}{N_{\\text{songs}}}`
* **PopularityBias** :math:`= \\text{mean}\\bigl(\\text{pop\\_norm}[r]\\bigr)`

The composite ``Overall_Score`` combines accuracy, ranking quality, catalog
coverage and an inverted popularity bias to favour balanced recommenders.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd


# ─── Building blocks ──────────────────────────────────────────────────────────

def dcg(hits: Sequence[float]) -> float:
    """Discounted Cumulative Gain of a binary hit vector."""
    h = np.asarray(hits, dtype=float)
    if h.sum() == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(h) + 2))
    return float(np.sum(h / discounts))


def precision_at_k(hit_vec: Sequence[int], k: int) -> float:
    """Fraction of the top-``k`` items that are relevant."""
    if k <= 0:
        return 0.0
    return float(np.sum(hit_vec[:k])) / k


# ─── Per-user records (for statistical testing) ───────────────────────────────

def evaluate_recs_per_user(
    recs_dict: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    pop_norm: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Return one row per evaluated user with all metrics at cut-off ``k``.

    Use this DataFrame for paired statistical tests (Wilcoxon, paired *t*).
    """
    rows: List[dict] = []
    for u, rec in recs_dict.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        top = list(rec[:k])
        h = [1 if s in gt else 0 for s in top]
        ndcg = dcg(h) / (dcg(sorted(h, reverse=True)) + 1e-9)
        rows.append({
            "u_idx":     u,
            "Recall@K":  len(set(top) & gt) / min(len(gt), k),
            "Precision@K": precision_at_k(h, k),
            "NDCG@K":    float(ndcg),
            "HitRate@K": float(any(s in gt for s in top)),
            "MRR":       next((1.0 / (r + 1) for r, s in enumerate(top) if s in gt), 0.0),
            "PopularityBias": float(np.mean(pop_norm[list(top)])) if top else 0.0,
        })
    return pd.DataFrame(rows)


# ─── Aggregate metrics ────────────────────────────────────────────────────────

def evaluate_recs(
    recs_dict: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    seen_dict: Mapping[int, Set[int]],
    n_songs: int,
    pop_norm: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Aggregate top-``k`` metrics over every user in ``recs_dict``.

    ``seen_dict`` is kept for API symmetry; it is not used here because the
    ``recs_dict`` is assumed to be already filtered of items the user saw at
    training time.
    """
    df = evaluate_recs_per_user(recs_dict, ground_truth, pop_norm, k)
    if df.empty:
        return {m: 0.0 for m in
                ("Recall@K", "Precision@K", "NDCG@K", "HitRate@K",
                 "MRR", "Coverage", "PopularityBias")}
    rec_set = {s for rec in recs_dict.values() for s in rec[:k]}
    return {
        "Recall@K":       float(df["Recall@K"].mean()),
        "Precision@K":    float(df["Precision@K"].mean()),
        "NDCG@K":         float(df["NDCG@K"].mean()),
        "HitRate@K":      float(df["HitRate@K"].mean()),
        "MRR":            float(df["MRR"].mean()),
        "Coverage":       len(rec_set) / n_songs,
        "PopularityBias": float(df["PopularityBias"].mean()),
    }


def overall_score(
    metrics: Mapping[str, float],
    *,
    w_recall:   float = 0.35,
    w_ndcg:     float = 0.35,
    w_cov:      float = 0.20,
    w_anti_pop: float = 0.10,
) -> float:
    """Weighted blend of accuracy, ranking, coverage and inverted popularity bias.

    Default weights → ``0.35·Recall@K + 0.35·NDCG@K + 0.20·Coverage + 0.10·(1−PopularityBias)``.
    """
    return (
        w_recall   * metrics["Recall@K"]
        + w_ndcg     * metrics["NDCG@K"]
        + w_cov      * metrics["Coverage"]
        + w_anti_pop * (1.0 - metrics["PopularityBias"])
    )


# ─── Multi-K table ────────────────────────────────────────────────────────────

def multi_k_evaluation(
    recs_dict_max_k: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    seen_dict: Mapping[int, Set[int]],
    n_songs: int,
    pop_norm: np.ndarray,
    *,
    ks: Iterable[int] = (5, 10, 20, 50),
    model_name: str = "model",
) -> pd.DataFrame:
    """Evaluate the same recommendation lists at multiple cut-offs ``K``.

    ``recs_dict_max_k`` must contain at least ``max(ks)`` items per user;
    each ``K`` slices the same list, avoiding repeated re-ranking.

    Returns a long-format ``DataFrame`` with columns ``[model, K, metric, value]``.
    """
    ks_sorted = sorted(set(int(k) for k in ks))
    rows = []
    for k in ks_sorted:
        m = evaluate_recs(recs_dict_max_k, ground_truth, seen_dict,
                          n_songs, pop_norm, k=k)
        m["Overall_Score"] = overall_score(m)
        for metric_name, value in m.items():
            rows.append({"model": model_name, "K": k,
                         "metric": metric_name, "value": float(value)})
    return pd.DataFrame(rows)
