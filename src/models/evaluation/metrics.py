"""Top-K recommendation evaluation metrics for simple and GNN-based models.

This module provides a unified evaluation interface that works regardless of how
recommendations are generated.

Simple models (popularity baseline, nearest-neighbour CF, matrix factorisation)
produce a recs_dict directly and can pass it straight to evaluate_recs or
multi_k_evaluation.

Embedding-based GNN models can use recs_from_embeddings to convert their
(user_emb, item_emb) tensors to a recs_dict, or call evaluate_from_embeddings
as a single-step convenience wrapper.

Score-matrix models (e.g. ALS that exposes a full U x I matrix) can use
recs_from_score_matrix instead.

All aggregate metrics are expressed as means over the evaluated users. The
composite Overall_Score blends accuracy (Recall, NDCG), catalogue coverage and
an inverted popularity bias to reward recommenders that surface diverse content.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch


# Building blocks

def dcg(hits: Sequence[float]) -> float:
    """Discounted Cumulative Gain of a binary hit vector."""
    h = np.asarray(hits, dtype=float)
    if h.sum() == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(h) + 2))
    return float(np.sum(h / discounts))


def precision_at_k(hit_vec: Sequence[int], k: int) -> float:
    """Fraction of the top-k items that are relevant."""
    if k <= 0:
        return 0.0
    return float(np.sum(hit_vec[:k])) / k


# Recommendation generation helpers

def recs_from_embeddings(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_ids: List[int],
    seen_dict: Optional[Mapping[int, Set[int]]] = None,
    top_n: int = 20,
    batch_size: int = 1024,
) -> Dict[int, List[int]]:
    """Generate a top-N recommendation list for each user from embedding vectors.

    Scores items for each user by computing a dot product between the user
    embedding and all item embeddings. Items present in seen_dict for a given
    user are masked to negative infinity so they cannot appear in the output.
    Users are processed in batches of batch_size to limit peak GPU memory.

    Parameters are:
      user_emb   -- (U, D) float tensor, one embedding row per user.
      item_emb   -- (I, D) float tensor, one embedding row per item.
      user_ids   -- list of user node indices to generate recommendations for.
      seen_dict  -- optional mapping from user id to the set of item ids seen
                    during training; these items are excluded from output.
      top_n      -- number of items to return per user.
      batch_size -- number of users scored per chunk.

    Returns a dict mapping user id to a list of item ids ordered best-first.
    """
    device = user_emb.device
    k = min(top_n, item_emb.size(0))
    recs: Dict[int, List[int]] = {}

    for start in range(0, len(user_ids), batch_size):
        batch_uids = user_ids[start : start + batch_size]
        u_idx = torch.tensor(batch_uids, device=device)
        scores = user_emb[u_idx] @ item_emb.t()

        if seen_dict is not None:
            for bi, uid in enumerate(batch_uids):
                seen = seen_dict.get(int(uid))
                if seen:
                    scores[bi, list(seen)] = float("-inf")

        _, top_idx = torch.topk(scores, k=k, dim=-1)
        for bi, uid in enumerate(batch_uids):
            recs[int(uid)] = top_idx[bi].tolist()

    return recs


def recs_from_score_matrix(
    score_matrix: torch.Tensor,
    user_ids: List[int],
    seen_dict: Optional[Mapping[int, Set[int]]] = None,
    top_n: int = 20,
) -> Dict[int, List[int]]:
    """Generate top-N recommendations from a pre-computed (U, I) score matrix.

    Useful for simple models such as ALS or popularity baselines that already
    expose a full user-item score matrix. Items in seen_dict are masked before
    ranking. The entire set of user_ids is processed in one shot, so keep
    the matrix on CPU if it does not fit in GPU memory.

    Parameters are:
      score_matrix -- (U, I) float tensor where entry [u, i] is the predicted
                      relevance of item i for user u.
      user_ids     -- list of user indices (rows) to extract recommendations for.
      seen_dict    -- optional training interactions per user to exclude.
      top_n        -- number of top-ranked items to return per user.

    Returns a dict mapping user id to an ordered list of item ids.
    """
    k = min(top_n, score_matrix.size(1))
    scores = score_matrix[user_ids].clone()

    if seen_dict is not None:
        for bi, uid in enumerate(user_ids):
            seen = seen_dict.get(int(uid))
            if seen:
                scores[bi, list(seen)] = float("-inf")

    _, top_idx = torch.topk(scores, k=k, dim=-1)
    return {int(uid): top_idx[bi].tolist() for bi, uid in enumerate(user_ids)}


# Per-user evaluation records

def evaluate_recs_per_user(
    recs_dict: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    pop_norm: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Return one row per evaluated user with all per-user metrics at cut-off k.

    Use the returned DataFrame for paired statistical tests such as Wilcoxon
    signed-rank or paired t-test when comparing two models.

    Parameters are:
      recs_dict    -- mapping from user id to their ranked recommendation list.
      ground_truth -- mapping from user id to the set of relevant item ids.
      pop_norm     -- 1-D array of normalised popularity scores, one per item.
                      Used to compute PopularityBias as the mean popularity of
                      the recommended items for each user.
      k            -- cut-off at which all metrics are evaluated.

    Returns a DataFrame with columns u_idx, Recall@K, Precision@K, NDCG@K,
    HitRate@K, MRR, and PopularityBias.
    """
    rows: List[dict] = []
    for u, rec in recs_dict.items():
        gt = ground_truth.get(u, set())
        if not gt:
            continue
        top = list(rec[:k])
        h = [1 if s in gt else 0 for s in top]
        ndcg_val = dcg(h) / (dcg(sorted(h, reverse=True)) + 1e-9)
        rows.append({
            "u_idx":          u,
            "Recall@K":       len(set(top) & gt) / min(len(gt), k),
            "Precision@K":    precision_at_k(h, k),
            "NDCG@K":         float(ndcg_val),
            "HitRate@K":      float(any(s in gt for s in top)),
            "MRR":            next((1.0 / (r + 1) for r, s in enumerate(top) if s in gt), 0.0),
            "PopularityBias": float(np.mean(pop_norm[list(top)])) if top else 0.0,
        })
    return pd.DataFrame(rows)


# Aggregate metrics

def evaluate_recs(
    recs_dict: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    seen_dict: Mapping[int, Set[int]],
    n_songs: int,
    pop_norm: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Aggregate top-k metrics over every user in recs_dict.

    Works identically for simple models and GNN models; the only requirement
    is that recs_dict already has training-seen items filtered out (or that
    seen_dict is applied upstream). seen_dict is kept in the signature for
    API compatibility but is not used inside this function.

    Parameters are:
      recs_dict    -- pre-computed recommendations, one ranked list per user.
      ground_truth -- held-out ground-truth item sets per user.
      seen_dict    -- training interactions per user (accepted but not used here).
      n_songs      -- total catalogue size, used to compute Coverage.
      pop_norm     -- normalised popularity array, one entry per song.
      k            -- evaluation cut-off.

    Returns a dict with keys Recall@K, Precision@K, NDCG@K, HitRate@K,
    MRR, Coverage, and PopularityBias.
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


def evaluate_from_embeddings(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_ids: List[int],
    ground_truth: Mapping[int, Set[int]],
    seen_dict: Mapping[int, Set[int]],
    n_songs: int,
    pop_norm: np.ndarray,
    k: int,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Convenience wrapper: generate recommendations from GNN embeddings and evaluate.

    Combines recs_from_embeddings and evaluate_recs into a single call. The
    caller passes pre-computed node embeddings (e.g. from model.encode(data))
    and gets back the full aggregate metric dict without needing to manage the
    intermediate recs_dict.

    Parameters are:
      user_emb     -- (U, D) tensor of user embeddings from the GNN.
      item_emb     -- (I, D) tensor of item embeddings from the GNN.
      user_ids     -- user indices to evaluate.
      ground_truth -- held-out positive item sets per user.
      seen_dict    -- training interactions per user used to mask recommendations.
      n_songs      -- catalogue size for Coverage computation.
      pop_norm     -- normalised popularity array.
      k            -- evaluation cut-off.
      batch_size   -- users processed per scoring chunk.

    Returns the same metric dict as evaluate_recs.
    """
    recs = recs_from_embeddings(
        user_emb, item_emb, user_ids,
        seen_dict=seen_dict, top_n=k, batch_size=batch_size,
    )
    return evaluate_recs(recs, ground_truth, seen_dict, n_songs, pop_norm, k)


def overall_score(
    metrics: Mapping[str, float],
    *,
    w_recall:   float = 0.35,
    w_ndcg:     float = 0.35,
    w_cov:      float = 0.20,
    w_anti_pop: float = 0.10,
) -> float:
    """Weighted composite score combining accuracy, ranking, coverage, and diversity.

    Default weights produce 0.35 * Recall@K + 0.35 * NDCG@K + 0.20 * Coverage
    + 0.10 * (1 - PopularityBias). The anti-popularity term rewards models that
    avoid concentrating all recommendations on already-popular tracks.
    """
    return (
        w_recall   * metrics["Recall@K"]
        + w_ndcg     * metrics["NDCG@K"]
        + w_cov      * metrics["Coverage"]
        + w_anti_pop * (1.0 - metrics["PopularityBias"])
    )


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
    """Evaluate the same recommendation lists at multiple cut-offs K.

    recs_dict_max_k must contain at least max(ks) items per user. Each K slices
    the same ranked list so no re-ranking is performed. This is the standard
    multi-cut-off evaluation used in recommender system benchmarking and works
    identically for simple models and GNN models as long as recs_dict_max_k is
    pre-built using the appropriate helper.

    Parameters are:
      recs_dict_max_k -- recommendation dict with at least max(ks) items per user.
      ground_truth    -- held-out item sets per user.
      seen_dict       -- training interactions per user (API compatibility only).
      n_songs         -- catalogue size.
      pop_norm        -- normalised popularity array.
      ks              -- iterable of integer cut-off values.
      model_name      -- label stored in the model column of the output DataFrame.

    Returns a long-format DataFrame with columns [model, K, metric, value].
    """
    ks_sorted = sorted(set(int(k) for k in ks))
    rows = []
    for k in ks_sorted:
        m = evaluate_recs(recs_dict_max_k, ground_truth, seen_dict, n_songs, pop_norm, k=k)
        m["Overall_Score"] = overall_score(m)
        for metric_name, value in m.items():
            rows.append({"model": model_name, "K": k,
                         "metric": metric_name, "value": float(value)})
    return pd.DataFrame(rows)
