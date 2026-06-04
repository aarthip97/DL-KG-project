"""Top-K recommendation evaluation metrics for simple and GNN-based models.

This module provides a unified evaluation interface that works regardless of how
recommendations are generated.

Simple models (popularity baseline, nearest-neighbour CF, matrix factorisation)
produce a recs_dict directly and can pass it straight to evaluate_recs or
multi_k_evaluation.

Embedding-based GNN models can use recs_from_embeddings to convert their
(user_emb, item_emb) tensors to a recs_dict, or call evaluate_from_embeddings
as a single-step convenience wrapper.  For large-scale K-sweeps that need to
stay fully on-device, evaluate_gnn_k_sweep performs a single top-max_K sort
and evaluates all requested cut-offs in one vectorised pass — substantially
faster than calling evaluate_from_embeddings once per K.

Score-matrix models (e.g. ALS that exposes a full U x I matrix) can use
recs_from_score_matrix instead.

All aggregate metrics are expressed as means over the evaluated users. The
composite Overall_Score blends accuracy (Recall, NDCG), catalogue coverage and
an inverted popularity bias to reward recommenders that surface diverse content.
"""
from __future__ import annotations

import re
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
        recall = len(set(top) & gt) / min(len(gt), k)
        prec   = precision_at_k(h, k)
        f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        rows.append({
            "u_idx":          u,
            "Recall@K":       recall,
            "Precision@K":    prec,
            "F1@K":           float(f1),
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
    MRR, Coverage, and PopularityBias@K.
    """
    df = evaluate_recs_per_user(recs_dict, ground_truth, pop_norm, k)
    if df.empty:
        return dict.fromkeys(
                (f"Mean_Recall@{k}", f"Mean_Precision@{k}", f"Mean_F1@{k}",
                 f"Mean_NDCG@{k}", f"Mean_HitRate@{k}", "MRR", "Coverage",
                 f"Mean_PopularityBias@{k}"), 0.0)
    rec_set = {s for rec in recs_dict.values() for s in rec[:k]}
    return {
        f"Mean_Recall@{k}":         float(df["Recall@K"].mean()),
        f"Mean_Precision@{k}":      float(df["Precision@K"].mean()),
        f"Mean_F1@{k}":             float(df["F1@K"].mean()),
        f"Mean_NDCG@{k}":           float(df["NDCG@K"].mean()),
        f"Mean_HitRate@{k}":        float(df["HitRate@K"].mean()),
        "MRR":                      float(df["MRR"].mean()),
        "Coverage":                 len(rec_set) / n_songs,
        f"Mean_PopularityBias@{k}": float(df["PopularityBias"].mean()),
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


# Vectorised GNN-specific multi-K sweep

def evaluate_gnn_k_sweep(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    eval_user_idx: torch.Tensor,
    eval_item_idx: torch.Tensor,
    train_edge_index: torch.Tensor,
    k_list: List[int],
) -> Dict[str, float]:
    """Evaluate GNN embeddings at multiple cut-offs K using a single top-K sort.

    This is the recommended function when you need Recall and NDCG at many
    different K values simultaneously (e.g. a full K-sweep from 5 to 100).
    It performs only one torch.topk call per user rather than re-sorting for
    every K, and keeps all tensors on-device throughout with no Python user loop.

    The approach is:
      1. Compute the full dot-product score matrix for the unique eval users.
      2. Mask training interactions to negative infinity so they cannot appear
         in the top-K, using a vectorised scatter via a global-to-local index map.
      3. Sort once up to max(k_list).
      4. Build the relevance matrix (U_eval, max_K) in a single vectorised pass
         using torch.isin on encoded (user, item) pairs — no Python loop over users.
      5. Compute cumulative DCG and recall with torch.cumsum batched over all users
         simultaneously, then slice at each K.

    This function does not compute Coverage or PopularityBias because those
    require catalogue-level information not needed here. Use evaluate_from_embeddings
    for single-K evaluation that includes those metrics.

    Parameters are:
      user_emb         -- (U, D) float tensor of user node embeddings.
      item_emb         -- (I, D) float tensor of item node embeddings.
      eval_user_idx    -- 1-D tensor of user node indices in the validation/test set.
                          May contain duplicates (one entry per positive interaction).
      eval_item_idx    -- 1-D tensor of the corresponding positive item indices.
                          Paired elementwise with eval_user_idx.
      train_edge_index -- (2, E) tensor with train_edge_index[0] = user indices and
                          train_edge_index[1] = item indices for all training edges.
                          These items are masked to -inf before scoring.
      k_list           -- list of integer cut-off values, e.g. [5, 10, 20, 50].

    Returns a flat dict with keys Recall@K and NDCG@K for each K in k_list,
    averaged over all users that have at least one positive in eval_item_idx.
    """
    device  = user_emb.device
    max_k   = max(k_list)
    I       = item_emb.size(0)

    # Unique eval users and a vectorised global→local index map
    eval_users_unique = torch.unique(eval_user_idx)
    U_eval = eval_users_unique.size(0)

    max_uid = int(max(eval_users_unique.max(), train_edge_index[0].max()).item()) + 1
    g2l = torch.full((max_uid,), -1, dtype=torch.long, device=device)
    g2l[eval_users_unique] = torch.arange(U_eval, device=device)

    # Score matrix (U_eval, I) — full dot-product
    scores = torch.matmul(user_emb[eval_users_unique], item_emb.t())

    # Mask training interactions: vectorised scatter using g2l map
    train_u, train_i = train_edge_index
    local_train_u = g2l[train_u]
    valid = local_train_u >= 0
    if valid.any():
        scores[local_train_u[valid], train_i[valid]] = float("-inf")

    # Single top-max_k sort (U_eval, max_k)
    _, topk_indices = torch.topk(scores, min(max_k, I), dim=-1)
    del scores  # free score matrix ASAP

    # ── Vectorised relevance matrix ───────────────────────────────────────────
    # Encode each (local_user, item) pair as a single int64 so we can use
    # torch.isin for membership testing — no Python loop over users.
    local_gt_u = g2l[eval_user_idx]                                   # (n_pos,)
    gt_enc     = local_gt_u.long() * I + eval_item_idx.long()         # (n_pos,)

    row_exp   = torch.arange(U_eval, device=device).unsqueeze(1).expand_as(topk_indices)
    pred_enc  = row_exp.reshape(-1).long() * I + topk_indices.reshape(-1).long()  # (U*max_k,)

    rel = torch.isin(pred_enc, gt_enc).float().view(U_eval, max_k)    # (U_eval, max_k)

    # n_pos per local user (vectorised scatter_add)
    n_pos = torch.zeros(U_eval, dtype=torch.float32, device=device)
    n_pos.scatter_add_(0, local_gt_u.long(),
                       torch.ones(local_gt_u.size(0), dtype=torch.float32, device=device))
    valid_users  = n_pos > 0                                           # (U_eval,)
    n_pos_safe   = n_pos.clamp(min=1.0).unsqueeze(1)                   # (U_eval, 1)

    # Discount factors: log2(rank+1), rank 1-indexed
    discounts = torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
    )  # (max_k,)

    # Cumulative recall: hits / n_pos  (U_eval, max_k)
    recalls = rel.cumsum(dim=1) / n_pos_safe

    # DCG: (2^rel - 1) / log2(rank+1), cumulative
    gains = (2.0 ** rel) - 1.0
    dcgs  = (gains / discounts).cumsum(dim=1)                          # (U_eval, max_k)

    # Ideal DCG: n_pos ones at the top, zeros elsewhere
    ideal_rel   = (torch.arange(max_k, device=device).unsqueeze(0)
                   < n_pos.long().unsqueeze(1)).float()                # (U_eval, max_k)
    idcgs       = ((2.0 ** ideal_rel - 1.0) / discounts).cumsum(dim=1)

    ndcgs = torch.where(idcgs > 0, dcgs / idcgs, torch.zeros_like(dcgs))

    # Extract metric value at each requested K, averaged only over valid users
    out: Dict[str, float] = {}
    for k in k_list:
        ki = k - 1
        r  = recalls[valid_users, ki]
        n  = ndcgs[valid_users, ki]
        out[f"Recall@{k}"] = float(r.mean()) if r.numel() > 0 else 0.0
        out[f"NDCG@{k}"]   = float(n.mean()) if n.numel() > 0 else 0.0
    return out


def overall_score(
    metrics: Mapping[str, float],
    *,
    w_ndcg:     float = 0.60,
    w_cov:      float = 0.20,
    w_anti_pop: float = 0.20,
    k: int,
) -> float:
    """Weighted composite score combining ranking, coverage, and diversity.

    Default weights produce 0.60 * NDCG@K + 0.20 * Coverage + 0.20 * (1 - PopularityBias).
    The anti-popularity term rewards models that avoid concentrating all recommendations
    on already-popular tracks.

    Accepts both the ``Mean_NDCG@K`` / ``Mean_PopularityBias@K`` keys returned by
    :func:`evaluate_recs` and the bare ``NDCG@K`` / ``PopularityBias@K`` keys
    returned by the vectorised :func:`fast_eval_top_k` path.
    """
    ndcg = metrics.get(f"NDCG@{k}", metrics.get(f"Mean_NDCG@{k}", 0.0))
    pop  = metrics.get(f"PopularityBias@{k}", metrics.get(f"Mean_PopularityBias@{k}", 0.0))
    cov  = metrics.get("Coverage", 0.0)
    return w_ndcg * ndcg + w_cov * cov + w_anti_pop * (1.0 - pop)


def multi_k_evaluation(
    recs_dict_max_k: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Set[int]],
    seen_dict: Mapping[int, Set[int]],
    n_songs: int,
    pop_norm: np.ndarray,
    *,
    # Linear from 5 to 100 (steps of 5), then 'exponential' jumps up to 5000
    ks: Sequence[int] = list(range(5, 105, 5)) + [200, 500, 1000, 2000, 5000],
    model_name: str,
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
    ks_sorted = sorted({int(k) for k in ks})
    rows = []
    for k in ks_sorted:
        m = evaluate_recs(recs_dict_max_k, ground_truth, seen_dict, n_songs, pop_norm, k=k)
        m["Overall_Score"] = overall_score(m, k=k)
        for metric_name, value in m.items():
            # Strip the embedded "@K" suffix (e.g. "Mean_Recall@5" → "Mean_Recall")
            # so the long-format "metric" column is K-agnostic; cutoff lives in "K".
            clean_name = re.sub(r"@\d+$", "", metric_name)
            rows.append({"model": model_name, "K": k,
                         "metric": clean_name, "value": float(value)})
    return pd.DataFrame(rows)
