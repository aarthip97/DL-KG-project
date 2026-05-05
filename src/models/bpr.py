"""Bayesian Personalized Ranking loss + full-ranking top-K evaluation.

The evaluation protocol follows the professor's m1_2 feedback: for every
held-out (user, track) positive, we score that user against **all** tracks
in the catalogue, mask out tracks the user already interacted with in the
training split, and compute Recall@K and NDCG@K.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
) -> torch.Tensor:
    """``-log σ(<u, i+> - <u, i->)`` averaged over the batch."""
    pos_score = (user_emb * pos_item_emb).sum(dim=-1)
    neg_score = (user_emb * neg_item_emb).sum(dim=-1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def sample_negative_items(
    pos_item_idx: torch.Tensor,
    num_items: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One uniform random negative per positive (no per-user filtering — fast & standard)."""
    return torch.randint(
        0, num_items, pos_item_idx.shape, device=pos_item_idx.device, generator=generator
    )


def _dcg_at_k(rel: torch.Tensor, k: int) -> torch.Tensor:
    rel = rel[:, :k].float()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=rel.device).float())
    return (rel * discounts).sum(dim=-1)


@torch.no_grad()
def evaluate_top_k(
    user_emb: torch.Tensor,             # (U, D)
    item_emb: torch.Tensor,             # (I, D)
    eval_user_idx: torch.Tensor,        # (E,)
    eval_item_idx: torch.Tensor,        # (E,)
    train_edge_index: torch.Tensor,     # (2, T)  — to mask seen items
    *,
    k_list: Iterable[int] = (10, 20),
    batch_users: int = 1024,
) -> dict[str, float]:
    """Return ``{'recall@k': ..., 'ndcg@k': ...}`` over distinct eval users.

    For each unique user in ``eval_user_idx``, scores all items, masks
    items already seen during training, and aggregates the held-out
    positives that fall in the top-K.
    """
    device = user_emb.device
    k_list = sorted(set(int(k) for k in k_list))
    k_max = max(k_list)

    # Build per-user training history (set of item ids) and held-out positives.
    train_by_user: dict[int, list[int]] = {}
    for u, i in train_edge_index.t().tolist():
        train_by_user.setdefault(int(u), []).append(int(i))

    eval_by_user: dict[int, list[int]] = {}
    for u, i in zip(eval_user_idx.tolist(), eval_item_idx.tolist()):
        eval_by_user.setdefault(int(u), []).append(int(i))

    users = torch.tensor(sorted(eval_by_user.keys()), device=device)
    if users.numel() == 0:
        return {f"{m}@{k}": 0.0 for m in ("recall", "ndcg") for k in k_list}

    metrics = {f"recall@{k}": 0.0 for k in k_list}
    metrics |= {f"ndcg@{k}": 0.0 for k in k_list}

    n_users = users.numel()
    for start in range(0, n_users, batch_users):
        chunk = users[start:start + batch_users]
        scores = user_emb[chunk] @ item_emb.t()           # (B, I)

        # Mask training positives so they cannot occupy top-K slots.
        for row, u in enumerate(chunk.tolist()):
            seen = train_by_user.get(int(u))
            if seen:
                scores[row, seen] = float("-inf")

        topk = torch.topk(scores, k=k_max, dim=-1).indices  # (B, k_max)

        # Build held-out positive mask (B, k_max)
        pos_sets = [set(eval_by_user[int(u)]) for u in chunk.tolist()]
        rel = torch.zeros_like(topk, dtype=torch.float32)
        for row, pos in enumerate(pos_sets):
            for col, item in enumerate(topk[row].tolist()):
                if item in pos:
                    rel[row, col] = 1.0

        n_pos = torch.tensor([len(p) for p in pos_sets], device=device, dtype=torch.float32).clamp(min=1)
        for k in k_list:
            hits_k = rel[:, :k].sum(dim=-1)
            recall_k = (hits_k / n_pos).sum().item()
            dcg_k = _dcg_at_k(rel, k)
            # Ideal DCG: top-min(k, n_pos) ones.
            idcg_k = torch.tensor(
                [_dcg_at_k(torch.ones(1, k, device=device), k).item()
                 if int(np.minimum(k, len(p))) >= k
                 else _dcg_at_k(
                     torch.cat([torch.ones(1, int(np.minimum(k, len(p))), device=device),
                                torch.zeros(1, k - int(np.minimum(k, len(p))), device=device)],
                               dim=-1),
                     k).item()
                 for p in pos_sets],
                device=device,
            )
            ndcg_k = (dcg_k / idcg_k.clamp(min=1e-9)).sum().item()
            metrics[f"recall@{k}"] += recall_k
            metrics[f"ndcg@{k}"]   += ndcg_k

    for key in metrics:
        metrics[key] /= n_users
    return metrics


__all__ = ("bpr_loss", "sample_negative_items", "evaluate_top_k")
