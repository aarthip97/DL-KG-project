"""Listwise recommendation loss with popularity debiasing and top-K ranking evaluation.

Training uses an Intensity-Weighted Listwise loss with Logit Adjustment. The key ideas are:

  - Embeddings are L2-normalised so the dot product is cosine similarity.
  - A temperature scalar sharpens the Softmax distribution over the full item catalogue.
  - A log-popularity term is subtracted from each logit to penalise recommending
    already-popular tracks, pushing the model toward broader catalogue coverage.
  - Soft target probabilities are built from log1p-transformed raw listen counts so that
    repeat listeners contribute a stronger signal with diminishing returns.

Evaluation follows the full-ranking protocol: for every held-out positive edge the user
is scored against the entire item catalogue, training-seen items are masked, and
Recall@K and NDCG@K are aggregated over all evaluated users.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def compute_log_pop_prior(
    total_track_listens: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the log-popularity prior vector used for logit adjustment.

    Adds Laplace smoothing (each count + 1) so that tracks with zero listens still
    receive a small non-zero probability and log(0) is avoided. Normalises the
    smoothed counts to a probability simplex and returns the element-wise log.

    The result is a 1-D tensor of length num_tracks on the requested device.
    Pre-compute this once before the training loop and pass it to
    debiased_listwise_loss every iteration.
    """
    counts = total_track_listens.float() + 1.0
    probs = counts / counts.sum()
    log_prior = torch.log(probs)
    if device is not None:
        log_prior = log_prior.to(device)
    return log_prior


def debiased_listwise_loss(
    user_embs: torch.Tensor,
    all_track_embs: torch.Tensor,
    user_indices: torch.Tensor,
    raw_counts_matrix: torch.Tensor,
    log_track_pop: torch.Tensor,
    lambda_reg: float = 0.2,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Intensity-Weighted Listwise Loss with Popularity Debiasing (Logit Adjustment).

    Steps performed internally:

      1. L2-normalise both user and track embeddings so the dot product becomes
         cosine similarity, removing magnitude bias from the embedding norms.

      2. Compute a (B, I) logit matrix and divide by the temperature scalar.
         A smaller temperature (e.g. 0.1) sharpens the resulting softmax
         distribution and produces a harder training signal.

      3. Apply logit adjustment: add lambda_reg * log_track_pop to each column.
         Because log_track_pop is negative (log of a probability less than 1),
         popular tracks receive a downward adjustment and less-heard tracks are
         relatively boosted, encouraging coverage diversity.

      4. Build soft target probabilities from the raw listen counts for the batch
         users. Counts are transformed with log1p to compress the range, then
         normalised row-wise so each user row is a proper probability distribution.

      5. Compute the cross-entropy between the adjusted logit distribution and the
         target distribution and return the mean over the user batch.

    Parameters are:
      user_embs         -- (B, D) float tensor, embeddings for the current user batch.
      all_track_embs    -- (I, D) float tensor, embeddings for all catalogue tracks.
      user_indices      -- (B,) long tensor, row indices into raw_counts_matrix.
      raw_counts_matrix -- (U, I) float or int tensor of raw per-user per-track listen
                           counts. May live on CPU; moved to the correct device internally.
      log_track_pop     -- (I,) float tensor from compute_log_pop_prior, on the same
                           device as user_embs.
      lambda_reg        -- scale of the popularity penalty.
      temperature       -- softmax temperature applied to cosine similarities.

    Returns a scalar loss tensor, mean over the user batch.
    """
    if not (0.0 <= lambda_reg <= 1.0):
        raise ValueError(f"lambda_reg must be in [0.0, 1.0], got {lambda_reg!r}")
    u_norm = F.normalize(user_embs, p=2, dim=1)
    i_norm = F.normalize(all_track_embs, p=2, dim=1)
    logits = torch.matmul(u_norm, i_norm.t()) / temperature
    adjusted_logits = logits + lambda_reg * log_track_pop

    # Gather this batch's relevance rows. raw_counts_matrix may be a dense
    # tensor OR a memory-light sparse tensor (CSR/COO) — densify only the small
    # (B, I) slice for the batch so the full (U, I) matrix never has to live as
    # a dense ~16 GB array in RAM.
    if raw_counts_matrix.layout == torch.strided:
        raw_targets = raw_counts_matrix[user_indices]
    else:
        idx_cpu = user_indices.detach().to(raw_counts_matrix.device)
        try:
            raw_targets = raw_counts_matrix.index_select(0, idx_cpu).to_dense()
        except (RuntimeError, NotImplementedError):
            # Some sparse layouts (e.g. CSR) lack index_select on dim 0;
            # fall back to COO which always supports row gathering.
            raw_targets = raw_counts_matrix.to_sparse_coo().index_select(
                0, idx_cpu
            ).to_dense()
    raw_targets = raw_targets.to(logits.device).float()
    weighted_targets = torch.log1p(raw_targets)
    target_probs = weighted_targets / weighted_targets.sum(dim=1, keepdim=True).clamp(min=1e-9)

    return -torch.sum(target_probs * F.log_softmax(adjusted_logits, dim=1), dim=1).mean()


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


__all__ = ("compute_log_pop_prior", "debiased_listwise_loss", "evaluate_top_k")
