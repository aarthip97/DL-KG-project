"""KNN collaborative-filtering baseline with k-sweep + caching.

Algorithm
---------
1. Convert the L2-normalised train interaction matrix to a PyTorch tensor
   and find the max(K_RANGE)+1 nearest neighbours for each query user via
   batched matrix multiplication (cosine similarity = dot product after L2
   normalisation).  This replaces sklearn's NearestNeighbors and runs on GPU
   when available, with automatic CPU fallback.
2. For every candidate ``k``, score items by summing the k nearest-neighbour
   rows using batched tensor operations, mask training-seen items, and return
   the top-``TOP_N`` as recommendations.
3. Pick the ``k`` that maximises ``Overall_Score`` on the validation set,
   then evaluate on the test set.

Results are persisted to CSV so subsequent notebook runs can skip the sweep.

Performance notes
-----------------
On CPU, the PyTorch path is still faster than sklearn because:
  - Batched matmul uses multi-threaded BLAS (MKL/OpenBLAS) end-to-end.
  - torch.topk is faster than np.argpartition + argsort for large tensors.
  - Scoring via gather+cumsum avoids Python object overhead per user.
On GPU the neighbour-finding step is 10-50x faster than CPU sklearn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.evaluation import evaluate_recs, overall_score


# ---------------------------------------------------------------------------
# Torch-based neighbour finding
# ---------------------------------------------------------------------------

def _matrix_to_tensor(
    train_matrix_norm,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, str]:
    """Convert a scipy sparse or numpy dense matrix to a dense tensor.

    Always builds the dense array in CPU float32 first (for arithmetic
    stability), converts to ``dtype`` before the device transfer, and
    returns ``(tensor, effective_device)``.  When a CUDA device is
    requested but the free VRAM is less than 120 % of the matrix size,
    the function automatically falls back to CPU and prints a warning —
    the caller receives the CPU tensor and the corrected device string so
    the rest of the pipeline runs on CPU without crashing.

    ``dtype=torch.float16`` halves VRAM (≈ 4 GB for a 285K × 7K matrix)
    at the cost of slightly lower cosine-similarity precision; the impact
    on recommendation quality is negligible for KNN-CF.
    """
    import warnings
    try:
        import scipy.sparse as sp
        if sp.issparse(train_matrix_norm):
            arr = train_matrix_norm.toarray().astype(np.float32)
        else:
            arr = np.asarray(train_matrix_norm, dtype=np.float32)
    except ImportError:
        arr = np.asarray(train_matrix_norm, dtype=np.float32)

    # Size after dtype conversion
    elem_bytes = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 4
    size_gb = arr.size * elem_bytes / 1e9

    if device.startswith("cuda"):
        try:
            free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
            if size_gb > free_gb * 0.80:
                warnings.warn(
                    f"[KNN] Matrix is {size_gb:.1f} GB but only "
                    f"{free_gb:.1f} GB VRAM free — falling back to CPU. "
                    f"Pass device='cpu' or dtype=torch.float16 to suppress.",
                    ResourceWarning,
                    stacklevel=3,
                )
                device = "cpu"
        except Exception:
            pass  # mem_get_info unavailable; attempt GPU transfer anyway

    t = torch.from_numpy(arr)
    if dtype != torch.float32:
        t = t.to(dtype)
    return t.to(device), device


def _find_neighbors_torch(
    query_tensor: torch.Tensor,
    train_tensor: torch.Tensor,
    n_neighbors: int,
    batch_size: int = 512,
) -> torch.Tensor:
    """Return the n_neighbors nearest-neighbour indices for each query row.

    Uses batched cosine similarity (dot product on L2-normalised rows).
    Both tensors must be dense float32 and on the same device.

    Batching over query rows keeps the intermediate similarity matrix at
    ``batch_size × n_train × 4 B`` (≈ 1.4 MB at batch_size=512, n_train=285K)
    rather than ``n_query × n_train`` all at once.

    Returns an (Q, n_neighbors) LongTensor of neighbour indices into train_tensor.
    """
    device = query_tensor.device
    Q = query_tensor.size(0)
    k = min(n_neighbors, train_tensor.size(0))
    nbrs = torch.empty(Q, k, dtype=torch.long, device=device)

    for start in range(0, Q, batch_size):
        end     = min(start + batch_size, Q)
        q_batch = query_tensor[start:end]           # (B, I)
        sim     = q_batch @ train_tensor.t()        # (B, U)
        _, idx  = torch.topk(sim, k, dim=-1)
        nbrs[start:end] = idx

    return nbrs  # (Q, n_neighbors)


# ---------------------------------------------------------------------------
# Torch-based recommendation scoring
# ---------------------------------------------------------------------------

def _build_recs_torch(
    k: int,
    user_list: Sequence[int],
    *,
    nbrs_tensor: torch.Tensor,
    qrow: Mapping[int, int],
    train_tensor: torch.Tensor,
    train_seen: Mapping[int, Set[int]],
    top_n: int,
    score_batch_size: int = 128,
) -> Dict[int, List[int]]:
    """Score items for a list of users using their k nearest neighbours.

    ``train_tensor`` is the same dense GPU tensor used for neighbour finding.
    The gather creates an intermediate tensor of shape (batch, k, I) which is
    reduced to (batch, I) immediately; peak extra VRAM is
    ``score_batch_size × k × n_items × 4 B`` (≈ 360 MB at defaults).

    Parameters are:
      k                -- number of nearest neighbours to aggregate.
      user_list        -- global user indices to generate recommendations for.
      nbrs_tensor      -- (Q, max_k+1) LongTensor from _find_neighbors_torch.
      qrow             -- mapping from global user id to row in nbrs_tensor.
      train_tensor     -- (U, I) dense float32 GPU tensor.
      train_seen       -- training interactions per user (for masking).
      top_n            -- number of items to return per user.
      score_batch_size -- users processed per scoring chunk.

    Returns a dict mapping global user id to an ordered list of item ids.
    """
    device = nbrs_tensor.device
    users  = list(user_list)
    recs: Dict[int, List[int]] = {}

    for start in range(0, len(users), score_batch_size):
        batch_users = users[start : start + score_batch_size]

        local_rows = torch.tensor([qrow[u] for u in batch_users],
                                  dtype=torch.long, device=device)

        # skip col-0 (self) → take cols 1…k
        nbr_k  = nbrs_tensor[local_rows, 1 : k + 1]        # (B, k)
        scores = train_tensor[nbr_k].sum(dim=1)             # (B, I)

        for bi, u in enumerate(batch_users):
            seen = train_seen.get(u)
            if seen:
                seen_t = torch.tensor(list(seen), dtype=torch.long, device=device)
                scores[bi].scatter_(0, seen_t, 0.0)

        _, top_idx = torch.topk(scores, min(top_n, scores.size(1)), dim=-1)
        for bi, u in enumerate(batch_users):
            recs[u] = top_idx[bi].tolist()

    return recs


# ---------------------------------------------------------------------------
# Legacy numpy fallback (kept for CPU environments without PyTorch BLAS)
# ---------------------------------------------------------------------------

def _build_recs_numpy(
    k: int,
    user_list: Sequence[int],
    *,
    all_nbrs: np.ndarray,
    qrow: Mapping[int, int],
    train_matrix_norm,
    train_seen: Mapping[int, Set[int]],
    top_n: int,
) -> Dict[int, list]:
    out: Dict[int, list] = {}
    for u in user_list:
        nbrs = [x for x in all_nbrs[qrow[u]] if x != u][:k]
        if not nbrs:
            out[u] = []
            continue
        sc = np.asarray(train_matrix_norm[nbrs].sum(axis=0)).ravel()
        for s in train_seen.get(u, set()):
            sc[s] = 0.0
        top = np.argpartition(sc, -top_n)[-top_n:]
        out[u] = top[np.argsort(sc[top])[::-1]].tolist()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_knn_sweep(
    *,
    train_matrix_norm,
    train_seen: Mapping[int, Set[int]],
    val_users: Sequence[int],
    test_users: Sequence[int],
    val_gt: Mapping[int, Set[int]],
    test_gt: Mapping[int, Set[int]],
    pop_norm: np.ndarray,
    n_songs: int,
    n_users: int,
    k_range: Iterable[int],
    top_n: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_csv: Path,
    test_csv: Path,
    nbrs_cache: Path | None = None,
    force_rebuild: bool = False,
    device: Optional[str] = None,
    nbr_batch_size: int = 512,
    score_batch_size: int = 128,
    matrix_dtype: torch.dtype = torch.float32,
) -> Tuple[pd.DataFrame, Dict[str, float], int, torch.Tensor, Dict[int, int]]:
    """Run (or load cached) KNN-CF k-sweep and final test evaluation.

    Uses PyTorch for all heavy computation so the neighbour-finding and item
    scoring steps run on GPU when available, and on CPU using multi-threaded
    BLAS otherwise (still faster than sklearn on CPU for large matrices).

    Parameters are:
      train_matrix_norm -- (U, I) L2-normalised scipy sparse or numpy array.
      device            -- 'cuda', 'cpu', or None (auto-detect).
      nbr_batch_size    -- query rows per chunk during neighbour finding.
      score_batch_size  -- users per chunk during item scoring.
      matrix_dtype      -- dtype for the dense GPU tensor. ``torch.float16``
                           halves VRAM (~4 GB for a 285K×7K matrix) at
                           negligible quality cost. Default: ``torch.float32``.
      (all other parameters match the previous signature)

    Returns
    -------
    val_results_df  -- pd.DataFrame indexed by k.
    test_metrics    -- dict of test-set metrics for the selected best_k.
    best_k          -- int.
    nbrs_tensor     -- (Q, max_k+1) LongTensor of neighbour indices on CPU.
    qrow            -- dict mapping global user id to row in nbrs_tensor.
    """
    if nbrs_cache is None:
        nbrs_cache = val_csv.with_name(val_csv.stem + "_nbrs.pt")

    # ── Cache hit ─────────────────────────────────────────────────────────────
    # Accept both the old .npz format and the new .pt format for backward compat.
    _old_cache = val_csv.with_name(val_csv.stem + "_nbrs.npz")
    if (val_csv.exists() and test_csv.exists()
            and (nbrs_cache.exists() or _old_cache.exists())
            and not force_rebuild):
        val_results_df = pd.read_csv(val_csv, index_col=0)
        test_summary   = pd.read_csv(test_csv).iloc[0].to_dict()
        best_k         = int(test_summary["best_k"])
        # Build test_metrics from whatever keys are actually in the CSV
        test_metrics = {
            k: float(v) for k, v in test_summary.items()
            if k not in ("best_k", "n_users", "n_songs",
                         "train_interactions", "val_interactions", "test_interactions")
        }
        # Load neighbour table — support both formats
        cache_path = nbrs_cache if nbrs_cache.exists() else _old_cache
        if cache_path.suffix == ".pt":
            saved      = torch.load(cache_path, map_location="cpu", weights_only=True)
            nbrs_t     = saved["nbrs_tensor"]
            all_query  = saved["all_query"].tolist()
        else:
            with np.load(cache_path) as z:
                nbrs_t    = torch.from_numpy(z["all_nbrs"])
                all_query = z["all_query"].tolist()
        qrow = {int(u): i for i, u in enumerate(all_query)}
        print(f"[SKIP] KNN results loaded. best_k={best_k}  "
              f"(cache: {cache_path.name})")
        return val_results_df, test_metrics, best_k, nbrs_t, qrow

    # ── Build neighbour table ─────────────────────────────────────────────────
    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = "float16" if matrix_dtype == torch.float16 else "float32"
    print(f"[KNN] building neighbours on {_device} (matrix dtype={dtype_name})")

    # _matrix_to_tensor checks free VRAM and falls back to CPU automatically
    # if the dense matrix would exceed 80% of available VRAM.  It returns the
    # effective device so downstream ops stay consistent.
    print(f"[KNN] loading matrix to {_device} (dense {dtype_name}) …")
    train_t, _device = _matrix_to_tensor(train_matrix_norm, _device, dtype=matrix_dtype)
    print(f"[KNN] train tensor: {tuple(train_t.shape)}  "
          f"({train_t.element_size() * train_t.nelement() / 1e9:.2f} GB)  "
          f"device={train_t.device}")

    all_query = sorted(set(val_users) | set(test_users))
    query_t   = train_t[torch.tensor(all_query, dtype=torch.long, device=_device)]

    k_list = list(k_range)
    max_k  = max(k_list)

    nbrs_t_dev = _find_neighbors_torch(
        query_t, train_t, n_neighbors=max_k + 1,
        batch_size=nbr_batch_size,
    )
    nbrs_t = nbrs_t_dev.cpu()   # keep CPU copy; nbrs_t_dev stays on GPU for scoring
    qrow   = {u: i for i, u in enumerate(all_query)}

    # ── k-sweep on validation set ─────────────────────────────────────────────
    val_results = []
    for k in tqdm(k_list, desc=f"k-sweep (val) [{_device}]"):
        recs = _build_recs_torch(
            k, val_users,
            nbrs_tensor=nbrs_t_dev, qrow=qrow,
            train_tensor=train_t, train_seen=train_seen,
            top_n=top_n, score_batch_size=score_batch_size,
        )
        m = evaluate_recs(recs, val_gt, train_seen, n_songs, pop_norm, k=top_n)
        m["k"] = k
        m["Overall_Score"] = overall_score(m, k=top_n)
        val_results.append(m)

    val_results_df = pd.DataFrame(val_results).set_index("k")
    best_k = int(val_results_df["Overall_Score"].idxmax())

    # ── Final test evaluation at best_k ──────────────────────────────────────
    test_recs    = _build_recs_torch(
        best_k, test_users,
        nbrs_tensor=nbrs_t_dev, qrow=qrow,
        train_tensor=train_t, train_seen=train_seen,
        top_n=top_n, score_batch_size=score_batch_size,
    )
    test_metrics = evaluate_recs(test_recs, test_gt, train_seen,
                                 n_songs, pop_norm, k=top_n)
    test_metrics["Overall_Score"] = overall_score(test_metrics, k=top_n)

    # ── Persist ──────────────────────────────────────────────────────────────
    val_results_df.to_csv(val_csv)
    pd.DataFrame([{
        "best_k": best_k,
        **test_metrics,
        "n_users": n_users,
        "n_songs": n_songs,
        "train_interactions": len(train_df),
        "val_interactions":   len(val_df),
        "test_interactions":  len(test_df),
    }]).to_csv(test_csv, index=False)
    torch.save({"nbrs_tensor": nbrs_t,
                "all_query":   torch.tensor(all_query, dtype=torch.long)},
               nbrs_cache)

    print(f"best_k={best_k}  (cache → {nbrs_cache.name})")
    return val_results_df, test_metrics, best_k, nbrs_t, qrow

