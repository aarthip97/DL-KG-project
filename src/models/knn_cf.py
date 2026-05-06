"""KNN collaborative-filtering baseline with k-sweep + caching.

Algorithm
---------
1. Fit a brute-force cosine k-NN on the L2-normalised train interaction matrix.
2. For each query user (validation ∪ test), retrieve the ``max(K_RANGE)+1``
   nearest neighbours once.
3. For every candidate ``k``, score items by summing the rows of the ``k``
   nearest neighbours, mask items already seen during training, and take the
   top-``TOP_N`` as recommendations.
4. Pick the ``k`` that maximises ``Overall_Score`` on the validation set,
   then evaluate it on the test set.

Results are persisted to CSV so subsequent notebook runs can skip the sweep.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from models.evaluation import evaluate_recs, overall_score


def _build_recs(
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
) -> Tuple[pd.DataFrame, Dict[str, float], int, np.ndarray, Dict[int, int]]:
    """Run (or load cached) KNN-CF k-sweep and final test evaluation.

    Returns
    -------
    val_results_df : pd.DataFrame indexed by k
        Per-k validation metrics (incl. ``Overall_Score``).
    test_metrics : dict
        Test-set metrics for the selected ``best_k`` (incl. ``Overall_Score``).
    best_k : int
    all_nbrs : np.ndarray, shape (len(val∪test users), max(k_range)+1)
        Pre-computed user→user nearest-neighbour indices (cosine, on
        ``train_matrix_norm``). Reused by downstream ``KNNRecommender``.
    qrow : dict[u_idx, row_in_all_nbrs]
        Lookup that maps a user index to its row inside ``all_nbrs``.
    """
    # Default sidecar cache for the (all_nbrs, all_query) tuple
    if nbrs_cache is None:
        nbrs_cache = val_csv.with_name(val_csv.stem + "_nbrs.npz")

    if (val_csv.exists() and test_csv.exists() and nbrs_cache.exists()
            and not force_rebuild):
        val_results_df = pd.read_csv(val_csv, index_col=0)
        test_summary   = pd.read_csv(test_csv).iloc[0].to_dict()
        best_k         = int(test_summary["best_k"])
        keys = ("Recall@K", "NDCG@K", "HitRate@K", "MRR",
                "Coverage", "PopularityBias", "Overall_Score")
        test_metrics = {k: float(test_summary[k]) for k in keys}
        with np.load(nbrs_cache) as z:
            all_nbrs  = z["all_nbrs"]
            all_query = z["all_query"]
        qrow = {int(u): i for i, u in enumerate(all_query.tolist())}
        print(f"[SKIP] KNN results loaded. best_k={best_k}  "
              f"test Recall@{top_n}={test_metrics['Recall@K']:.4f}  "
              f"(neighbour cache: {nbrs_cache.name})")
        return val_results_df, test_metrics, best_k, all_nbrs, qrow

    k_list = list(k_range)
    max_k  = max(k_list)
    knn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine",
                           algorithm="brute", n_jobs=-1)
    knn.fit(train_matrix_norm)
    all_query = sorted(set(val_users) | set(test_users))
    _, all_nbrs = knn.kneighbors(train_matrix_norm[all_query],
                                 n_neighbors=max_k + 1)
    qrow = {u: i for i, u in enumerate(all_query)}

    val_results = []
    for k in tqdm(k_list, desc="k-sweep (val)"):
        recs = _build_recs(k, val_users,
                           all_nbrs=all_nbrs, qrow=qrow,
                           train_matrix_norm=train_matrix_norm,
                           train_seen=train_seen, top_n=top_n)
        m = evaluate_recs(recs, val_gt, train_seen, n_songs, pop_norm, k=top_n)
        m["k"] = k
        m["Overall_Score"] = overall_score(m)
        val_results.append(m)
    val_results_df = pd.DataFrame(val_results).set_index("k")
    best_k = int(val_results_df["Overall_Score"].idxmax())

    test_recs = _build_recs(best_k, test_users,
                            all_nbrs=all_nbrs, qrow=qrow,
                            train_matrix_norm=train_matrix_norm,
                            train_seen=train_seen, top_n=top_n)
    test_metrics = evaluate_recs(test_recs, test_gt, train_seen,
                                 n_songs, pop_norm, k=top_n)
    test_metrics["Overall_Score"] = overall_score(test_metrics)

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
    np.savez_compressed(nbrs_cache,
                        all_nbrs=all_nbrs,
                        all_query=np.asarray(all_query, dtype=np.int64))
    print(f"best_k={best_k}  test Recall@{top_n}={test_metrics['Recall@K']:.4f}  "
          f"(neighbour cache → {nbrs_cache.name})")
    return val_results_df, test_metrics, best_k, all_nbrs, qrow
