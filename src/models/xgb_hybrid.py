"""Two-stage XGBoost LTR hybrid baseline.

Architecture
------------
Stage 1 — Candidate generation : mean-AE-profile × track-AE dot-product (top-N)
Stage 2 — Feature assembly     : flat (user, track) table; 128 user + 128 track AE dims
Stage 3 — XGBoost LTR          : rank:ndcg; re-sorts each candidate list non-linearly
Stage 4 — Evaluation           : recs_dict → multi_k_evaluation K-sweep

Memory guidance
---------------
Feature-table size = n_xgb_train_users × n_candidates × 2 × emb_dim × 4 bytes.
With defaults (5 000 users, 200 candidates, 128-dim AE) that is ~1 GB — comfortable
on a 16 GB machine.  Raise n_xgb_train_users / n_candidates for richer training data
if your machine has more RAM.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.evaluation import multi_k_evaluation
from models.evaluation.metrics import recs_from_embeddings


_XGB_DEFAULTS: dict = {
    "objective":          "rank:ndcg",
    "eval_metric":        "ndcg@10",
    "tree_method":        "hist",
    "learning_rate":      0.1,
    "max_depth":          6,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ae_matrix(
    ae_path: Path,
    song_id_to_sidx: Mapping[str, int],
    n_songs: int,
) -> Tuple[np.ndarray, int]:
    """Load AE embeddings parquet → (n_songs, emb_dim) float32 matrix indexed by s_idx.

    Tracks absent from the parquet keep zero embeddings.
    """
    ae_df = pd.read_parquet(ae_path)
    ae_cols = [c for c in ae_df.columns if c.startswith("ae_")]
    emb_dim = len(ae_cols)
    ae_matrix = np.zeros((n_songs, emb_dim), dtype=np.float32)
    for _, row in ae_df.iterrows():
        sidx = song_id_to_sidx.get(str(row["song_id"]))
        if sidx is not None and 0 <= int(sidx) < n_songs:
            ae_matrix[int(sidx)] = row[ae_cols].to_numpy(dtype=np.float32)
    return ae_matrix, emb_dim


def _build_user_profiles(
    user_ids: Sequence[int],
    train_seen: Mapping[int, Set[int]],
    ae_matrix: np.ndarray,
) -> np.ndarray:
    """Mean-pool AE track embeddings into a user-profile matrix.

    Returns an array of shape (max(user_ids)+1, emb_dim); rows for users not
    in user_ids are zero.
    """
    emb_dim = ae_matrix.shape[1]
    n_items = ae_matrix.shape[0]
    profiles = np.zeros((max(user_ids) + 1, emb_dim), dtype=np.float32)
    for u in user_ids:
        items = train_seen.get(u)
        if items:
            valid = [s for s in items if 0 <= s < n_items]
            if valid:
                profiles[u] = ae_matrix[valid].mean(axis=0)
    return profiles


def _candidates_ae(
    user_ids: Sequence[int],
    profiles: np.ndarray,
    ae_matrix: np.ndarray,
    n_candidates: int,
    seen_dict: Optional[Mapping[int, Set[int]]] = None,
    batch_size: int = 1024,
) -> Dict[int, List[int]]:
    """Top-n_candidates per user via user-profile × item-AE dot-product."""
    k = min(n_candidates, ae_matrix.shape[0])
    return recs_from_embeddings(
        torch.from_numpy(profiles),
        torch.from_numpy(ae_matrix),
        list(user_ids),
        seen_dict=seen_dict,
        top_n=k,
        batch_size=batch_size,
    )


def _make_feature_matrix(
    user_ids: Sequence[int],
    candidates: Dict[int, List[int]],
    profiles: np.ndarray,
    ae_matrix: np.ndarray,
    interactions_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X, y, qid) arrays for XGBoost DMatrix.

    X   : (N, 2*emb_dim) float32 — [user_profile | track_ae_embedding]
    y   : (N,)            float32 — log1p(play_count); 0 for unobserved pairs
    qid : (N,)            int64   — u_idx; rows pre-sorted by qid (XGBoost requirement)

    interactions_df must have columns u_idx, s_idx, play_count.
    """
    pairs = [
        (u, s)
        for u in sorted(user_ids)
        for s in candidates.get(u, [])
    ]
    if not pairs:
        w = ae_matrix.shape[1] * 2
        return (
            np.empty((0, w), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    pairs_df = pd.DataFrame(pairs, columns=["u_idx", "s_idx"])
    merged = pairs_df.merge(
        interactions_df[["u_idx", "s_idx", "play_count"]],
        on=["u_idx", "s_idx"],
        how="left",
    ).fillna({"play_count": 0.0})

    u_arr = merged["u_idx"].to_numpy(dtype=np.int64)
    s_arr = merged["s_idx"].to_numpy(dtype=np.int64)

    # Clip to valid ranges (safety guard for stale indices)
    vu = np.clip(u_arr, 0, profiles.shape[0] - 1)
    vs = np.clip(s_arr, 0, ae_matrix.shape[0] - 1)

    X = np.hstack([profiles[vu], ae_matrix[vs]])                     # (N, 2*D)
    y = np.log1p(merged["play_count"].to_numpy(dtype=np.float32))
    return X, y, u_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_xgb_hybrid(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ae_embeddings_path: Path | str,
    song_id_to_sidx: Mapping[str, int],
    train_seen: Mapping[int, Set[int]],
    val_users: Sequence[int],
    test_users: Sequence[int],
    val_gt: Mapping[int, Set[int]],
    test_gt: Mapping[int, Set[int]],
    pop_norm: np.ndarray,
    n_songs: int,
    n_users: int,
    top_n: int,
    n_candidates: int = 200,
    n_xgb_train_users: int = 5_000,
    xgb_params: Optional[dict] = None,
    num_boost_round: int = 100,
    ks: Sequence[int] = (5, 10, 20, 50, 100),
    results_csv: Optional[Path] = None,
    model_cache: Optional[Path] = None,
    force_rebuild: bool = False,
    seed: int = 42,
    infer_batch_users: int = 1_000,
) -> Tuple[pd.DataFrame, "xgboost.Booster"]:  # type: ignore[name-defined]
    """Train and evaluate the XGBoost LTR hybrid baseline.

    Parameters
    ----------
    train_df, val_df, test_df:
        Split DataFrames with columns u_idx, s_idx, play_count.
    ae_embeddings_path:
        Path to ae_embeddings.parquet (song_id + ae_0…ae_127 columns).
    song_id_to_sidx:
        Mapping from string song_id to integer s_idx.  Build with:
        ``dict(zip(train_df.song_id, train_df.s_idx))`` after deduplication.
    train_seen:
        ``{u_idx: {s_idx, …}}`` — training interactions per user.
    val_users, test_users:
        User indices for validation / test evaluation.
    val_gt, test_gt:
        Ground-truth item sets for validation / test users.
    pop_norm:
        Normalised popularity array, one entry per song.
    n_songs, n_users:
        Catalogue and user counts from the split metadata.
    top_n:
        Maximum recommendation list length (should be ≥ max(ks)).
    n_candidates:
        Candidate pool size per user (Stage 1).  Must be ≥ max(ks).
        Increase for better recall at the cost of higher memory usage.
    n_xgb_train_users:
        Random subsample of training users used to build the XGBoost
        training set.  Larger values improve XGBoost fit but use more RAM:
        ``n_xgb_train_users × n_candidates × 256 × 4 bytes``.
    xgb_params:
        Override dictionary merged into the default XGBoost params.
    num_boost_round:
        Number of boosting iterations.
    ks:
        Cut-off values passed to multi_k_evaluation.
    results_csv:
        If provided, the multi_k_evaluation DataFrame is saved here (and
        loaded on subsequent calls when force_rebuild=False).
    model_cache:
        If provided, the trained XGBoost model is saved here (``*.ubj``).
    force_rebuild:
        Re-train even if cached results exist.
    seed:
        Random seed for user subsampling and XGBoost.
    infer_batch_users:
        Users processed per inference batch to control peak memory.

    Returns
    -------
    results_df : long-format DataFrame from multi_k_evaluation.
    model      : trained xgboost.Booster.
    """
    import xgboost as xgb

    ae_embeddings_path = Path(ae_embeddings_path)
    n_candidates = max(n_candidates, top_n, max(ks))

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if (results_csv is not None and model_cache is not None
            and results_csv.exists() and model_cache.exists()
            and not force_rebuild):
        results_df = pd.read_csv(results_csv)
        model = xgb.Booster()
        model.load_model(str(model_cache))
        print(f"[SKIP] XGBoost results loaded from {results_csv.name}")
        return results_df, model

    # ── Stage 1: load AE embeddings & build user profiles ────────────────────
    print("[XGB] loading AE embeddings …")
    ae_matrix, emb_dim = _load_ae_matrix(ae_embeddings_path, song_id_to_sidx, n_songs)

    all_train_users = list(train_seen.keys())
    rng = np.random.default_rng(seed)
    n_sample = min(n_xgb_train_users, len(all_train_users))
    xgb_train_users = rng.choice(all_train_users, size=n_sample, replace=False).tolist()

    print(f"[XGB] building user AE profiles for {n_sample:,} train users …")
    profiles = _build_user_profiles(xgb_train_users, train_seen, ae_matrix)

    # ── Stage 2: generate candidates (no masking — seen items needed for labels)
    print(f"[XGB] generating {n_candidates} candidates per train user …")
    train_candidates = _candidates_ae(
        xgb_train_users, profiles, ae_matrix,
        n_candidates=n_candidates, seen_dict=None,
    )

    # ── Stage 3: build feature table & train XGBoost ─────────────────────────
    print("[XGB] assembling feature table …")
    X_train, y_train, qid_train = _make_feature_matrix(
        xgb_train_users, train_candidates, profiles, ae_matrix, train_df,
    )
    gc.collect()

    params = {**_XGB_DEFAULTS, "seed": seed}
    if xgb_params:
        params.update(xgb_params)

    print(f"[XGB] training XGBoost  (rows={len(X_train):,}, "
          f"features={X_train.shape[1]}, rounds={num_boost_round}) …")
    dtrain = xgb.DMatrix(data=X_train, label=y_train, qid=qid_train)
    del X_train, y_train, qid_train
    gc.collect()

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                      verbose_eval=False)
    del dtrain
    gc.collect()

    # ── Stage 4: inference on test users ─────────────────────────────────────
    print(f"[XGB] scoring test users ({len(test_users):,}) …")
    recs_dict: Dict[int, List[int]] = {}

    # Build profiles only for test users (most already in train_seen)
    test_profiles = _build_user_profiles(list(test_users), train_seen, ae_matrix)

    for start in tqdm(range(0, len(test_users), infer_batch_users),
                      desc="[XGB] inference batches"):
        batch = list(test_users[start : start + infer_batch_users])

        # Candidates with seen items masked
        cands = _candidates_ae(
            batch, test_profiles, ae_matrix,
            n_candidates=n_candidates, seen_dict=train_seen,
        )

        # Build inference feature table (no labels needed; pass empty df)
        X_inf, _, qid_inf = _make_feature_matrix(
            batch, cands, test_profiles, ae_matrix,
            pd.DataFrame(columns=["u_idx", "s_idx", "play_count"]),
        )
        if X_inf.shape[0] == 0:
            continue

        dinf = xgb.DMatrix(data=X_inf)
        scores = model.predict(dinf)
        del X_inf, dinf

        # Reconstruct per-user ranked lists from flat scores
        pos = 0
        for u in sorted(batch):
            n = len(cands.get(u, []))
            if n == 0:
                recs_dict[u] = []
                continue
            u_scores = scores[pos : pos + n]
            u_cands = np.asarray(cands[u])
            order = np.argsort(u_scores)[::-1]
            recs_dict[u] = u_cands[order[:top_n]].tolist()
            pos += n

    gc.collect()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[XGB] running multi_k_evaluation …")
    results_df = multi_k_evaluation(
        recs_dict_max_k=recs_dict,
        ground_truth=test_gt,
        seen_dict=train_seen,
        n_songs=n_songs,
        pop_norm=pop_norm,
        ks=list(ks),
        model_name="XGBoost-Hybrid",
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    if results_csv is not None:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv, index=False)
    if model_cache is not None:
        model_cache.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_cache))
        print(f"[XGB] model saved → {model_cache.name}")

    return results_df, model


__all__ = ("run_xgb_hybrid",)
