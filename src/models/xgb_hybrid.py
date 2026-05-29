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
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.evaluation import multi_k_evaluation
from models.evaluation.metrics import recs_from_embeddings


def _xgb_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_XGB_DEFAULTS: dict = {
    "objective":          "rank:ndcg",
    "eval_metric":        "ndcg@10",
    # Our relevance label is log1p(play_count) -- a *continuous* graded signal.
    # rank:ndcg defaults to exponential gain (2^rel - 1), which newer XGBoost
    # only allows for non-negative *integer* relevance grades. Linear gain
    # (ndcg_exp_gain=False) uses the label value directly, so it accepts our
    # continuous play-count relevance instead of rejecting it.
    "ndcg_exp_gain":      False,
    "tree_method":        "hist",   # works for both CPU and GPU (device= controls which)
    "device":             _xgb_device(),   # "cuda" on T4/GPU, "cpu" otherwise
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
    # Vectorised: map all song_ids at once, then batch-assign valid rows.
    sidx_series = ae_df["song_id"].astype(str).map(song_id_to_sidx)
    valid = sidx_series.notna() & sidx_series.between(0, n_songs - 1)
    dest_idx = sidx_series[valid].astype(int).to_numpy()
    ae_matrix[dest_idx] = ae_df.loc[valid, ae_cols].to_numpy(dtype=np.float32)
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


def build_track_categorical_features(
    track_meta: pd.DataFrame,
    train_df: pd.DataFrame,
    song_id_to_sidx: Mapping[str, int],
    n_songs: int,
    *,
    onehot_cols: Sequence[str] = ("key", "mode", "tempo_class"),
    target_encode_cols: Sequence[str] = ("primary_genre", "artist_id"),
    multihot_col: Optional[str] = "midi_instrument_names",
    top_instruments: int = 20,
    smoothing: float = 20.0,
    te_folds: int = 5,
    te_seed: int = 42,
    target_col: str = "play_count",
) -> Tuple[np.ndarray, List[str]]:
    """Build a per-track ``(n_songs, F)`` categorical/content feature matrix.

    Encoding strategy by cardinality / shape:

    * ``onehot_cols`` (low cardinality, e.g. key/mode/tempo_class) -> one-hot.
    * ``target_encode_cols`` (high cardinality, e.g. genre/artist) -> *out-of-fold
      smoothed mean-target encoding* fit on ``train_df`` interactions only. Each
      category is encoded by its mean ``log1p(play_count)`` -- the graded-relevance
      analogue of Weight of Evidence (WoE needs a binary target; ours is
      continuous). To avoid a track's own interactions leaking into its own
      feature, tracks are partitioned into ``te_folds`` folds and each track's
      category statistic is computed only from tracks in the *other* folds,
      smoothed toward the global mean. This removes self-leakage while keeping a
      single static per-track matrix; fitting on the training split only keeps
      the held-out val/test interactions out of the features entirely.
    * ``multihot_col`` (a list-valued column, e.g. MIDI instruments) -> multi-hot
      over the ``top_instruments`` most frequent values.

    Rows are indexed by ``s_idx``; tracks absent from ``track_meta`` keep zeros
    (target-encoded columns default to the global mean). Returns
    ``(matrix float32, feature_names)``.
    """
    from collections import Counter

    # Only pull the columns we actually need — avoids copying the full 900-col merged df.
    _need = ["song_id"]
    for c in list(onehot_cols) + list(target_encode_cols):
        if c in track_meta.columns:
            _need.append(c)
    if multihot_col and multihot_col in track_meta.columns:
        _need.append(multihot_col)
    meta = track_meta[_need].copy()

    meta["s_idx"] = meta["song_id"].map(song_id_to_sidx)
    meta = meta.dropna(subset=["s_idx"])
    meta["s_idx"] = meta["s_idx"].astype(int)
    meta = meta[(meta["s_idx"] >= 0) & (meta["s_idx"] < n_songs)]
    meta = meta.drop_duplicates("s_idx").set_index("s_idx")
    _rows = meta.index.to_numpy()

    feats: List[np.ndarray] = []
    names: List[str] = []

    # ── one-hot low-cardinality columns ──────────────────────────────────────
    for col in onehot_cols:
        if col not in meta.columns:
            continue
        col_vals = meta[col].astype("string")
        for c in sorted(col_vals.dropna().unique()):
            arr = np.zeros(n_songs, dtype=np.float32)
            arr[_rows] = (col_vals == c).to_numpy(dtype=np.float32)
            names.append(f"{col}={c}")
            feats.append(arr)

    # ── out-of-fold smoothed mean-target encoding (no self-leakage) ──────────
    # Fit on train interactions only. Tracks are split into ``te_folds`` folds;
    # each track's category statistic is computed from tracks in the *other*
    # folds, so a track's own play counts never inform its own feature.
    tr = train_df[["s_idx", target_col]].copy()
    tr["_y"] = np.log1p(tr[target_col].astype(float))
    global_mean = float(tr["_y"].mean()) if len(tr) else 0.0

    if any(c in meta.columns for c in target_encode_cols):
        _te_rng = np.random.default_rng(te_seed)
        _tg = tr.groupby("s_idx")["_y"]
        track_sum = _tg.sum()
        track_cnt = _tg.count()

        # Per-track scaffold: fold assignment + that track's own train totals.
        base = pd.DataFrame({"s_idx": _rows})
        base["fold"] = _te_rng.integers(0, max(1, te_folds), size=len(base))
        base["tsum"] = base["s_idx"].map(track_sum).fillna(0.0).to_numpy()
        base["tcnt"] = base["s_idx"].map(track_cnt).fillna(0.0).to_numpy()

        for col in target_encode_cols:
            if col not in meta.columns:
                continue
            base["_cat"] = base["s_idx"].map(meta[col]).to_numpy()
            cat_tot  = base.groupby("_cat")[["tsum", "tcnt"]].sum()
            cat_fold = base.groupby(["_cat", "fold"])[["tsum", "tcnt"]].sum()
            td = base.merge(
                cat_tot.rename(columns={"tsum": "c_sum", "tcnt": "c_cnt"}),
                left_on="_cat", right_index=True, how="left",
            ).merge(
                cat_fold.rename(columns={"tsum": "f_sum", "tcnt": "f_cnt"}).reset_index(),
                on=["_cat", "fold"], how="left",
            )
            # Out-of-fold totals = category totals minus this track's own fold.
            oof_sum = (td["c_sum"] - td["f_sum"]).to_numpy(dtype=np.float64)
            oof_cnt = (td["c_cnt"] - td["f_cnt"]).to_numpy(dtype=np.float64)
            enc = (oof_sum + smoothing * global_mean) / (oof_cnt + smoothing)
            enc = np.where(np.isfinite(enc), enc, global_mean)   # unknown cat -> global mean
            arr = np.full(n_songs, global_mean, dtype=np.float32)
            arr[td["s_idx"].to_numpy()] = enc.astype(np.float32)
            names.append(f"te_{col}")
            feats.append(arr)

    # ── multi-hot over the top-N most frequent list values (instruments) ─────
    if multihot_col and multihot_col in meta.columns:
        counter: Counter = Counter()
        for lst in meta[multihot_col].dropna():
            if isinstance(lst, (list, np.ndarray)):
                counter.update(str(v) for v in lst)
        top_insts = [inst for inst, _ in counter.most_common(top_instruments)]
        inst_col  = {inst: j for j, inst in enumerate(top_insts)}
        # Single pass over tracks: O(n_songs × avg_instruments_per_track)
        # vs the old O(n_songs × top_instruments) nested loop.
        multihot = np.zeros((n_songs, len(top_insts)), dtype=np.float32)
        for sidx, lst in meta[multihot_col].items():
            if isinstance(lst, (list, np.ndarray)):
                for v in lst:
                    j = inst_col.get(str(v))
                    if j is not None:
                        multihot[sidx, j] = 1.0
        for j, inst in enumerate(top_insts):
            names.append(f"instr={inst}")
            feats.append(multihot[:, j])

    if not feats:
        return np.zeros((n_songs, 0), dtype=np.float32), []
    return np.column_stack(feats).astype(np.float32), names


def _make_feature_matrix(
    user_ids: Sequence[int],
    candidates: Dict[int, List[int]],
    profiles: np.ndarray,
    ae_matrix: np.ndarray,
    interactions_df: pd.DataFrame,
    track_extra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X, y, qid) arrays for XGBoost DMatrix.

    X   : (N, F) float32 — [user_profile | track_ae_embedding | track_extra?]
    y   : (N,)   float32 — log1p(play_count); 0 for unobserved pairs
    qid : (N,)   int64   — u_idx; rows pre-sorted by qid (XGBoost requirement)

    track_extra, when given, is an (n_songs, F_cat) matrix of per-track
    categorical/content features (one-hot + target-encoded + multi-hot) that is
    concatenated on the *track* side, so the ranker also sees genre/artist/
    instrument/key/mode/tempo context. interactions_df must have columns
    u_idx, s_idx, play_count.
    """
    pairs = [
        (u, s)
        for u in sorted(user_ids)
        for s in candidates.get(u, [])
    ]
    if not pairs:
        w = ae_matrix.shape[1] * 2 + (track_extra.shape[1] if track_extra is not None else 0)
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

    blocks = [profiles[vu], ae_matrix[vs]]
    if track_extra is not None:
        blocks.append(track_extra[vs])
    X = np.hstack(blocks)                                            # (N, F)
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
    early_stopping_rounds: Optional[int] = None,
    xgb_val_frac: float = 0.1,
    track_extra: Optional[np.ndarray] = None,
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
    print("[xgb] loading AE embeddings …")
    ae_matrix, emb_dim = _load_ae_matrix(ae_embeddings_path, song_id_to_sidx, n_songs)

    all_train_users = list(train_seen.keys())
    rng = np.random.default_rng(seed)
    n_sample = min(n_xgb_train_users, len(all_train_users))
    xgb_train_users = rng.choice(all_train_users, size=n_sample, replace=False).tolist()

    print(f"[xgb] building user AE profiles for {n_sample:,} train users …")
    profiles = _build_user_profiles(xgb_train_users, train_seen, ae_matrix)

    # ── Stage 2: generate candidates (no masking — seen items needed for labels)
    print(f"[xgb] generating {n_candidates} candidates per train user …")
    train_candidates = _candidates_ae(
        xgb_train_users, profiles, ae_matrix,
        n_candidates=n_candidates, seen_dict=None,
    )

    # ── Stage 3: build feature table & train XGBoost ─────────────────────────
    # When early stopping is requested, hold out a fraction of the LTR training
    # *users* (whole query groups, never split within a user) as a validation
    # set so the booster can stop once val ndcg stops improving.
    _use_es = early_stopping_rounds is not None and n_sample >= 10
    if _use_es:
        n_val_u      = max(1, int(n_sample * xgb_val_frac))
        val_users_es = xgb_train_users[:n_val_u]
        tr_users_es  = xgb_train_users[n_val_u:]
    else:
        tr_users_es, val_users_es = xgb_train_users, []

    print("[xgb] assembling feature table …")
    X_train, y_train, qid_train = _make_feature_matrix(
        tr_users_es, train_candidates, profiles, ae_matrix, train_df,
        track_extra=track_extra,
    )
    gc.collect()

    params = {**_XGB_DEFAULTS, "seed": seed}
    if xgb_params:
        params.update(xgb_params)

    dtrain = xgb.DMatrix(data=X_train, label=y_train, qid=qid_train)
    del X_train, y_train, qid_train
    gc.collect()

    evals = [(dtrain, "train")]
    dval = None
    if val_users_es:
        X_val_es, y_val_es, qid_val_es = _make_feature_matrix(
            val_users_es, train_candidates, profiles, ae_matrix, train_df,
            track_extra=track_extra,
        )
        dval = xgb.DMatrix(data=X_val_es, label=y_val_es, qid=qid_val_es)
        del X_val_es, y_val_es, qid_val_es
        gc.collect()
        evals.append((dval, "val"))

    print(f"[xgb] training XGBoost  (rounds={num_boost_round}, "
          f"early_stopping={'val' if dval is not None else 'off'}) …")
    t_start = time.time()
    _evals_result: dict = {}
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=evals, evals_result=_evals_result,
        early_stopping_rounds=(early_stopping_rounds if dval is not None else None),
        verbose_eval=False,
    )
    t_xgb_elapsed = time.time() - t_start

    # Slice the booster to the best iteration so every downstream predict()
    # (here and in XGBHybridRecommender) uses the early-stopped model.
    if dval is not None and getattr(model, "best_iteration", None) is not None:
        try:
            model = model[: model.best_iteration + 1]
            print(f"[xgb] early-stopped at iteration {model.num_boosted_rounds()}")
        except Exception:  # noqa: BLE001  (older xgboost without slicing)
            pass

    del dtrain, dval
    gc.collect()

    # ── Stage 4: inference on test users ─────────────────────────────────────
    print(f"[xgb] scoring test users ({len(test_users):,}) …")
    recs_dict: Dict[int, List[int]] = {}

    # Build profiles only for test users (most already in train_seen)
    test_profiles = _build_user_profiles(list(test_users), train_seen, ae_matrix)

    for start in tqdm(range(0, len(test_users), infer_batch_users),
                      desc="[xgb] inference batches"):
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
            track_extra=track_extra,
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
    print("[xgb] running multi_k_evaluation …")
    results_df = multi_k_evaluation(
        recs_dict_max_k=recs_dict,
        ground_truth=test_gt,
        seen_dict=train_seen,
        n_songs=n_songs,
        pop_norm=pop_norm,
        ks=list(ks),
        model_name="XGBoost-Hybrid",
    )
    results_df["training_time_seconds"] = t_xgb_elapsed

    # ── Persist ───────────────────────────────────────────────────────────────
    if results_csv is not None:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv, index=False)
        # Per-round metric curve (rank:ndcg eval_metric) for plotting. Save the
        # train curve, plus the val curve when early stopping was used.
        _curve_cols = {}
        for _split in ("train", "val"):
            _sp = _evals_result.get(_split, {})
            if _sp:
                _mname = next(iter(_sp))
                _curve_cols[f"{_split}_{_mname}"] = _sp[_mname]
        if _curve_cols:
            _n_rounds = len(next(iter(_curve_cols.values())))
            pd.DataFrame({"round": range(1, _n_rounds + 1), **_curve_cols}).to_csv(
                results_csv.with_name("xgb_loss_history.csv"), index=False)
            print(f"[xgb] training curve saved -> xgb_loss_history.csv "
                  f"({', '.join(_curve_cols)})")
    if model_cache is not None:
        model_cache.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_cache))
        print(f"[xgb] model saved → {model_cache.name}")

    return results_df, model


__all__ = ("run_xgb_hybrid", "build_track_categorical_features")
