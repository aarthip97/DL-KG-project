"""HGT ablation harnesses for the pipeline notebook (§8, §9).

Three ablations, lifted verbatim from the notebook so the cells stay short:

* :func:`run_phase1_ablation` — short one-factor-at-a-time sweep over capacity /
  temperature / novelty, scored by Overall_Score@10, emitting the winning config.
* :func:`run_init_ablation` — RotatE vs random node initialisation (same topology,
  hyper-params and seed) to isolate the value of the learned KGE structure.
* :func:`run_direction_ablation` — directed vs undirected forward of the SAME
  trained weights (no retraining), quantifying the message-passing-direction gap.

All take explicit inputs (no notebook globals); ``train_fn`` is ``train_hgt``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .checkpoints import cache_pickle, load_pickle

_INT_KEYS = ("hidden_channels", "out_channels", "num_heads", "num_layers")


def default_ablation_axes() -> dict:
    """The one-factor-at-a-time sweep used in §8b (each axis perturbs one field)."""
    return {
        "capacity": [{"num_layers": nl, "num_heads": nh}
                     for nl in (2, 3) for nh in (2, 4, 8)],
        "temperature": [{"temperature": t} for t in (0.05, 0.1, 0.2)],
        "lambda_reg": [{"lambda_reg": lam} for lam in (0.0, 0.1, 0.2, 0.5)],
    }


def _overall_at_k(val_metrics: Mapping[str, float], k: int) -> float:
    """Overall_Score@k from a TrainResult.best_val dict (the §8 criterion)."""
    from ..evaluation.metrics import overall_score
    return overall_score(
        {f"NDCG@{k}": val_metrics.get(f"ndcg@{k}", 0.0),
         f"PopularityBias@{k}": val_metrics.get(f"pop_bias@{k}", 0.0),
         "Coverage": val_metrics.get(f"coverage@{k}", 0.0)}, k=k)


def run_phase1_ablation(
    train_fn: Callable,
    *,
    data,
    train_inputs: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    axes: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    epochs: int = 50,
    primary_k: int = 10,
    eval_every: int = 10,
    out_csv: Optional[Path] = None,
    out_json: Optional[Path] = None,
    int_keys: Sequence[str] = _INT_KEYS,
    progress: bool = True,
    per_run_verbose: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Phase-1 sweep: short runs scored by Overall_Score@``primary_k``.

    Each candidate trains ``train_fn(data, **train_inputs, epochs=epochs, **cfg)``
    for ``epochs`` epochs (no early stopping) on the shared stratified split
    (``val_gt``/``test_gt`` must be inside ``train_inputs``) → leak-free and
    comparable. The winner seeds the long Phase-2 run.

    Candidates run **sequentially** (each ``train_fn`` already saturates the GPU
    with full-graph matmuls). ``progress=True`` shows one tqdm bar over all
    candidates with the live current/best Overall_Score; ``per_run_verbose=True``
    additionally shows each run's own epoch bar.

    Returns:
        ``(abl_df, best_params)`` — the full sorted sweep table and the winning
        tunable arch/loss config (arch dims cast back to ``int``). Persisted to
        ``out_csv`` / ``out_json`` when given.
    """
    axes = dict(axes or default_ablation_axes())
    score_col = f"overall@{primary_k}"
    candidates = [(ax, ov) for ax, ovs in axes.items() for ov in ovs]
    rows: list[dict] = []

    _bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(total=len(candidates), desc="Phase-1 sweep", unit="run")
        except Exception:       # noqa: BLE001
            _bar = None
    _write = _bar.write if _bar is not None else print

    best_so_far = -float("inf")
    for axis, ov in candidates:
        cfg = {**base_cfg, **ov}
        r = train_fn(data, **train_inputs, epochs=epochs,
                     early_stopping_patience=None, eval_every=eval_every,
                     verbose=per_run_verbose, **cfg)
        score = _overall_at_k(r.best_val, primary_k)
        best_so_far = max(best_so_far, score)
        if verbose:
            _write(f"[{axis}] {ov}  →  Overall@{primary_k}={score:.4f}  "
                   f"(ndcg={r.best_val.get(f'ndcg@{primary_k}', 0):.4f}, "
                   f"cov={r.best_val.get(f'coverage@{primary_k}', 0):.4f}, "
                   f"pop={r.best_val.get(f'pop_bias@{primary_k}', 0):.4f})")
        rows.append({"axis": axis, **cfg, score_col: score,
                     **{f"val_{kk}": vv for kk, vv in r.best_val.items()}})
        if _bar is not None:
            _bar.set_postfix_str(f"{axis} {score:.3f} | best {best_so_far:.3f}")
            _bar.update(1)
    if _bar is not None:
        _bar.close()

    abl_df = (pd.DataFrame(rows)
              .sort_values(score_col, ascending=False).reset_index(drop=True))
    best_row = abl_df.iloc[0]
    best_params = {k: (int(best_row[k]) if k in int_keys else float(best_row[k]))
                   for k in base_cfg}
    if out_csv is not None:
        abl_df.to_csv(out_csv, index=False)
    if out_json is not None:
        with open(out_json, "w") as f:
            json.dump(best_params, f, indent=2)
    if verbose and (out_csv is not None or out_json is not None):
        _names = " + ".join(p.name for p in (out_csv, out_json) if p is not None)
        print(f"\nBEST_HGT_PARAMS (seeds Phase-2): {best_params}")
        print(f"[SAVED] {_names}")
    return abl_df, best_params


def run_init_ablation(
    train_fn: Callable,
    build_graph_fn: Callable,
    *,
    kge_dict: Mapping[str, np.ndarray],
    edge_dict: Mapping[str, Any],
    track_audio: Mapping[str, np.ndarray],
    train_u_kg: np.ndarray,
    train_t_kg: np.ndarray,
    train_cnt: np.ndarray,
    main_test_metrics: Mapping[str, float],
    train_inputs: Mapping[str, Any],
    result_path: Path,
    cfg: Optional[Mapping[str, Any]] = None,
    epochs: int = 200,
    early_stopping_patience: int = 5,
    force: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[Any, pd.DataFrame]:
    """RotatE-vs-random node-init ablation (cached; no retraining when present).

    Rebuilds an identical graph with the KGE slice replaced by random vectors of
    matched scale, pins the same train-only interaction edges, trains an otherwise
    identical HGT, and reports the per-metric gap against ``main_test_metrics``.

    Returns ``(result_random, init_cmp_df)``.
    """
    import torch

    cfg = dict(cfg or dict(hidden_channels=128, out_channels=64, num_heads=4,
                           num_layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                           lambda_reg=0.2, temperature=0.1))
    kge_mat = np.stack(list(kge_dict.values()))
    mean, std, dim = float(kge_mat.mean()), float(kge_mat.std()), kge_mat.shape[1]
    rng = np.random.default_rng(seed)
    random_embeddings = {
        uri: (mean + std * rng.standard_normal(dim)).astype("float32")
        for uri in kge_dict
    }
    if verbose:
        print(f"[rand] {len(random_embeddings):,} random {dim}D vectors "
              f"(mean={mean:.3f}, std={std:.3f})")

    data_random = build_graph_fn(
        edge_dict=edge_dict, rotate_embeddings=random_embeddings,
        track_audio_features=track_audio, listen_counts=None)
    # Same stratified-TRAIN-only interaction edges as the main graph → leak-free.
    data_random["user", "listened_to", "track"].edge_index = torch.tensor(
        np.vstack([train_u_kg, train_t_kg]), dtype=torch.long)
    data_random["user", "listened_to", "track"].edge_weight = torch.log1p(
        torch.from_numpy(np.minimum(train_cnt, 1000.0)))

    result_path = Path(result_path)
    if result_path.exists() and not force:
        if verbose:
            print(f"[SKIP] Loading random-init HGT from {result_path.name}")
        result_random = load_pickle(result_path)
    else:
        result_random = train_fn(
            data_random, **train_inputs, epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose, **cfg)
        cache_pickle(result_random, result_path, force=force, label="random-init HGT")

    cmp = pd.DataFrame({"RotatE": dict(main_test_metrics),
                        "random": result_random.test_metrics})
    cmp["delta"] = cmp["RotatE"] - cmp["random"]
    return result_random, cmp


def run_direction_ablation(
    model,
    data,
    *,
    train_seen: Mapping[int, set],
    user_to_kg: Mapping[int, int],
    song2kg: Mapping[int, int],
    track_kg_to_song: Mapping[int, int],
    test_users: Sequence[int],
    test_gt: Mapping[int, set],
    n_songs: int,
    pop_norm: np.ndarray,
    k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Directed vs undirected forward of the SAME trained HGT (no retraining).

    Forwards ``model`` on the directed graph (where ``user`` nodes have no
    incoming edges → starved embeddings) and on the training-consistent
    undirected graph, builds a recommender from each and reports the metric gap
    at ``k``. Returns a DataFrame with ``undirected``/``directed``/``delta`` cols.
    """
    import torch_geometric.transforms as T

    from ..evaluation.metrics import multi_k_evaluation
    from .benchmark import build_hgt_recommender, to_kg_seen

    train_seen_kg = to_kg_seen(train_seen, user_to_kg, song2kg)

    proto: dict = {}
    for tag, undirected in (("undirected", True), ("directed", False)):
        rec = build_hgt_recommender(
            model, data, train_seen_kg=train_seen_kg, user_to_kg=user_to_kg,
            track_kg_to_song=track_kg_to_song, undirected=undirected)
        bulk = rec.recommend(list(test_users), k)
        m = multi_k_evaluation(bulk, test_gt, train_seen, n_songs, pop_norm,
                               ks=[k], model_name=tag)
        proto[tag] = m.set_index("metric")["value"]
    df = pd.DataFrame(proto)
    df["delta (undir−dir)"] = df["undirected"] - df["directed"]
    if verbose:
        print("(directed starves `user` nodes of incoming edges; undirected is "
              "the training-consistent protocol used for the main row)")
    return df


__all__ = [
    "default_ablation_axes",
    "run_phase1_ablation", "run_init_ablation", "run_direction_ablation",
]
