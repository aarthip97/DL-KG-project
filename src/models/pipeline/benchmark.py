"""Benchmark assembly + evaluation for the pipeline notebook (§9).

Centralizes the recommender-assembly ladder and the multi-K evaluation so the §9
cell reads as ``ctx = load_eval_context(...) → recs = assemble_recommenders(...)
→ run_benchmark(recs, ctx)``. The fallback ladders (in-memory → in-memory
ingredients → assembled pickle → reconstruct from each model's own artefact)
wrap the existing :mod:`models.evaluation.rehydrate` helpers — nothing is
retrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .checkpoints import cache_pickle, load_pickle


@dataclass
class EvalContext:
    """Split-derived inputs every model is scored against (§9)."""
    train_seen: dict
    test_gt: dict
    test_users: list
    n_songs: int
    n_users: int
    pop_norm: np.ndarray
    train_matrix_norm: Any
    song_popularity: np.ndarray


@dataclass
class RecommenderSet:
    """Assembled recommenders + any globals rebuilt from disk along the way."""
    all: list
    hgt: Any
    baselines: list
    extras: dict = field(default_factory=dict)


def to_kg_seen(train_seen: Mapping[int, set], user_to_kg: Mapping[int, int],
               song2kg: Mapping[int, int]) -> dict:
    """Map ``{u_idx: {s_idx}}`` train history into KG-node index space."""
    return {user_to_kg[u]: {song2kg[s] for s in seen if s in song2kg}
            for u, seen in train_seen.items() if u in user_to_kg}


def load_eval_context(
    splits_dir,
    *,
    in_memory: Optional[Mapping[str, Any]] = None,
    verbose: bool = True,
) -> EvalContext:
    """Build the :class:`EvalContext`, reusing in-memory globals when present.

    If ``in_memory`` already carries every field (warm kernel), they are wrapped
    directly; otherwise everything is rebuilt from the cached splits — the same
    derivation the §6 cells used (L2-normalised train matrix, song popularity,
    train-seen / test ground-truth dicts). No upstream rerun is forced.
    """
    g = in_memory or {}
    keys = ("test_users", "test_gt", "train_seen", "N_SONGS", "N_USERS",
            "pop_norm", "train_matrix_norm", "song_popularity")
    if all(k in g for k in keys):
        return EvalContext(
            train_seen=g["train_seen"], test_gt=g["test_gt"],
            test_users=g["test_users"], n_songs=g["N_SONGS"], n_users=g["N_USERS"],
            pop_norm=g["pop_norm"], train_matrix_norm=g["train_matrix_norm"],
            song_popularity=g["song_popularity"])

    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    from ..train_val_test_split import load_splits

    sp = load_splits(splits_dir, fmt="parquet")
    tr, va, te = sp["train"], sp["val"], sp["test"]
    n_users = int(max(tr["u_idx"].max(), va["u_idx"].max(), te["u_idx"].max())) + 1
    n_songs = int(max(tr["s_idx"].max(), va["s_idx"].max(), te["s_idx"].max())) + 1
    m = csr_matrix((np.ones(len(tr)), (tr["u_idx"], tr["s_idx"])),
                   shape=(n_users, n_songs), dtype=np.float32)
    song_pop = np.asarray(m.sum(axis=0)).ravel()
    if verbose:
        print(f"[rehydrate] eval context ← {Path(splits_dir).name}/")
    return EvalContext(
        train_seen=tr.groupby("u_idx")["s_idx"].apply(set).to_dict(),
        test_gt=te.groupby("u_idx")["s_idx"].apply(set).to_dict(),
        test_users=sorted(te.groupby("u_idx")["s_idx"].apply(set).to_dict().keys()),
        n_songs=n_songs, n_users=n_users,
        pop_norm=song_pop / (song_pop.max() + 1e-9),
        train_matrix_norm=normalize(m, norm="l2"), song_popularity=song_pop)


def build_hgt_recommender(
    model,
    data,
    *,
    train_seen_kg: Mapping[int, set],
    user_to_kg: Mapping[int, int],
    track_kg_to_song: Mapping[int, int],
    undirected: bool = True,
):
    """Forward ``model`` once and wrap the embeddings as an :class:`HGTRecommender`.

    With ``undirected=True`` (default, training-consistent) the reverse
    ``track → user`` relation is added so ``user`` nodes receive messages.
    """
    import torch
    import torch_geometric.transforms as T

    from ..evaluation.recommenders import HGTRecommender

    g = T.ToUndirected(merge=False)(data) if undirected else data
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        emb = model(
            {nt: g[nt].x.to(dev) for nt in g.node_types if g[nt].get("x") is not None},
            {et: g[et].edge_index.to(dev) for et in g.edge_types})
    return HGTRecommender(
        user_emb=emb["user"].cpu().numpy().astype(np.float32),
        track_emb=emb["track"].cpu().numpy().astype(np.float32),
        train_seen_kg=train_seen_kg, user_to_kg=user_to_kg,
        track_kg_to_song=track_kg_to_song)


def assemble_recommenders(
    ctx: EvalContext,
    *,
    in_memory: Optional[Mapping[str, Any]] = None,
    models_dir,
    splits_dir,
    knn_nbrs_cache,
    knn_summary_json,
    xgb_model_cache,
    ae_embeddings_path,
    kg_input_path,
    hgt_model_path,
    kge_rotate_path,
    edge_dict_path,
    xgb_n_candidates: int = 200,
    device: str = "cpu",
    force: bool = False,
    verbose: bool = True,
) -> RecommenderSet:
    """Assemble baselines + HGT via the no-retrain fallback ladder.

    Baselines: live ``RECOMMENDERS`` → in-memory ingredients → assembled pickle →
    :func:`rebuild_baselines_from_disk`. HGT: live ``result``+``data``+bridges
    (undirected forward) → ``hgt_recommender.pkl`` →
    :func:`rebuild_hgt_recommender_from_disk`. Any globals rebuilt from disk
    (``data``/``edge_dict``/``idx2song``/bridges/``result``) are returned in
    ``extras`` for the notebook to re-expose for §10–14.
    """
    from ..evaluation.recommenders import PopularityRecommender, KNNRecommender
    from ..evaluation.rehydrate import (
        rebuild_baselines_from_disk, rebuild_hgt_recommender_from_disk)

    g = in_memory or {}
    models_dir = Path(models_dir)
    baseline_cache = models_dir / "baseline_recommenders.pkl"
    hgt_cache = models_dir / "hgt_recommender.pkl"
    log = print if verbose else (lambda *_: None)

    # ── 1. baselines (+ XGB) ──────────────────────────────────────────────────
    loaded_pkl = False
    if "RECOMMENDERS" in g:
        baselines = list(g["RECOMMENDERS"])
    elif all(n in g for n in ("song_popularity", "train_matrix_norm",
                              "train_seen", "all_nbrs", "qrow", "best_k")):
        baselines = [
            PopularityRecommender(g["song_popularity"], g["train_seen"]),
            KNNRecommender(g["train_matrix_norm"], g["train_seen"],
                           g["all_nbrs"], g["qrow"], best_k=g["best_k"]),
        ]
        if "xgb_recommender" in g:
            baselines.append(g["xgb_recommender"])
    elif baseline_cache.exists():
        baselines = load_pickle(baseline_cache)
        loaded_pkl = True
        log(f"[rehydrate] baselines ← {baseline_cache.name}")
    else:
        baselines = rebuild_baselines_from_disk(
            train_seen=ctx.train_seen, song_popularity=ctx.song_popularity,
            train_matrix_norm=ctx.train_matrix_norm, n_songs=ctx.n_songs,
            splits_dir=splits_dir, knn_nbrs_cache=knn_nbrs_cache,
            knn_summary_json=knn_summary_json, xgb_model_cache=xgb_model_cache,
            ae_embeddings_path=ae_embeddings_path, kg_input_path=kg_input_path,
            xgb_n_candidates=xgb_n_candidates)
        if len(baselines) <= 1:
            raise RuntimeError(
                "Only MostPopular could be rebuilt — KNN/XGB artefacts not found "
                f"on disk and none in memory. Expected {knn_nbrs_cache} and "
                f"{xgb_model_cache}. Run the KNN/XGB cells once (no retraining).")
    if not loaded_pkl and (force or not baseline_cache.exists()):
        cache_pickle(baselines, baseline_cache, force=force, label="baselines")

    # ── 2. HGT recommender ────────────────────────────────────────────────────
    extras: dict = {}
    if all(n in g for n in ("result", "data", "_user_to_kg",
                            "_song2kg", "_track_kg_to_song")):
        train_seen_kg = to_kg_seen(ctx.train_seen, g["_user_to_kg"], g["_song2kg"])
        recs_hgt = build_hgt_recommender(
            g["result"].model, g["data"], train_seen_kg=train_seen_kg,
            user_to_kg=g["_user_to_kg"], track_kg_to_song=g["_track_kg_to_song"],
            undirected=True)
        if force or not hgt_cache.exists():
            cache_pickle(recs_hgt, hgt_cache, force=force, label="HGT")
    elif hgt_cache.exists():
        recs_hgt = load_pickle(hgt_cache)
        log(f"[rehydrate] HGT ← {hgt_cache.name}")
    elif Path(hgt_model_path).exists():
        hgt = rebuild_hgt_recommender_from_disk(
            hgt_model_path=hgt_model_path, kge_rotate_path=kge_rotate_path,
            edge_dict_path=edge_dict_path, kg_input_path=kg_input_path,
            ae_embeddings_path=ae_embeddings_path, splits_dir=splits_dir,
            device=device)
        recs_hgt = hgt["recommender"]
        extras = {"data": hgt["data"], "edge_dict": hgt["edge_dict"],
                  "idx2song": hgt["idx2song"], "_user_to_kg": hgt["user_to_kg"],
                  "_song2kg": hgt["song2kg"], "_track_kg_to_song": hgt["track_kg_to_song"]}
        if "result" not in g:
            import types
            extras["result"] = types.SimpleNamespace(model=hgt["model"])
        cache_pickle(recs_hgt, hgt_cache, force=force, label="HGT")
    else:
        raise RuntimeError(
            f"HGT not in memory, no cache at {hgt_cache}, and no weights at "
            f"{hgt_model_path}. Run the HGT training/eval cell once.")

    all_recs = [r for r in baselines if r.name != recs_hgt.name] + [recs_hgt]
    log(f"Recommenders: {[r.name for r in all_recs]}")
    return RecommenderSet(all=all_recs, hgt=recs_hgt, baselines=baselines, extras=extras)


def run_benchmark(
    recommenders: Sequence[Any],
    ctx: EvalContext,
    *,
    k_list: Sequence[int] = (5, 10, 20, 50),
    top_n: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Full multi-K evaluation + Overall_Score@top_n significance + selection.

    Returns ``{bulk, multi_k, agg, pairwise, comparison, per_user, selection}``.
    ``selection`` ranks models by Overall_Score@``top_n`` — the §8 criterion,
    already computed per model/K by :func:`multi_k_evaluation`.

    ``pairwise`` tests *only* the selection criterion and its components at
    ``top_n`` (NDCG, Coverage, 1−PopularityBias, and the composite Overall_Score)
    via :func:`overall_significance` — per-user Wilcoxon for the per-user
    components, a paired user bootstrap for the set-level Coverage / composite.
    ``comparison`` keeps the broader Wilcoxon/Friedman view over the per-user
    ranking metrics for completeness.
    """
    from ..evaluation.metrics import multi_k_evaluation, evaluate_recs_per_user
    from ..evaluation.comparison import summarise_comparison, overall_significance

    bulk = {r.name: r.recommend(ctx.test_users, max(k_list)) for r in recommenders}
    multi_k = pd.concat(
        [multi_k_evaluation(bulk[n], ctx.test_gt, ctx.train_seen, ctx.n_songs,
                            ctx.pop_norm, ks=list(k_list), model_name=n)
         for n in bulk], ignore_index=True)
    per_user = {n: evaluate_recs_per_user(bulk[n], ctx.test_gt, ctx.pop_norm, k=top_n)
                for n in bulk}
    # Headline significance: the Overall_Score@top_n criterion + its components only.
    pairwise = overall_significance(
        bulk, ctx.test_gt, ctx.pop_norm, ctx.n_songs,
        k=top_n, n_boot=n_boot, seed=seed)
    # Broader per-user ranking view (Wilcoxon pairwise + Friedman/Nemenyi).
    comparison = summarise_comparison(
        per_user, metrics=("Recall@K", "NDCG@K", "F1@K"))
    agg = (multi_k.groupby(["model", "K", "metric"])["value"]
           .mean().unstack("metric").round(4))
    selection = (multi_k[(multi_k["K"] == top_n) & (multi_k["metric"] == "Overall_Score")]
                 .set_index("model")["value"].sort_values(ascending=False))
    return {"bulk": bulk, "multi_k": multi_k, "agg": agg,
            "pairwise": pairwise, "comparison": comparison,
            "per_user": per_user, "selection": selection}


__all__ = [
    "EvalContext", "RecommenderSet", "to_kg_seen",
    "load_eval_context", "build_hgt_recommender",
    "assemble_recommenders", "run_benchmark",
]
