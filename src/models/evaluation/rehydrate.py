"""Reconstruct the baseline recommenders from their on-disk training artefacts.

After a kernel restart the in-memory recommender objects are gone, but the
*trained* artefacts each baseline produced are still on disk:

* **MostPopular** — needs only ``song_popularity`` + ``train_seen`` (both cheap
  to rebuild from the cached stratified splits).
* **KNN-CF** — the neighbour table (``torch.save`` cache with ``nbrs_tensor`` +
  ``all_query``) and the chosen ``best_k`` (in the KNN/pop summary JSON).
* **XGBoost-Hybrid** — the trained booster (``.ubj``), the AE embedding matrix
  (parquet) and the per-track categorical features, which are *deterministically*
  rebuilt from ``kg_input.parquet`` + the train split (``te_seed`` is fixed) — so
  no model is retrained.

:func:`rebuild_baselines_from_disk` assembles whichever of these it can, so the
final-benchmark cell can run standalone without re-executing the baseline
training cells.  Anything whose artefacts are missing is skipped with a note
rather than raising, so a partial set still lets the benchmark proceed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping, Optional, Set

import numpy as np

from .recommenders import KNNRecommender, PopularityRecommender, XGBHybridRecommender


def _exists(p) -> bool:
    return p is not None and Path(p).exists()


def rebuild_baselines_from_disk(
    *,
    train_seen: Mapping[int, Set[int]],
    song_popularity: np.ndarray,
    train_matrix_norm,
    n_songs: int,
    splits_dir,
    knn_nbrs_cache=None,
    knn_summary_json=None,
    xgb_model_cache=None,
    ae_embeddings_path=None,
    kg_input_path=None,
    xgb_n_candidates: int = 200,
    top_instruments: int = 20,
    verbose: bool = True,
) -> List:
    """Rebuild ``[MostPopular, KNN-CF, XGBoost-Hybrid]`` from cached artefacts.

    Only the recommenders whose artefacts are present are returned; none of them
    triggers training.  ``song_popularity`` / ``train_matrix_norm`` / ``train_seen``
    are expected to already be rebuilt from the cached splits by the caller.

    Returns the list of successfully reconstructed recommenders (MostPopular is
    always included as it has no separate artefact).
    """
    log = print if verbose else (lambda *_: None)
    recs: List = [PopularityRecommender(song_popularity, train_seen)]
    log("[rehydrate] MostPopular ← splits")

    # ── KNN-CF ────────────────────────────────────────────────────────────────
    if _exists(knn_nbrs_cache) and _exists(knn_summary_json):
        try:
            import torch  # noqa: PLC0415
            saved = torch.load(knn_nbrs_cache, map_location="cpu", weights_only=True)
            all_nbrs = saved["nbrs_tensor"].cpu().numpy()
            qrow = {int(u): i for i, u in enumerate(saved["all_query"].tolist())}
            best_k = int(json.loads(Path(knn_summary_json).read_text())["best_k"])
            recs.append(KNNRecommender(train_matrix_norm, train_seen,
                                       all_nbrs, qrow, best_k=best_k))
            log(f"[rehydrate] KNN-CF ← {Path(knn_nbrs_cache).name} (best_k={best_k})")
        except Exception as e:  # noqa: BLE001
            log(f"[rehydrate] KNN-CF skipped ({e.__class__.__name__}: {e})")
    else:
        log("[rehydrate] KNN-CF skipped (neighbours cache / summary JSON missing)")

    # ── XGBoost-Hybrid (booster reloaded, features rebuilt deterministically) ──
    if _exists(xgb_model_cache) and _exists(ae_embeddings_path) and _exists(kg_input_path):
        try:
            import pandas as pd  # noqa: PLC0415
            import xgboost as xgb  # noqa: PLC0415
            from ..xgb_hybrid import (  # noqa: PLC0415
                _load_ae_matrix, build_track_categorical_features,
            )
            splits_dir = Path(splits_dir)
            sid2sidx = (
                pd.concat([
                    pd.read_parquet(splits_dir / f"{s}.parquet", columns=["song_id", "s_idx"])
                    for s in ("train", "val", "test")
                ])
                .drop_duplicates("song_id")
                .set_index("song_id")["s_idx"]
                .to_dict()
            )
            train_df = pd.read_parquet(splits_dir / "train.parquet")
            merged = pd.read_parquet(kg_input_path)
            track_extra, _ = build_track_categorical_features(
                track_meta=merged, train_df=train_df, song_id_to_sidx=sid2sidx,
                n_songs=n_songs, top_instruments=top_instruments)
            ae_matrix, _ = _load_ae_matrix(ae_embeddings_path, sid2sidx, n_songs)
            booster = xgb.Booster()
            booster.load_model(str(xgb_model_cache))
            recs.append(XGBHybridRecommender(
                model=booster, ae_matrix=ae_matrix, train_seen=train_seen,
                n_candidates=xgb_n_candidates, track_extra=track_extra))
            log(f"[rehydrate] XGBoost-Hybrid ← {Path(xgb_model_cache).name} (no retrain)")
        except Exception as e:  # noqa: BLE001
            log(f"[rehydrate] XGBoost-Hybrid skipped ({e.__class__.__name__}: {e})")
    else:
        log("[rehydrate] XGBoost-Hybrid skipped (booster / AE / kg_input cache missing)")

    return recs


# ─────────────────────────────────────────────────────────────────────────────
#  HGT recommender reconstruction (no retraining)
# ─────────────────────────────────────────────────────────────────────────────
# URI templates — must match what KGBuilder mints (see the graph-build cell).
_TRACK_URI_TPL = "http://purl.org/ontology/mrc/resource/track/{tid}"
_USER_URI_TPL = "http://purl.org/ontology/mrc/resource/user/{uid}"


def _infer_hgt_arch(state_dict) -> dict:
    """Recover (hidden_channels, out_channels, num_layers) from a saved
    ``RecommenderHGT`` ``state_dict`` so the architecture adapts automatically.

    ``num_heads`` cannot be read back from parameter shapes (HGTConv folds heads
    into ``out_channels``), so it stays an explicit argument of the caller.
    """
    from collections import defaultdict

    num_layers = max(int(k.split(".")[1]) for k in state_dict
                     if k.startswith("convs.")) + 1
    lin_w = next(v for k, v in state_dict.items()
                 if k.startswith("lin_dict.") and k.endswith(".weight"))
    hidden = int(lin_w.shape[0])
    heads_out: "defaultdict[str, list]" = defaultdict(list)
    for k in state_dict:
        if k.startswith("head_dict.") and k.endswith(".weight"):
            p = k.split(".")
            heads_out[p[1]].append((int(p[2]), k))
    nt0 = next(iter(heads_out))
    _, last_key = max(heads_out[nt0])
    out_channels = int(state_dict[last_key].shape[0])
    return {"hidden_channels": hidden, "out_channels": out_channels,
            "num_layers": num_layers}


def rebuild_hgt_recommender_from_disk(
    *,
    hgt_model_path,
    kge_rotate_path,
    edge_dict_path,
    kg_input_path,
    ae_embeddings_path,
    splits_dir,
    device: str = "cpu",
    num_heads: int = 4,
    dropout: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Reconstruct the frozen HGT recommender from disk **without retraining**.

    Mirrors the notebook's graph-build + HGT cells: it rebuilds the rich
    heterograph (KGE node features ← RotatE checkpoint, audio ← AE parquet,
    topology + node URIs ← ``node_dict.json``), re-derives the ``u_idx``/``s_idx``
    ↔ KG-node index bridges straight from the cached splits, re-instantiates
    :class:`~models.hgt.RecommenderHGT` (architecture inferred from the
    ``state_dict``; ``num_heads`` supplied) and loads ``model.pt``, then forwards
    on the directed graph — exactly as the benchmark cell does when the model is
    in memory — to materialise the user/track embeddings.

    Returns a dict with the ``recommender`` plus the rebuilt ``model``, ``data``,
    ``edge_dict``, the three index-bridge maps and ``idx2song``, so the
    explainability / latent-space sections can reuse them after a kernel restart.
    """
    import json as _json

    import pandas as pd
    import torch
    from torch_geometric.transforms import ToUndirected

    from ..hgt import RecommenderHGT
    from ..kg_embeddings import load_kge_checkpoint
    from ..kg_to_hetero import build_rich_hetero_graph
    from .recommenders import HGTRecommender

    log = print if verbose else (lambda *_: None)
    splits_dir = Path(splits_dir)

    # song_id → track_id (track URIs are minted from track_id, not song_id).
    merged = pd.read_parquet(kg_input_path)
    sid2tid = (merged[["song_id", "track_id"]].drop_duplicates("song_id")
               .set_index("song_id")["track_id"].to_dict())

    # Split-derived index space (no need to re-run the data-prep filtering): the
    # split parquets already carry the (user_id, song_id) ↔ (u_idx, s_idx) maps.
    sp = {s: pd.read_parquet(splits_dir / f"{s}.parquet") for s in ("train", "val", "test")}
    allsp = pd.concat(sp.values())
    idx2song = allsp.drop_duplicates("s_idx").set_index("s_idx")["song_id"].to_dict()
    u2user = allsp.drop_duplicates("u_idx").set_index("u_idx")["user_id"].to_dict()
    train_df = sp["train"]

    # Node features + topology.
    edge_dict = _json.load(open(edge_dict_path))
    kge_dict = load_kge_checkpoint(kge_rotate_path)
    ae_df = pd.read_parquet(ae_embeddings_path)
    ae_cols = [c for c in ae_df.columns if c.startswith("ae_")]
    ae_vals = ae_df[ae_cols].to_numpy(dtype="float32")
    track_audio = {
        _TRACK_URI_TPL.format(tid=sid2tid[sid]): ae_vals[i]
        for i, sid in enumerate(ae_df["song_id"].astype(str))
        if sid in sid2tid
    }
    data = build_rich_hetero_graph(
        edge_dict=edge_dict, rotate_embeddings=kge_dict,
        track_audio_features=track_audio, listen_counts=None)

    # Index bridges (u_idx/s_idx → KG node) from the split maps + URI templates.
    user_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["user"])}
    track_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["track"])}
    user_to_kg = {}
    for u_idx, uid in u2user.items():
        kg = user_uri2kg.get(_USER_URI_TPL.format(uid=uid))
        if kg is not None:
            user_to_kg[int(u_idx)] = kg
    song2kg, track_kg_to_song = {}, {}
    for s_idx, sid in idx2song.items():
        tid = sid2tid.get(sid)
        if tid is None:
            continue
        kg = track_uri2kg.get(_TRACK_URI_TPL.format(tid=tid))
        if kg is not None:
            song2kg[int(s_idx)] = kg
            track_kg_to_song[kg] = int(s_idx)
    log(f"[rehydrate] HGT index bridge: {len(user_to_kg):,} users, "
        f"{len(song2kg):,} songs mapped to KG nodes")

    # TRAIN-only user→track edges (leak-free), matching the graph-build cell.
    tr_u = train_df["u_idx"].map(user_to_kg)
    tr_t = train_df["s_idx"].map(song2kg)
    ok = tr_u.notna() & tr_t.notna()
    data["user", "listened_to", "track"].edge_index = torch.tensor(
        np.vstack([tr_u[ok].to_numpy("int64"), tr_t[ok].to_numpy("int64")]),
        dtype=torch.long)

    # Re-instantiate the model (arch from the state_dict, heads supplied) + load.
    state_dict = torch.load(hgt_model_path, map_location="cpu", weights_only=True)
    arch = _infer_hgt_arch(state_dict)
    meta = ToUndirected(merge=False)(data).metadata()      # model was built undirected
    model = RecommenderHGT(meta, num_heads=num_heads, dropout=dropout, **arch)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Forward on the DIRECTED graph (same as the benchmark cell's in-memory path).
    with torch.no_grad():
        emb = model(
            {nt: data[nt].x.to(device) for nt in data.node_types
             if data[nt].get("x") is not None},
            {et: data[et].edge_index.to(device) for et in data.edge_types})
    user_emb = emb["user"].cpu().numpy().astype(np.float32)
    track_emb = emb["track"].cpu().numpy().astype(np.float32)

    seen = train_df.groupby("u_idx")["s_idx"].apply(set).to_dict()
    train_seen_kg = {
        user_to_kg[u]: {song2kg[s] for s in items if s in song2kg}
        for u, items in seen.items() if u in user_to_kg
    }
    rec = HGTRecommender(user_emb=user_emb, track_emb=track_emb,
                         train_seen_kg=train_seen_kg, user_to_kg=user_to_kg,
                         track_kg_to_song=track_kg_to_song)
    log(f"[rehydrate] HGT ← {Path(hgt_model_path).name} "
        f"(hidden={arch['hidden_channels']}, out={arch['out_channels']}, "
        f"heads={num_heads}, layers={arch['num_layers']}; no retrain)")

    return {
        "recommender": rec, "model": model, "data": data, "edge_dict": edge_dict,
        "user_to_kg": user_to_kg, "song2kg": song2kg,
        "track_kg_to_song": track_kg_to_song, "idx2song": idx2song,
        "embeddings": {"user": user_emb, "track": track_emb},
    }


def load_index_bridges_from_disk(
    *,
    edge_dict_path,
    kg_input_path,
    splits_dir,
) -> dict:
    """Rebuild the ``u_idx``/``s_idx`` ↔ KG-node bridges **without** the model.

    The bridge part of :func:`rebuild_hgt_recommender_from_disk`, isolated so the
    persona / latent cells can resolve KG track nodes → songs after a restart
    cheaply — it reads only ``node_dict.json`` (URI ordering), the split parquets
    (the ``u_idx``/``s_idx`` maps) and ``kg_input.parquet`` (``song_id → track_id``).
    No KGE checkpoint, AE parquet, graph build or weight load.

    Returns ``{edge_dict, idx2song, user_to_kg, song2kg, track_kg_to_song}``.
    """
    import json as _json

    import pandas as pd

    splits_dir = Path(splits_dir)
    merged = pd.read_parquet(kg_input_path, columns=["song_id", "track_id"])
    sid2tid = (merged.drop_duplicates("song_id")
               .set_index("song_id")["track_id"].to_dict())
    sp = pd.concat([
        pd.read_parquet(splits_dir / f"{s}.parquet",
                        columns=["song_id", "s_idx", "user_id", "u_idx"])
        for s in ("train", "val", "test")
    ])
    idx2song = {int(k): v for k, v in
                sp.drop_duplicates("s_idx").set_index("s_idx")["song_id"].to_dict().items()}
    u2user = sp.drop_duplicates("u_idx").set_index("u_idx")["user_id"].to_dict()

    edge_dict = _json.load(open(edge_dict_path))
    user_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["user"])}
    track_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["track"])}

    user_to_kg = {}
    for u_idx, uid in u2user.items():
        kg = user_uri2kg.get(_USER_URI_TPL.format(uid=uid))
        if kg is not None:
            user_to_kg[int(u_idx)] = kg
    song2kg, track_kg_to_song = {}, {}
    for s_idx, sid in idx2song.items():
        tid = sid2tid.get(sid)
        if tid is None:
            continue
        kg = track_uri2kg.get(_TRACK_URI_TPL.format(tid=tid))
        if kg is not None:
            song2kg[int(s_idx)] = kg
            track_kg_to_song[kg] = int(s_idx)
    return {"edge_dict": edge_dict, "idx2song": idx2song, "user_to_kg": user_to_kg,
            "song2kg": song2kg, "track_kg_to_song": track_kg_to_song}


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight split / metadata globals (for the qualitative + explainability cells)
# ─────────────────────────────────────────────────────────────────────────────
def load_song_meta(
    kg_input_path,
    *,
    song_id_col: str = "song_id",
    cols: tuple = ("song_id", "title", "artist_name", "primary_genre",
                   "year", "key", "mode", "Mean_Tempo", "tempo_class"),
):
    """Rebuild the per-song metadata table (indexed by ``song_id``) from
    ``kg_input.parquet`` — the same construction the §7 cell does, so the
    qualitative / explainability cells can recover ``song_meta`` after a kernel
    restart. Only the columns that actually exist are kept.
    """
    import pandas as pd  # noqa: PLC0415
    df = pd.read_parquet(kg_input_path)
    keep = [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates(song_id_col).set_index(song_id_col)


def load_eval_ground_truth(splits_dir):
    """Return ``(train_seen, test_gt, test_users)`` from the cached splits.

    ``train_seen`` / ``test_gt`` are ``{u_idx: {s_idx, …}}`` and ``test_users`` is
    the sorted list of evaluation users — the split-derived globals the
    qualitative / explainability cells consume.
    """
    import pandas as pd  # noqa: PLC0415
    splits_dir = Path(splits_dir)
    tr = pd.read_parquet(splits_dir / "train.parquet", columns=["u_idx", "s_idx"])
    te = pd.read_parquet(splits_dir / "test.parquet", columns=["u_idx", "s_idx"])
    train_seen = tr.groupby("u_idx")["s_idx"].apply(set).to_dict()
    test_gt = te.groupby("u_idx")["s_idx"].apply(set).to_dict()
    return train_seen, test_gt, sorted(test_gt.keys())


__all__ = (
    "rebuild_baselines_from_disk",
    "rebuild_hgt_recommender_from_disk",
    "load_index_bridges_from_disk",
    "load_song_meta",
    "load_eval_ground_truth",
)
