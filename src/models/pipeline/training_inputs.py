"""Rebuild the HGT training bundle for §8 (the §7 graph-build cell, from disk).

Lets §8 run standalone (env-setup → §8) without re-running §6/§7: it reuses the
in-memory bundle when warm, else rebuilds every input the training cells consume
— the rich heterograph (TRAIN-only interaction edges), the sparse interaction
matrix + per-track listen counts, the KG-space val/test ground truth, and the
graph-build ingredients (`kge_dict` / `edge_dict` / `track_audio` / `_train_*`) +
index bridges — straight from ``node_dict.json`` + the RotatE KGE checkpoint + the
AE parquet + the split parquets + ``kg_input.parquet``. No retraining of the KGE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

#: URI templates — must match what KGBuilder mints (track_id, not song_id).
_TRACK_URI_TPL = "http://purl.org/ontology/mrc/resource/track/{tid}"
_USER_URI_TPL = "http://purl.org/ontology/mrc/resource/user/{uid}"


@dataclass
class HGTTrainingInputs:
    """Everything the §8 training cells consume (graph + targets + ingredients)."""
    data: Any
    user_interaction_matrix: Any
    track_listen_counts: Any
    val_gt_kg: dict
    test_gt_kg: dict
    edge_dict: dict
    kge_dict: dict
    track_audio: dict
    train_u_kg: Any
    train_t_kg: Any
    train_cnt: Any
    user_to_kg: dict
    song2kg: dict
    track_kg_to_song: dict
    idx2song: dict

    def exposed(self) -> dict:
        """Globals to publish so every §8 cell (and §9+ later) reuses this state."""
        return {
            "data": self.data,
            "user_interaction_matrix": self.user_interaction_matrix,
            "track_listen_counts": self.track_listen_counts,
            "val_gt_kg": self.val_gt_kg, "test_gt_kg": self.test_gt_kg,
            "edge_dict": self.edge_dict, "kge_dict": self.kge_dict,
            "track_audio": self.track_audio,
            "_train_u_kg": self.train_u_kg, "_train_t_kg": self.train_t_kg,
            "_train_cnt": self.train_cnt,
            "_user_to_kg": self.user_to_kg, "_song2kg": self.song2kg,
            "_track_kg_to_song": self.track_kg_to_song, "idx2song": self.idx2song,
        }


_NEEDED = ("data", "user_interaction_matrix", "track_listen_counts", "val_gt_kg",
           "test_gt_kg", "edge_dict", "kge_dict", "track_audio", "_train_u_kg",
           "_train_t_kg", "_train_cnt", "_user_to_kg", "_song2kg",
           "_track_kg_to_song", "idx2song")


def ensure_hgt_training_inputs(
    in_memory: Optional[Mapping[str, Any]],
    *,
    kg_input_path,
    splits_dir,
    edge_dict_path,
    kge_rotate_path,
    ae_embeddings_path,
    verbose: bool = True,
) -> HGTTrainingInputs:
    """Reuse the in-memory §7 bundle, else rebuild it from disk (no §6/§7 rerun).

    Mirrors the §7 graph-build cell: TRAIN-only ``user→track`` edges (leak-free),
    a sparse ``(U, I)`` interaction matrix + per-track listen counts, KG-indexed
    val/test ground truth, and the graph-build ingredients + index bridges.
    """
    import json as _json

    import numpy as np
    import pandas as pd
    import scipy.sparse as sps
    import torch

    g = in_memory or {}
    log = print if verbose else (lambda *_: None)

    if all(n in g for n in _NEEDED):
        log("[hgt-inputs] reusing in-memory §7 training bundle (no rebuild)")
        return HGTTrainingInputs(
            g["data"], g["user_interaction_matrix"], g["track_listen_counts"],
            g["val_gt_kg"], g["test_gt_kg"], g["edge_dict"], g["kge_dict"],
            g["track_audio"], g["_train_u_kg"], g["_train_t_kg"], g["_train_cnt"],
            g["_user_to_kg"], g["_song2kg"], g["_track_kg_to_song"], g["idx2song"])

    from ..kg_to_hetero import build_rich_hetero_graph
    from ..kg_embeddings import load_kge_checkpoint

    splits_dir = Path(splits_dir)

    # song_id → track_id (URIs are minted from track_id).
    merged = pd.read_parquet(kg_input_path, columns=["song_id", "track_id"])
    sid2tid = (merged.drop_duplicates("song_id")
               .set_index("song_id")["track_id"].to_dict())

    # Split-derived index space + ground truth (u_idx/s_idx space).
    sp = {s: pd.read_parquet(splits_dir / f"{s}.parquet",
                             columns=["user_id", "song_id", "play_count", "u_idx", "s_idx"])
          for s in ("train", "val", "test")}
    allsp = pd.concat(sp.values())
    idx2song = {int(k): str(v) for k, v in
                allsp.drop_duplicates("s_idx").set_index("s_idx")["song_id"].to_dict().items()}
    u2user = allsp.drop_duplicates("u_idx").set_index("u_idx")["user_id"].to_dict()
    train_df = sp["train"]
    val_gt = sp["val"].groupby("u_idx")["s_idx"].apply(set).to_dict()
    test_gt = sp["test"].groupby("u_idx")["s_idx"].apply(set).to_dict()

    edge_dict = _json.load(open(edge_dict_path))
    kge_dict = load_kge_checkpoint(kge_rotate_path)

    # Audio features: track_uri → AE embedding.
    ae_df = pd.read_parquet(ae_embeddings_path)
    ae_cols = [c for c in ae_df.columns if c.startswith("ae_")]
    ae_vals = ae_df[ae_cols].to_numpy(dtype="float32")
    track_audio = {
        _TRACK_URI_TPL.format(tid=sid2tid[sid]): ae_vals[i]
        for i, sid in enumerate(ae_df["song_id"].astype(str))
        if sid in sid2tid
    }

    # Heterograph (train-only interaction edges set below).
    data = build_rich_hetero_graph(
        edge_dict=edge_dict, rotate_embeddings=kge_dict,
        track_audio_features=track_audio, listen_counts=None)
    user_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["user"])}
    track_uri2kg = {uri: i for i, uri in enumerate(edge_dict["node_mappings"]["track"])}
    n_users_kg = data["user"].num_nodes
    n_tracks_kg = data["track"].num_nodes

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

    # TRAIN-only user→track edges + interaction matrix (leak-free).
    tr_u = train_df["u_idx"].map(user_to_kg)
    tr_t = train_df["s_idx"].map(song2kg)
    ok = tr_u.notna() & tr_t.notna()
    train_u_kg = tr_u[ok].to_numpy(dtype=np.int64)
    train_t_kg = tr_t[ok].to_numpy(dtype=np.int64)
    train_cnt = train_df["play_count"][ok].to_numpy(dtype=np.float32)

    data["user", "listened_to", "track"].edge_index = torch.tensor(
        np.vstack([train_u_kg, train_t_kg]), dtype=torch.long)
    data["user", "listened_to", "track"].edge_weight = torch.log1p(
        torch.from_numpy(np.minimum(train_cnt, 1000.0)))

    user_interaction_matrix = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([train_u_kg, train_t_kg])),
        torch.from_numpy(train_cnt),
        size=(n_users_kg, n_tracks_kg),
    ).coalesce()
    track_listen_counts = torch.from_numpy(
        np.asarray(sps.csr_matrix((train_cnt, (train_u_kg, train_t_kg)),
                                  shape=(n_users_kg, n_tracks_kg)).sum(axis=0))
        .ravel().astype(np.float32))

    def _gt_to_kg(gt):
        out = {}
        for u, items in gt.items():
            ku = user_to_kg.get(u)
            if ku is None:
                continue
            ks = {song2kg[s] for s in items if s in song2kg}
            if ks:
                out[ku] = ks
        return out

    val_gt_kg, test_gt_kg = _gt_to_kg(val_gt), _gt_to_kg(test_gt)
    log(f"[hgt-inputs] rebuilt from disk — graph + interaction matrix "
        f"(nnz={user_interaction_matrix._nnz():,}, train_edges={len(train_u_kg):,}) "
        f"+ val/test_gt_kg ({len(val_gt_kg):,}/{len(test_gt_kg):,} users); "
        "no §6/§7 rerun, no KGE retrain")
    return HGTTrainingInputs(
        data, user_interaction_matrix, track_listen_counts, val_gt_kg, test_gt_kg,
        edge_dict, kge_dict, track_audio, train_u_kg, train_t_kg, train_cnt,
        user_to_kg, song2kg, track_kg_to_song, idx2song)


__all__ = ["HGTTrainingInputs", "ensure_hgt_training_inputs"]
