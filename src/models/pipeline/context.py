"""Standalone-section rehydration for the pipeline notebook (§12–14).

A single entrypoint that gathers everything the explainability / latent /
persona sections need — ``song_meta``, the eval ground truth, and the live
model + graph + index bridges — reusing in-memory globals when warm and
otherwise rebuilding from disk (``model.pt`` + RotatE KGE + ``node_dict`` + AE +
splits) with NO retraining. Wraps the existing
:mod:`models.evaluation.rehydrate` helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class HGTContext:
    """Live model + graph + bridges + eval ground truth for §12–14."""
    song_meta: Any
    train_seen: dict
    test_gt: dict
    test_users: list
    result: Any
    data: Any
    edge_dict: dict
    idx2song: dict
    user_to_kg: dict
    song2kg: dict
    track_kg_to_song: dict

    def exposed(self) -> dict:
        """Globals to re-publish so later sections reuse this state after a restart."""
        return {
            "song_meta": self.song_meta, "train_seen": self.train_seen,
            "test_gt": self.test_gt, "test_users": self.test_users,
            "result": self.result, "data": self.data, "edge_dict": self.edge_dict,
            "idx2song": self.idx2song, "_user_to_kg": self.user_to_kg,
            "_song2kg": self.song2kg, "_track_kg_to_song": self.track_kg_to_song,
        }


def ensure_hgt_context(
    in_memory: Optional[Mapping[str, Any]],
    *,
    kg_input_path,
    splits_dir,
    hgt_model_path,
    kge_rotate_path,
    edge_dict_path,
    ae_embeddings_path,
    num_heads: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Optional[HGTContext]:
    """Assemble an :class:`HGTContext`, rebuilding only the missing pieces.

    Returns ``None`` when the HGT weights are absent (``model.pt`` missing) and
    nothing is in memory — the caller should then fall back to a no-HGT path.

    ``num_heads`` is forwarded to :func:`rebuild_hgt_recommender_from_disk` when a
    rebuild is needed. It is the only architecture knob that cannot be recovered
    from the saved ``state_dict`` (HGTConv folds heads into ``out_channels``), so
    it should come from the HGT training best params; ``None`` keeps the default.
    """
    from ..evaluation import (
        load_song_meta, load_eval_ground_truth, rebuild_hgt_recommender_from_disk)

    g = in_memory or {}
    log = print if verbose else (lambda *_: None)

    song_meta = g.get("song_meta")
    if song_meta is None:
        song_meta = load_song_meta(kg_input_path)
        log(f"[rehydrate] song_meta ← {Path(kg_input_path).name}")

    if all(n in g for n in ("train_seen", "test_gt", "test_users")):
        train_seen, test_gt, test_users = g["train_seen"], g["test_gt"], g["test_users"]
    else:
        train_seen, test_gt, test_users = load_eval_ground_truth(splits_dir)
        log(f"[rehydrate] train_seen/test_gt/test_users ← {Path(splits_dir).name}/")

    need = ("result", "data", "edge_dict", "_user_to_kg",
            "_song2kg", "_track_kg_to_song", "idx2song")
    if all(n in g for n in need):
        result, data, edge_dict, idx2song = (g["result"], g["data"],
                                             g["edge_dict"], g["idx2song"])
        u2k, s2k, tk2s = g["_user_to_kg"], g["_song2kg"], g["_track_kg_to_song"]
    elif Path(hgt_model_path).exists():
        h = rebuild_hgt_recommender_from_disk(
            hgt_model_path=hgt_model_path, kge_rotate_path=kge_rotate_path,
            edge_dict_path=edge_dict_path, kg_input_path=kg_input_path,
            ae_embeddings_path=ae_embeddings_path, splits_dir=splits_dir,
            device=device,
            **({} if num_heads is None else {"num_heads": int(num_heads)}))
        data, edge_dict, idx2song = h["data"], h["edge_dict"], h["idx2song"]
        u2k, s2k, tk2s = h["user_to_kg"], h["song2kg"], h["track_kg_to_song"]
        result = g.get("result")
        if result is None:
            import types
            result = types.SimpleNamespace(model=h["model"])
    else:
        return None

    return HGTContext(song_meta, train_seen, test_gt, test_users, result, data,
                      edge_dict, idx2song, u2k, s2k, tk2s)


__all__ = ["HGTContext", "ensure_hgt_context"]
