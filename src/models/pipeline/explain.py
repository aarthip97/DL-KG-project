"""Per-user explanation glue for the pipeline notebook (§12).

Builds an :class:`~models.evaluation.explainability.HGTExplainer` over the
training-consistent undirected graph, wires the track-label / artist-name maps
from ``song_meta`` + index bridges, and exposes a small
:class:`UserExplainer` that turns a ``u_idx`` into its top recommendation's
faithful-attention :class:`Explanation`. Keeps the §12 cell to a display loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..evaluation.explainability import HGTExplainer, Explanation


def build_track_label_fn(song_meta, data, *, track_kg_to_song, idx2song):
    """``(track_label_fn, label_overrides)`` from song metadata + index bridges.

    ``track_label_fn(track_kg) -> "title — artist"``; ``label_overrides`` maps
    artist KG nodes to a representative name (via the ``performed_by`` edges).
    """
    sid2artist = (song_meta["artist_name"].astype(str).to_dict()
                  if "artist_name" in song_meta.columns else {})
    sid2title = (song_meta["title"].astype(str).to_dict()
                 if "title" in song_meta.columns else {})

    def track_label(track_kg):
        s = track_kg_to_song.get(int(track_kg))
        sid = idx2song.get(s) if s is not None else None
        if sid is None:
            return None
        return f"{sid2title.get(sid, sid)} — {sid2artist.get(sid, '?')}"

    artist_name: dict = {}
    if ("track", "performed_by", "artist") in data.edge_types:
        pa = data["track", "performed_by", "artist"].edge_index
        for ti, ai in zip(pa[0].tolist(), pa[1].tolist()):
            if ai in artist_name:
                continue
            s = track_kg_to_song.get(ti)
            nm = sid2artist.get(idx2song.get(s)) if s is not None else None
            if nm:
                artist_name[ai] = nm
    return track_label, {"artist": artist_name}


@dataclass
class UserExplainer:
    """Wraps an :class:`HGTExplainer` with the maps to explain a user's top pick."""
    explainer: HGTExplainer
    user_to_kg: Mapping[int, int]
    seen_kg: Mapping[int, set]
    test_kg: Mapping[int, set]

    def top_track_kg(self, user_kg: int, k: int = 1) -> list:
        """Top-``k`` unseen tracks for ``user_kg`` by embedding dot-product."""
        import torch
        u_emb = self.explainer.emb[self.explainer.user_type]
        t_emb = self.explainer.emb[self.explainer.item_type]
        scores = t_emb @ u_emb[user_kg]
        for t in self.seen_kg.get(user_kg, ()):
            scores[t] = float("-inf")
        return torch.topk(scores, k).indices.tolist()

    def explain_user(self, u_idx: int, *, top_k_anchors: int = 6) -> Optional[Explanation]:
        """Explain ``u_idx``'s top recommendation, or ``None`` if it has no KG node."""
        user_kg = self.user_to_kg.get(int(u_idx))
        if user_kg is None:
            return None
        tops = self.top_track_kg(user_kg, k=1)
        if not tops:
            return None
        track_kg = tops[0]
        is_hit = track_kg in self.test_kg.get(user_kg, set())
        return self.explainer.explain(user_kg, track_kg,
                                      top_k_anchors=top_k_anchors, is_hit=is_hit)


def build_user_explainer(
    model,
    data,
    *,
    edge_dict: Mapping[str, Any],
    song_meta,
    train_seen: Mapping[int, set],
    test_gt: Mapping[int, set],
    user_to_kg: Mapping[int, int],
    song2kg: Mapping[int, int],
    track_kg_to_song: Mapping[int, int],
    idx2song: Mapping[int, str],
    undirected: bool = True,
) -> UserExplainer:
    """Capture attention on the undirected graph and wrap it as a UserExplainer."""
    import torch_geometric.transforms as T

    g = T.ToUndirected(merge=False)(data) if undirected else data
    track_label_fn, overrides = build_track_label_fn(
        song_meta, data, track_kg_to_song=track_kg_to_song, idx2song=idx2song)
    explainer = HGTExplainer.from_model(
        model=model, data=g, node_mappings=edge_dict["node_mappings"],
        track_label_fn=track_label_fn, label_overrides=overrides)

    seen_kg = {user_to_kg[u]: {song2kg[s] for s in seen if s in song2kg}
               for u, seen in train_seen.items() if u in user_to_kg}
    test_kg = {user_to_kg[u]: {song2kg[s] for s in gt if s in song2kg}
               for u, gt in test_gt.items() if u in user_to_kg}
    return UserExplainer(explainer=explainer, user_to_kg=user_to_kg,
                         seen_kg=seen_kg, test_kg=test_kg)


__all__ = ["UserExplainer", "build_track_label_fn", "build_user_explainer"]
