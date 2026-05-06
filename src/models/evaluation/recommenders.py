"""Uniform recommender interface so every model can be evaluated identically.

Each concrete recommender implements::

    recommend(users: Sequence[int], top_n: int) -> dict[int, list[int]]

The returned dict maps user index → ordered list of recommended item indices,
**already filtered of items the user saw during training**. Downstream metric
and qualitative-analysis code consumes the same shape regardless of model.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Set

import numpy as np


class Recommender(Protocol):
    """Minimal duck-typed interface every model has to satisfy."""

    name: str

    def recommend(
        self, users: Sequence[int], top_n: int,
    ) -> Dict[int, List[int]]: ...


# ─── 1. Popularity baseline ───────────────────────────────────────────────────

class PopularityRecommender:
    """Recommend the globally most popular unseen items."""

    name = "MostPopular"

    def __init__(
        self,
        song_popularity: np.ndarray,
        train_seen: Mapping[int, Set[int]],
    ):
        # Pre-sort once: index 0 is the most popular song.
        self._pop_rank = np.argsort(-np.asarray(song_popularity)).astype(int)
        self._train_seen = train_seen

    def recommend(self, users, top_n):
        out: Dict[int, List[int]] = {}
        for u in users:
            seen = self._train_seen.get(int(u), set())
            picks: List[int] = []
            for s in self._pop_rank:
                if int(s) not in seen:
                    picks.append(int(s))
                    if len(picks) >= top_n:
                        break
            out[int(u)] = picks
        return out


# ─── 2. KNN-CF baseline ───────────────────────────────────────────────────────

class KNNRecommender:
    """User-user KNN scoring on a pre-computed neighbour table.

    Re-uses the cached ``all_nbrs`` / ``qrow`` produced by ``run_knn_sweep``,
    so no new sklearn fit is required.
    """

    name = "KNN-CF"

    def __init__(
        self,
        train_matrix_norm,
        train_seen: Mapping[int, Set[int]],
        all_nbrs: np.ndarray,
        qrow: Mapping[int, int],
        best_k: int,
        pop_norm: Optional[np.ndarray] = None,
    ):
        self._tm = train_matrix_norm
        self._train_seen = train_seen
        self._all_nbrs = all_nbrs
        self._qrow = qrow
        self._best_k = int(best_k)
        # Used as a popularity fallback for users without a precomputed row.
        self._pop_norm = pop_norm

    def recommend(self, users, top_n):
        out: Dict[int, List[int]] = {}
        for u in users:
            u = int(u)
            seen = self._train_seen.get(u, set())
            if u in self._qrow:
                nbrs = [x for x in self._all_nbrs[self._qrow[u]] if x != u][:self._best_k]
                if not nbrs:
                    out[u] = []
                    continue
                sc = np.asarray(self._tm[nbrs].sum(axis=0)).ravel()
            elif self._pop_norm is not None:
                sc = self._pop_norm.copy()
            else:
                out[u] = []
                continue
            for s in seen:
                sc[s] = 0.0
            top = np.argpartition(sc, -top_n)[-top_n:]
            out[u] = top[np.argsort(sc[top])[::-1]].astype(int).tolist()
        return out


# ─── 3. HGT recommender ───────────────────────────────────────────────────────

class HGTRecommender:
    """Score user × track inner-products from frozen HGT embeddings.

    Parameters
    ----------
    user_emb, track_emb : ``np.ndarray`` (CPU, float32)
        Final node embeddings for the *track-recommendation* subgraph.
    train_seen : ``{user_kg_idx: {track_kg_idx, ...}}``
        Items to mask before computing the top-``K``.
    user_to_kg, track_kg_to_song : mappings between the dataset's integer
        ``u_idx`` / ``s_idx`` and the HeteroData node indices, plus the inverse
        for tracks (so the returned recs use the dataset's ``s_idx``).
    """

    name = "HGT"

    def __init__(
        self,
        user_emb: np.ndarray,
        track_emb: np.ndarray,
        train_seen_kg: Mapping[int, Set[int]],
        user_to_kg: Mapping[int, int],
        track_kg_to_song: Mapping[int, int],
    ):
        self._U = np.asarray(user_emb, dtype=np.float32)
        self._T = np.asarray(track_emb, dtype=np.float32)
        self._seen = train_seen_kg
        self._u_map = user_to_kg
        self._t_map = track_kg_to_song

    def recommend(self, users, top_n):
        out: Dict[int, List[int]] = {}
        for u in users:
            u = int(u)
            kg_u = self._u_map.get(u)
            if kg_u is None:
                out[u] = []
                continue
            scores = self._T @ self._U[kg_u]
            for kg_t in self._seen.get(kg_u, set()):
                scores[kg_t] = -np.inf
            top_kg = np.argpartition(scores, -top_n)[-top_n:]
            top_kg = top_kg[np.argsort(scores[top_kg])[::-1]]
            out[u] = [self._t_map[int(t)] for t in top_kg if int(t) in self._t_map]
        return out
