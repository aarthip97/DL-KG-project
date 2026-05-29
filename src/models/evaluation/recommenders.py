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


# ─── 3. XGBoost hybrid recommender ───────────────────────────────────────────

class XGBHybridRecommender:
    """Re-rank AE-based candidates with a trained XGBoost LTR model.

    Parameters
    ----------
    model:
        Trained ``xgboost.Booster`` (``rank:ndcg`` objective).
    ae_matrix:
        (n_songs, emb_dim) float32 array of track AE embeddings.
    train_seen:
        ``{u_idx: {s_idx, …}}`` — training interactions per user (for masking
        and user-profile construction).
    n_candidates:
        Candidate pool size retrieved by the AE dot-product stage before XGBoost
        re-ranking.  Should be ≥ the largest top_n you will request.
    """

    name = "XGBoost-Hybrid"

    def __init__(
        self,
        model: "xgboost.Booster",  # type: ignore[name-defined]
        ae_matrix: np.ndarray,
        train_seen: Mapping[int, Set[int]],
        n_candidates: int = 200,
    ):
        self._model = model
        self._ae = np.asarray(ae_matrix, dtype=np.float32)
        self._seen = train_seen
        self._n_cands = n_candidates

    def recommend(
        self,
        users: Sequence[int],
        top_n: int,
        batch_size: int = 512,
    ) -> Dict[int, List[int]]:
        """Return top_n recommendations per user, processed in batches.

        batch_size controls peak memory: 512 users × 200 candidates × 256 features
        ≈ 105 MB per batch — safe on a 16 GB machine.
        """
        import xgboost as xgb
        from models.xgb_hybrid import _build_user_profiles, _candidates_ae

        users = list(users)
        n_cands = max(self._n_cands, top_n)
        # Build profiles once for all users so batches share the same array
        profiles = _build_user_profiles(users, self._seen, self._ae)
        out: Dict[int, List[int]] = {}

        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            cands = _candidates_ae(batch, profiles, self._ae,
                                   n_candidates=n_cands, seen_dict=self._seen)

            u_list: List[int] = []
            s_list: List[int] = []
            for u in sorted(batch):
                for s in cands.get(u, []):
                    u_list.append(u)
                    s_list.append(s)

            if not u_list:
                for u in batch:
                    out[u] = []
                continue

            u_arr = np.asarray(u_list, dtype=np.int64)
            s_arr = np.asarray(s_list, dtype=np.int64)
            vu = np.clip(u_arr, 0, profiles.shape[0] - 1)
            vs = np.clip(s_arr, 0, self._ae.shape[0] - 1)
            X = np.hstack([profiles[vu], self._ae[vs]])

            scores = self._model.predict(xgb.DMatrix(data=X))

            pos = 0
            for u in sorted(batch):
                n = len(cands.get(u, []))
                if n == 0:
                    out[u] = []
                    pos += n
                    continue
                u_scores = scores[pos : pos + n]
                order = np.argsort(u_scores)[::-1]
                out[u] = np.asarray(cands[u])[order[:top_n]].tolist()
                pos += n

        return out


# ─── 4. HGT recommender ───────────────────────────────────────────────────────

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
