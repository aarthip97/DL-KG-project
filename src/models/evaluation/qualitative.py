"""Model-agnostic qualitative analysis (per-user + population-level).

The cell-level §6.7 / §6.8 logic that used to be hard-coded against the KNN
recommender now lives here as two reusable functions:

* :func:`analyze_user`       — single-user explanation (reference profile,
  cosine similarity, attribute-distribution comparison).
* :func:`analyze_population` — runs the same comparison across every test
  user and returns a per-user :class:`pd.DataFrame` ready for aggregation
  or statistical testing.

Both accept any object that implements the
:class:`~src.models.evaluation.recommenders.Recommender` protocol, so the same
code drives MostPopular, KNN-CF, HGT, and any future model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm

from .recommenders import Recommender


# ─────────────────────────────────────────────────────────────────────────────
#  Shared attribute lookup arrays  (one-time pre-computation)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AttributeArrays:
    """Vectorised song-attribute lookup tables (indexed by ``s_idx``).

    Build once with :meth:`from_song_meta` and re-use across all models.
    """
    genre:        np.ndarray   # int16 categorical codes
    mode:         np.ndarray
    tempo_class:  np.ndarray
    decade:       np.ndarray
    mean_tempo:   np.ndarray   # float32, may contain NaNs
    mean_decade:  np.ndarray   # float32, may contain NaNs
    genre_cats:   np.ndarray
    mode_cats:    np.ndarray
    tempo_cats:   np.ndarray
    decade_cats:  np.ndarray
    n_songs:      int

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def n_genre(self):  return len(self.genre_cats)
    @property
    def n_mode(self):   return len(self.mode_cats)
    @property
    def n_tempo(self):  return len(self.tempo_cats)
    @property
    def n_decade(self): return len(self.decade_cats)

    @classmethod
    def from_song_meta(
        cls,
        song_meta: pd.DataFrame,
        idx2song: Mapping[int, str],
        *,
        genre_col: str = "primary_genre",
        mode_col: str = "mode",
        tempo_class_col: str = "tempo_class",
        year_col: str = "year",
        mean_tempo_col: str = "Mean_Tempo",
    ) -> "AttributeArrays":
        n = max(idx2song.keys()) + 1
        sids = [idx2song.get(i) for i in range(n)]

        df = song_meta.copy()
        df["__decade"] = (pd.to_numeric(df[year_col], errors="coerce")
                          .fillna(0).astype(int) // 10 * 10).astype(str)
        df["__genre"]  = df[genre_col].fillna("unk").astype(str)
        df["__mode"]   = df[mode_col].fillna("unk").astype(str)
        df["__tempo"]  = df[tempo_class_col].fillna("unk").astype(str)

        def _enc(col: str, default="unk"):
            vals = [str(df.at[s, col]) if (s is not None and s in df.index) else default
                    for s in sids]
            cats = np.array(sorted(set(vals)))
            enc  = {c: np.int16(i) for i, c in enumerate(cats)}
            arr  = np.array([enc[v] for v in vals], dtype=np.int16)
            return arr, cats

        genre_arr,  genre_cats  = _enc("__genre")
        mode_arr,   mode_cats   = _enc("__mode")
        tempo_arr,  tempo_cats  = _enc("__tempo")
        decade_arr, decade_cats = _enc("__decade")

        mean_tempo = np.array(
            [float(df.at[s, mean_tempo_col])
             if s is not None and s in df.index and pd.notna(df.at[s, mean_tempo_col])
             else np.nan for s in sids], dtype=np.float32)
        mean_decade = np.array(
            [float(df.at[s, "__decade"])
             if s is not None and s in df.index and df.at[s, "__decade"] not in ("0", "unk", "nan")
             else np.nan for s in sids], dtype=np.float32)

        return cls(
            genre=genre_arr, mode=mode_arr, tempo_class=tempo_arr, decade=decade_arr,
            mean_tempo=mean_tempo, mean_decade=mean_decade,
            genre_cats=genre_cats, mode_cats=mode_cats,
            tempo_cats=tempo_cats, decade_cats=decade_cats, n_songs=n,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Vectorised primitives
# ─────────────────────────────────────────────────────────────────────────────
def _js_div(a: np.ndarray, b: np.ndarray, n_cat: int) -> float:
    """JS divergence between two integer-encoded 1-D arrays via ``np.bincount``."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    p = np.bincount(a.astype(np.intp), minlength=n_cat).astype(np.float64)
    q = np.bincount(b.astype(np.intp), minlength=n_cat).astype(np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def _dominant(enc: np.ndarray, cats: np.ndarray, n_cat: int) -> str:
    if len(enc) == 0:
        return "unk"
    return str(cats[np.bincount(enc.astype(np.intp), minlength=n_cat).argmax()])


def _cosine_to_profile(
    train_si: np.ndarray, top_si: np.ndarray,
    song_vectors, song_norms: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of recommendations to the user's mean training profile.

    Computed in the user-co-occurrence space using sparse-dense dot products
    (no full ``.toarray()`` of the rec matrix).
    """
    sv_train = song_vectors[train_si]
    p_sum    = np.asarray(sv_train.sum(axis=0)).ravel()
    p_norm   = np.linalg.norm(p_sum) + 1e-12
    sv_recs  = song_vectors[top_si]
    dots     = np.asarray(sv_recs.dot(p_sum)).ravel()
    return dots / (song_norms[top_si] * p_norm + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-user qualitative summary  (the §6.7 routine, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_user(
    user: int,
    recommender: Recommender,
    *,
    top_n: int,
    train_seen: Mapping[int, Set[int]],
    test_gt:    Mapping[int, Set[int]],
    attrs: AttributeArrays,
    song_vectors,
    song_norms: np.ndarray,
) -> dict:
    """Per-user explanation: ground-truth hits + cosine + attribute drift.

    Returns a small dict suitable for ``pd.DataFrame([analyze_user(...)])``.
    """
    train_si = np.fromiter(train_seen.get(user, ()), dtype=np.int32)
    test_si  = np.fromiter(test_gt.get(user,    ()), dtype=np.int32)
    if len(train_si) == 0:
        return {}

    recs = recommender.recommend([user], top_n).get(user, [])
    top_si = np.asarray(recs, dtype=np.int32)

    cos_u = (_cosine_to_profile(train_si, top_si, song_vectors, song_norms)
             if len(top_si) else np.array([np.nan]))

    return {
        "model":   recommender.name,
        "u_idx":   user,
        "n_train": len(train_si),
        "n_test":  len(test_si),
        "n_hits":  int(len(np.intersect1d(top_si, test_si))) if len(top_si) else 0,
        "cos_mean":   float(np.nanmean(cos_u)),
        "cos_median": float(np.nanmedian(cos_u)),
        "cos_max":    float(np.nanmax(cos_u)),
        "js_genre":       _js_div(attrs.genre[train_si],  attrs.genre[top_si],  attrs.n_genre),
        "js_mode":        _js_div(attrs.mode[train_si],   attrs.mode[top_si],   attrs.n_mode),
        "js_tempo_class": _js_div(attrs.tempo_class[train_si], attrs.tempo_class[top_si], attrs.n_tempo),
        "js_decade":      _js_div(attrs.decade[train_si], attrs.decade[top_si], attrs.n_decade),
        "dom_genre_train": _dominant(attrs.genre[train_si], attrs.genre_cats, attrs.n_genre),
        "dom_genre_rec":   _dominant(attrs.genre[top_si],   attrs.genre_cats, attrs.n_genre),
        "dom_mode_train":  _dominant(attrs.mode[train_si],  attrs.mode_cats,  attrs.n_mode),
        "dom_mode_rec":    _dominant(attrs.mode[top_si],    attrs.mode_cats,  attrs.n_mode),
        "mean_tempo_train":  float(np.nanmean(attrs.mean_tempo[train_si])) if len(train_si) else np.nan,
        "mean_tempo_rec":    float(np.nanmean(attrs.mean_tempo[top_si]))   if len(top_si)   else np.nan,
        "mean_decade_train": float(np.nanmean(attrs.mean_decade[train_si])) if len(train_si) else np.nan,
        "mean_decade_rec":   float(np.nanmean(attrs.mean_decade[top_si]))   if len(top_si)   else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Population-level summary  (the §6.8 routine, model-agnostic + parallel)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_population(
    recommender: Recommender,
    test_users: Sequence[int],
    *,
    top_n: int,
    train_seen: Mapping[int, Set[int]],
    test_gt:    Mapping[int, Set[int]],
    attrs: AttributeArrays,
    song_vectors,
    song_norms: np.ndarray,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run :func:`analyze_user` over every user in ``test_users`` (threaded)."""
    import os
    n_workers = n_workers or min(os.cpu_count() or 4, 10)

    # Bulk-recommend once: most recommenders amortise this much better than
    # one call per user.
    bulk = recommender.recommend(test_users, top_n)

    def _one(u):
        train_si = np.fromiter(train_seen.get(u, ()), dtype=np.int32)
        test_si  = np.fromiter(test_gt.get(u, ()), dtype=np.int32)
        if len(train_si) == 0:
            return None
        top_si = np.asarray(bulk.get(u, []), dtype=np.int32)
        cos_u = (_cosine_to_profile(train_si, top_si, song_vectors, song_norms)
                 if len(top_si) else np.array([np.nan]))
        rec = {
            "model":   recommender.name,
            "u_idx":   u,
            "n_train": len(train_si),
            "n_test":  len(test_si),
            "n_hits":  int(len(np.intersect1d(top_si, test_si))) if len(top_si) else 0,
            "cos_mean":   float(np.nanmean(cos_u)),
            "cos_median": float(np.nanmedian(cos_u)),
            "cos_max":    float(np.nanmax(cos_u)),
            "js_genre":       _js_div(attrs.genre[train_si],       attrs.genre[top_si],       attrs.n_genre),
            "js_mode":        _js_div(attrs.mode[train_si],        attrs.mode[top_si],        attrs.n_mode),
            "js_tempo_class": _js_div(attrs.tempo_class[train_si], attrs.tempo_class[top_si], attrs.n_tempo),
            "js_decade":      _js_div(attrs.decade[train_si],      attrs.decade[top_si],      attrs.n_decade),
            "dom_genre_train": _dominant(attrs.genre[train_si], attrs.genre_cats, attrs.n_genre),
            "dom_genre_rec":   _dominant(attrs.genre[top_si],   attrs.genre_cats, attrs.n_genre),
            "dom_mode_train":  _dominant(attrs.mode[train_si],  attrs.mode_cats,  attrs.n_mode),
            "dom_mode_rec":    _dominant(attrs.mode[top_si],    attrs.mode_cats,  attrs.n_mode),
            "mean_tempo_train":  float(np.nanmean(attrs.mean_tempo[train_si])) if len(train_si) else np.nan,
            "mean_tempo_rec":    float(np.nanmean(attrs.mean_tempo[top_si]))   if len(top_si)   else np.nan,
            "mean_decade_train": float(np.nanmean(attrs.mean_decade[train_si])) if len(train_si) else np.nan,
            "mean_decade_rec":   float(np.nanmean(attrs.mean_decade[top_si]))   if len(top_si)   else np.nan,
        }
        if len(test_si):
            rec["js_genre_test"]  = _js_div(attrs.genre[train_si],  attrs.genre[test_si],  attrs.n_genre)
            rec["js_mode_test"]   = _js_div(attrs.mode[train_si],   attrs.mode[test_si],   attrs.n_mode)
            rec["js_decade_test"] = _js_div(attrs.decade[train_si], attrs.decade[test_si], attrs.n_decade)
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_one, u) for u in test_users]
        it = as_completed(futs)
        if show_progress:
            it = tqdm(it, total=len(futs), desc=f"pop-qual[{recommender.name}]")
        for fut in it:
            r = fut.result()
            if r is not None:
                out.append(r)

    df = pd.DataFrame(out)
    if not df.empty:
        df["dom_genre_match"] = df["dom_genre_train"].str.lower() == df["dom_genre_rec"].str.lower()
        df["dom_mode_match"]  = df["dom_mode_train"].str.lower()  == df["dom_mode_rec"].str.lower()
    return df
