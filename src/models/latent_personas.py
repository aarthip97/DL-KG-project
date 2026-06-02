"""Persona-cluster cold-start recommender on top of the HGT latent space.

Everything here is **pure inference** over embeddings the trained HGT already
produced — no retraining, no gradient steps.  It turns the multi-type latent
space (the same space :mod:`models.latent_space` clusters with a GMM) into an
onboarding flow:

1. :func:`build_attribute_vectors` — every *attribute* node (genre / tempo /
   mode / key / decade / instrument) already has a 64-d embedding, so we index
   them ``{type: {label: vector}}``.  These are the semantic anchors a brand-new
   user can point at.
2. :func:`build_track_table` — the recommendable pool: each KG track's latent
   vector aligned to its song metadata (title / artist / genre / …).
3. :func:`summarise_clusters` — turns each GMM centroid into a *persona*: the
   dominant genre / tempo / mode / decade of its nearest tracks plus its nearest
   semantic-anchor nodes, and (optionally) its node-type composition.
4. :func:`taste_vector_from_selections` — average the embeddings of the
   attributes a user ticks on a form → one synthetic *taste vector* in the same
   space.
5. :func:`assign_persona` + :func:`recommend_for_vector` — snap that vector to
   the closest persona centroid and rank the nearest tracks.
6. :func:`explain_recommendations` — for every recommended track, which of the
   user's stated preferences it shares and how strongly it aligns (cosine in the
   HGT space) — the explainability story, extended to cold-start users.

:class:`PersonaPack` bundles all of the above so it can be pickled once and
reloaded after a kernel restart (the GMM/UMAP cell does not have to re-run).
:func:`launch_persona_gui` wires it to an ``ipywidgets`` form for a live demo.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Attribute node types a user can express a preference over, in priority order.
ATTR_TYPES: Tuple[str, ...] = ("genre", "tempo_class", "mode", "key",
                               "decade", "instrument")

# Default per-type weight when blending selected attributes into a taste vector.
# Genre carries the most taste signal; key/decade the least.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre": 1.0, "instrument": 0.7, "tempo_class": 0.5,
    "mode": 0.4, "decade": 0.4, "key": 0.3,
}

_NODE_ID_RE = re.compile(r"^(?P<type>.+)_(?P<idx>\d+)$")


# ─── small helpers ────────────────────────────────────────────────────────────

def _split_node_id(node_id: str) -> Tuple[str, int]:
    """``"tempo_class_5"`` → ``("tempo_class", 5)``."""
    m = _NODE_ID_RE.match(str(node_id))
    if not m:
        raise ValueError(f"un-parseable Node_ID {node_id!r}")
    return m.group("type"), int(m.group("idx"))


def _uri_label(uri: str) -> str:
    """Last path segment of a resource URI (``.../genre/rock`` → ``rock``)."""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _l2norm(M: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Row-wise L2 normalisation (so dot products become cosine sims)."""
    M = np.asarray(M, dtype=np.float32)
    if M.ndim == 1:
        n = np.linalg.norm(M)
        return M / max(n, eps)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.clip(n, eps, None)


def _norm_label(x) -> str:
    """Case/format-insensitive key for matching metadata against KG labels."""
    return re.sub(r"[\s_]+", "", str(x).strip().lower())


def _year_to_decade(year) -> Optional[str]:
    """``1993`` → ``"1990s"`` (``None`` when the year is missing/invalid)."""
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return None
    if y <= 0:
        return None
    return f"{(y // 10) * 10}s"


# ─── 1. attribute (anchor) vectors ────────────────────────────────────────────

def type_embedding_matrix(emb_df: pd.DataFrame, node_type: str) -> np.ndarray:
    """All embeddings of one node type, ordered by KG node index.

    Args:
        emb_df: output of :func:`models.latent_space.extract_node_embeddings`
            (columns ``Node_ID``, ``Node_Type``, ``Embedding``).
        node_type: lower-case KG type, e.g. ``"track"`` or ``"tempo_class"``.

    Returns:
        ``(n_nodes, d)`` float32 array; empty ``(0, 0)`` when the type is absent.
    """
    sub = emb_df[emb_df["Node_Type"].str.lower() == node_type.lower()]
    if sub.empty:
        return np.zeros((0, 0), dtype=np.float32)
    idx = np.fromiter((_split_node_id(nid)[1] for nid in sub["Node_ID"]),
                      dtype=np.int64, count=len(sub))
    order = np.argsort(idx)
    return np.vstack(sub["Embedding"].to_numpy()[order]).astype(np.float32)


def build_attribute_vectors(
    emb_df: pd.DataFrame,
    edge_dict: Mapping,
    *,
    attr_types: Sequence[str] = ATTR_TYPES,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Index every attribute node as ``{type: {label: latent_vector}}``.

    The label is the resource URI's last segment, taken from
    ``edge_dict["node_mappings"][type]`` (index-aligned to the embeddings).
    """
    node_mappings = edge_dict["node_mappings"]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for nt in attr_types:
        uris = node_mappings.get(nt)
        if not uris:
            continue
        M = type_embedding_matrix(emb_df, nt)
        if len(M) == 0:
            continue
        out[nt] = {_uri_label(uris[i]): M[i] for i in range(min(len(uris), len(M)))}
    return out


# ─── 2. recommendable track table ─────────────────────────────────────────────

def build_track_table(
    emb_df: pd.DataFrame,
    *,
    track_kg_to_song: Mapping[int, int],
    idx2song: Mapping[int, str],
    song_meta: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Track latent vectors aligned to song metadata.

    Only KG tracks that resolve all the way to a ``song_meta`` row are kept, so
    every recommendable track has a title/artist/genre to show and explain.

    Returns:
        ``(track_emb, track_meta)`` — ``track_emb`` is ``(n, d)`` float32 and
        ``track_meta`` is a row-aligned frame with ``kg_idx``, ``song_id``,
        ``title``, ``artist``, ``genre``, ``tempo_class``, ``mode``, ``key``,
        ``decade``.
    """
    M = type_embedding_matrix(emb_df, "track")
    rows: List[dict] = []
    vecs: List[np.ndarray] = []
    for kg_idx in range(len(M)):
        s_idx = track_kg_to_song.get(kg_idx)
        if s_idx is None:
            continue
        song_id = idx2song.get(s_idx)
        if song_id is None or song_id not in song_meta.index:
            continue
        meta = song_meta.loc[song_id]
        if isinstance(meta, pd.DataFrame):          # de-dupe safety
            meta = meta.iloc[0]
        rows.append({
            "kg_idx": kg_idx,
            "s_idx": int(s_idx),
            "song_id": song_id,
            "title": meta.get("title"),
            "artist": meta.get("artist_name"),
            "genre": meta.get("primary_genre"),
            "tempo_class": meta.get("tempo_class"),
            "mode": meta.get("mode"),
            "key": meta.get("key"),
            "decade": _year_to_decade(meta.get("year")),
        })
        vecs.append(M[kg_idx])
    if not vecs:
        raise ValueError("no KG track mapped through to song_meta — check the "
                         "track_kg_to_song / idx2song / song_meta bridges")
    return np.vstack(vecs).astype(np.float32), pd.DataFrame(rows)


# ─── 3. cluster → persona ─────────────────────────────────────────────────────

def _top_counts(series: pd.Series, top: int) -> List[Tuple[str, int]]:
    vc = series.dropna().astype(str).value_counts()
    return [(k, int(v)) for k, v in vc.head(top).items()]


def nearest_anchors(
    vec: np.ndarray,
    attr_vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    per_type: int = 1,
    types: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Closest attribute label(s) to ``vec`` per attribute type (cosine)."""
    v = _l2norm(vec)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for nt, d in attr_vectors.items():
        if types is not None and nt not in types:
            continue
        if not d:
            continue
        labels = list(d.keys())
        M = _l2norm(np.vstack([d[l] for l in labels]))
        sims = M @ v
        order = np.argsort(-sims)[:per_type]
        out[nt] = [(labels[i], float(sims[i])) for i in order]
    return out


def summarise_clusters(
    centroids: np.ndarray,
    track_emb: np.ndarray,
    track_meta: pd.DataFrame,
    attr_vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    composition: Optional[pd.DataFrame] = None,
    top_tracks: int = 200,
    top_labels: int = 3,
) -> pd.DataFrame:
    """Describe each GMM centroid as a persona.

    For every centroid we take its ``top_tracks`` nearest tracks (cosine) and
    read off the dominant genre / tempo / mode / decade, attach the nearest
    semantic-anchor node per attribute type, and — when ``composition`` (a
    ``cluster × Node_Type`` count frame, e.g. from ``pd.crosstab``) is given —
    its dominant node type and user count.  Returns one row per cluster.
    """
    Tn = _l2norm(track_emb)
    rows: List[dict] = []
    for c in range(len(centroids)):
        sims = Tn @ _l2norm(centroids[c])
        near = np.argsort(-sims)[:top_tracks]
        sub = track_meta.iloc[near]
        genres = _top_counts(sub["genre"], top_labels)
        tempos = _top_counts(sub["tempo_class"], top_labels)
        modes = _top_counts(sub["mode"], top_labels)
        decades = _top_counts(sub["decade"], top_labels)
        anchors = nearest_anchors(centroids[c], attr_vectors, per_type=1,
                                  types=("genre", "tempo_class", "mode", "decade"))
        top_genre = genres[0][0] if genres else (anchors.get("genre", [("?", 0)])[0][0])
        top_tempo = tempos[0][0] if tempos else (anchors.get("tempo_class", [("?", 0)])[0][0])
        top_decade = decades[0][0] if decades else "?"
        name = " · ".join(str(x) for x in (top_genre, top_tempo, top_decade)
                          if x and x != "?")
        row = {
            "cluster": c,
            "persona": name or f"cluster-{c}",
            "top_genres": ", ".join(f"{g}({n})" for g, n in genres),
            "top_tempo": ", ".join(f"{t}({n})" for t, n in tempos),
            "top_mode": ", ".join(f"{m}({n})" for m, n in modes),
            "top_decades": ", ".join(f"{d}({n})" for d, n in decades),
            "anchors": ", ".join(f"{lab}" for v in anchors.values() for lab, _ in v),
        }
        if composition is not None and c in composition.index:
            comp = composition.loc[c]
            row["dominant_node_type"] = str(comp.idxmax())
            row["n_users"] = int(comp.get("User", 0))
            row["n_nodes"] = int(comp.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


# ─── 4-5. taste vector → persona → recommendations ────────────────────────────

def taste_vector_from_selections(
    selections: Mapping[str, object],
    attr_vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Blend the selected attribute embeddings into one taste vector.

    ``selections`` maps an attribute type to the chosen labels. Two forms are
    accepted (and may be mixed across types):

    * a plain list — ``{"genre": ["rock", "metal"]}`` — all picks weighted equally;
    * a ``{label: weight}`` mapping — ``{"genre": {"rock": 9, "classical": 10,
      "pop": 7}}`` — a per-item importance (any positive scale, e.g. 1–10).

    Within a type the picks are combined as a **weighted mean** of their anchor
    vectors (equal weights for the list form), then scaled by the per-*type*
    ``weights`` (defaults to :data:`DEFAULT_WEIGHTS` — genre counts more than key);
    the sum across types is L2-normalised. Unknown labels and non-positive
    weights are ignored.

    Raises:
        ValueError: if nothing valid was selected.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    dim = next((len(next(iter(d.values()))) for d in attr_vectors.values() if d), None)
    if dim is None:
        raise ValueError("attr_vectors is empty")
    acc = np.zeros(dim, dtype=np.float32)
    used = 0
    for nt, picks in selections.items():
        table = attr_vectors.get(nt)
        if not table or not picks:
            continue
        if isinstance(picks, Mapping):                      # {label: weight}
            items = [(l, float(w)) for l, w in picks.items() if l in table and float(w) > 0]
        else:                                               # [label, ...] → equal weights
            items = [(l, 1.0) for l in picks if l in table]
        if not items:
            continue
        wsum = sum(w for _, w in items) or 1.0
        vec = sum(w * table[l] for l, w in items) / wsum    # within-type weighted mean
        acc += float(weights.get(nt, 0.5)) * vec.astype(np.float32)
        used += 1
    if used == 0:
        raise ValueError("no recognised attributes selected")
    return _l2norm(acc)


def assign_persona(
    vec: np.ndarray, centroids: np.ndarray
) -> Tuple[int, List[Tuple[int, float]]]:
    """Closest persona centroid to ``vec`` by cosine.

    Returns ``(best_cluster, ranked)`` where ``ranked`` is every
    ``(cluster, cosine)`` sorted high→low.
    """
    sims = _l2norm(centroids) @ _l2norm(vec)
    order = np.argsort(-sims)
    ranked = [(int(c), float(sims[c])) for c in order]
    return ranked[0][0], ranked


def recommend_for_vector(
    vec: np.ndarray,
    track_emb: np.ndarray,
    track_meta: pd.DataFrame,
    *,
    k: int = 10,
    restrict_rows: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Top-``k`` tracks nearest to ``vec`` (cosine), as a metadata frame + ``sim``.

    ``restrict_rows`` optionally limits the candidate pool to a subset of
    ``track_meta`` row positions (e.g. a single persona's tracks).
    """
    sims = _l2norm(track_emb) @ _l2norm(vec)
    pool = (np.asarray(restrict_rows) if restrict_rows is not None
            else np.arange(len(track_meta)))
    pool = pool[np.argsort(-sims[pool])][:k]
    out = track_meta.iloc[pool].copy()
    out["sim"] = sims[pool]
    return out.reset_index(drop=True)


# ─── 5b. user-manifold cold start: persona-centroid vs k-NN user blend ────────

def nearest_users(
    query: np.ndarray,
    user_emb: np.ndarray,
    *,
    k: int = 15,
    temperature: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``k`` nearest real users to ``query`` (cosine) + soft blend weights.

    Args:
        query: a taste vector in the shared HGT latent space (e.g. the output of
            :func:`taste_vector_from_selections`).
        user_emb: ``(n_users, d)`` matrix of **user** node embeddings.
        k: neighbourhood size.
        temperature: softmax temperature on the cosine similarities — lower
            concentrates weight on the closest users, higher averages more evenly.

    Returns:
        ``(idx, weights, sims)`` — neighbour row indices (high→low similarity),
        their softmax blend weights (sum to 1) and their raw cosine similarities.
    """
    q = _l2norm(query)
    U = _l2norm(user_emb)
    sims = U @ q                                          # [n_users]
    k = int(min(max(k, 1), len(sims)))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    s = sims[idx]
    w = np.exp((s - s.max()) / max(temperature, 1e-6))
    w = (w / w.sum()).astype(np.float32)
    return idx.astype(np.int64), w, s.astype(np.float32)


def blend_user_embedding(
    query: np.ndarray,
    user_emb: np.ndarray,
    *,
    k: int = 15,
    temperature: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a brand-new user's embedding from existing users — no retrain.

    The new vector is the similarity-weighted mean of the ``k`` nearest real
    users (see :func:`nearest_users`), so it lands *on the learned user manifold*
    (inside its convex hull) rather than drifting into track/attribute regions.
    This is a non-parametric / Nadaraya–Watson inductive embedding: the HGT is
    never touched.

    Returns:
        ``(blended_vec, idx, weights, sims)`` — the L2-normalised blend plus the
        neighbour indices / weights / similarities backing it (for explanation).
    """
    idx, w, s = nearest_users(query, user_emb, k=k, temperature=temperature)
    blended = (w[:, None] * np.asarray(user_emb, dtype=np.float32)[idx]).sum(0)
    return _l2norm(blended), idx, w, s


def neighbor_dispersion(user_emb: np.ndarray, idx: np.ndarray) -> float:
    """Mean pairwise cosine among the chosen neighbour users (blend confidence).

    Near 1.0 → the neighbours agree, so the blend is a reliable estimate; lower →
    it is averaging dissimilar users, so the (coarser) persona centroid is safer.
    """
    idx = np.asarray(idx)
    if len(idx) < 2:
        return 1.0
    M = _l2norm(user_emb[idx])
    G = M @ M.T
    iu = np.triu_indices(len(idx), k=1)
    return float(G[iu].mean())


# ─── 6. explanation ───────────────────────────────────────────────────────────

def explain_recommendations(
    selections: Mapping[str, Sequence[str]],
    recs: pd.DataFrame,
    *,
    persona_id: int,
    personas: pd.DataFrame,
    ranked_personas: Sequence[Tuple[int, float]],
    attr_vectors: Mapping[str, Mapping[str, np.ndarray]],
    taste_vec: np.ndarray,
) -> dict:
    """Build a structured, faithful rationale for a recommendation set.

    For each recommended track we record which of the user's stated preferences
    (genre / tempo / mode / decade) it actually shares and its cosine alignment
    in the HGT latent space; plus the matched persona, the runner-up persona,
    and the user's nearest semantic anchors.
    """
    sel_norm = {nt: {_norm_label(x) for x in labs}
                for nt, labs in selections.items() if labs}
    match_cols = [("genre", "genre"), ("tempo_class", "tempo_class"),
                  ("mode", "mode"), ("decade", "decade")]

    per_track: List[dict] = []
    n_to_persona = 0
    persona_genres = {_norm_label(g.split("(")[0])
                      for g in str(personas.at[persona_id, "top_genres"]).split(",")} \
        if persona_id in personas.index else set()
    for _, r in recs.iterrows():
        shared: List[str] = []
        for sel_key, col in match_cols:
            want = sel_norm.get(sel_key)
            if want and _norm_label(r.get(col)) in want:
                shared.append(f"{col}={r.get(col)}")
        if _norm_label(r.get("genre")) in persona_genres:
            n_to_persona += 1
        per_track.append({
            "title": r.get("title"), "artist": r.get("artist"),
            "genre": r.get("genre"), "tempo_class": r.get("tempo_class"),
            "mode": r.get("mode"), "decade": r.get("decade"),
            "sim": float(r.get("sim", float("nan"))),
            "shared": shared, "n_shared": len(shared),
        })

    anchors = nearest_anchors(taste_vec, attr_vectors, per_type=1)
    persona_name = (personas.at[persona_id, "persona"]
                    if persona_id in personas.index else f"cluster-{persona_id}")
    runner = next(((c, s) for c, s in ranked_personas if c != persona_id), None)
    return {
        "persona_id": int(persona_id),
        "persona_name": persona_name,
        "persona_confidence": float(ranked_personas[0][1]) if ranked_personas else float("nan"),
        "runner_up": runner,
        "n_align_persona": int(n_to_persona),
        "n_recs": int(len(recs)),
        "anchors": {nt: v[0] for nt, v in anchors.items() if v},
        "tracks": per_track,
    }


def format_explanation_text(expl: dict) -> str:
    """Plain-text rendering of :func:`explain_recommendations`."""
    lines: List[str] = []
    conf = expl["persona_confidence"]
    lines.append(f"You map onto persona #{expl['persona_id']} — "
                 f"“{expl['persona_name']}” (fit {conf:.2f}).")
    if expl.get("runner_up"):
        rc, rs = expl["runner_up"]
        lines.append(f"Runner-up persona #{rc} (fit {rs:.2f}).")
    lines.append(f"{expl['n_align_persona']}/{expl['n_recs']} picks sit in that "
                 f"persona's core genres.")
    if expl.get("anchors"):
        anchor_str = ", ".join(f"{lab} ({sim:.2f})"
                               for lab, sim in expl["anchors"].values())
        lines.append(f"Nearest taste anchors: {anchor_str}.")
    lines.append("")
    for i, t in enumerate(expl["tracks"], 1):
        why = ("matches " + ", ".join(t["shared"])) if t["shared"] else "latent-space match"
        title = t["title"] or "?"
        artist = t["artist"] or "?"
        lines.append(f"  {i:>2}. {title} — {artist}  [sim={t['sim']:.3f}; {why}]")
    return "\n".join(lines)


def format_explanation_html(expl: dict) -> str:
    """HTML rendering for notebook display."""
    conf = expl["persona_confidence"]
    head = (f"<b>Persona #{expl['persona_id']} — “{expl['persona_name']}”</b> "
            f"<span style='color:#666'>(fit {conf:.2f}; "
            f"{expl['n_align_persona']}/{expl['n_recs']} picks on-persona)</span>")
    if expl.get("anchors"):
        head += "<br><span style='color:#666'>anchors: " + ", ".join(
            f"{lab} ({sim:.2f})" for lab, sim in expl["anchors"].values()) + "</span>"
    cells = []
    for i, t in enumerate(expl["tracks"], 1):
        tags = " ".join(
            f"<span style='background:#e8f0fe;border-radius:4px;padding:1px 5px;"
            f"margin:0 2px;font-size:11px'>{s}</span>" for s in t["shared"])
        cells.append(
            f"<tr><td style='text-align:right;color:#999'>{i}</td>"
            f"<td><b>{t['title'] or '?'}</b><br>"
            f"<span style='color:#666;font-size:12px'>{t['artist'] or '?'}</span></td>"
            f"<td style='font-size:12px'>{t['genre']} · {t['tempo_class']} · "
            f"{t['mode']} · {t['decade']}</td>"
            f"<td style='text-align:right'>{t['sim']:.3f}</td>"
            f"<td>{tags}</td></tr>")
    table = ("<table style='border-collapse:collapse;width:100%'>"
             "<tr style='border-bottom:1px solid #ccc;text-align:left'>"
             "<th></th><th>track</th><th>genre·tempo·mode·decade</th>"
             "<th>sim</th><th>shared with you</th></tr>" + "".join(cells) + "</table>")
    return f"<div style='font-family:sans-serif'>{head}<br><br>{table}</div>"


# ─── persona pack: bundle + persist ───────────────────────────────────────────

@dataclass
class PersonaPack:
    """Everything the cold-start recommender needs, picklable in one object."""

    attr_vectors: Dict[str, Dict[str, np.ndarray]]
    centroids: np.ndarray
    best_params: dict
    track_emb: np.ndarray
    track_meta: pd.DataFrame
    personas: pd.DataFrame
    track_cluster: np.ndarray
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    composition: Optional[pd.DataFrame] = None
    # ── user-manifold cold start (optional; set when a users-only GMM is built) ──
    user_emb: Optional[np.ndarray] = None          # (n_users, d) user embeddings
    user_centroids: Optional[np.ndarray] = None    # users-only GMM means (archetypes)
    user_personas: Optional[pd.DataFrame] = None   # summarise_clusters(user_centroids)
    user_composition: Optional[pd.DataFrame] = None
    blend_k: int = 15
    blend_temperature: float = 0.07

    # convenience: the labels a form can offer, sorted
    def options(self, attr_type: str) -> List[str]:
        return sorted(self.attr_vectors.get(attr_type, {}))

    @property
    def has_user_manifold(self) -> bool:
        """True when the users-only GMM artefacts are present (centroid/blend modes)."""
        return self.user_emb is not None and self.user_centroids is not None

    # ── user-manifold recommendation modes (all pure inference, no retrain) ──────
    def recommend_user_persona(
        self, selections: Mapping[str, Sequence[str]], *, k: int = 10,
    ) -> Tuple[pd.DataFrame, int, List[Tuple[int, float]]]:
        """Snap the form to the nearest **user archetype** centroid and recommend.

        Returns ``(recs, persona_id, ranked)`` — recs are the tracks nearest the
        matched users-only centroid.
        """
        if self.user_centroids is None:
            raise ValueError("pack has no user_centroids — build a users-only GMM first")
        vec = taste_vector_from_selections(selections, self.attr_vectors, weights=self.weights)
        pid, ranked = assign_persona(vec, self.user_centroids)
        recs = recommend_for_vector(self.user_centroids[pid], self.track_emb,
                                    self.track_meta, k=k)
        return recs, pid, ranked

    def recommend_user_blend(
        self, selections: Mapping[str, Sequence[str]], *, k: int = 10,
        k_users: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a bespoke new-user vector via :func:`blend_user_embedding`, recommend.

        Returns ``(recs, blend_vec, neighbor_idx, neighbor_weights, neighbor_sims)``.
        """
        if self.user_emb is None:
            raise ValueError("pack has no user_emb — build the user manifold first")
        vec = taste_vector_from_selections(selections, self.attr_vectors, weights=self.weights)
        b, idx, w, s = blend_user_embedding(
            vec, self.user_emb, k=int(k_users or self.blend_k),
            temperature=self.blend_temperature)
        recs = recommend_for_vector(b, self.track_emb, self.track_meta, k=k)
        return recs, b, idx, w, s

    def compare_user_modes(
        self, selections: Mapping[str, Sequence[str]], *, k: int = 10,
        k_users: Optional[int] = None,
    ) -> dict:
        """Run persona-centroid **and** k-NN blend, with a "which is more informative"
        diagnostic.

        The diagnostic quantifies how much the personalized blend adds over the
        coarse archetype:

        * ``residual`` = ``1 - cos(blend, centroid)`` — how far the blend sits from
          its archetype prototype (≈0 → persona already captures the user).
        * ``overlap`` = Jaccard of the two top-``k`` recommendation sets.
        * ``neighbor_dispersion`` = agreement among the blend's neighbour users.

        ``verdict`` is ``"blend"`` when the blend looks materially more individual
        (residual high or rec overlap low) **and** its neighbours are coherent,
        else ``"persona"``.
        """
        recs_p, pid, ranked = self.recommend_user_persona(selections, k=k)
        recs_b, b, idx, w, s = self.recommend_user_blend(selections, k=k, k_users=k_users)
        cvec = self.user_centroids[pid]
        residual = 1.0 - float(_l2norm(b) @ _l2norm(cvec))
        set_p, set_b = set(recs_p["kg_idx"].tolist()), set(recs_b["kg_idx"].tolist())
        overlap = len(set_p & set_b) / max(len(set_p | set_b), 1)
        disp = neighbor_dispersion(self.user_emb, idx)
        pname = (self.user_personas.at[pid, "persona"]
                 if self.user_personas is not None and pid in self.user_personas.index
                 else f"user-cluster-{pid}")
        individual = (residual >= 0.15) or (overlap <= 0.5)
        verdict = "blend" if (individual and disp >= 0.5) else "persona"
        return {
            "persona_id": int(pid), "persona_name": pname, "ranked": ranked,
            "recs_persona": recs_p, "recs_blend": recs_b,
            "blend_vec": b, "residual": residual, "overlap": overlap,
            "neighbor_idx": idx, "neighbor_weights": w, "neighbor_sims": s,
            "neighbor_dispersion": disp, "verdict": verdict,
        }

    def recommend(
        self,
        selections: Mapping[str, Sequence[str]],
        *,
        k: int = 10,
        restrict_to_persona: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """Form selections → ``(recommendations, explanation)``."""
        vec = taste_vector_from_selections(selections, self.attr_vectors,
                                           weights=self.weights)
        pid, ranked = assign_persona(vec, self.centroids)
        restrict = (np.where(self.track_cluster == pid)[0]
                    if restrict_to_persona else None)
        recs = recommend_for_vector(vec, self.track_emb, self.track_meta,
                                    k=k, restrict_rows=restrict)
        expl = explain_recommendations(
            selections, recs, persona_id=pid, personas=self.personas,
            ranked_personas=ranked, attr_vectors=self.attr_vectors, taste_vec=vec)
        return recs, expl


def build_persona_pack(
    emb_df: pd.DataFrame,
    edge_dict: Mapping,
    centroids: np.ndarray,
    best_params: dict,
    *,
    track_kg_to_song: Mapping[int, int],
    idx2song: Mapping[int, str],
    song_meta: pd.DataFrame,
    composition: Optional[pd.DataFrame] = None,
    weights: Optional[Mapping[str, float]] = None,
    top_tracks: int = 200,
    user_centroids: Optional[np.ndarray] = None,
    user_composition: Optional[pd.DataFrame] = None,
) -> PersonaPack:
    """Assemble a :class:`PersonaPack` from the latent-space artefacts.

    ``centroids`` are the all-node-type GMM means (``GMMResult.means``);
    ``composition`` is an optional ``cluster × Node_Type`` count frame
    (``pd.crosstab`` of the subsample's ``Cluster_ID`` and ``Node_Type``) used to
    label personas with their dominant node type and user count.

    When ``user_centroids`` (the means of a **users-only** GMM) are supplied, the
    pack also gains the user-manifold cold-start modes: every ``user`` node's
    embedding is indexed for the k-NN blend, and the user centroids are summarised
    into listener-archetype personas. ``user_composition`` optionally labels those
    archetypes (a ``cluster × Node_Type`` crosstab of the users-only subsample).
    """
    attr_vectors = build_attribute_vectors(emb_df, edge_dict)
    track_emb, track_meta = build_track_table(
        emb_df, track_kg_to_song=track_kg_to_song,
        idx2song=idx2song, song_meta=song_meta)
    personas = summarise_clusters(
        centroids, track_emb, track_meta, attr_vectors,
        composition=composition, top_tracks=top_tracks)
    # each track's own nearest persona (for optional persona-restricted recs)
    track_cluster = (_l2norm(track_emb) @ _l2norm(np.asarray(centroids)).T).argmax(1)

    # ── optional user manifold (users-only GMM) ─────────────────────────────────
    user_emb = type_embedding_matrix(emb_df, "user")
    user_emb = user_emb if len(user_emb) else None
    user_personas = None
    if user_centroids is not None:
        user_centroids = np.asarray(user_centroids, dtype=np.float32)
        user_personas = summarise_clusters(
            user_centroids, track_emb, track_meta, attr_vectors,
            composition=user_composition, top_tracks=top_tracks)

    return PersonaPack(
        attr_vectors=attr_vectors, centroids=np.asarray(centroids, dtype=np.float32),
        best_params=dict(best_params), track_emb=track_emb, track_meta=track_meta,
        personas=personas, track_cluster=np.asarray(track_cluster),
        weights=dict(DEFAULT_WEIGHTS if weights is None else weights),
        composition=composition,
        user_emb=user_emb, user_centroids=user_centroids,
        user_personas=user_personas, user_composition=user_composition)


def save_persona_pack(pack: PersonaPack, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pack, f)
    return path


def load_persona_pack(path) -> PersonaPack:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── faithful HGT-attention rationale for cold-start users ────────────────────

class ColdStartHGT:
    """Inductive cold-start recommender with **faithful HGT attention** rationale.

    A brand-new user has no node in the trained graph, so we add a *temporary*
    one and let the trained HGT build its embedding by real message passing:

    1. The form's taste vector selects ``seed_k`` seed tracks (nearest tracks in
       the persona/HGT space) — the synthetic user's stand-in listening history.
    2. A temporary user node is appended, fed a KGE-space feature averaged from
       the selected attribute nodes (the same RotatE space real users are fed),
       and wired to the seeds by ``track → user`` (reverse-listened) edges **only**
       — so every other node's embedding stays byte-identical to the base graph.
    3. One forward pass with :func:`capture_hgt_attention` yields the synthetic
       user's embedding **and** the genuine softmax attention on its incoming
       edges. Recommendations are the nearest tracks to that inductive embedding.
    4. :class:`~models.evaluation.explainability.HGTExplainer` is wrapped around
       the *same* captured attention, so ``explain(synthetic_user, rec)`` reuses
       the §12 machinery verbatim: its "anchors" are the seed tracks the model
       actually attended to, its "reasons" the attributes they share with the rec.

    Needs the live model + (directed or undirected) graph in memory; it is the
    attention-grounded counterpart to :meth:`PersonaPack.recommend`'s cosine path.
    """

    def __init__(
        self,
        model,
        data,
        node_mappings: Mapping[str, Sequence[str]],
        pack: PersonaPack,
        *,
        user_type: str = "user",
        item_type: str = "track",
    ) -> None:
        import torch
        from torch_geometric.transforms import ToUndirected

        # Ensure the reverse (track → user) relation exists.
        rev = next((et for et in data.edge_types
                    if et[0] == item_type and et[2] == user_type), None)
        if rev is None:
            data = ToUndirected(merge=False)(data)
            rev = next(et for et in data.edge_types
                       if et[0] == item_type and et[2] == user_type)
        fwd = next(et for et in data.edge_types
                   if et[0] == user_type and et[2] == item_type)

        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.base = data
        self.node_mappings = node_mappings
        self.pack = pack
        self.user_type, self.item_type = user_type, item_type
        self.rev_rel, self.fwd_rel = rev, fwd

        base_user_x = data[user_type].x
        self._gdev = base_user_x.device
        self.n_users = int(base_user_x.size(0))
        self._kge_dim = int(base_user_x.size(1))
        self._base_rev = data[rev].edge_index
        # Pre-grow user features once (reused every query; only the last row moves).
        self._user_x_aug = torch.empty(self.n_users + 1, self._kge_dim,
                                       dtype=base_user_x.dtype, device=self._gdev)
        self._user_x_aug[:self.n_users] = base_user_x

        # attribute label → KG node index, per attribute type (for the synth feature).
        self._attr_label2kg: Dict[str, Dict[str, int]] = {}
        for nt in ATTR_TYPES:
            uris = node_mappings.get(nt)
            if uris and nt in data.node_types and data[nt].get("x") is not None:
                self._attr_label2kg[nt] = {_uri_label(u): i for i, u in enumerate(uris)}

        # pack track-row → KG track index, and a KG track index → label map.
        self._row_kg = pack.track_meta["kg_idx"].to_numpy().astype(np.int64)
        self._label_by_kg = {
            int(r.kg_idx): f"{r.title or '?'} — {r.artist or '?'}"
            for r in pack.track_meta.itertuples()
        }

    def _track_label(self, kg_idx: int):
        return self._label_by_kg.get(int(kg_idx))

    def _synth_feature(self, selections: Mapping[str, Sequence[str]]):
        import torch
        vecs = []
        for nt, labels in selections.items():
            label2kg = self._attr_label2kg.get(nt)
            if not label2kg:
                continue
            for lab in labels:
                i = label2kg.get(lab)
                if i is not None:
                    vecs.append(self.base[nt].x[i])
        if not vecs:
            return torch.zeros(self._kge_dim, dtype=self._user_x_aug.dtype,
                               device=self._gdev)
        return torch.stack([v.to(self._gdev) for v in vecs]).mean(0)

    def tracks_listened_by(self, user_idx, *, top: Optional[int] = None) -> np.ndarray:
        """KG track indices the given existing users listened to, most-frequent first.

        Read straight off the base reverse (track → user) edges, so it lets a
        cold-start user be seeded from the *real* histories of its nearest
        neighbour users — the most faithful grounding for the k-NN-blend rationale.
        """
        rev = self._base_rev.cpu().numpy()
        users = np.asarray(list({int(u) for u in np.asarray(user_idx).ravel()}),
                           dtype=rev.dtype)
        tracks = rev[0][np.isin(rev[1], users)]
        if len(tracks) == 0:
            return np.array([], dtype=np.int64)
        uniq, cnt = np.unique(tracks, return_counts=True)
        out = uniq[np.argsort(-cnt)].astype(np.int64)
        return out[:top] if top else out

    def recommend(
        self,
        selections: Mapping[str, Sequence[str]],
        *,
        k: int = 10,
        seed_k: int = 8,
        exclude_seeds: bool = True,
        seed_vec: Optional[np.ndarray] = None,
        seed_kg: Optional[Sequence[int]] = None,
    ) -> dict:
        """Inductive cold-start recommendation + a ready :class:`HGTExplainer`.

        The synthetic user's stand-in history (the seed tracks it is wired to) can
        be chosen three ways, so every vector-building strategy can be routed
        through this *same* attention-grounded forward pass:

        * ``seed_kg`` — explicit KG track indices (e.g. the listening edges of the
          k nearest users, from :meth:`tracks_listened_by`);
        * ``seed_vec`` — any output-space vector to rank tracks by (a **persona
          centroid** or a **k-NN user blend**);
        * neither — the attribute taste vector (the default behaviour).

        Returns a dict with ``recs`` (``track_meta`` rows + ``score``),
        ``synth_idx`` (the temporary user's KG index), ``explainer`` (wrapping the
        captured attention), ``seed_rows``/``seed_kg`` and the seeding ``taste_vec``.
        """
        import torch
        from torch_geometric.data import HeteroData

        from .evaluation.explainability import HGTExplainer, capture_hgt_attention

        query_vec = (np.asarray(seed_vec, dtype=np.float32) if seed_vec is not None
                     else taste_vector_from_selections(
                         selections, self.pack.attr_vectors, weights=self.pack.weights))
        if seed_kg is not None:
            _row_of_kg = {int(kg): r for r, kg in enumerate(self._row_kg)}
            seed_rows = np.array([_row_of_kg[int(kg)] for kg in np.asarray(seed_kg)
                                  if int(kg) in _row_of_kg][:seed_k], dtype=np.int64)
        else:
            seed_rows = np.array([], dtype=np.int64)
        if len(seed_rows) == 0:                          # default / fallback seeding
            sims = _l2norm(self.pack.track_emb) @ _l2norm(query_vec)
            seed_rows = np.argsort(-sims)[:seed_k]
        seed_kg = self._row_kg[seed_rows]

        U = self.n_users
        self._user_x_aug[U] = self._synth_feature(selections)
        seed_t = torch.as_tensor(seed_kg, dtype=torch.long, device=self._gdev)
        add = torch.stack([seed_t, torch.full_like(seed_t, U)])
        rev_aug = torch.cat([self._base_rev.to(self._gdev), add], dim=1)

        g = HeteroData()
        for nt in self.base.node_types:
            g[nt].x = self._user_x_aug if nt == self.user_type else self.base[nt].x
        for et in self.base.edge_types:
            g[et].edge_index = rev_aug if et == self.rev_rel else self.base[et].edge_index

        x_dict = {nt: g[nt].x for nt in g.node_types if g[nt].get("x") is not None}
        eid = {et: g[et].edge_index for et in g.edge_types}
        emb, layers = capture_hgt_attention(self.model, x_dict, eid)

        synth = emb[self.user_type][U]                       # (out_dim,)
        item_emb = emb[self.item_type]                       # (n_kg_tracks, out_dim)
        scores = (item_emb[torch.as_tensor(self._row_kg)] @ synth).numpy()
        order = np.argsort(-scores)
        seed_set = set(seed_rows.tolist()) if exclude_seeds else set()
        picked = [int(r) for r in order if int(r) not in seed_set][:k]

        recs = self.pack.track_meta.iloc[picked].copy()
        recs["score"] = scores[picked]
        explainer = HGTExplainer(
            g, layers[-1], self.node_mappings, embeddings=emb,
            track_label_fn=self._track_label,
            user_type=self.user_type, item_type=self.item_type)
        return {"recs": recs.reset_index(drop=True), "synth_idx": U,
                "explainer": explainer, "seed_rows": seed_rows, "seed_kg": seed_kg,
                "taste_vec": query_vec, "scores": scores}


# ─── ipywidgets GUI ───────────────────────────────────────────────────────────

def _html_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _recs_table_html(recs: pd.DataFrame, score_col: str) -> str:
    """Ranked recommendations as a compact HTML table."""
    rows = []
    for i, (_, r) in enumerate(recs.iterrows(), 1):
        rows.append(
            f"<tr><td style='color:#999;text-align:right'>{i}</td>"
            f"<td><b>{_html_escape(r.get('title') or '?')}</b><br>"
            f"<span style='color:#666;font-size:12px'>{_html_escape(r.get('artist') or '?')}</span></td>"
            f"<td style='font-size:12px'>{r.get('genre')} · {r.get('tempo_class')} · "
            f"{r.get('mode')} · {r.get('decade')}</td>"
            f"<td style='text-align:right'>{float(r.get(score_col, float('nan'))):.3f}</td></tr>")
    return ("<table style='border-collapse:collapse;width:100%;font-family:sans-serif'>"
            "<tr style='text-align:left;border-bottom:1px solid #ccc'>"
            f"<th></th><th>track</th><th>genre·tempo·mode·decade</th><th>{score_col}</th></tr>"
            + "".join(rows) + "</table>")


def _neighbors_html(idx: np.ndarray, weights: np.ndarray, sims: np.ndarray) -> str:
    """The real users backing a k-NN taste blend, as chips (index · sim · weight)."""
    chips = " ".join(
        f"<span style='background:#eef7ee;border-radius:4px;padding:1px 6px;margin:0 2px;"
        f"font-size:11px'>user&nbsp;#{int(i)} <span style='color:#888'>"
        f"(sim {float(s):.2f}, w {float(w):.2f})</span></span>"
        for i, w, s in zip(idx, weights, sims))
    return ("<div style='font-family:sans-serif;font-size:12px;margin-top:6px'>"
            f"<b>Blended from {len(idx)} nearest users:</b><br>{chips}</div>")


def _compare_diag_html(cmp: dict) -> str:
    """The "which is more informative" verdict panel for the compare mode."""
    verdict = cmp["verdict"]
    indiv = ("your taste sits well inside this archetype — the persona centroid "
             "already captures you" if verdict == "persona" else
             "your taste sits between archetypes — the personalized blend carries "
             "individual signal the single centroid discards")
    badge = ("#1f7a1f" if verdict == "blend" else "#7a5a1f")
    return (
        "<div style='padding:10px;background:#f7f7fb;border-radius:6px;"
        "font-family:sans-serif;font-size:13px'>"
        f"<b>More informative here: "
        f"<span style='color:{badge}'>"
        f"{'k-NN taste blend' if verdict=='blend' else 'persona centroid'}</span></b><br>"
        f"<span style='color:#555'>{indiv}.</span><br><br>"
        f"residual&nbsp;(blend↔centroid cosine gap): <b>{cmp['residual']:.3f}</b> "
        "<span style='color:#888'>— 0 ≈ identical to the archetype</span><br>"
        f"recommendation overlap@k: <b>{cmp['overlap']*100:.0f}%</b> "
        "<span style='color:#888'>— how much the two lists agree</span><br>"
        f"neighbour agreement: <b>{cmp['neighbor_dispersion']:.2f}</b> "
        "<span style='color:#888'>— mean pairwise cosine of the blend's users "
        "(high ⇒ blend reliable)</span>"
        "</div>")


def _compare_tables_html(cmp: dict) -> str:
    """Side-by-side persona-centroid vs k-NN-blend recommendation tables."""
    left = _recs_table_html(cmp["recs_persona"], "sim")
    right = _recs_table_html(cmp["recs_blend"], "sim")
    return (
        "<div style='display:flex;gap:16px;font-family:sans-serif'>"
        f"<div style='flex:1'><h4 style='margin:4px 0'>Persona centroid "
        f"#{cmp['persona_id']} — “{_html_escape(cmp['persona_name'])}”</h4>{left}</div>"
        f"<div style='flex:1'><h4 style='margin:4px 0'>My taste "
        f"(k-NN user blend)</h4>{right}</div></div>")


def launch_persona_gui(pack: PersonaPack, *, cold_start: "ColdStartHGT | None" = None,
                       default_k: int = 10, default_seed_k: int = 8):
    """A 3-page ipywidgets app rendered **inline** in the notebook output.

    Pages: **1 · Your taste** (the form), **2 · Recommendations** (ranked tracks +
    matched persona), **3 · Why these?** (the rationale).

    The form's **mode** selector exposes whichever cold-start strategies the pack
    supports (all pure inference, no retraining):

    * **User persona (centroid)** / **My taste (k-NN blend)** / **Compare** —
      available when ``pack`` carries a users-only GMM (``user_centroids``): snap
      to a listener-archetype centroid, synthesise a bespoke vector from the
      nearest real users, or run both with a "which is more informative"
      diagnostic (residual, rec-overlap, neighbour agreement).
    * **HGT attention (inductive)** — available when ``cold_start`` (a
      :class:`ColdStartHGT`) is supplied: treats the new user as a temporary graph
      node and shows the faithful softmax edge-attention (the §12 story).
    * **Persona (cosine)** — the always-available attribute-anchor baseline.

    No server/localhost — everything renders in the cell output.

    Returns the :class:`ipywidgets.Tab` (display it, or leave as the last cell
    expression). Falls back to a printed message if ``ipywidgets`` is missing.
    """
    try:
        import ipywidgets as W
        from IPython.display import HTML, display
    except Exception as e:  # noqa: BLE001
        print(f"[persona-gui] ipywidgets unavailable ({e}); use pack.recommend(...) "
              "or cold_start.recommend(...) directly.")
        return None

    has_hgt = cold_start is not None
    has_user = pack.has_user_manifold      # users-only GMM → centroid/blend/compare

    # ── page 1: the form ──────────────────────────────────────────────────────
    def _ms(attr, rows=6):
        opts = pack.options(attr)
        return W.SelectMultiple(options=opts, rows=min(rows, max(2, len(opts))),
                                description=attr, style={"description_width": "92px"},
                                layout=W.Layout(width="330px"))

    def _dd(attr):
        return W.Dropdown(options=["(any)"] + pack.options(attr), description=attr,
                          style={"description_width": "92px"},
                          layout=W.Layout(width="330px"))

    w_genre = _ms("genre", rows=8)
    w_instr = _ms("instrument", rows=6) if pack.options("instrument") else None
    w_tempo, w_mode, w_decade = _dd("tempo_class"), _dd("mode"), _dd("decade")
    w_k = W.IntSlider(value=default_k, min=3, max=30, description="top-k",
                      style={"description_width": "92px"})
    w_seed = W.IntSlider(value=default_seed_k, min=3, max=20, description="seed tracks",
                         style={"description_width": "92px"},
                         layout=W.Layout(display="" if has_hgt else "none"))
    w_kusers = W.IntSlider(value=pack.blend_k, min=3, max=50, description="blend users",
                           style={"description_width": "92px"},
                           layout=W.Layout(display="" if has_user else "none"))
    # axis 1 — how the new-user vector is built
    _sources = ((["Persona centroid", "k-NN blend", "Compare: persona vs blend"]
                 if has_user else []) + ["Attribute taste"])
    w_source = W.ToggleButtons(
        options=_sources, value=_sources[0], description="recommend via",
        style={"description_width": "92px"},
        layout=W.Layout(display="" if len(_sources) > 1 else "none"))
    # axis 2 — how that recommendation is explained
    _explains = ["Cosine / anchors"] + (["HGT attention"] if has_hgt else [])
    w_explain = W.ToggleButtons(
        options=_explains, value=_explains[0], description="explain via",
        style={"description_width": "92px"},
        layout=W.Layout(display="" if has_hgt else "none"))
    w_btn = W.Button(description="Get recommendations  →", button_style="primary",
                     icon="music", layout=W.Layout(width="240px"))
    w_status = W.HTML("")

    # ── page 2/3 outputs ──────────────────────────────────────────────────────
    recs_out, expl_out = W.Output(), W.Output()
    w_pick = W.Dropdown(options=[], description="explain rec",
                        style={"description_width": "92px"},
                        layout=W.Layout(width="380px"))
    state: dict = {"res": None, "recs": None, "expl": None, "mode": None}

    def _collect():
        sel = {"genre": list(w_genre.value),
               "tempo_class": [] if w_tempo.value == "(any)" else [w_tempo.value],
               "mode": [] if w_mode.value == "(any)" else [w_mode.value],
               "decade": [] if w_decade.value == "(any)" else [w_decade.value]}
        if w_instr is not None:
            sel["instrument"] = list(w_instr.value)
        return sel

    def _banner_html(sel):
        vec = taste_vector_from_selections(sel, pack.attr_vectors, weights=pack.weights)
        pid, ranked = assign_persona(vec, pack.centroids)
        name = (pack.personas.at[pid, "persona"] if pid in pack.personas.index
                else f"cluster-{pid}")
        chips = " · ".join(f"{k}: {'/'.join(v)}" for k, v in sel.items() if v)
        return (f"<div style='padding:9px;background:#f3f6ff;border-radius:6px;"
                f"font-family:sans-serif'><b>Your taste</b> — {chips}<br>"
                f"<b>Closest persona:</b> #{pid} “{name}” "
                f"<span style='color:#666'>(fit {ranked[0][1]:.2f})</span></div>")

    def _render_cosine_expl():
        expl_out.clear_output(wait=True)
        with expl_out:
            display(HTML(format_explanation_html(state["expl"])))
            if not has_hgt:
                display(HTML("<div style='color:#888;font-size:12px;margin-top:8px;"
                             "font-family:sans-serif'>Tip: pass a <code>ColdStartHGT</code> "
                             "to this GUI to get faithful edge-attention rationales "
                             "(treat-the-new-user-as-a-temporary-node).</div>"))

    def _render_user_expl():
        """Persona-centroid / k-NN-blend rationale (same HTML as cosine + neighbours)."""
        expl_out.clear_output(wait=True)
        with expl_out:
            display(HTML(format_explanation_html(state["expl"])))
            nb = state.get("neighbors")
            if nb is not None:
                display(HTML(_neighbors_html(*nb)))

    def _render_compare_expl():
        expl_out.clear_output(wait=True)
        with expl_out:
            cmp = state.get("cmp")
            if cmp is None:
                print("Run a comparison first."); return
            display(HTML(_compare_diag_html(cmp)))
            display(HTML(_neighbors_html(cmp["neighbor_idx"], cmp["neighbor_weights"],
                                         cmp["neighbor_sims"])))

    def _render_hgt_expl(idx):
        expl_out.clear_output(wait=True)
        with expl_out:
            res = state["res"]
            if res is None:
                print("Run a recommendation first."); return
            recs = res["recs"]
            if idx is None or idx >= len(recs):
                return
            kg = int(recs.iloc[idx]["kg_idx"])
            ex = res["explainer"]
            e = ex.explain(res["synth_idx"], kg, top_k_anchors=6)
            n_seed = len(res["seed_kg"])
            display(HTML(
                f"<div style='padding:8px;background:#fff7e6;border-radius:6px;"
                f"font-family:sans-serif'><b>Faithful HGT attention rationale.</b> "
                f"Your profile was seeded from {n_seed} tracks matching your taste; "
                f"the trained HGT built your embedding by attending to them. The "
                f"weights below are the <i>real</i> softmax attention the model "
                f"applied (same capture as §12), not a surrogate.</div>"))
            display(HTML("<pre style='font-size:12px;white-space:pre-wrap;"
                         "font-family:monospace'>" + _html_escape(ex.explain_text(e))
                         + "</pre>"))
            import matplotlib.pyplot as plt
            for fn in ("plot_explanation_graph", "plot_explanation"):
                try:
                    fig = getattr(ex, fn)(e); display(fig); plt.close(fig)
                except Exception as ex_err:  # noqa: BLE001
                    print(f"[{fn} skipped: {ex_err}]")

    def _show_hgt(res):
        """Render an inductive-HGT result: recs table + per-rec attention picker."""
        state.update(res=res, recs=res["recs"], mode="hgt")
        display(HTML(_recs_table_html(res["recs"], "score")))
        w_pick.options = [(f"{i+1}. {t.title or '?'} — {t.artist or '?'}", i)
                          for i, t in enumerate(res["recs"].itertuples())]
        if w_pick.options:
            w_pick.value = 0
        _render_hgt_expl(0)

    def _on_pick(change):
        if state.get("mode") == "hgt" and change["name"] == "value":
            _render_hgt_expl(change["new"])

    w_pick.observe(_on_pick, names="value")

    def _on_click(_):
        sel = _collect()
        if not any(sel.values()):
            w_status.value = ("<span style='color:#b00'>Pick at least one preference "
                              "(a genre is a good start).</span>")
            return
        w_status.value = "<span style='color:#666'>scoring…</span>"
        src = str(w_source.value)
        want_hgt = has_hgt and str(w_explain.value).startswith("HGT") \
            and not src.startswith("Compare")
        k, k_users, seed_k = int(w_k.value), int(w_kusers.value), int(w_seed.value)
        recs_out.clear_output(wait=True)
        try:
            with recs_out:
                display(HTML(_banner_html(sel)))
                w_pick.options = []                       # only HGT mode uses per-rec pick
                vec = taste_vector_from_selections(sel, pack.attr_vectors, weights=pack.weights)

                if src.startswith("Compare"):
                    # cosine-only diagnostic (attention is not defined for two lists)
                    cmp = pack.compare_user_modes(sel, k=k, k_users=k_users)
                    state.update(cmp=cmp, mode="compare")
                    display(HTML(_compare_tables_html(cmp)))
                    _render_compare_expl()

                elif src.startswith("Persona"):          # nearest user-archetype centroid
                    pid, ranked = assign_persona(vec, pack.user_centroids)
                    cvec = pack.user_centroids[pid]
                    if want_hgt:
                        _show_hgt(cold_start.recommend(sel, k=k, seed_k=seed_k, seed_vec=cvec))
                    else:
                        recs = recommend_for_vector(cvec, pack.track_emb, pack.track_meta, k=k)
                        expl = explain_recommendations(
                            sel, recs, persona_id=pid, personas=pack.user_personas,
                            ranked_personas=ranked, attr_vectors=pack.attr_vectors, taste_vec=cvec)
                        state.update(recs=recs, expl=expl, neighbors=None, mode="user_persona")
                        display(HTML(_recs_table_html(recs, "sim")))
                        _render_user_expl()

                elif src.startswith("k-NN"):             # blend of nearest real users
                    b, idx, w, s = blend_user_embedding(
                        vec, pack.user_emb, k=k_users, temperature=pack.blend_temperature)
                    if want_hgt:
                        # seed the temp node from the neighbours' real listening edges
                        # (falls back to the blend vector inside recommend() if empty)
                        seed_kg = cold_start.tracks_listened_by(idx, top=seed_k)
                        _show_hgt(cold_start.recommend(
                            sel, k=k, seed_k=seed_k, seed_vec=b,
                            seed_kg=(seed_kg if len(seed_kg) else None)))
                    else:
                        recs = recommend_for_vector(b, pack.track_emb, pack.track_meta, k=k)
                        pid, ranked = assign_persona(b, pack.user_centroids)
                        expl = explain_recommendations(
                            sel, recs, persona_id=pid, personas=pack.user_personas,
                            ranked_personas=ranked, attr_vectors=pack.attr_vectors, taste_vec=b)
                        state.update(recs=recs, expl=expl, neighbors=(idx, w, s), mode="user_blend")
                        display(HTML(_recs_table_html(recs, "sim")))
                        _render_user_expl()

                else:                                     # Attribute taste (all-type baseline)
                    if want_hgt:
                        _show_hgt(cold_start.recommend(sel, k=k, seed_k=seed_k))
                    else:
                        recs, expl = pack.recommend(sel, k=k)
                        state.update(recs=recs, expl=expl, mode="cosine")
                        display(HTML(_recs_table_html(recs, "sim")))
                        _render_cosine_expl()
            w_status.value = ""
            tab.selected_index = 1
        except Exception as ex_err:  # noqa: BLE001
            w_status.value = f"<span style='color:#b00'>Failed: {_html_escape(ex_err)}</span>"

    w_btn.on_click(_on_click)

    left = [w_genre] + ([w_instr] if w_instr is not None else [])
    right = ([w_tempo, w_mode, w_decade, w_k]
             + ([w_kusers] if has_user else [])
             + ([w_seed] if has_hgt else []))
    controls = ([w_source] if len(_sources) > 1 else []) + ([w_explain] if has_hgt else [])
    _src_note = ("snap to a listener-archetype <b>centroid</b>, synthesise your own "
                 "vector via a <b>k-NN blend</b> of similar users, <b>compare</b> the two, "
                 "or use the attribute <b>taste</b> vector"
                 if has_user else "attribute <b>taste</b> vector")
    _exp_note = (" — explain each with cosine/anchors <b>or</b> faithful "
                 "<b>HGT edge-attention</b> (new user as a temporary graph node)"
                 if has_hgt else "")
    mode_note = f"recommend via: {_src_note}{_exp_note}"
    page1 = W.VBox([
        W.HTML("<h4 style='margin:4px 0'>🎧 Tell us your taste</h4>"
               f"<div style='color:#666;font-size:12px'>{mode_note}</div>"),
        W.HBox([W.VBox(left), W.VBox(right)]),
        W.VBox(controls),
        W.HBox([w_btn, w_status]),
    ])
    page2 = W.VBox([recs_out])
    page3 = W.VBox([W.HBox([w_pick]), expl_out])

    tab = W.Tab(children=[page1, page2, page3])
    for i, t in enumerate(["1 · Your taste", "2 · Recommendations", "3 · Why these?"]):
        tab.set_title(i, t)
    return tab


# ─── Gradio web app (separate browser tab; public link via share=True) ─────────

def _show_share_qr(url: Optional[str]) -> None:
    """Render a scannable QR code for the public share URL (live-demo friendly).

    Shows the QR as an image in a notebook (and an ASCII fallback to stdout), so
    in a live demo people can scan the *.gradio.live link with a phone instead of
    typing it. Best-effort: auto-installs ``qrcode`` and degrades to just printing
    the URL if anything is unavailable.
    """
    if not url:
        return
    try:
        import qrcode
    except ImportError:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "qrcode[pil]"],
                           check=True)
            import qrcode
        except Exception as e:  # noqa: BLE001
            print(f"[app] QR code unavailable ({e}); open this link instead:\n  {url}")
            return
    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)
    print(f"\n[app] 📱 Scan to open the demo on any device — {url}\n")
    shown = False
    try:                                              # crisp image in notebooks
        from IPython.display import display
        display(qr.make_image(fill_color="black", back_color="white"))
        shown = True
    except Exception:                                 # noqa: BLE001
        pass
    if not shown:                                     # ASCII fallback (terminal)
        import io
        buf = io.StringIO()
        qr.print_ascii(out=buf)
        print(buf.getvalue())


def launch_persona_app(
    pack: PersonaPack,
    *,
    cold_start: "ColdStartHGT | None" = None,
    share: bool = True,
    inline: bool = False,
    server_port: Optional[int] = None,
    default_k: int = 10,
    default_seed_k: int = 8,
    default_weight: int = 7,
    max_picks: int = 16,
    qr: bool = True,
    **launch_kwargs,
):
    """Serve the cold-start recommender as a **Gradio web app** (not inline).

    Unlike :func:`launch_persona_gui` (inline ipywidgets), this launches a real
    web server.  With ``share=True`` Gradio prints a public ``*.gradio.live`` URL
    you can open in a separate browser tab **from any device** — so in Colab you
    just run the build cell, then this cell, and open the link (no need to keep
    the notebook tab focused).

    Features beyond the inline GUI:

    * **per-item preference weights** — pick several labels per category and give
      each a 1–10 importance (e.g. rock 9, classical 10, pop 7); a slider appears
      for every pick and feeds the ``{label: weight}`` taste vector;
    * **searchable multi-select dropdowns** — type to autocomplete/filter the
      attribute labels;
    * the same modes (persona centroid / k-NN blend / compare / attribute taste)
      and explanations (cosine-anchors or faithful HGT-attention) as the inline GUI.

    Returns the ``gradio.Blocks`` (already launched), or ``None`` if gradio is
    missing (``pip install gradio``).
    """
    try:
        import gradio as gr
    except Exception as e:  # noqa: BLE001
        print(f"[persona-app] gradio not installed ({e}). Run `pip install gradio`, "
              "or use launch_persona_gui(...) for the inline ipywidgets GUI.")
        return None
    import matplotlib
    matplotlib.use("Agg")        # headless figures for gr.Plot on a server/Colab

    has_hgt = cold_start is not None
    has_user = pack.has_user_manifold
    attr_types = [nt for nt in ATTR_TYPES if pack.options(nt)]
    sources = ((["Persona centroid", "k-NN blend", "Compare: persona vs blend"]
                if has_user else []) + ["Attribute taste"])
    explains = ["Cosine / anchors"] + (["HGT attention"] if has_hgt else [])

    # ── compute: (weighted selections, controls) → (recs_html, why_html, figure) ──
    def _hgt_outputs(res):
        recs = res["recs"]
        ex, kg = res["explainer"], int(recs.iloc[0]["kg_idx"])
        e = ex.explain(res["synth_idx"], kg, top_k_anchors=6)
        try:
            fig = ex.plot_explanation_graph(e)
        except Exception as _err:  # noqa: BLE001
            fig = None
        why = ("<div style='font-family:sans-serif;font-size:12px'><b>Faithful HGT "
               "attention</b> — the graph shows how your temporary node's embedding "
               "was built; edge labels are the real softmax attention. Explaining the "
               f"top pick “{_html_escape(recs.iloc[0].get('title') or '?')}”.</div>"
               "<pre style='font-size:12px;white-space:pre-wrap;font-family:monospace'>"
               + _html_escape(ex.explain_text(e)) + "</pre>")
        return _recs_table_html(recs, "score"), why, fig

    def _compute(selections, source, explain, k, k_users, seed_k):
        if not selections:
            return ("<div style='color:#b00;font-family:sans-serif'>Pick at least one "
                    "preference and give it a weight &gt; 0.</div>", "", None)
        want_hgt = (has_hgt and str(explain).startswith("HGT")
                    and not str(source).startswith("Compare"))
        vec = taste_vector_from_selections(selections, pack.attr_vectors, weights=pack.weights)

        if str(source).startswith("Compare"):
            cmp = pack.compare_user_modes(selections, k=k, k_users=k_users)
            why = (_compare_diag_html(cmp) + _neighbors_html(
                cmp["neighbor_idx"], cmp["neighbor_weights"], cmp["neighbor_sims"]))
            return _compare_tables_html(cmp), why, None

        if str(source).startswith("Persona"):
            pid, ranked = assign_persona(vec, pack.user_centroids)
            cvec = pack.user_centroids[pid]
            if want_hgt:
                return _hgt_outputs(cold_start.recommend(
                    selections, k=k, seed_k=seed_k, seed_vec=cvec))
            recs = recommend_for_vector(cvec, pack.track_emb, pack.track_meta, k=k)
            why = format_explanation_html(explain_recommendations(
                selections, recs, persona_id=pid, personas=pack.user_personas,
                ranked_personas=ranked, attr_vectors=pack.attr_vectors, taste_vec=cvec))
            return _recs_table_html(recs, "sim"), why, None

        if str(source).startswith("k-NN"):
            b, idx, w, s = blend_user_embedding(
                vec, pack.user_emb, k=k_users, temperature=pack.blend_temperature)
            if want_hgt:
                seed_kg = cold_start.tracks_listened_by(idx, top=seed_k)
                return _hgt_outputs(cold_start.recommend(
                    selections, k=k, seed_k=seed_k, seed_vec=b,
                    seed_kg=(seed_kg if len(seed_kg) else None)))
            recs = recommend_for_vector(b, pack.track_emb, pack.track_meta, k=k)
            pid, ranked = assign_persona(b, pack.user_centroids)
            why = (format_explanation_html(explain_recommendations(
                selections, recs, persona_id=pid, personas=pack.user_personas,
                ranked_personas=ranked, attr_vectors=pack.attr_vectors, taste_vec=b))
                + _neighbors_html(idx, w, s))
            return _recs_table_html(recs, "sim"), why, None

        # Attribute taste (all-type baseline)
        if want_hgt:
            return _hgt_outputs(cold_start.recommend(selections, k=k, seed_k=seed_k))
        recs, e = pack.recommend(selections, k=k)
        return _recs_table_html(recs, "sim"), format_explanation_html(e), None

    # ── per-pick weight wiring ───────────────────────────────────────────────────
    # A FIXED pool of weight sliders (shown/hidden by a dropdown `.change`) instead
    # of `gr.render` dynamic components. `gr.render` re-registers the button's click
    # on every edit — stacking stale handlers that fire with old selections (same
    # recs regardless of preferences) and double-write the outputs (duplicate
    # table). One pool + one click handler + a State map fixes all of that, and
    # lets the sliders sit directly under the dropdowns.
    def _meta_from_dropdowns(picks_per_type):
        meta: List[Tuple[str, str]] = []
        for nt, picks in zip(attr_types, picks_per_type):
            for lab in (picks or []):
                meta.append((nt, lab))
        return meta[:max_picks]

    # ── UI ──────────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Persona cold-start recommender") as demo:
        gr.Markdown("# 🎧 Persona cold-start recommender\n"
                    "Pick a few preferences per category (type to search), set how "
                    "much each matters (1–10), choose how to build & explain the "
                    "recommendation, then hit **Get recommendations**.")
        dropdowns: Dict[str, object] = {}
        with gr.Row():
            for nt in attr_types:
                dropdowns[nt] = gr.Dropdown(
                    choices=pack.options(nt), label=nt, multiselect=True,
                    filterable=True, allow_custom_value=False)

        # Per-pick weights, RIGHT BELOW the selections they belong to.
        gr.Markdown("#### Importance of each pick (1–10)")
        weight_hint = gr.Markdown(
            "*Select at least one preference above to set its weight.*")
        sliders: List[object] = []
        with gr.Row():
            for j in range(max_picks):
                sliders.append(gr.Slider(
                    1, 10, value=default_weight, step=1,
                    label=f"slot {j}", visible=False))
        # ordered [(node_type, label), ...] currently mapped onto the slider slots
        sel_state = gr.State([])

        gr.Markdown("#### How to build & explain")
        with gr.Row():
            w_mode = gr.Radio(sources, value=sources[0], label="recommend via")
            w_explain = gr.Radio(explains, value=explains[0], label="explain via",
                                 visible=has_hgt)
        with gr.Row():
            w_k = gr.Slider(3, 30, value=default_k, step=1, label="top-k")
            w_kusers = gr.Slider(3, 50, value=pack.blend_k, step=1,
                                 label="blend users", visible=has_user)
            w_seed = gr.Slider(3, 20, value=default_seed_k, step=1,
                               label="HGT seed tracks", visible=has_hgt)
        w_btn = gr.Button("Get recommendations  →", variant="primary")

        out_recs = gr.HTML(label="Recommendations")
        out_why = gr.HTML(label="Why these?")
        out_plot = gr.Plot(label="How the HGT built it", visible=has_hgt)

        # Dropdown edits reconfigure the slider pool: relabel/show one slider per
        # pick (carrying any weight the user already set for that label forward),
        # hide the rest, and store the slot→(type,label) map in State.
        def _sync(*args):
            n_dd = len(attr_types)
            picks_per_type = args[:n_dd]
            cur_vals = args[n_dd:n_dd + max_picks]
            prev_meta = args[-1] or []
            prev_w = {tuple(m): v for m, v in zip(prev_meta, cur_vals)}
            new_meta = _meta_from_dropdowns(picks_per_type)
            updates = []
            for j in range(max_picks):
                if j < len(new_meta):
                    nt, lab = new_meta[j]
                    updates.append(gr.update(
                        visible=True, label=f"{nt} · {lab}",
                        value=prev_w.get((nt, lab), default_weight)))
                else:
                    updates.append(gr.update(visible=False))
            hint = gr.update(visible=(len(new_meta) == 0))
            return updates + [hint, new_meta]

        sync_inputs = list(dropdowns.values()) + sliders + [sel_state]
        sync_outputs = sliders + [weight_hint, sel_state]
        for dd in dropdowns.values():
            dd.change(_sync, inputs=sync_inputs, outputs=sync_outputs)

        # One click handler, reading the State map + the full slider pool.
        def _run(meta, *vals):
            slider_vals = vals[:max_picks]
            mode_v, expl_v, k_v, ku_v, sd_v = vals[max_picks:]
            selections: Dict[str, Dict[str, float]] = {}
            for (nt, lab), wv in zip(meta or [], slider_vals):
                selections.setdefault(nt, {})[lab] = float(wv)
            try:
                return _compute(selections, mode_v, expl_v,
                                int(k_v), int(ku_v), int(sd_v))
            except Exception as err:  # noqa: BLE001
                return (f"<div style='color:#b00;font-family:sans-serif'>Failed: "
                        f"{_html_escape(err)}</div>", "", None)

        w_btn.click(
            _run,
            inputs=[sel_state] + sliders + [w_mode, w_explain, w_k, w_kusers, w_seed],
            outputs=[out_recs, out_why, out_plot])

    demo.queue()
    launch_kwargs.setdefault("show_error", True)
    if server_port is not None:
        launch_kwargs["server_port"] = server_port
    demo.launch(share=share, inline=inline, **launch_kwargs)
    if qr and share:
        _show_share_qr(getattr(demo, "share_url", None) or getattr(demo, "local_url", None))
    return demo


__all__ = [
    "ATTR_TYPES", "DEFAULT_WEIGHTS", "PersonaPack", "ColdStartHGT",
    "type_embedding_matrix", "build_attribute_vectors", "build_track_table",
    "nearest_anchors", "summarise_clusters", "taste_vector_from_selections",
    "nearest_users", "blend_user_embedding", "neighbor_dispersion",
    "assign_persona", "recommend_for_vector", "explain_recommendations",
    "format_explanation_text", "format_explanation_html",
    "build_persona_pack", "save_persona_pack", "load_persona_pack",
    "launch_persona_gui", "launch_persona_app",
]
