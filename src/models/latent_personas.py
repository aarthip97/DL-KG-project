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
    selections: Mapping[str, Sequence[str]],
    attr_vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Blend the selected attribute embeddings into one taste vector.

    ``selections`` maps an attribute type to the chosen labels, e.g.
    ``{"genre": ["rock", "metal"], "tempo_class": ["Allegro"], "mode": ["Major"]}``.
    Each type contributes the mean of its selected anchor vectors, scaled by
    ``weights`` (defaults to :data:`DEFAULT_WEIGHTS`); the sum is L2-normalised.
    Unknown labels are ignored.

    Raises:
        ValueError: if nothing valid was selected.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    dim = next((len(next(iter(d.values()))) for d in attr_vectors.values() if d), None)
    if dim is None:
        raise ValueError("attr_vectors is empty")
    acc = np.zeros(dim, dtype=np.float32)
    used = 0
    for nt, labels in selections.items():
        table = attr_vectors.get(nt)
        if not table or not labels:
            continue
        picked = [table[l] for l in labels if l in table]
        if not picked:
            continue
        acc += float(weights.get(nt, 0.5)) * np.mean(picked, axis=0).astype(np.float32)
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

    # convenience: the labels a form can offer, sorted
    def options(self, attr_type: str) -> List[str]:
        return sorted(self.attr_vectors.get(attr_type, {}))

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
) -> PersonaPack:
    """Assemble a :class:`PersonaPack` from the latent-space artefacts.

    ``centroids`` are the GMM means (``GMMResult.means``); ``composition`` is an
    optional ``cluster × Node_Type`` count frame (``pd.crosstab`` of the
    subsample's ``Cluster_ID`` and ``Node_Type``) used to label personas with
    their dominant node type and user count.
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
    return PersonaPack(
        attr_vectors=attr_vectors, centroids=np.asarray(centroids, dtype=np.float32),
        best_params=dict(best_params), track_emb=track_emb, track_meta=track_meta,
        personas=personas, track_cluster=np.asarray(track_cluster),
        weights=dict(DEFAULT_WEIGHTS if weights is None else weights),
        composition=composition)


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

    def recommend(
        self,
        selections: Mapping[str, Sequence[str]],
        *,
        k: int = 10,
        seed_k: int = 8,
        exclude_seeds: bool = True,
    ) -> dict:
        """Inductive cold-start recommendation + a ready :class:`HGTExplainer`.

        Returns a dict with ``recs`` (``track_meta`` rows + ``score``),
        ``synth_idx`` (the temporary user's KG index), ``explainer`` (wrapping the
        captured attention), ``seed_rows``/``seed_kg`` and the ``taste_vec``.
        """
        import torch
        from torch_geometric.data import HeteroData

        from .evaluation.explainability import HGTExplainer, capture_hgt_attention

        taste = taste_vector_from_selections(selections, self.pack.attr_vectors,
                                             weights=self.pack.weights)
        sims = _l2norm(self.pack.track_emb) @ _l2norm(taste)
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
                "taste_vec": taste, "scores": scores}


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


def launch_persona_gui(pack: PersonaPack, *, cold_start: "ColdStartHGT | None" = None,
                       default_k: int = 10, default_seed_k: int = 8):
    """A 3-page ipywidgets app rendered **inline** in the notebook output.

    Pages: **1 · Your taste** (the form), **2 · Recommendations** (ranked tracks +
    matched persona), **3 · Why these?** (the rationale). When ``cold_start`` (a
    :class:`ColdStartHGT`) is supplied, the form offers an *HGT-attention* mode
    that treats the new user as a temporary graph node and shows the faithful
    edge-attention rationale (the §12 story); otherwise it uses the persona/cosine
    explanation. No server/localhost — everything renders in the cell output.

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
    _modes = (["HGT attention (inductive)", "Persona (cosine)"] if has_hgt
              else ["Persona (cosine)"])
    w_mode_sel = W.ToggleButtons(
        options=_modes, value=_modes[0], description="explain via",
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
        use_hgt = has_hgt and str(w_mode_sel.value).startswith("HGT")
        recs_out.clear_output(wait=True)
        try:
            with recs_out:
                display(HTML(_banner_html(sel)))
                if use_hgt:
                    res = cold_start.recommend(sel, k=int(w_k.value), seed_k=int(w_seed.value))
                    state.update(res=res, recs=res["recs"], mode="hgt")
                    display(HTML(_recs_table_html(res["recs"], "score")))
                    w_pick.options = [(f"{i+1}. {t.title or '?'} — {t.artist or '?'}", i)
                                      for i, t in enumerate(res["recs"].itertuples())]
                    if w_pick.options:
                        w_pick.value = 0
                    _render_hgt_expl(0)
                else:
                    recs, expl = pack.recommend(sel, k=int(w_k.value))
                    state.update(recs=recs, expl=expl, mode="cosine")
                    display(HTML(_recs_table_html(recs, "sim")))
                    w_pick.options = []
                    _render_cosine_expl()
            w_status.value = ""
            tab.selected_index = 1
        except Exception as ex_err:  # noqa: BLE001
            w_status.value = f"<span style='color:#b00'>Failed: {_html_escape(ex_err)}</span>"

    w_btn.on_click(_on_click)

    left = [w_genre] + ([w_instr] if w_instr is not None else [])
    right = [w_tempo, w_mode, w_decade, w_k] + ([w_seed, w_mode_sel] if has_hgt else [])
    mode_note = ("two explanation modes: faithful HGT edge-attention or persona-cosine"
                 if has_hgt else "persona-cosine explanation "
                 "(add a ColdStartHGT for attention rationales)")
    page1 = W.VBox([
        W.HTML("<h4 style='margin:4px 0'>🎧 Tell us your taste</h4>"
               f"<div style='color:#666;font-size:12px'>{mode_note}</div>"),
        W.HBox([W.VBox(left), W.VBox(right)]),
        W.HBox([w_btn, w_status]),
    ])
    page2 = W.VBox([recs_out])
    page3 = W.VBox([W.HBox([w_pick]), expl_out])

    tab = W.Tab(children=[page1, page2, page3])
    for i, t in enumerate(["1 · Your taste", "2 · Recommendations", "3 · Why these?"]):
        tab.set_title(i, t)
    return tab


__all__ = [
    "ATTR_TYPES", "DEFAULT_WEIGHTS", "PersonaPack", "ColdStartHGT",
    "type_embedding_matrix", "build_attribute_vectors", "build_track_table",
    "nearest_anchors", "summarise_clusters", "taste_vector_from_selections",
    "assign_persona", "recommend_for_vector", "explain_recommendations",
    "format_explanation_text", "format_explanation_html",
    "build_persona_pack", "save_persona_pack", "load_persona_pack",
    "launch_persona_gui",
]
