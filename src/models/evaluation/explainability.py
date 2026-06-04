"""Attention-based explainability for the HGT recommender.

The whole point of choosing a Heterogeneous Graph **Transformer** over a black-box
model is that every message-passing step is an *attention* over typed edges: we
can read, for a given user, *which* of their listened tracks (and which shared
attributes — artist, genre, decade, tempo …) the model leaned on when it built
the embedding that scored a recommendation.

PyG's :class:`~torch_geometric.nn.HGTConv` computes that attention inside
``message()`` but never returns it.  :func:`capture_hgt_attention` recovers it
**faithfully** — it re-evaluates the exact same softmax formula PyG uses on the
exact same inputs (captured via a temporary ``message`` wrapper), so the numbers
are the attention the model actually applied, not a surrogate.  The captured
per-edge weights are aligned column-for-column with ``data[edge_type].edge_index``
by exploiting the fact that :func:`construct_bipartite_edge_index` concatenates
the relations in ``edge_index_dict`` iteration order.

:class:`HGTExplainer` turns those weights into a human-readable, attention-grounded
answer to *"why was track T recommended to user U?"*:

* **anchors**  — the user's own listened tracks that most shaped their embedding;
* **reasons**  — attributes shared between those anchors and T (same artist /
  genre / decade / tempo …), weighted by the anchor attention;
* **drivers**  — the neighbours that most shaped T's embedding;
* **text + graph** renderings of the above.

Everything is model-faithful and dependency-light (matplotlib only; ``networkx``
is used opportunistically if present).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

EdgeType = Tuple[str, str, str]

# Forward (track -> attribute) relations that describe what a track *is*.
# Kept as an ordered mapping so explanations always list attributes the same way.
_TRACK_ATTR_RELS: Dict[str, EdgeType] = {
    "artist":      ("track", "performed_by", "artist"),
    "tempo_class": ("track", "has_tempo", "tempo_class"),
    "key":         ("track", "has_key", "key"),
    "mode":        ("track", "has_mode", "mode"),
    "instrument":  ("track", "has_instrument", "instrument"),
    "decade":      ("track", "in_decade", "decade"),
}
# Genre hangs off the artist, so it is resolved as a 2-hop (track->artist->genre).
_ARTIST_GENRE_REL: EdgeType = ("artist", "has_genre", "genre")


# ─────────────────────────────────────────────────────────────────────────────
#  1. Faithful attention capture
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EdgeAttention:
    """Per-edge attention for a single HGT layer.

    ``by_edge_type[et]`` is a 1-D CPU tensor of length ``edge_index_dict[et].size(1)``
    holding the (head-reduced) attention weight of every edge of that relation,
    in the *same column order* as ``data[et].edge_index``.
    """

    by_edge_type: Dict[EdgeType, Tensor]

    def __getitem__(self, et: EdgeType) -> Tensor:
        return self.by_edge_type[et]

    def __contains__(self, et: EdgeType) -> bool:
        return et in self.by_edge_type


def capture_hgt_attention(
    model: torch.nn.Module,
    x_dict: Mapping[str, Tensor],
    edge_index_dict: Mapping[EdgeType, Tensor],
    *,
    head_reduce: str = "mean",
) -> Tuple[Dict[str, Tensor], List[EdgeAttention]]:
    """Run one forward pass and recover per-edge attention from every HGTConv.

    The model is expected to expose its message-passing layers as
    ``model.convs`` (an iterable of :class:`~torch_geometric.nn.HGTConv`), which
    is how :class:`~models.hgt.RecommenderHGT` is built.

    Args:
        model: A trained ``RecommenderHGT`` (or any module whose ``convs`` are
            ``HGTConv`` layers).
        x_dict: ``node_type -> features`` for the forward pass.
        edge_index_dict: ``edge_type -> [2, E]`` connectivity.  The captured
            attention is aligned to *this* dict's iteration order, so pass the
            same ordering you index with afterwards (e.g. build it by iterating
            ``data.edge_types``).
        head_reduce: ``"mean"`` (default) or ``"max"`` reduction over heads.

    Returns:
        ``(embeddings, layers)`` where ``embeddings`` is the model output moved
        to CPU and ``layers[i]`` is the :class:`EdgeAttention` for conv ``i``.
    """
    from torch_geometric.utils import softmax as pyg_softmax

    device = next(model.parameters()).device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    edge_types = list(edge_index_dict.keys())
    counts = [int(edge_index_dict[et].size(1)) for et in edge_types]
    offsets = np.cumsum([0] + counts)

    convs = list(getattr(model, "convs"))
    stores: List[dict] = [{} for _ in convs]
    originals: List[Callable] = []

    def _make_patch(orig_message, store):
        def patched(k_j, q_i, v_j, edge_attr, index, ptr, size_i):
            # Exact replica of HGTConv.message's attention (PyG 2.x):
            alpha = (q_i * k_j).sum(dim=-1) * edge_attr
            alpha = alpha / math.sqrt(q_i.size(-1))
            alpha = pyg_softmax(alpha, index, ptr, size_i)   # [E, heads]
            store["alpha"] = alpha.detach()
            # Defer to the real message so the forward output is bit-identical.
            return orig_message(k_j, q_i, v_j, edge_attr, index, ptr, size_i)
        return patched

    model.eval()
    try:
        for conv, store in zip(convs, stores):
            originals.append(conv.message)
            conv.message = _make_patch(conv.message, store)   # type: ignore[assignment]
        with torch.no_grad():
            emb = model(x_dict, edge_index_dict)
    finally:
        for conv, orig in zip(convs, originals):
            conv.message = orig   # type: ignore[assignment]

    layers: List[EdgeAttention] = []
    for store in stores:
        alpha = store["alpha"]                       # [E_total, heads]
        if alpha.size(0) != int(offsets[-1]):
            raise RuntimeError(
                f"captured {alpha.size(0)} edges but edge_index_dict has "
                f"{int(offsets[-1])}; attention/edge alignment broke."
            )
        red = alpha.mean(dim=1) if head_reduce == "mean" else alpha.max(dim=1).values
        red = red.cpu()
        by_et = {et: red[offsets[i]:offsets[i + 1]].clone()
                 for i, et in enumerate(edge_types)}
        layers.append(EdgeAttention(by_et))

    emb = {k: v.detach().cpu() for k, v in emb.items()}
    return emb, layers


# ─────────────────────────────────────────────────────────────────────────────
#  1b. Faithful, PREDICTION-LEVEL attribution (occlusion + optional IG)
# ─────────────────────────────────────────────────────────────────────────────
# Attention says what the model *looked at*; it does not say how much each input
# *moved the recommendation score*.  Here we attribute the actual score
# ``s(u,t) = <z_u, z_t>`` (z = the model's L2-normalised output) back to edges by
# re-running the model with one connection removed at a time (occlusion) — a
# faithful counterfactual, not a surrogate.  The result is EdgeAttention-shaped,
# so it drops straight into HGTExplainer / the existing plots.

@dataclass
class FaithfulAttribution:
    """Prediction-level edge importances for one ``(user, track)`` rec.

    EdgeAttention-compatible: ``by_edge_type[et]`` is a per-edge tensor aligned to
    ``data[et].edge_index`` columns, so it can replace captured attention inside
    :class:`HGTExplainer`.  An edge's value is its **Δscore** — how much the rec
    score drops when that connection (both directions) is removed and the model is
    re-run.  Positive ⇒ the edge *supported* the rec; **negative ⇒ it pushed
    against it** (counter-evidence attention can never express).

    Attributes:
        by_edge_type:   per-edge Δscore, EdgeAttention-aligned.
        edge_type_delta: Δscore when a whole relation (both directions) is dropped.
        score:          ``s(u, T)``.
        contrast_score: ``s(u, T′)`` when a contrastive target was given.
        contrastive:    True ⇒ every Δ is attributed to the gap ``s(u,T)−s(u,T′)``.
    """

    by_edge_type: Dict[EdgeType, Tensor]
    edge_type_delta: Dict[EdgeType, float]
    score: float
    contrast_score: Optional[float] = None
    contrastive: bool = False

    def __getitem__(self, et: EdgeType) -> Tensor:
        return self.by_edge_type[et]

    def __contains__(self, et: EdgeType) -> bool:
        return et in self.by_edge_type


def _reverse_et(edge_types, et: EdgeType) -> Optional[EdgeType]:
    """Find the ToUndirected reverse of ``et`` (so we can drop both directions)."""
    a, r, b = et
    for cand in ((b, f"rev_{r}", a), (b, r, a)):
        if cand in edge_types:
            return cand
    for e2 in edge_types:                       # generic mirror fallback
        if e2 != et and e2[0] == b and e2[2] == a:
            return e2
    return None


def faithful_attribution(
    model: torch.nn.Module,
    data,
    user_kg: int,
    track_kg: int,
    *,
    contrast_track_kg: Optional[int] = None,
    k_anchor_candidates: int = 12,
    eval_undirected: bool = True,
    user_type: str = "user",
    item_type: str = "track",
    device: Optional[torch.device | str] = None,
) -> FaithfulAttribution:
    """Occlusion-based, model-faithful attribution for one recommendation.

    Re-runs the HGT with one connection removed at a time and records the drop in
    the recommendation score ``s(u,t)=<z_u, z_t>``.  Cheap (≈ one forward per
    occluded edge — a dozen or so) and exact (the model's real behaviour).  Two
    granularities are returned together:

    * ``edge_type_delta`` — drop each *relation* (both directions) → "which edge
      types drove this rec".
    * ``by_edge_type`` — drop each individual edge in the explanation subgraph
      (the user's most-attended listened tracks + the track's attribute
      neighbours) → per-edge "Δscore if removed", EdgeAttention-shaped.

    With ``contrast_track_kg`` every Δ is attributed to the **gap**
    ``s(u,T) − s(u,T′)`` — a contrastive "why T and not T′" explanation.

    The model is run on the **undirected** graph (training-consistent), matching
    how recommendations are scored.  Pure inference — no retraining.
    """
    import torch_geometric.transforms as T

    g = T.ToUndirected(merge=False)(data) if eval_undirected else data
    device = device or next(model.parameters()).device
    x_dict = {nt: g[nt].x.to(device) for nt in g.node_types
              if g[nt].get("x") is not None}
    base_eid = {et: g[et].edge_index.to(device) for et in g.edge_types}
    u, t = int(user_kg), int(track_kg)
    tc = None if contrast_track_kg is None else int(contrast_track_kg)
    model.eval()

    @torch.no_grad()
    def _val(eid) -> float:
        emb = model(x_dict, eid)
        s = float(torch.dot(emb[user_type][u], emb[item_type][t]))
        if tc is not None:
            s -= float(torch.dot(emb[user_type][u], emb[item_type][tc]))
        return s

    with torch.no_grad():
        base_emb = model(x_dict, base_eid)
        raw_sT = float(torch.dot(base_emb[user_type][u], base_emb[item_type][t]))
        raw_sTc = (None if tc is None
                   else float(torch.dot(base_emb[user_type][u], base_emb[item_type][tc])))
    s_full = raw_sT - (raw_sTc if tc is not None else 0.0)

    def _drop(eid, removals):
        out = dict(eid)
        for et, cols in removals.items():
            keep = torch.ones(eid[et].size(1), dtype=torch.bool, device=device)
            keep[cols] = False
            out[et] = eid[et][:, keep]
        return out

    def _with_reverse(et, cols):
        """Removal dict for ``cols`` of ``et`` plus the mirrored reverse edges."""
        removals = {et: cols if torch.is_tensor(cols)
                    else torch.as_tensor(cols, device=device)}
        rev = _reverse_et(g.edge_types, et)
        if rev is not None:
            ei, rei = base_eid[et], base_eid[rev]
            src = ei[0, removals[et]]; dst = ei[1, removals[et]]
            mask = torch.zeros(rei.size(1), dtype=torch.bool, device=device)
            for sgl, dgl in zip(src.tolist(), dst.tolist()):
                mask |= (rei[0] == dgl) & (rei[1] == sgl)
            rc = mask.nonzero(as_tuple=True)[0]
            if rc.numel():
                removals[rev] = rc
        return removals

    # ── edge-TYPE importance: drop each relation + its reverse together ───────
    edge_type_delta: Dict[EdgeType, float] = {}
    done: set = set()
    for et in g.edge_types:
        if et in done:
            continue
        rev = _reverse_et(g.edge_types, et)
        removals = {et: torch.arange(base_eid[et].size(1), device=device)}
        done.add(et)
        if rev is not None:
            removals[rev] = torch.arange(base_eid[rev].size(1), device=device)
            done.add(rev)
        edge_type_delta[et] = s_full - _val(_drop(base_eid, removals))

    # ── per-EDGE importance over the explanation subgraph ─────────────────────
    by_et: Dict[EdgeType, Tensor] = {
        et: torch.zeros(base_eid[et].size(1)) for et in g.edge_types}
    candidates: List[Tuple[EdgeType, int]] = []

    # (1) anchors — reverse-listened edges into the user; rank by attention and
    #     cap, so a heavy listener doesn't blow up the forward-pass budget.
    anchor_ets = [et for et in g.edge_types
                  if et[2] == user_type and et[0] == item_type]
    attn_rank = None
    if anchor_ets:
        try:
            _, _layers = capture_hgt_attention(model, x_dict, base_eid)
            attn_rank = _layers[-1]
        except Exception:                       # noqa: BLE001
            attn_rank = None
    for et in anchor_ets:
        cols = (base_eid[et][1] == u).nonzero(as_tuple=True)[0]
        if (attn_rank is not None and et in attn_rank
                and cols.numel() > k_anchor_candidates):
            w = attn_rank[et][cols.cpu()]
            cols = cols[torch.topk(w, k_anchor_candidates).indices.to(device)]
        candidates += [(et, int(c)) for c in cols.tolist()]

    # (2) drivers — the track's attribute neighbours (non-user sources), all.
    for et in g.edge_types:
        if et[2] == item_type and et[0] != user_type:
            cols = (base_eid[et][1] == t).nonzero(as_tuple=True)[0]
            candidates += [(et, int(c)) for c in cols.tolist()]

    for et, col in candidates:
        by_et[et][col] = s_full - _val(_drop(base_eid, _with_reverse(
            et, torch.tensor([col], device=device))))

    return FaithfulAttribution(
        by_edge_type=by_et, edge_type_delta=edge_type_delta,
        score=raw_sT, contrast_score=raw_sTc, contrastive=tc is not None)


def faithful_attribution_ig(
    model: torch.nn.Module,
    data,
    user_kg: int,
    track_kg: int,
    *,
    steps: int = 24,
    contrast_track_kg: Optional[int] = None,
    eval_undirected: bool = True,
    user_type: str = "user",
    item_type: str = "track",
    device: Optional[torch.device | str] = None,
) -> FaithfulAttribution:
    """Integrated-Gradients edge attribution (additive; sums to the score gap).

    Injects a per-edge mask ``m_e ∈ [0,1]`` into every ``HGTConv.message`` (the
    same patch trick as :func:`capture_hgt_attention`, but scaling the message),
    then integrates ``∂s/∂m`` along ``m: 0 → 1``.  By the completeness axiom the
    per-edge attributions **sum to** ``s(m=1) − s(m=0)`` (m=0 = no message
    passing), giving faithful additive shares.  Heavier than occlusion (``steps``
    forward+backward passes on the full graph); use when you want shares that add
    up rather than independent counterfactuals.  Returns the same
    :class:`FaithfulAttribution` shape (``edge_type_delta`` left empty).
    """
    import torch_geometric.transforms as T

    g = T.ToUndirected(merge=False)(data) if eval_undirected else data
    device = device or next(model.parameters()).device
    x_dict = {nt: g[nt].x.to(device) for nt in g.node_types
              if g[nt].get("x") is not None}
    eid = {et: g[et].edge_index.to(device) for et in g.edge_types}
    edge_types = list(eid.keys())
    counts = [int(eid[et].size(1)) for et in edge_types]
    offsets = np.cumsum([0] + counts)
    u, t = int(user_kg), int(track_kg)
    tc = None if contrast_track_kg is None else int(contrast_track_kg)

    convs = list(getattr(model, "convs"))
    mask = torch.zeros(int(offsets[-1]), device=device, requires_grad=True)

    def _make_patch(orig_message):
        # Defer to the real message (correct shape / version-proof), then gate each
        # edge's contribution by its mask value → differentiable edge-presence knob.
        def patched(k_j, q_i, v_j, edge_attr, index, ptr, size_i):
            msg = orig_message(k_j, q_i, v_j, edge_attr, index, ptr, size_i)
            return msg * mask.view(-1, *([1] * (msg.dim() - 1)))
        return patched

    model.eval()
    originals = [c.message for c in convs]
    grad_accum = torch.zeros_like(mask)
    try:
        for c in convs:
            c.message = _make_patch(c.message)   # type: ignore[assignment]
        for step in range(steps):
            a = (step + 0.5) / steps
            mask.data.fill_(a)
            if mask.grad is not None:
                mask.grad.zero_()
            emb = model(x_dict, eid)
            s = torch.dot(emb[user_type][u], emb[item_type][t])
            if tc is not None:
                s = s - torch.dot(emb[user_type][u], emb[item_type][tc])
            s.backward()
            grad_accum += mask.grad.detach()
    finally:
        for c, orig in zip(convs, originals):
            c.message = orig                     # type: ignore[assignment]

    attributions = (grad_accum / steps).detach().cpu()      # × (1 − 0) edge mask
    by_et = {et: attributions[offsets[i]:offsets[i + 1]].clone()
             for i, et in enumerate(edge_types)}
    return FaithfulAttribution(
        by_edge_type=by_et, edge_type_delta={}, score=float("nan"),
        contrastive=tc is not None)


def plot_edge_type_importance(fa: FaithfulAttribution, *, top: Optional[int] = None,
                              figsize=(7, 4), label_fn: Optional[Callable] = None):
    """Horizontal bar of Δscore per relation type (occlusion).  >0 supported the rec."""
    import matplotlib.pyplot as plt

    items = sorted(fa.edge_type_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if top:
        items = items[:top]
    items = items[::-1]
    labs = [(label_fn(et) if label_fn else f"{et[0]} —{et[1]}→ {et[2]}")
            for et, _ in items]
    vals = [d for _, d in items]
    cols = ["#2c7fb8" if v >= 0 else "#c0392b" for v in vals]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(labs)), vals, color=cols)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Δ score if this relation is removed  ( > 0 → it supported the rec )")
    ttl = "Which relation types drove this recommendation"
    if fa.contrastive:
        ttl += "  (contrastive: T vs T′)"
    ax.set_title(ttl, fontsize=10)
    fig.tight_layout()
    return fig


def plot_attention_vs_faithful(
    attn: "EdgeAttention",
    fa: FaithfulAttribution,
    *,
    label_fn: Optional[Callable] = None,
    top: Optional[int] = None,
    figsize=(9.0, 4.8),
    title: Optional[str] = None,
):
    """Side-by-side **attention vs faithful Δscore** for the *same* recommendation.

    The attention graph and the Δscore graph look almost identical (same anchors,
    same layout), so the *difference between the two rationales* is hard to see.
    This collapses both onto one axis: per relation type, the attention the HGT
    put on that relation's explanation edges versus the Δscore that relation
    actually contributed (drop-it-and-re-run). Bars are each normalised to their
    own max magnitude so the two very different scales are visually comparable;
    the **raw** value of each is annotated next to its bar, so you read the actual
    numbers. Where a tall attention bar sits next to a short/opposite Δscore bar,
    attention attended to something that did not move the score — the precise
    "attention ≠ explanation" gap.

    ``attn`` is the :class:`EdgeAttention` from :meth:`HGTExplainer.from_model`'s
    capture (``explainer.attn``); ``fa`` is the :class:`FaithfulAttribution` for
    the same ``(user, track)`` (``faithful_explainer.faithful``). Both index edges
    the same way, so attention is summed over exactly the edges Δscore scored.
    """
    import matplotlib.pyplot as plt

    ets_all = list(fa.by_edge_type)
    att = {et: 0.0 for et in fa.edge_type_delta}
    for et in ets_all:
        dvec = fa.by_edge_type[et]
        mask = dvec != 0
        if not bool(mask.any()):
            continue
        avec = getattr(attn, "by_edge_type", {}).get(et)
        if avec is None:
            continue
        a_sum = float(avec[mask].abs().sum())
        # Route onto the canonical relation that edge_type_delta keys on (it folds
        # each forward/reverse pair into one), via the reverse-relation finder.
        key = et if et in att else _reverse_et(ets_all, et)
        if key in att:
            att[key] += a_sum

    rows = [(et, att.get(et, 0.0), float(fa.edge_type_delta[et]))
            for et in fa.edge_type_delta]
    rows = [r for r in rows if r[1] != 0.0 or r[2] != 0.0]
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    if top:
        rows = rows[:top]
    rows = rows[::-1]
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no shared attention / Δscore edges to compare",
                ha="center", va="center")
        ax.axis("off")
        return fig

    labs = [(label_fn(et) if label_fn else f"{et[0]} —{et[1]}→ {et[2]}")
            for et, _, _ in rows]
    a = np.array([r[1] for r in rows], dtype=float)
    d = np.array([r[2] for r in rows], dtype=float)
    an = a / (np.abs(a).max() or 1.0)
    dn = d / (np.abs(d).max() or 1.0)

    y = np.arange(len(rows))
    h = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + h / 2, an, height=h, color="#9ecae1", label="attention (∑ on these edges)")
    ax.barh(y - h / 2, dn, height=h,
            color=["#2c7fb8" if v >= 0 else "#c0392b" for v in dn],
            label="faithful Δscore (drop & re-run)")
    for yi, (av, dv, avn, dvn) in enumerate(zip(a, d, an, dn)):
        ax.text(avn + 0.02 * (1 if avn >= 0 else -1), yi + h / 2, f"{av:.3g}",
                va="center", ha="left" if avn >= 0 else "right", fontsize=7, color="#225577")
        ax.text(dvn + 0.02 * (1 if dvn >= 0 else -1), yi - h / 2, f"{dv:+.3g}",
                va="center", ha="left" if dvn >= 0 else "right", fontsize=7, color="#222")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("normalised magnitude (bar = signal ÷ its own max; text = raw value)")
    ax.set_title(title or "Same recommendation, two rationales: attention vs faithful Δscore",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. Explanation containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Reason:
    """One attribute shared between the user's anchor tracks and the rec."""

    kind: str                       # "artist" | "genre" | "decade" | ...
    value: str                      # human-readable attribute label
    strength: float                 # Σ anchor-attention of supporting anchors (raw, deduped)
    anchors: List[Tuple[str, float]]  # [(anchor track label, anchor attn), ...]
    support: float = 0.0            # strength ÷ Σ attention on the shown anchors → [0,1]


@dataclass
class Explanation:
    """Structured, attention-grounded rationale for one (user, track) pair."""

    user_kg: int
    track_kg: int
    track_label: str
    score: Optional[float]
    is_hit: Optional[bool]
    anchors: List[dict] = field(default_factory=list)   # user's defining tracks
    drivers: List[dict] = field(default_factory=list)   # track's defining neighbours
    reasons: List[Reason] = field(default_factory=list)
    track_attrs: Dict[str, List[str]] = field(default_factory=dict)
    # Attention coverage: how much of each node's *total* incoming attention the
    # shown top-k edges account for — so the plot can be honest about the rest.
    # ``{"anchors"|"drivers": {"shown": Σattn_shown, "total": Σattn_all, "n_total": int}}``
    coverage: Dict[str, dict] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  3. The explainer
# ─────────────────────────────────────────────────────────────────────────────
class HGTExplainer:
    """Explain HGT recommendations from faithfully-captured edge attention.

    Build it once per trained model with :meth:`from_model` (which runs the
    capture), then call :meth:`explain` for any ``(user_kg, track_kg)`` pair.
    """

    def __init__(
        self,
        data,
        attention: EdgeAttention,
        node_mappings: Mapping[str, Sequence[str]],
        *,
        embeddings: Optional[Mapping[str, Tensor]] = None,
        track_label_fn: Optional[Callable[[int], Optional[str]]] = None,
        label_overrides: Optional[Mapping[str, Mapping[int, str]]] = None,
        user_type: str = "user",
        item_type: str = "track",
    ) -> None:
        self.data = data
        self.attn = attention
        self.node_mappings = node_mappings
        self.emb = embeddings
        self.user_type = user_type
        self.item_type = item_type
        self._track_label_fn = track_label_fn
        self._overrides = {k: dict(v) for k, v in (label_overrides or {}).items()}
        # Pre-index edge types by their destination / source node type for fast
        # incoming / outgoing look-ups.
        self._in_ets: Dict[str, List[EdgeType]] = {}
        self._out_ets: Dict[str, List[EdgeType]] = {}
        for et in data.edge_types:
            self._in_ets.setdefault(et[2], []).append(et)
            self._out_ets.setdefault(et[0], []).append(et)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        data,
        node_mappings: Mapping[str, Sequence[str]],
        *,
        layer: int = -1,
        head_reduce: str = "mean",
        keep_embeddings: bool = True,
        **kwargs,
    ) -> "HGTExplainer":
        """Capture attention from ``model`` over ``data`` and wrap it.

        Args:
            layer: Which conv layer's attention to explain with.  ``-1`` (the
                last layer, default) is closest to the scored embedding and the
                usual choice for a one-hop rationale.
            keep_embeddings: Store the forward-pass embeddings so :meth:`explain`
                can report the recommendation score.
        """
        x_dict = {nt: data[nt].x for nt in data.node_types
                  if data[nt].get("x") is not None}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        emb, layers = capture_hgt_attention(
            model, x_dict, edge_index_dict, head_reduce=head_reduce)
        return cls(data, layers[layer], node_mappings,
                   embeddings=emb if keep_embeddings else None, **kwargs)

    @classmethod
    def from_faithful(
        cls,
        model: torch.nn.Module,
        data,
        node_mappings: Mapping[str, Sequence[str]],
        user_kg: int,
        track_kg: int,
        *,
        contrast_track_kg: Optional[int] = None,
        eval_undirected: bool = True,
        **kwargs,
    ) -> "HGTExplainer":
        """Build an explainer whose edge weights are faithful **Δscore** values.

        Runs :func:`faithful_attribution` for this one ``(user_kg, track_kg)`` pair
        (occlusion on the undirected graph) and wraps the result so the *same*
        :meth:`explain` / :meth:`plot_explanation_graph` work — every edge weight
        is now "Δscore if removed" instead of attention.  The
        :class:`FaithfulAttribution` is stored on ``.faithful`` (use it with
        :func:`plot_edge_type_importance`).  Single-pair by construction, since
        occlusion is specific to the recommendation being explained.
        """
        import torch_geometric.transforms as T

        user_type = kwargs.get("user_type", "user")
        item_type = kwargs.get("item_type", "track")
        fa = faithful_attribution(
            model, data, user_kg, track_kg, contrast_track_kg=contrast_track_kg,
            eval_undirected=eval_undirected, user_type=user_type, item_type=item_type)
        g = T.ToUndirected(merge=False)(data) if eval_undirected else data
        dev = next(model.parameters()).device
        x_dict = {nt: g[nt].x.to(dev) for nt in g.node_types
                  if g[nt].get("x") is not None}
        eid = {et: g[et].edge_index.to(dev) for et in g.edge_types}
        model.eval()
        with torch.no_grad():
            emb = {k: v.detach().cpu() for k, v in model(x_dict, eid).items()}
        self = cls(g, fa, node_mappings, embeddings=emb, **kwargs)
        self.faithful = fa
        return self

    @classmethod
    def from_attribution(
        cls,
        model: torch.nn.Module,
        data,
        node_mappings: Mapping[str, Sequence[str]],
        attribution: "FaithfulAttribution",
        *,
        eval_undirected: bool = True,
        **kwargs,
    ) -> "HGTExplainer":
        """Wrap an **already-computed** :class:`FaithfulAttribution` as an explainer.

        Like :meth:`from_faithful`, but for an attribution you produced yourself
        (e.g. :func:`faithful_attribution_ig`) — it just runs one forward pass for
        the scoring embeddings and stores the attribution as the edge weights.
        ``attribution.by_edge_type`` must be aligned to ``data`` (already undirected
        when ``eval_undirected=False``), matching how the attribution was computed.
        """
        import torch_geometric.transforms as T

        g = T.ToUndirected(merge=False)(data) if eval_undirected else data
        dev = next(model.parameters()).device
        x_dict = {nt: g[nt].x.to(dev) for nt in g.node_types
                  if g[nt].get("x") is not None}
        eid = {et: g[et].edge_index.to(dev) for et in g.edge_types}
        model.eval()
        with torch.no_grad():
            emb = {k: v.detach().cpu() for k, v in model(x_dict, eid).items()}
        self = cls(g, attribution, node_mappings, embeddings=emb, **kwargs)
        self.faithful = attribution
        return self

    # ── labels ───────────────────────────────────────────────────────────────
    @staticmethod
    def _uri_tail(uri: str) -> str:
        """Readable tail of a URI: last ``/`` or ``#`` segment, de-slugged."""
        tail = uri.rstrip("/#").replace("#", "/").rsplit("/", 1)[-1]
        return tail.replace("_", " ").replace("%20", " ") or uri

    def node_label(self, ntype: str, nid: int) -> str:
        """Human-readable label for a KG node (overrides → track fn → URI tail)."""
        nid = int(nid)
        if ntype in self._overrides and nid in self._overrides[ntype]:
            return self._overrides[ntype][nid]
        if ntype == self.item_type and self._track_label_fn is not None:
            lab = self._track_label_fn(nid)
            if lab:
                return lab
        uris = self.node_mappings.get(ntype)
        if uris is not None and 0 <= nid < len(uris):
            return self._uri_tail(str(uris[nid]))
        return f"{ntype}#{nid}"

    # ── graph queries ────────────────────────────────────────────────────────
    def incoming(
        self, dst_type: str, dst_id: int, *, relations: Optional[Sequence[EdgeType]] = None,
    ) -> List[Tuple[EdgeType, int, float]]:
        """Edges pointing *into* ``dst_id`` with their captured attention.

        Returns ``[(edge_type, src_id, attention), ...]`` sorted high→low.
        Attention is normalised across all of ``dst_id``'s incoming edges, so it
        reads as "share of this node's incoming message".
        """
        dst_id = int(dst_id)
        ets = relations if relations is not None else self._in_ets.get(dst_type, [])
        out: List[Tuple[EdgeType, int, float]] = []
        for et in ets:
            if et not in self.attn:
                continue
            ei = self.data[et].edge_index
            mask = ei[1] == dst_id
            if not bool(mask.any()):
                continue
            src = ei[0][mask].tolist()
            w = self.attn[et][mask].tolist()
            out.extend((et, int(s), float(a)) for s, a in zip(src, w))
        out.sort(key=lambda t: t[2], reverse=True)
        return out

    def outgoing(self, src_type: str, src_id: int, relation: EdgeType) -> List[int]:
        """Destination node ids reachable from ``src_id`` via ``relation``."""
        ei = self.data[relation].edge_index
        mask = ei[0] == int(src_id)
        return ei[1][mask].tolist()

    def track_attributes(self, track_kg: int) -> Dict[str, List[Tuple[int, str]]]:
        """All descriptive attributes of a track as ``kind -> [(node_id, label)]``.

        ``genre`` is resolved through the track's artist(s); the rest are direct
        ``track -> attribute`` edges.  Silently skips relations absent from the
        graph so it works on reduced schemas too.
        """
        attrs: Dict[str, List[Tuple[int, str]]] = {}
        for kind, rel in _TRACK_ATTR_RELS.items():
            if rel not in self.data.edge_types:
                continue
            ids = self.outgoing("track", track_kg, rel)
            if ids:
                attrs[kind] = [(int(i), self.node_label(rel[2], i)) for i in ids]
        # genre via artist (2-hop)
        if _ARTIST_GENRE_REL in self.data.edge_types:
            genres: Dict[int, str] = {}
            for aid, _ in attrs.get("artist", []):
                for gid in self.outgoing("artist", aid, _ARTIST_GENRE_REL):
                    genres[int(gid)] = self.node_label("genre", gid)
            if genres:
                attrs["genre"] = sorted(genres.items())
        return attrs

    # ── the explanation ──────────────────────────────────────────────────────
    def explain(
        self,
        user_kg: int,
        track_kg: int,
        *,
        top_k_anchors: int = 5,
        top_k_drivers: int = 8,
        is_hit: Optional[bool] = None,
    ) -> Explanation:
        """Build a full attention-grounded explanation for ``(user, track)``."""
        user_kg, track_kg = int(user_kg), int(track_kg)

        score = None
        if self.emb is not None:
            u = self.emb[self.user_type][user_kg]
            t = self.emb[self.item_type][track_kg]
            score = float(torch.dot(u, t))

        # 1) User anchors — the listened tracks that most shaped U's embedding.
        anchor_edges = [(s, a) for et, s, a in self.incoming(self.user_type, user_kg)
                        if et[0] == self.item_type]        # track→user messages, sorted
        anchors: List[dict] = [{
            "track_kg": s,
            "label": self.node_label(self.item_type, s),
            "attn": a,
            "attrs": self.track_attributes(s),
        } for s, a in anchor_edges[:top_k_anchors]]

        # 2) Track drivers — the neighbours that most shaped T's embedding.
        driver_edges = self.incoming(self.item_type, track_kg)   # all relations, sorted
        drivers: List[dict] = [{
            "rel": et[1],
            "ntype": et[0],
            "node_kg": s,
            "label": self.node_label(et[0], s),
            "attn": a,
        } for et, s, a in driver_edges[:top_k_drivers]]

        # Coverage — what fraction of each node's total incoming attention is shown.
        coverage = {
            "anchors": {"shown": float(sum(a for _, a in anchor_edges[:top_k_anchors])),
                        "total": float(sum(a for _, a in anchor_edges)),
                        "n_total": len(anchor_edges)},
            "drivers": {"shown": float(sum(a for *_, a in driver_edges[:top_k_drivers])),
                        "total": float(sum(a for *_, a in driver_edges)),
                        "n_total": len(driver_edges)},
        }

        # 3) Reasons — attributes shared between the anchors and the rec.
        # Each anchor contributes its attention to a shared attribute AT MOST ONCE
        # (dedupe by node id), so an anchor that exposes the same attribute through
        # several edges — common for `instrument` — can't inflate the total.
        track_attrs = self.track_attributes(track_kg)
        track_attr_ids = {k: {i for i, _ in v} for k, v in track_attrs.items()}
        anchor_attn_used = float(sum(a["attn"] for a in anchors)) or 1.0
        reasons: List[Reason] = []
        for kind, t_ids in track_attr_ids.items():
            buckets: Dict[int, Tuple[str, List[Tuple[str, float]]]] = {}
            for a in anchors:
                for nid in {i for i, _ in a["attrs"].get(kind, [])} & t_ids:
                    lab = next(l for i, l in a["attrs"][kind] if i == nid)
                    _, supp = buckets.setdefault(nid, (lab, []))
                    supp.append((a["label"], a["attn"]))
            for nid, (lab, supp) in buckets.items():
                strength = float(sum(w for _, w in supp))
                reasons.append(Reason(
                    kind=kind, value=lab, strength=strength,
                    support=strength / anchor_attn_used,
                    anchors=sorted(supp, key=lambda x: x[1], reverse=True),
                ))
        reasons.sort(key=lambda r: r.support, reverse=True)

        return Explanation(
            user_kg=user_kg, track_kg=track_kg,
            track_label=self.node_label(self.item_type, track_kg),
            score=score, is_hit=is_hit,
            anchors=anchors, drivers=drivers, reasons=reasons,
            track_attrs={k: [lab for _, lab in v] for k, v in track_attrs.items()},
            coverage=coverage,
        )

    def reweight_like(self, canonical: "Explanation") -> "Explanation":
        """Re-express ``canonical``'s anchors/reasons with **this** explainer's weights.

        Keeps the *same* anchor set, ordering and shared-attribute reasons as
        ``canonical`` (so several strategies draw on identical layouts), but swaps
        every weight for the one this explainer holds — its captured attention,
        occlusion Δscore or IG attribution. Each anchor's value becomes this
        explainer's ``track → user`` edge weight; each reason's strength/support is
        re-summed from the re-weighted backing anchors. The score is reused (the
        recommendation is the same ``(user, track)`` under the same model).
        """
        w_by_track: Dict[int, float] = {}
        for et, src, w in self.incoming(self.user_type, canonical.user_kg):
            if et[0] == self.item_type:
                w_by_track.setdefault(int(src), float(w))
        new_anchors = [dict(a, attn=w_by_track.get(int(a["track_kg"]), 0.0))
                       for a in canonical.anchors]
        w_by_label = {a["label"]: a["attn"] for a in new_anchors}
        denom = float(sum(abs(a["attn"]) for a in new_anchors)) or 1.0
        new_reasons: List[Reason] = []
        for r in canonical.reasons:
            backing = [(lab, float(w_by_label.get(lab, 0.0))) for lab, _ in r.anchors]
            strength = float(sum(w for _, w in backing))
            new_reasons.append(Reason(
                kind=r.kind, value=r.value, strength=strength,
                support=strength / denom,
                anchors=sorted(backing, key=lambda x: abs(x[1]), reverse=True)))
        return Explanation(
            user_kg=canonical.user_kg, track_kg=canonical.track_kg,
            track_label=canonical.track_label, score=canonical.score,
            is_hit=canonical.is_hit, anchors=new_anchors, drivers=canonical.drivers,
            reasons=new_reasons, track_attrs=canonical.track_attrs,
            coverage=canonical.coverage)

    # ── renderings ───────────────────────────────────────────────────────────
    def explain_text(self, expl: Explanation, *, max_reasons: int = 6) -> str:
        """Render an :class:`Explanation` as a readable multi-line string."""
        lines: List[str] = []
        verdict = ("" if expl.is_hit is None
                   else "  [HIT ✓]" if expl.is_hit else "  [miss ✗]")
        score = "" if expl.score is None else f"  (score={expl.score:+.3f})"
        lines.append(f"Why was “{expl.track_label}” recommended to "
                     f"user#{expl.user_kg}?{score}{verdict}")

        if expl.track_attrs:
            attr_str = "; ".join(
                f"{k}={', '.join(v[:3])}" for k, v in expl.track_attrs.items())
            lines.append(f"  • the track is: {attr_str}")

        if expl.reasons:
            lines.append("  • shared with your most-attended tracks "
                         "(support = share of your attention on tracks that share it):")
            for r in expl.reasons[:max_reasons]:
                who = ", ".join(f"“{lab}” (attn {w:.0%})" for lab, w in r.anchors[:2])
                lines.append(f"      – same {r.kind} “{r.value}” "
                             f"(support {r.support:.0%}) ← {who}")
        else:
            lines.append("  • no explicit attribute overlap with your top tracks — "
                         "ranked via latent KGE/audio similarity rather than a "
                         "shared artist/genre/decade.")

        if expl.anchors:
            top = ", ".join(f"“{a['label']}” ({a['attn']:.0%})"
                            for a in expl.anchors[:4])
            lines.append(f"  • your taste anchors (attention on listened tracks): {top}")
        return "\n".join(lines)

    def plot_explanation(self, expl: Explanation, *, figsize=(12, 5),
                         attn_label: str = "attention"):
        """Two-panel matplotlib figure: anchor weights + reason strengths.

        ``attn_label`` renames the anchor-axis (pass e.g. ``"Δscore"`` when the
        weights came from :func:`faithful_attribution` rather than attention).
        """
        import matplotlib.pyplot as plt

        # The weights are *attention shares* (0–1) or *signed Δscores* depending on
        # the strategy — derive the number format from the supplied label so the
        # attention and faithful figures read differently (and show real values).
        is_share = any(t in attn_label.lower() for t in ("attention", "share", "%"))
        fmt_w = (lambda v: f"{v:.0%}") if is_share else (lambda v: f"{v:+.3f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1 — user anchors by the strategy's weight, flagged if they back a reason.
        backing = {lab for r in expl.reasons for lab, _ in r.anchors}
        labs = [a["label"][:34] for a in expl.anchors][::-1]
        vals = [a["attn"] for a in expl.anchors][::-1]
        cols = ["seagreen" if a["label"] in backing else "steelblue"
                for a in expl.anchors][::-1]
        if labs:
            ax1.barh(range(len(labs)), vals, color=cols)
            ax1.set_yticks(range(len(labs)))
            ax1.set_yticklabels(labs, fontsize=8)
            ax1.set_xlabel(f"{attn_label}: U → listened track")
            # Print the actual value at each bar tip so the two strategies are
            # numerically comparable for the SAME anchors/recommendation.
            _vmax = max((abs(v) for v in vals), default=1.0) or 1.0
            for i, v in enumerate(vals):
                ax1.text(v + (0.01 * _vmax if v >= 0 else -0.01 * _vmax), i, fmt_w(v),
                         va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
            ax1.margins(x=0.18)
        ax1.set_title(f"Your taste anchors — by {attn_label}\n"
                      "(green = shares an attribute with the rec)", fontsize=9)

        # Panel 2 — reasons (shared attributes) by support (normalised share).
        rlabs = [f"{r.kind}: {r.value}"[:34] for r in expl.reasons][:8][::-1]
        rvals = [r.support for r in expl.reasons][:8][::-1]
        if rlabs:
            ax2.barh(range(len(rlabs)), rvals, color="darkorange")
            ax2.set_yticks(range(len(rlabs)))
            ax2.set_yticklabels(rlabs, fontsize=8)
            ax2.set_xlim(0, 1)
            ax2.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
            ax2.set_xlabel(f"support — share of your {attn_label} on tracks that share it")
        else:
            ax2.text(0.5, 0.5, "no shared-attribute reasons", ha="center",
                     va="center", fontsize=10, style="italic")
            ax2.set_axis_off()
        ax2.set_title("Why this track (shared-attribute reasons)", fontsize=9)

        verdict = ("" if expl.is_hit is None else
                   "  [HIT]" if expl.is_hit else "  [miss]")
        fig.suptitle(f"HGT explanation [{attn_label}] — “{expl.track_label}” "
                     f"→ user#{expl.user_kg}{verdict}", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_explanation_graph(self, expl: Explanation, *, top_k: int = 5,
                               top_k_drivers: int = 5, figsize=(14, 8),
                               attn_label: str = "attention",
                               caption: Optional[str] = None, ax=None):
        """Draw *how the HGT built this recommendation* as a left->right pipeline::

            [USER] ->attn-> tracks you played ->links-> shared attributes ->support-> [REC]

        It reads in reading order: the listener; the listened tracks the model
        most attends to (edge label = that track's share of the user's incoming
        attention); the attributes those tracks share with the candidate (thin
        links mark which track carries which attribute); and the recommended
        track, where each shared attribute's arrow is labelled with its *support*
        - the share of the user's attention sitting on tracks that carry it, i.e.
        the attribute's weight on the final selection. Anchors that back a shared
        attribute are highlighted; a caption reports attention coverage. Pure
        matplotlib, no deps.
        """
        import textwrap

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        anchors = expl.anchors[:top_k]
        reasons = expl.reasons[:top_k_drivers]          # shared attributes, by support
        backing = {lab for r in reasons for lab, _ in r.anchors}

        # Attention shares (0–1) print as %, signed Δscores print with a sign — so
        # the faithful graph never looks like a relabelled copy of the attention one.
        is_share = any(t in attn_label.lower() for t in ("attention", "share", "%"))
        fmt_w = (lambda v: f"{v:.0%}") if is_share else (lambda v: f"{v:+.3f}")

        xU, xA, xS, xT = 0.0, 1.9, 3.9, 5.7
        yU = yT = 0.5
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        def _ys(n, lo=0.12, hi=0.88):
            if n == 0:
                return []
            if n == 1:
                return [(lo + hi) / 2]
            return list(np.linspace(hi, lo, n))

        yA, yS = _ys(len(anchors)), _ys(len(reasons))
        anchor_y = {a["label"]: y for y, a in zip(yA, anchors)}

        def _box(x, y, text, fc, *, ec="#444", lw=1.0, width=15, fs=9, bold=False):
            ax.text(x, y, textwrap.fill(str(text), width), ha="center", va="center",
                    fontsize=fs, zorder=5, fontweight=("bold" if bold else "normal"),
                    bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=lw))

        def _edge(x0, y0, x1, y1, frac, color, label=None, *, shrinkA=30, shrinkB=30):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.85,
                                        lw=1.3 + 5.0 * frac, shrinkA=shrinkA,
                                        shrinkB=shrinkB))
            if label is not None:
                ax.text(x0 + 0.46 * (x1 - x0), y0 + 0.46 * (y1 - y0), label,
                        fontsize=9.5, color=color, ha="center", va="center", zorder=6,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=color, lw=0.8))

        # 1) USER -> listened tracks (label = the track's attention share of U).
        amax_a = max((abs(a["attn"]) for a in anchors), default=1.0) or 1.0
        for y, a in zip(yA, anchors):
            _edge(xU, yU, xA, y, abs(a["attn"]) / amax_a, "#2c7fb8",
                  fmt_w(a["attn"]), shrinkA=34, shrinkB=26)

        # 2) listened track -> shared attribute (thin links: which track carries it)
        #    and 3) shared attribute -> rec (label = support = weight on selection).
        smax = max((r.support for r in reasons), default=1.0) or 1.0
        for ys_, r in zip(yS, reasons):
            for lab, _w in r.anchors:
                if lab in anchor_y:
                    ax.annotate("", xy=(xS, ys_), xytext=(xA, anchor_y[lab]), zorder=1,
                                arrowprops=dict(arrowstyle="-", color="#bdbdbd",
                                                alpha=0.7, lw=1.0,
                                                shrinkA=26, shrinkB=24))
            _edge(xS, ys_, xT, yT, r.support / smax, "#c97a14",
                  f"{r.support:.0%}", shrinkA=24, shrinkB=34)

        # nodes
        _box(xU, yU, f"USER #{expl.user_kg}", "#aacbe6", ec="#1f5e85", lw=1.6,
             width=12, bold=True)
        for y, a in zip(yA, anchors):
            _hot = a["label"] in backing
            _box(xA, y, a["label"], "#cdeac0" if _hot else "#d7ebf7",
                 ec="#3f8f3f" if _hot else "#5a8bb0", lw=1.4 if _hot else 0.9, width=15)
        for ys_, r in zip(yS, reasons):
            _box(xS, ys_, f"{r.kind}: {r.value}", "#ffe1a8", ec="#cc9a06",
                 lw=1.4, width=15, bold=True)
        _box(xT, yT, expl.track_label, "#f6c6c6", ec="#9c2b2b", lw=1.6,
             width=16, bold=True)

        # No shared-attribute path -> connect directly, latent-similarity note.
        if not reasons:
            ax.annotate("", xy=(xT, yT), xytext=(xA if anchors else xU, yU), zorder=1,
                        arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.6,
                                        ls="--", shrinkA=30, shrinkB=34))
            ax.text(xS, yU + 0.12, "no shared attribute -\nranked by latent "
                    "KGE/audio similarity", fontsize=8.5, ha="center", va="center",
                    style="italic", color="#666")

        # score chip under the recommended track
        if expl.score is not None:
            ax.text(xT, yT - 0.14, f"score {expl.score:+.2f}", fontsize=9,
                    ha="center", va="top", color="#9c2b2b", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="#fdeaea",
                              ec="#9c2b2b", lw=0.7))

        # column headers
        for x, lab in [(xU, "listener"), (xA, f"tracks you played\n({attn_label})"),
                       (xS, "shared attributes"), (xT, "recommended track")]:
            ax.text(x, 0.99, lab, fontsize=9, ha="center", va="bottom",
                    fontweight="bold", color="#333")

        ax.set_xlim(-0.7, 6.4)
        ax.set_ylim(-0.04, 1.10)
        ax.axis("off")

        legend = [
            Line2D([0], [0], color="#2c7fb8", lw=3,
                   label=f"{attn_label}: how much each track defines you"),
            Line2D([0], [0], color="#bdbdbd", lw=2,
                   label="track carries this attribute"),
            Line2D([0], [0], color="#c97a14", lw=3,
                   label="support: the attribute's weight on the rec"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#cdeac0",
                   markeredgecolor="#3f8f3f", markersize=12,
                   label="anchor that shares an attribute"),
        ]
        ax.legend(handles=legend, loc="lower center", ncol=2, fontsize=8.5,
                  frameon=False, bbox_to_anchor=(0.5, -0.08))

        def _cov(side, shown_n):
            c = expl.coverage.get(side, {})
            tot = c.get("total", 0.0) or 1.0
            return (f"top {shown_n} of {c.get('n_total', shown_n)} "
                    f"({c.get('shown', 0.0) / tot:.0%} of its attention)")
        if caption is not None:
            cap = caption
        elif is_share:
            cap = ("Edge labels = genuine HGT softmax attention (averaged over heads). "
                   "USER->track = the track's share of your incoming message "
                   f"({_cov('anchors', len(anchors))}); attribute->rec = support, the "
                   "share of your attention on tracks that carry that attribute.")
        else:
            cap = (f"Edge labels = {attn_label}: the counterfactual change in the "
                   "recommendation score when that listened-track edge is removed "
                   "(> 0 = it supported the rec). attribute->rec = support, the "
                   "share of that signed weight sitting on tracks carrying the attribute.")
        if own_fig:
            fig.text(0.5, 0.01, cap, ha="center", va="bottom", fontsize=8,
                     color="#555", wrap=True)
        else:
            ax.text(0.5, -0.06, cap, transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color="#555", wrap=True)

        verdict = "" if expl.is_hit is None else ("  [HIT]" if expl.is_hit else "  [miss]")
        ax.set_title(f"How the HGT built this recommendation — {attn_label}{verdict}",
                     fontsize=12, fontweight="bold")
        if own_fig:
            fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.12)
        return fig


def build_attribution_panels(
    model: torch.nn.Module,
    data,
    node_mappings: Mapping[str, Sequence[str]],
    user_kg: int,
    track_kg: int,
    *,
    strategies: Sequence[str] = ("attention", "delta", "ig"),
    attn_explainer: "Optional[HGTExplainer]" = None,
    contrast_track_kg: Optional[int] = None,
    ig_steps: int = 24,
    top_k_anchors: int = 5,
    top_k_drivers: int = 5,
    eval_undirected: bool = True,
    user_type: str = "user",
    item_type: str = "track",
    **explainer_kwargs,
):
    """Build aligned per-strategy explanation panels for ONE ``(user, track)`` rec.

    Returns ``(panels, canonical)`` where ``panels`` is a list of
    ``(label, HGTExplainer, Explanation)`` — one per requested strategy, all sharing
    the **same** anchor set / ordering (taken from the attention rationale) so only
    the per-edge *values* differ:

    * ``"attention"`` — captured softmax (what the model looked at);
    * ``"delta"``     — occlusion Δscore (what moved the score, signed);
    * ``"ig"``        — integrated-gradients edge attribution (additive).

    Pass ``attn_explainer`` to reuse an already-built attention explainer (e.g.
    :class:`ColdStartHGT`'s, whose graph already carries the synthetic user node);
    its ``.data`` is then used as the evaluation graph and the occlusion/IG passes
    run on it directly (``eval_undirected`` is ignored in that case). Otherwise the
    undirected graph is built from ``data``. Costs a handful of extra forward
    passes (occlusion ≈ a dozen, IG ≈ ``ig_steps``), so it is meant for one rec at
    a time / behind a button.
    """
    import torch_geometric.transforms as T

    if attn_explainer is not None:
        attn_ex = attn_explainer
        g = attn_ex.data                       # already the eval graph (undirected)
        eu = False
    else:
        g = T.ToUndirected(merge=False)(data) if eval_undirected else data
        attn_ex = HGTExplainer.from_model(
            model, g, node_mappings, user_type=user_type, item_type=item_type,
            **explainer_kwargs)
        eu = False                             # g is already the eval graph

    canonical = attn_ex.explain(user_kg, track_kg, top_k_anchors=top_k_anchors,
                                top_k_drivers=top_k_drivers)
    panels: List[Tuple[str, "HGTExplainer", "Explanation"]] = []
    for s in strategies:
        if s == "attention":
            panels.append(("attention", attn_ex, canonical))
        elif s == "delta":
            fa = faithful_attribution(
                model, g, user_kg, track_kg, contrast_track_kg=contrast_track_kg,
                eval_undirected=eu, user_type=user_type, item_type=item_type)
            ex = HGTExplainer.from_attribution(
                model, g, node_mappings, fa, eval_undirected=eu,
                user_type=user_type, item_type=item_type, **explainer_kwargs)
            panels.append(("Δscore", ex, ex.reweight_like(canonical)))
        elif s == "ig":
            fa = faithful_attribution_ig(
                model, g, user_kg, track_kg, steps=ig_steps,
                contrast_track_kg=contrast_track_kg, eval_undirected=eu,
                user_type=user_type, item_type=item_type)
            ex = HGTExplainer.from_attribution(
                model, g, node_mappings, fa, eval_undirected=eu,
                user_type=user_type, item_type=item_type, **explainer_kwargs)
            panels.append(("IG", ex, ex.reweight_like(canonical)))
        else:
            raise ValueError(f"unknown strategy {s!r} (use attention/delta/ig)")
    return panels, canonical


def plot_explanation_graphs(
    panels,
    *,
    suptitle: Optional[str] = None,
    figsize_per=(8.5, 7.5),
    top_k: int = 5,
    top_k_drivers: int = 5,
):
    """Draw several strategies' rationale graphs **side by side** for one rec.

    ``panels`` is the ``[(label, HGTExplainer, Explanation), ...]`` list from
    :func:`build_attribution_panels`. Every panel uses the same anchors/layout, so
    reading left→right across panels shows how attention, Δscore and IG put
    *different values on the same edges*.
    """
    import matplotlib.pyplot as plt

    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]
    for ax, (label, ex, expl) in zip(axes, panels):
        ex.plot_explanation_graph(expl, ax=ax, attn_label=label, top_k=top_k,
                                  top_k_drivers=top_k_drivers)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


__all__ = (
    "EdgeAttention",
    "capture_hgt_attention",
    "FaithfulAttribution",
    "faithful_attribution",
    "faithful_attribution_ig",
    "plot_edge_type_importance",
    "plot_attention_vs_faithful",
    "build_attribution_panels",
    "plot_explanation_graphs",
    "Reason",
    "Explanation",
    "HGTExplainer",
)
