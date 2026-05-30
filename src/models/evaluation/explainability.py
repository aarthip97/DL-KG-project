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
#  2. Explanation containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Reason:
    """One attribute shared between the user's anchor tracks and the rec."""

    kind: str                       # "artist" | "genre" | "decade" | ...
    value: str                      # human-readable attribute label
    strength: float                 # Σ anchor-attention of supporting anchors
    anchors: List[Tuple[str, float]]  # [(anchor track label, anchor attn), ...]


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
        anchors: List[dict] = []
        for et, src_id, attn in self.incoming(self.user_type, user_kg):
            if et[0] != self.item_type:        # only track→user messages
                continue
            anchors.append({
                "track_kg": src_id,
                "label": self.node_label(self.item_type, src_id),
                "attn": attn,
                "attrs": self.track_attributes(src_id),
            })
            if len(anchors) >= top_k_anchors:
                break

        # 2) Track drivers — the neighbours that most shaped T's embedding.
        drivers: List[dict] = []
        for et, src_id, attn in self.incoming(self.item_type, track_kg)[:top_k_drivers]:
            drivers.append({
                "rel": et[1],
                "ntype": et[0],
                "node_kg": src_id,
                "label": self.node_label(et[0], src_id),
                "attn": attn,
            })

        # 3) Reasons — attributes shared between the anchors and the rec.
        track_attrs = self.track_attributes(track_kg)
        track_attr_ids = {k: {i for i, _ in v} for k, v in track_attrs.items()}
        reasons: List[Reason] = []
        for kind, t_ids in track_attr_ids.items():
            buckets: Dict[int, Tuple[str, List[Tuple[str, float]]]] = {}
            for a in anchors:
                for nid, lab in a["attrs"].get(kind, []):
                    if nid in t_ids:
                        _, supp = buckets.setdefault(nid, (lab, []))
                        supp.append((a["label"], a["attn"]))
            for nid, (lab, supp) in buckets.items():
                reasons.append(Reason(
                    kind=kind, value=lab,
                    strength=float(sum(w for _, w in supp)),
                    anchors=sorted(supp, key=lambda x: x[1], reverse=True),
                ))
        reasons.sort(key=lambda r: r.strength, reverse=True)

        return Explanation(
            user_kg=user_kg, track_kg=track_kg,
            track_label=self.node_label(self.item_type, track_kg),
            score=score, is_hit=is_hit,
            anchors=anchors, drivers=drivers, reasons=reasons,
            track_attrs={k: [lab for _, lab in v] for k, v in track_attrs.items()},
        )

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
            lines.append("  • shared with your most-attended tracks:")
            for r in expl.reasons[:max_reasons]:
                who = ", ".join(f"“{lab}” (attn {w:.0%})" for lab, w in r.anchors[:2])
                lines.append(f"      – same {r.kind} “{r.value}” "
                             f"(strength {r.strength:.2f}) ← {who}")
        else:
            lines.append("  • no explicit attribute overlap with your top tracks — "
                         "ranked via latent KGE/audio similarity rather than a "
                         "shared artist/genre/decade.")

        if expl.anchors:
            top = ", ".join(f"“{a['label']}” ({a['attn']:.0%})"
                            for a in expl.anchors[:4])
            lines.append(f"  • your taste anchors (attention on listened tracks): {top}")
        return "\n".join(lines)

    def plot_explanation(self, expl: Explanation, *, figsize=(12, 5)):
        """Two-panel matplotlib figure: anchor attention + reason strengths."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1 — user anchors by attention, flagged if they back a reason.
        backing = {lab for r in expl.reasons for lab, _ in r.anchors}
        labs = [a["label"][:34] for a in expl.anchors][::-1]
        vals = [a["attn"] for a in expl.anchors][::-1]
        cols = ["seagreen" if a["label"] in backing else "steelblue"
                for a in expl.anchors][::-1]
        if labs:
            ax1.barh(range(len(labs)), vals, color=cols)
            ax1.set_yticks(range(len(labs)))
            ax1.set_yticklabels(labs, fontsize=8)
            ax1.set_xlabel("attention U → listened track")
        ax1.set_title("Your taste anchors\n(green = shares an attribute with the rec)",
                      fontsize=9)

        # Panel 2 — reasons (shared attributes) by strength.
        rlabs = [f"{r.kind}: {r.value}"[:34] for r in expl.reasons][:8][::-1]
        rvals = [r.strength for r in expl.reasons][:8][::-1]
        if rlabs:
            ax2.barh(range(len(rlabs)), rvals, color="darkorange")
            ax2.set_yticks(range(len(rlabs)))
            ax2.set_yticklabels(rlabs, fontsize=8)
            ax2.set_xlabel("Σ anchor-attention supporting this reason")
        else:
            ax2.text(0.5, 0.5, "no shared-attribute reasons", ha="center",
                     va="center", fontsize=10, style="italic")
            ax2.set_axis_off()
        ax2.set_title("Why this track (shared-attribute reasons)", fontsize=9)

        verdict = ("" if expl.is_hit is None else
                   "  [HIT]" if expl.is_hit else "  [miss]")
        fig.suptitle(f"HGT explanation — “{expl.track_label}” → user#{expl.user_kg}"
                     f"{verdict}", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_explanation_graph(self, expl: Explanation, *, top_k: int = 4,
                               figsize=(11, 6)):
        """Draw the attention-weighted rationale as an actual node-edge graph.

        Layout (left→right): user · top anchor tracks · shared-attribute nodes ·
        recommended track.  Edge thickness/label on the user→anchor edges is the
        captured attention; anchor→attribute→rec links trace the shared
        attributes that justify the recommendation.  Pure matplotlib, no deps.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        anchors = expl.anchors[:top_k]
        reasons = expl.reasons[:6]

        def _wrap(s, n=22):
            s = s if len(s) <= n else s[: n - 1] + "…"
            return s

        # Column x-positions.
        xU, xA, xR, xT = 0.0, 1.0, 2.0, 3.0
        fig, ax = plt.subplots(figsize=figsize)

        # Node y-positions, vertically centred per column.
        def _ys(n):
            return list(np.linspace(0.9, 0.1, n)) if n else []

        yU = [0.5]
        yA = _ys(len(anchors))
        yR = _ys(len(reasons))
        yT = [0.5]

        # Edges: user → anchors (width = attention).
        amax = max((a["attn"] for a in anchors), default=1.0) or 1.0
        for ya, a in zip(yA, anchors):
            ax.plot([xU, xA], [yU[0], ya], color="steelblue",
                    lw=0.5 + 3.5 * a["attn"] / amax, alpha=0.7, zorder=1)
            ax.text((xU + xA) / 2, (yU[0] + ya) / 2, f"{a['attn']:.0%}",
                    fontsize=7, color="steelblue", ha="center", va="center")

        # Edges: anchors → reason (if the anchor backs it) and reason → track.
        reason_y = {f"{r.kind}: {r.value}": yr for r, yr in zip(reasons, yR)}
        for r, yr in zip(reasons, yR):
            key = f"{r.kind}: {r.value}"
            backing_labels = {lab for lab, _ in r.anchors}
            for ya, a in zip(yA, anchors):
                if a["label"] in backing_labels:
                    ax.plot([xA, xR], [ya, yr], color="seagreen", lw=1.0,
                            alpha=0.55, zorder=1)
            ax.plot([xR, xT], [yr, yT[0]], color="darkorange", lw=1.4,
                    alpha=0.8, zorder=1)

        # Nodes.
        def _node(x, y, text, fc):
            ax.scatter([x], [y], s=420, color=fc, edgecolor="black",
                       zorder=3, linewidths=0.6)
            ax.text(x, y, _wrap(text), fontsize=7.5, ha="center", va="center",
                    zorder=4)

        _node(xU, yU[0], f"user#{expl.user_kg}", "lightsteelblue")
        for ya, a in zip(yA, anchors):
            _node(xA, ya, a["label"], "#cfe3f3")
        for r, yr in zip(reasons, yR):
            _node(xR, yr, f"{r.kind}\n{r.value}", "#ffe6c2")
        _node(xT, yT[0], expl.track_label, "#ffd0d0")

        for x, lab in [(xU, "user"), (xA, "taste anchors"),
                       (xR, "shared attributes"), (xT, "recommended")]:
            ax.text(x, 1.02, lab, fontsize=9, ha="center", va="bottom",
                    fontweight="bold")

        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(0.0, 1.12)
        ax.axis("off")
        legend = [
            Line2D([0], [0], color="steelblue", lw=3, label="U→track attention"),
            Line2D([0], [0], color="seagreen", lw=2, label="anchor shares attribute"),
            Line2D([0], [0], color="darkorange", lw=2, label="attribute of rec"),
        ]
        ax.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
                  frameon=False, bbox_to_anchor=(0.5, -0.06))
        verdict = ("" if expl.is_hit is None else
                   "  [HIT]" if expl.is_hit else "  [miss]")
        ax.set_title(f"Attention rationale{verdict}", fontsize=11)
        fig.tight_layout()
        return fig


__all__ = (
    "EdgeAttention",
    "capture_hgt_attention",
    "Reason",
    "Explanation",
    "HGTExplainer",
)
