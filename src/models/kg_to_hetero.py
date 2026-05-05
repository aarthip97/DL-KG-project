"""Convert the populated TTL + listening sidecar into a PyG ``HeteroData``.

We model the recommender graph with **typed nodes** (one type per concept
class in the ontology) and **typed edges** (one relation per predicate).
This is the input shape expected by ``torch_geometric.nn.HGTConv``.

Node types we extract:
    * ``user``         — listeners
    * ``track``        — MSDTrack instances
    * ``artist``       — MusicArtist instances
    * ``genre``        — Genre concepts (local + Wikidata anchors)
    * ``instrument``   — Instrument concepts
    * ``decade``       — Decade concepts
    * ``key``          — Key individuals (12 chromatic)
    * ``mode``         — Mode individuals (Major / Minor)
    * ``tempo_class``  — TempoClass individuals (Allegro, …)

Edge types we extract (subset — easy to extend):
    * ``(user, listened_to, track)``      ← from listening sidecar
    * ``(track, has_genre, genre)``       ← inferred from artist→genre
    * ``(artist, has_genre, genre)``      ← mrc:hasGenre
    * ``(track, by_artist, artist)``      ← reverse of mo:performer
    * ``(track, has_instrument, instrument)``
    * ``(track, in_decade, decade)``
    * ``(track, has_key, key)``
    * ``(track, has_mode, mode)``
    * ``(track, has_tempo_class, tempo_class)``

Each node type ships with a contiguous integer index (0..N_t-1). The
returned :class:`KGEncoding` carries those mappings so downstream code can
look up a track URI -> integer ID and back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Predicates we care about (URI -> (src_type, edge_name, dst_type)) ─────
# The "src/dst type" tells us how to look up the head/tail node in the
# right type-specific index. ``None`` means "infer from the URI prefix".
PREDICATE_MAP: dict[str, tuple[str, str, str]] = {
    "http://purl.org/ontology/mrc/listenedTo":
        ("user", "listened_to", "track"),
    "http://purl.org/ontology/mrc/onTrack":
        ("__bnode__", "on_track", "track"),       # rich variant: bnode -> track
    "http://purl.org/ontology/mrc/hasListeningInteraction":
        ("user", "has_listening", "__bnode__"),   # rich variant: user -> bnode
    "http://purl.org/ontology/mrc/listenCount":
        ("__bnode__", "listen_count", "__lit__"),
    "http://purl.org/ontology/mrc/hasGenre":
        ("artist", "has_genre", "genre"),
    "http://purl.org/ontology/mrc/hasInstrument":
        ("track", "has_instrument", "instrument"),
    "http://purl.org/ontology/mo/instrument":
        ("track", "has_instrument", "instrument"),
    "http://purl.org/ontology/mrc/inDecade":
        ("track", "in_decade", "decade"),
    "http://purl.org/ontology/mrc/hasKey":
        ("track", "has_key", "key"),
    "http://purl.org/ontology/mrc/hasMode":
        ("track", "has_mode", "mode"),
    "http://purl.org/ontology/mrc/hasTempoClass":
        ("track", "has_tempo_class", "tempo_class"),
    "http://purl.org/ontology/mo/performer":
        ("track", "by_artist", "artist"),
}

# URI substring -> node type (used as a fallback / sanity guess).
URI_TYPE_HINTS = (
    ("/user/",            "user"),
    ("/track/",           "track"),
    ("/artist/",          "artist"),
    ("/genre/",           "genre"),
    ("/instrument/",      "instrument"),
    ("/decade/",          "decade"),
    ("/key/",             "key"),
    ("/mode/",            "mode"),
    ("/tempo/",           "tempo_class"),
    ("/tempoclass/",      "tempo_class"),
)


def _infer_type(uri: str) -> Optional[str]:
    low = uri.lower()
    for needle, t in URI_TYPE_HINTS:
        if needle in low:
            return t
    return None


# ── Encoding container ────────────────────────────────────────────────────
@dataclass
class KGEncoding:
    """All the index bookkeeping needed to round-trip URIs ↔ integer IDs."""
    uri_to_id: dict[str, dict[str, int]] = field(default_factory=dict)
    id_to_uri: dict[str, list[str]] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], np.ndarray] = field(default_factory=dict)

    def num_nodes(self, node_type: str) -> int:
        return len(self.uri_to_id.get(node_type, {}))

    def encode(self, node_type: str, uri: str) -> Optional[int]:
        return self.uri_to_id.get(node_type, {}).get(uri)

    def decode(self, node_type: str, idx: int) -> Optional[str]:
        bucket = self.id_to_uri.get(node_type, [])
        return bucket[idx] if 0 <= idx < len(bucket) else None

    def summary(self) -> pd.DataFrame:
        rows = [{"node_type": t, "n_nodes": len(uris)}
                for t, uris in self.id_to_uri.items()]
        rows += [{"edge_type": "→".join(et), "n_edges": e.shape[1]}
                 for et, e in self.edges.items()]
        return pd.DataFrame(rows)


# ── Loaders ───────────────────────────────────────────────────────────────
def _iter_triples_from_ttl(ttl_path: Path):
    import rdflib
    g = rdflib.Graph()
    g.parse(str(ttl_path), format="turtle")
    for s, p, o in g:
        yield str(s), str(p), str(o), isinstance(o, rdflib.Literal), \
              isinstance(s, rdflib.BNode), isinstance(o, rdflib.BNode)


def _iter_triples_from_nt(nt_path: Path):
    """Streaming N-Triples reader — avoids loading the whole sidecar into rdflib."""
    import re
    LIT_RE  = re.compile(r'^"(.*)"(?:\^\^<[^>]+>|@\w+)?$')
    URI_RE  = re.compile(r'^<([^>]+)>$')
    with open(nt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            # Drop trailing " ."
            if line.endswith(" ."):
                line = line[:-2]
            # Cheap split: find the first two whitespace-delimited tokens that
            # are URIs or blank-node IDs. The remainder is the object.
            parts = line.split(" ", 2)
            if len(parts) != 3:
                continue
            s_raw, p_raw, o_raw = parts
            s_blank = s_raw.startswith("_:")
            s = s_raw[2:] if s_blank else (URI_RE.match(s_raw).group(1) if URI_RE.match(s_raw) else s_raw)
            p = URI_RE.match(p_raw).group(1) if URI_RE.match(p_raw) else p_raw
            o_blank = o_raw.startswith("_:")
            if o_blank:
                o = o_raw[2:]; is_lit = False
            elif (m := URI_RE.match(o_raw)):
                o = m.group(1); is_lit = False
            else:
                o = o_raw; is_lit = True
            yield s, p, o, is_lit, s_blank, o_blank


def load_kg_as_hetero(
    ttl_path: str | Path,
    listening_nt_path: Optional[str | Path] = None,
    *,
    track_features: Optional[pd.DataFrame] = None,
    track_id_col: str = "track_id",
    track_uri_template: str = "http://example.org/mrc/track/{track_id}",
    user_uri_prefix: str = "http://example.org/mrc/user/",
    embed_dim: int = 32,
    seed: int = 42,
    verbose: bool = True,
):
    """Build a PyG ``HeteroData`` from a populated TTL + listening sidecar.

    Returns
    -------
    data : torch_geometric.data.HeteroData
    enc  : :class:`KGEncoding` with ``uri_to_id`` / ``id_to_uri`` lookups.

    Notes
    -----
    * Node features default to a deterministic random embedding of size
      ``embed_dim`` for every type. Pass ``track_features`` (DataFrame
      indexed by ``track_id``) to inject autoencoder-derived vectors on
      ``track`` nodes — they replace the random init for tracks only.
    * The rich-variant blank-node listening events are *folded* on the fly:
      each ``user --hasListening--> _:ev --onTrack--> track`` pair becomes
      a direct ``(user, listened_to, track)`` edge. The simple variant
      already emits those edges directly.
    * Pruning happens here (literals, OWL admin) — no need to call
      :func:`prune_rdf_graph` separately if you start from the TTL.
    """
    import torch
    from torch_geometric.data import HeteroData

    ttl_path = Path(ttl_path)
    enc = KGEncoding()
    # type -> {uri: id}
    buckets: dict[str, dict[str, int]] = {}

    def _add_node(node_type: str, uri: str) -> int:
        b = buckets.setdefault(node_type, {})
        idx = b.get(uri)
        if idx is None:
            idx = len(b); b[uri] = idx
        return idx

    # Per-edge-type accumulators
    edge_lists: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    # bnode -> (user_uri, track_uri) accumulator for rich-listening folding
    bnode_user: dict[str, str] = {}
    bnode_track: dict[str, str] = {}

    def _record_edge(et: tuple[str, str, str], h: int, t: int) -> None:
        edge_lists.setdefault(et, []).append((h, t))

    def _process(s, p, o, is_lit, s_blank, o_blank):
        if is_lit:
            return  # literals skipped (track features come from `track_features`)
        rel = PREDICATE_MAP.get(p)
        if rel is None:
            return  # ignore predicates we didn't whitelist (admin / SKOS labels)
        src_type, name, dst_type = rel

        # Rich-listening blank-node folding
        if name == "has_listening":
            bnode_user[o] = s
            return
        if name == "on_track":
            bnode_track[s] = o
            return
        if name == "listen_count":
            return  # count is ignored in this implicit-feedback formulation

        if s_blank or o_blank:
            return  # any other blank-node payload is dropped

        if src_type and src_type != "__bnode__":
            h_idx = _add_node(src_type, s)
        else:
            t = _infer_type(s)
            if t is None:
                return
            h_idx = _add_node(t, s)

        if dst_type and dst_type != "__bnode__":
            t_idx = _add_node(dst_type, o)
        else:
            t = _infer_type(o)
            if t is None:
                return
            t_idx = _add_node(t, o)

        _record_edge((src_type, name, dst_type), h_idx, t_idx)

    # ── pass 1: TTL (all non-listening triples + rich schema) ────────────
    if verbose:
        print(f"[hetero] parsing TTL {ttl_path} …")
    for tup in _iter_triples_from_ttl(ttl_path):
        _process(*tup)

    # ── pass 2: listening sidecar (streamed) ──────────────────────────────
    if listening_nt_path is not None:
        listening_nt_path = Path(listening_nt_path)
        if verbose:
            print(f"[hetero] streaming listening sidecar {listening_nt_path} …")
        for tup in _iter_triples_from_nt(listening_nt_path):
            _process(*tup)

    # ── fold rich blank-node listening into direct user→track edges ──────
    if bnode_user and bnode_track:
        if verbose:
            print(f"[hetero] folding {len(bnode_user):,} blank-node listening events …")
        et = ("user", "listened_to", "track")
        for bnode, user_uri in bnode_user.items():
            track_uri = bnode_track.get(bnode)
            if track_uri is None:
                continue
            u_idx = _add_node("user", user_uri)
            t_idx = _add_node("track", track_uri)
            _record_edge(et, u_idx, t_idx)

    # ── finalise encoding ────────────────────────────────────────────────
    enc.uri_to_id = buckets
    enc.id_to_uri = {t: [None] * len(b) for t, b in buckets.items()}
    for t, b in buckets.items():
        for uri, idx in b.items():
            enc.id_to_uri[t][idx] = uri

    enc.edges = {
        et: np.asarray(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), dtype=np.int64)
        for et, pairs in edge_lists.items()
    }

    # ── build the HeteroData object ──────────────────────────────────────
    rng = np.random.default_rng(seed)
    data = HeteroData()
    for node_type, b in buckets.items():
        n = len(b)
        x = torch.from_numpy(
            rng.standard_normal((n, embed_dim)).astype(np.float32) * 0.1
        )
        data[node_type].num_nodes = n
        data[node_type].x = x
        data[node_type].node_id = torch.arange(n)

    # Inject precomputed track features if provided
    if track_features is not None and "track" in buckets:
        tf = track_features.copy()
        if track_id_col not in tf.columns:
            tf = tf.reset_index().rename(columns={tf.index.name or "index": track_id_col})
        feat_cols = [c for c in tf.columns if c != track_id_col]
        feat_dim = len(feat_cols)
        x = torch.zeros(buckets["track"].__len__(), feat_dim, dtype=torch.float32)
        hits = 0
        for _, row in tf.iterrows():
            uri = track_uri_template.format(track_id=row[track_id_col])
            idx = buckets["track"].get(uri)
            if idx is not None:
                x[idx] = torch.from_numpy(row[feat_cols].to_numpy(dtype=np.float32))
                hits += 1
        data["track"].x = x
        if verbose:
            print(f"[hetero] injected AE features into {hits:,}/{buckets['track'].__len__():,} tracks "
                  f"(dim={feat_dim})")

    for et, edges in enc.edges.items():
        data[et].edge_index = __import__("torch").from_numpy(edges).long()

    if verbose:
        print("[hetero] node types : "
              + ", ".join(f"{t}={len(b)}" for t, b in buckets.items()))
        print("[hetero] edge types : "
              + ", ".join(f"{'/'.join(et)}={e.shape[1]}" for et, e in enc.edges.items()))

    return data, enc


__all__ = ("load_kg_as_hetero", "KGEncoding", "PREDICATE_MAP")
