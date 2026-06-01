"""Pull tabular artefacts out of GraphDB.

These functions know nothing about the graph structure beyond what the
SPARQL queries in :mod:`graphdb.queries` already encode.  Their job is just
to turn query results into the on-disk shapes that downstream tooling
expects:

* ``triples.tsv``      — head/relation/tail TSV ready for ``pykeen.triples``
                         (bare, headerless, canonicalised IRIs)
* ``node_dict.json``   — the PyG ``edge_dict`` (per-type ``node_mappings`` +
                         integer edge-index pairs + weight/count arrays +
                         ``canonical_map``) that ``build_rich_hetero_graph``
                         consumes directly. Produced by :func:`export_edge_dict`
                         with the *same* canonicalisation as the PyKEEN TSV, so
                         KGE entity labels and node URIs match exactly.

Legacy helpers (:func:`export_node_dict` → ``{type: {uri: id}}`` and
:func:`export_hetero_edges` → parquet) are retained for backwards
compatibility but are no longer part of the default export, because their
output shape is not what the notebook's KGE → HGT path consumes.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from ..canonicalize import build_canonical_map, pick_canonical
from . import queries
from .client import GraphDBClient

log = logging.getLogger(__name__)

_INT_RE = re.compile(r"-?\d+")


def _strip_iri(u: object) -> str:
    """Strip the angle brackets GraphDB's SPARQL-TSV serialisation puts around
    IRIs (``<http://...>`` → ``http://...``). No-op for already-bare strings."""
    s = str(u)
    return s[1:-1] if len(s) >= 2 and s[0] == "<" and s[-1] == ">" else s


def _parse_int_literal(cell: object, default: int = 1) -> int:
    """Pull an integer out of a SPARQL-TSV literal cell.

    Handles ``"5"^^<xsd:integer>``, bare ``5``, and empty/None (→ ``default``).
    """
    if cell is None:
        return default
    m = _INT_RE.search(str(cell))
    return int(m.group()) if m else default


# Canonical URI helpers (delegated to canonicalize module).
# Both this exporter and the in-memory kg_to_hetero exporter share one
# union-find + preference-order definition.  Thin private aliases preserve
# the historical module API.
_pick_canonical = pick_canonical


def _build_canonical_map(equiv_df: pd.DataFrame) -> Dict[str, str]:
    """Build a {uri -> canonical_uri} map from an equivalence-pair DataFrame.

    Thin wrapper around canonicalize.build_canonical_map that handles the
    SPARQL column-name variance ("a"/"b" vs "?a"/"?b").
    """
    a_col = "a" if "a" in equiv_df.columns else "?a"
    b_col = "b" if "b" in equiv_df.columns else "?b"
    pairs = zip(
        equiv_df[a_col].astype(str),
        equiv_df[b_col].astype(str),
    )
    return build_canonical_map(pairs)


# ── PyKEEN ──────────────────────────────────────────────────────────────
def export_pykeen_tsv(client: GraphDBClient, out_path: Path) -> Path:
    """Stream all (head, relation, tail) triples to a TSV file, with
    equivalent nodes merged into a single canonical URI.

    Steps
    -----
    1. Fetch all ``owl:sameAs`` (instance) and ``owl:equivalentClass`` (class)
       pairs and build a union-find canonical map
       (e.g. Wikidata genre URI → mrc: genre URI).
    2. Stream triples from GraphDB (``infer=True`` for RDFS+ memberships).
    3. Replace every h/r/t URI with its canonical form.
    4. Drop self-loops that arose from the merge (``h == t``).
    5. Deduplicate — two triples may become identical after canonicalisation.

    Output is suitable for ``pykeen.triples.TriplesFactory.from_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting PyKEEN triples → %s", out_path)

    # ── Step 1: build canonical map ───────────────────────────────────
    log.info("Fetching equivalence pairs (owl:sameAs + owl:equivalentClass) …")
    equiv_df = client.select_df(queries.QUERY_EQUIV_PAIRS, infer=False)
    canon_map: Dict[str, str] = _build_canonical_map(equiv_df)
    n_components = len(set(canon_map.values()))
    log.info(
        "Equivalence pairs: %d  →  %d components  (%d URIs will be remapped)",
        len(equiv_df),
        n_components,
        sum(1 for k, v in canon_map.items() if k != v),
    )

    # ── Step 2: stream triples to a temp file ─────────────────────────
    tmp_path = out_path.with_suffix(".raw.tsv")
    client.select_tsv(queries.QUERY_PYKEEN_TRIPLES, tmp_path, infer=True)
    log.info("Raw triples streamed → %s", tmp_path)

    # ── Steps 3-5: canonicalise, filter, dedup ────────────────────────
    df = pd.read_csv(tmp_path, sep="\t", dtype=str)

    # Rename SPARQL variable columns to conventional names
    df.columns = ["head", "rel", "tail"]

    # GraphDB's SPARQL *TSV* serialisation wraps every IRI in angle brackets
    # (``<http://...>``), whereas the *CSV* serialisation used by
    # export_node_dict (select_df) returns bare URIs. Strip the brackets here
    # so that:
    #   (a) the canonical_map (built from select_df → bare URIs) actually
    #       matches the head/rel/tail values and the owl:sameAs /
    #       equivalentClass merges take effect; and
    #   (b) the entity labels PyKEEN learns are byte-identical to the bare
    #       URIs stored in node_dict.json. Otherwise the KGE lookup in
    #       build_rich_hetero_graph misses 100% of nodes (every node ends up
    #       with an all-zero KGE slice).
    for _col in ("head", "rel", "tail"):
        df[_col] = df[_col].map(_strip_iri)

    before = len(df)
    def _canon(u: object) -> str:
        s = str(u)
        return canon_map.get(s, s)

    df["head"] = df["head"].map(_canon)
    df["rel"]  = df["rel"].map(_canon)
    df["tail"] = df["tail"].map(_canon)

    # Drop self-loops introduced by the merge
    df = df[df["head"] != df["tail"]]
    # Deduplicate
    df = df.drop_duplicates()
    after = len(df)

    # Headerless: pykeen.triples.load_triples reads with header=None, so a
    # header row would be parsed as a spurious ("head","rel","tail") triple.
    df.to_csv(out_path, sep="\t", index=False, header=False)
    tmp_path.unlink(missing_ok=True)

    log.info(
        "PyKEEN triples: %d raw  →  %d after canonicalise+dedup  "
        "(-%d self-loops/duplicates)",
        before, after, before - after,
    )
    return out_path


# ── Edge dict (PyG-ready, notebook-consumable) ───────────────────────────
def export_edge_dict(client: GraphDBClient, out_path: Path) -> Path:
    """Build the ``edge_dict`` JSON that ``build_rich_hetero_graph`` consumes,
    sourced entirely from GraphDB (no in-memory rdflib graph required).

    This is the GraphDB-native equivalent of
    ``models.kg_to_hetero.extract_dl_artifacts``: it produces the *same*
    structure (per-type ``node_mappings`` + integer edge index pairs +
    weight/count arrays + ``canonical_map``) so the notebook's KGE → HGT path
    can consume the GraphDB export directly. URIs are canonicalised with the
    **same** union-find map used by :func:`export_pykeen_tsv`, so the entity
    labels match the PyKEEN TSV exactly (no KGE-lookup misses).

    Memory profile: every edge query except listening is small and fetched via
    ``select_df``; the large user→track listening relation is *streamed* to a
    temp TSV and parsed line-by-line, so the host never materialises millions
    of rows in pandas. Listening play counts are preserved (``listenCount``,
    defaulting to 1 when absent).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting PyG edge_dict → %s", out_path)

    # Same canonical map as export_pykeen_tsv so node URIs == KGE entity labels.
    equiv_df = client.select_df(queries.QUERY_EQUIV_PAIRS, infer=False)
    canon_map: Dict[str, str] = _build_canonical_map(equiv_df)

    def _canon(uri: object) -> str:
        s = _strip_iri(uri)
        return canon_map.get(s, s)

    node_to_idx: Dict[str, Dict[str, int]] = defaultdict(dict)
    node_mappings: Dict[str, list] = defaultdict(list)

    def nidx(node_type: str, uri: object) -> int:
        u = _canon(uri)
        bucket = node_to_idx[node_type]
        i = bucket.get(u)
        if i is None:
            i = len(node_mappings[node_type])
            bucket[u] = i
            node_mappings[node_type].append(u)
        return i

    edge_dict: Dict[str, object] = {
        "user_track":        [[], []],
        "track_artist":      [[], []],
        "track_tempo":       [[], []],
        "track_key":         [[], []],
        "track_mode":        [[], []],
        "track_instrument":  [[], []],
        "track_decade":      [[], []],
        "artist_genre":      [[], []],
        "genre_parent":      [[], []],
        "instrument_parent": [[], []],
        # Weight arrays. track_key/track_mode are intentionally left empty to
        # match extract_dl_artifacts (the simple graph carries no key/mode
        # confidence); artist_genre and user_track ARE populated below.
        "track_key_weights":    [],
        "track_mode_weights":   [],
        "artist_genre_weights": [],
        "user_track_counts":    [],
    }

    # ── Small, dense edge relations (fetched whole via select_df) ─────────
    # Column ORDER is the contract (see queries.py): read positionally so a
    # GraphDB CSV header naming quirk ("user" vs "?user") can't break us.
    def _add_pairs(sparql: str, key: str, ta: str, tb: str) -> int:
        df = client.select_df(sparql, infer=False)
        if df.empty:
            return 0
        a = df.iloc[:, 0].to_numpy()
        b = df.iloc[:, 1].to_numpy()
        for x, y in zip(a, b):
            edge_dict[key][0].append(nidx(ta, x))
            edge_dict[key][1].append(nidx(tb, y))
        return len(df)

    _add_pairs(queries.QUERY_EDGE_TRACK_ARTIST,     "track_artist",      "track", "artist")
    _add_pairs(queries.QUERY_EDGE_TRACK_KEY,        "track_key",         "track", "key")
    _add_pairs(queries.QUERY_EDGE_TRACK_MODE,       "track_mode",        "track", "mode")
    _add_pairs(queries.QUERY_EDGE_TRACK_TEMPO,      "track_tempo",       "track", "tempo_class")
    _add_pairs(queries.QUERY_EDGE_TRACK_INSTRUMENT, "track_instrument",  "track", "instrument")
    _add_pairs(queries.QUERY_EDGE_TRACK_DECADE,     "track_decade",      "track", "decade")
    _add_pairs(queries.QUERY_EDGE_GENRE_PARENT,     "genre_parent",      "genre", "genre")
    _add_pairs(queries.QUERY_EDGE_INSTRUMENT_PARENT,"instrument_parent", "instrument", "instrument")

    # artist → genre, with per-edge weight (rich) or 1.0 (simple/missing).
    ag = client.select_df(queries.QUERY_EDGE_ARTIST_GENRE, infer=False)
    if not ag.empty:
        a = ag.iloc[:, 0].to_numpy()
        g = ag.iloc[:, 1].to_numpy()
        w = ag.iloc[:, 2].to_numpy() if ag.shape[1] > 2 else [None] * len(ag)
        for x, y, wt in zip(a, g, w):
            edge_dict["artist_genre"][0].append(nidx("artist", x))
            edge_dict["artist_genre"][1].append(nidx("genre", y))
            try:
                edge_dict["artist_genre_weights"].append(
                    1.0 if wt is None or pd.isna(wt) else float(wt)
                )
            except (TypeError, ValueError):
                edge_dict["artist_genre_weights"].append(1.0)

    # ── user → track listening (streamed; preserves play counts) ─────────
    tmp = out_path.with_name(out_path.stem + "._listen.tsv")
    client.select_tsv(queries.QUERY_EDGE_USER_TRACK, tmp, infer=False)
    n_listen = 0
    with open(tmp, "r", encoding="utf-8") as fh:
        fh.readline()  # skip the "?user ?track ?count" TSV header row
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            edge_dict["user_track"][0].append(nidx("user",  parts[0]))
            edge_dict["user_track"][1].append(nidx("track", parts[1]))
            edge_dict["user_track_counts"].append(
                _parse_int_literal(parts[2]) if len(parts) > 2 else 1
            )
            n_listen += 1
    tmp.unlink(missing_ok=True)

    # ── Serialise (plain lists; same shape extract_dl_artifacts writes) ───
    edge_dict["node_mappings"] = {nt: list(uris) for nt, uris in node_mappings.items()}
    edge_dict["canonical_map"] = canon_map
    out_path.write_text(json.dumps(edge_dict))

    log.info(
        "edge_dict: %s nodes | user→track=%d | track→artist=%d | artist→genre=%d",
        {nt: len(u) for nt, u in node_mappings.items()},
        n_listen, len(edge_dict["track_artist"][0]), len(edge_dict["artist_genre"][0]),
    )
    return out_path


# ── Node dictionary ─────────────────────────────────────────────────────
def export_node_dict(client: GraphDBClient, out_path: Path) -> Path:
    """Build ``{node_type: {uri: int_id}}`` from the live repo.

    Persisted as JSON for portability; PyG side just calls ``json.load``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = client.select_df(queries.QUERY_ALL_ENTITIES)
    grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
    # pandas may return either "?e/?type" or "e/type" depending on server
    # version — handle both.
    e_col = "e" if "e" in df.columns else "?e"
    t_col = "type" if "type" in df.columns else "?type"

    for typ, sub in df.groupby(t_col):
        mapping = {uri: i for i, uri in enumerate(sorted(sub[e_col].astype(str)))}
        grouped[str(typ)] = mapping

    out_path.write_text(json.dumps(grouped, indent=2))
    log.info("Wrote node dict (%d types, %d URIs) → %s",
             len(grouped), sum(len(v) for v in grouped.values()), out_path)
    return out_path


# ── Heterogeneous edge list ─────────────────────────────────────────────
def export_hetero_edges(
    triples_tsv: Path,
    node_dict_json: Path,
    out_path: Path,
) -> Path:
    """Re-encode the TSV triples as integer ids partitioned by edge type.

    Output schema (parquet):
        rel       : str   — full relation URI
        head      : int64 — id within its node-type bucket
        head_type : str
        tail      : int64
        tail_type : str

    This is the format that the PyG-side notebook reads to build a
    ``HeteroData`` object.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # header=None: export_pykeen_tsv now writes a headerless TSV (pykeen-ready),
    # so the three columns are positional head/rel/tail.
    triples = pd.read_csv(triples_tsv, sep="\t", dtype=str, header=None,
                          names=["head", "rel", "tail"])
    node_dict: Dict[str, Dict[str, int]] = json.loads(Path(node_dict_json).read_text())

    # Invert the dict once: uri → (type, id)
    uri_lookup: Dict[str, tuple] = {}
    for ntype, mapping in node_dict.items():
        for uri, idx in mapping.items():
            uri_lookup[uri] = (ntype, idx)

    # Vectorised join via pandas merge would build huge intermediate
    # frames; a simple comprehension is faster for a few-million-row TSV
    # and keeps memory predictable.
    rows = []
    skipped = 0
    for h, r, t in triples.itertuples(index=False, name=None):
        h_info = uri_lookup.get(h)
        t_info = uri_lookup.get(t)
        if h_info is None or t_info is None:
            skipped += 1
            continue
        rows.append((r, h_info[1], h_info[0], t_info[1], t_info[0]))

    out = pd.DataFrame(
        rows,
        columns=["rel", "head", "head_type", "tail", "tail_type"],
    )
    out.to_parquet(out_path, index=False)
    log.info("Wrote %d hetero edges (skipped %d untyped) → %s",
             len(out), skipped, out_path)
    return out_path
