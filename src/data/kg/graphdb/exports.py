"""Pull tabular artefacts out of GraphDB.

These functions know nothing about the graph structure beyond what the
SPARQL queries in :mod:`graphdb.queries` already encode.  Their job is just
to turn query results into the on-disk shapes that downstream tooling
expects:

* ``triples.tsv``      — head/relation/tail TSV ready for ``pykeen.triples``
* ``node_dict.json``   — URI → integer id, partitioned by node type
* ``hetero_edges.parquet`` — long-format edge list with integer ids
                              ready to feed into a PyG ``HeteroData``
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from . import queries
from .client import GraphDBClient

log = logging.getLogger(__name__)


# ── Canonical URI helpers ────────────────────────────────────────────────
_CANON_PREFERENCE = (
    "purl.org/ontology/mrc/resource/",  # our own data nodes — highest priority
    "purl.org/ontology/mrc/",           # our ontology terms
    "wikidata.org/entity/",             # Wikidata entities
    "purl.org/ontology/mo/",            # Music Ontology
    "xmlns.com/foaf/",                  # FOAF
)


def _pick_canonical(a: str, b: str) -> str:
    """Return whichever URI is the preferred canonical representative.

    Priority: mrc:resource > mrc: > wd: > mo: > foaf: > lexicographic min.
    """
    for prefix in _CANON_PREFERENCE:
        if prefix in a and prefix not in b:
            return a
        if prefix in b and prefix not in a:
            return b
    return min(a, b)  # lexicographic fallback — at least deterministic


def _build_canonical_map(equiv_df: pd.DataFrame) -> Dict[str, str]:
    """Build a {uri → canonical_uri} map from an equivalence-pair DataFrame.

    Uses path-compressed union-find so the whole thing is O(n·α(n)).
    Any URI that has no equivalences maps to itself implicitly (callers
    should use ``canon_map.get(uri, uri)``).
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])   # path compression
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        winner = _pick_canonical(ra, rb)
        loser  = rb if winner == ra else ra
        parent[loser] = winner

    # Column names may come back as "a"/"b" or "?a"/"?b"
    a_col = "a" if "a" in equiv_df.columns else "?a"
    b_col = "b" if "b" in equiv_df.columns else "?b"
    for a, b in zip(equiv_df[a_col].astype(str), equiv_df[b_col].astype(str)):
        union(a, b)

    # Materialise fully-compressed map for all nodes seen
    return {k: find(k) for k in parent}


# ── PyKEEN ──────────────────────────────────────────────────────────────
def export_pykeen_tsv(client: GraphDBClient, out_path: Path) -> Path:
    """Stream all (head, relation, tail) triples to a TSV file, with
    equivalent nodes merged into a single canonical URI.

    Steps
    -----
    1. Fetch all ``owl:sameAs`` / ``skos:exactMatch`` pairs and build a
       union-find canonical map (e.g. Wikidata genre URI → mrc: genre URI).
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
    log.info("Fetching equivalence pairs (owl:sameAs + skos:exactMatch) …")
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

    df.to_csv(out_path, sep="\t", index=False)
    tmp_path.unlink(missing_ok=True)

    log.info(
        "PyKEEN triples: %d raw  →  %d after canonicalise+dedup  "
        "(-%d self-loops/duplicates)",
        before, after, before - after,
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

    triples = pd.read_csv(triples_tsv, sep="\t", dtype=str)
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
