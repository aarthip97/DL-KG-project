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


# ── PyKEEN ──────────────────────────────────────────────────────────────
def export_pykeen_tsv(client: GraphDBClient, out_path: Path) -> Path:
    """Stream all (head, relation, tail) triples to a TSV file.

    Output is suitable for ``pykeen.triples.TriplesFactory.from_path``.
    """
    out_path = Path(out_path)
    log.info("Exporting PyKEEN triples → %s", out_path)
    client.select_tsv(queries.QUERY_PYKEEN_TRIPLES, out_path)

    # The streamed TSV uses the SPARQL variable names as the header
    # (`?h\t?r\t?t`); rewrite to the conventional `head\trel\ttail`.
    with out_path.open("rb+") as fh:
        rest = fh.read().split(b"\n", 1)[1]
        fh.seek(0)
        fh.write(b"head\trel\ttail\n")
        fh.write(rest)
        fh.truncate()
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
