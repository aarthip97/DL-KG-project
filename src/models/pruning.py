"""RDF/OWL pruning to keep only payload triples for KGE / GNN training.

Why prune?
----------
A populated OWL ontology carries a *lot* of administrative triples that
inflate the entity vocabulary and create huge "hub" nodes (e.g. every
entity is a ``rdf:type owl:NamedIndividual``). Embedding methods then
collapse most of the signal into the hub, hurting downstream link
prediction. We strip those before exporting to PyKEEN / PyG.

The function below takes either:
* a TSV file path (``head\\trelation\\ttail``), or
* an ``rdflib.Graph`` (we'll iterate it directly).

…and returns a clean ``pandas.DataFrame`` of triples plus, optionally,
writes it back to a TSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

# ── Triples we want gone ──────────────────────────────────────────────────
ADMIN_RELATIONS = frozenset({
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
})

# Tail values that signal "this is just a class/individual declaration"
ADMIN_TAILS = frozenset({
    "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#Ontology",
    "http://www.w3.org/2000/01/rdf-schema#Class",
})


def _is_literal(s: str) -> bool:
    """Heuristic: a literal in N-Triples form contains ``"`` or a ``^^`` cast."""
    return s.startswith('"') or "^^" in s or s.startswith("'")


def _is_blank(s: str) -> bool:
    return s.startswith("_:")


def prune_rdf_graph(
    triples: Union[str, Path, pd.DataFrame, Iterable[tuple[str, str, str]]],
    *,
    out_tsv: Optional[Union[str, Path]] = None,
    keep_subclass: bool = False,
    drop_literals: bool = True,
    drop_blanks: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove administrative OWL/RDF triples.

    Parameters
    ----------
    triples : path to a TSV (head\\trelation\\ttail), a DataFrame with those
        columns, or an iterable of tuples.
    out_tsv : if given, write the pruned table there (no header).
    keep_subclass : keep ``rdfs:subClassOf`` triples (useful if you want to
        embed the genre / instrument hierarchy explicitly).
    drop_literals : drop triples whose tail looks like a literal.
    drop_blanks : drop triples touching blank nodes (e.g. listening events).
    """
    if isinstance(triples, (str, Path)):
        df = pd.read_csv(triples, sep="\t", header=None,
                         names=["head", "relation", "tail"], dtype=str)
    elif isinstance(triples, pd.DataFrame):
        df = triples[["head", "relation", "tail"]].astype(str).copy()
    else:
        df = pd.DataFrame(list(triples), columns=["head", "relation", "tail"], dtype=str)

    n0 = len(df)

    # 1) drop admin relations (optionally keeping subClassOf for hierarchy)
    bad_rels = set(ADMIN_RELATIONS)
    if keep_subclass:
        bad_rels.discard("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    df = df[~df["relation"].isin(bad_rels)]

    # 2) drop "is-a NamedIndividual / Class" tails (in case rdf:type was kept)
    df = df[~df["tail"].isin(ADMIN_TAILS)]

    # 3) drop literal tails (audio numbers, labels, dates…)
    if drop_literals:
        df = df[~df["tail"].map(_is_literal)]

    # 4) drop blank nodes (listening events live there)
    if drop_blanks:
        df = df[~(df["head"].map(_is_blank) | df["tail"].map(_is_blank))]

    df = df.reset_index(drop=True)

    if verbose:
        print(f"[prune] {n0:,} -> {len(df):,} triples "
              f"({(n0 - len(df)) / max(n0, 1):.1%} pruned)")

    if out_tsv is not None:
        out_tsv = Path(out_tsv)
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_tsv, sep="\t", index=False, header=False)
        if verbose:
            print(f"[prune] wrote {out_tsv} ({out_tsv.stat().st_size/1024:,.1f} KiB)")

    return df


__all__ = ("prune_rdf_graph", "ADMIN_RELATIONS", "ADMIN_TAILS")
