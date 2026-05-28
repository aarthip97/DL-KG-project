"""
Decade modelling for the KG.

We mint one ``mrc:Decade`` individual per distinct release decade observed
in the dataset (e.g. 1960, 1970, ..., 2010).  Each decade forms a
*sequence* via the Wikidata properties ``wdt:P155`` (follows) /
``wdt:P156`` (followed by), so SPARQL can walk forward/backward in time.

When Wikidata enrichment runs, every local decade also gets:

* ``owl:sameAs wd:Q...``          - canonical Wikidata decade entity
* ``rdfs:label "<en-label>"@en``  - label from Wikidata
* ``rdfs:subClassOf wd:Q<century>`` - parent century (Wikidata P361)
* ``wdt:P31 wd:Q39825``           - "decade" type, lifted from Wikidata

Track ↔ decade is asserted with ``mrc:inDecade``.

Wikidata calls are parallelised through the helpers in
:mod:`data.kg.wikidata_mapping`.
"""
from __future__ import annotations

import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm.auto import tqdm

from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from .kg_builder import KGBuilder, MRC
from .wikidata_mapping import (
    WD, WDT,
    _attach_qid_metadata, _session,
    fetch_qid_metadata,
)


# Wikidata QIDs we use as RDF constants.
WD_DECADE_TYPE = "Q39911"            # "decade"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def decade_for_year(year: int) -> int:
    """1960 -> 1960, 1969 -> 1960, 2014 -> 2010."""
    return (int(year) // 10) * 10


def decade_label(start_year: int) -> str:
    """1960 -> '1960s'."""
    return f"{int(start_year)}s"


def unique_decades_from_dataframe(
    df: pd.DataFrame, year_col: str = "year",
    min_year: int = 1900, max_year: int = 2030,
) -> list[int]:
    """Distinct, sorted decade-starts present in ``df[year_col]``."""
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    years = years[(years >= min_year) & (years <= max_year)]
    return sorted({decade_for_year(int(y)) for y in years.unique()})


# ---------------------------------------------------------------------------
# Wikidata decade resolution (parallel)
# ---------------------------------------------------------------------------
_SPARQL_URL = "https://query.wikidata.org/sparql"


def _resolve_decade_qid(start_year: int) -> Optional[dict]:
    """
    SPARQL: find the Wikidata decade entity whose canonical English label
    is ``"<start>s"`` (e.g. ``"2010s"``) and whose ``wdt:P31/wdt:P279*``
    chain bottoms out at ``Q39911`` ("decade").

    The previous strategy ("``wdt:P31 wd:Q39911 ; wdt:P580 ?start ; YEAR``)
    missed most decades because (a) many decade entities are typed via a
    *subclass* of Q39911 (e.g. *decade of the Gregorian calendar*) and
    (b) ``wdt:P580``/``wdt:P582`` is inconsistently populated. Matching on
    the canonical label is robust: every decade entity carries
    ``rdfs:label "1960s"@en`` etc.

    Also pulls:
      * ``century_qid``  - parent century via ``wdt:P361`` (part of)
      * ``follows_qid``  - previous decade via ``wdt:P155``
      * ``followed_qid`` - next decade via ``wdt:P156``
      * ``start_iso``    - ``wdt:P580`` start time literal (if set)
      * ``end_iso``      - ``wdt:P582`` end time literal (if set)
    """
    label = f"{int(start_year)}s"
    q = f"""
    SELECT ?d ?dLabel ?century ?centuryLabel
           ?follows ?followed ?start ?end WHERE {{
      ?d rdfs:label "{label}"@en ;
         wdt:P31/wdt:P279* wd:{WD_DECADE_TYPE} .
      OPTIONAL {{ ?d wdt:P361 ?century .
                  ?century wdt:P31/wdt:P279* wd:Q578 . }}
      OPTIONAL {{ ?d wdt:P155 ?follows  . }}
      OPTIONAL {{ ?d wdt:P156 ?followed . }}
      OPTIONAL {{ ?d wdt:P580 ?start    . }}
      OPTIONAL {{ ?d wdt:P582 ?end      . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    try:
        r = _session().get(_SPARQL_URL, params={"query": q, "format": "json"},
                           timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    rows = r.json().get("results", {}).get("bindings", [])
    if not rows:
        return None
    row = rows[0]

    def _qid(key: str) -> Optional[str]:
        if key not in row:
            return None
        return row[key]["value"].rsplit("/", 1)[-1]

    def _lit(key: str) -> Optional[str]:
        return row[key]["value"] if key in row else None

    return {
        "qid":          _qid("d"),
        "century_qid":  _qid("century"),
        "follows_qid":  _qid("follows"),
        "followed_qid": _qid("followed"),
        "start_iso":    _lit("start"),
        "end_iso":      _lit("end"),
        "label":        row.get("dLabel", {}).get("value"),
    }


def resolve_decades(
    decade_starts: Iterable[int],
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    parallel: int = 4,
    verbose: bool = True,
) -> dict[int, Optional[dict]]:
    """
    Parallel ``decade_start -> {qid, century_qid, label}`` lookup with
    JSON cache. Misses persist as ``null``.
    """
    cache: dict[str, Optional[dict]] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        cache = json.loads(cache_path.read_text())
        if verbose:
            print(f"[decades] loaded {len(cache):,} cached decade entries "
                  f"from {cache_path.name}")

    todo = [int(d) for d in dict.fromkeys(decade_starts) if str(d) not in cache]
    if verbose:
        print(f"[decades] resolving {len(todo)} new decades "
              f"(workers={parallel}) ...")

    if todo:
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_resolve_decade_qid, d): d for d in todo}
            bar = tqdm(as_completed(futures), total=len(futures),
                       desc="decades", unit="dec", disable=not verbose,
                       leave=True)
            for fut in bar:
                d = futures[fut]
                try:
                    cache[str(d)] = fut.result()
                except Exception:                              # noqa: BLE001
                    cache[str(d)] = None
                with lock:
                    bar.set_postfix(last=f"{d}s", refresh=False)
            bar.close()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[decades] cache -> {cache_path}")

    # Normalise keys back to int for the caller.
    return {int(k): v for k, v in cache.items()}


# ---------------------------------------------------------------------------
# Folding into the KG
# ---------------------------------------------------------------------------
def add_decades_to_graph(
    builder: KGBuilder,
    df: pd.DataFrame,
    year_col: str = "year",
    track_id_col: str = "track_id",
    *,
    decade_qids: Optional[dict[int, Optional[dict]]] = None,
    qid_metadata: Optional[dict[str, dict]] = None,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Mint Decade nodes, link them in a sequence, and attach every track
    in ``df`` to its decade via ``mrc:inDecade``.

    Parameters
    ----------
    decade_qids : optional mapping ``decade_start -> {qid, century_qid, label}``
        produced by :func:`resolve_decades`. When provided, each decade
        gets ``skos:exactMatch wd:Q...``, ``wdt:P31 wd:Q39825``,
        ``wdt:P155``/``wdt:P156`` to its sibling decades on Wikidata,
        and ``skos:broader wd:Q<century>``.
    qid_metadata : optional mapping ``QID -> {label, description, aliases}``
        produced by :func:`fetch_qid_metadata`. Used to label decade and
        century QID nodes properly (so we don't end up with bare
        ``wd:Q6927`` nodes).
    """
    g       = builder.g
    counts  = {"decades": 0, "track_links": 0, "qid_concepts": 0}
    decade_qids  = decade_qids  or {}
    qid_metadata = dict(qid_metadata or {})  # local copy — never mutate caller's dict

    # ── 1. Declare mrc:Decade as an OWL class ──────────────────────────────
    DECADE_CLS = MRC["Decade"]
    g.add((DECADE_CLS, RDF.type, OWL.Class))
    g.add((DECADE_CLS, RDFS.label, Literal("Decade", lang="en")))

    # ── 2. Annotate referenced Wikidata QID nodes ──────────────────────────
    referenced_qids: set[str] = set()
    for entry in decade_qids.values():
        if not entry:
            continue
        for key in ("qid", "century_qid", "follows_qid", "followed_qid"):
            v = entry.get(key)
            if v:
                referenced_qids.add(v)

    # Auto-fetch metadata for any QIDs the caller did not supply.  This
    # prevents century / neighbour-decade nodes from falling back to the
    # bare QID string as their rdfs:label (e.g. "Q6927" instead of
    # "20th century").  fetch_qid_metadata also fetches Portuguese labels
    # by default so the KG gets @pt rdfs:label triples for free.
    missing = referenced_qids - set(qid_metadata)
    if missing:
        if verbose:
            print(f"[decades] auto-fetching metadata for {len(missing)} "
                  f"QIDs not supplied in qid_metadata ...")
        qid_metadata.update(fetch_qid_metadata(list(missing), verbose=False))

    for qid in referenced_qids:
        wd_node = WD[qid]
        _attach_qid_metadata(g, wd_node, qid,
                             qid_metadata.get(qid, {}),
                             fallback_label=None)
        counts["qid_concepts"] += 1

    # ── 3. mint local Decade individuals + sequence them ───────────────────
    starts_present = unique_decades_from_dataframe(df, year_col=year_col)
    decade_node: dict[int, URIRef] = {}

    for start in starts_present:
        node = builder.decade_uri(start)
        decade_node[start] = node
        g.add((node, RDF.type,      DECADE_CLS))
        g.add((node, RDF.type,      OWL.NamedIndividual))
        g.add((node, RDFS.label,    Literal(decade_label(start), lang="en")))
        # First-class temporal anchor
        g.add((node, MRC["startYear"],
               Literal(int(start), datatype=XSD.gYear)))
        g.add((node, MRC["endYear"],
               Literal(int(start) + 9, datatype=XSD.gYear)))
        counts["decades"] += 1

        # Wikidata-backed enrichment for this decade
        entry = decade_qids.get(int(start))
        if entry and entry.get("qid"):
            wd_node = WD[entry["qid"]]
            g.add((node, OWL.sameAs, wd_node))
            # decade typed as "decade" via P31
            g.add((wd_node, WDT["P31"], WD[WD_DECADE_TYPE]))

            # Parent century: wd:<decade> wdt:P361 wd:<century>;
            # surface on local node via rdfs:subClassOf.
            if entry.get("century_qid"):
                century_node = WD[entry["century_qid"]]
                g.add((wd_node, WDT["P361"], century_node))
                g.add((node,    RDFS.subClassOf, century_node))
                g.add((century_node, WDT["P31"], WD["Q578"]))  # century

            # Neighbour decades on Wikidata (mirrors the local sequence below).
            if entry.get("follows_qid"):
                g.add((wd_node, WDT["P155"], WD[entry["follows_qid"]]))
            if entry.get("followed_qid"):
                g.add((wd_node, WDT["P156"], WD[entry["followed_qid"]]))

            # Time bounds straight from Wikidata (P580 / P582), as xsd:dateTime.
            if entry.get("start_iso"):
                g.add((wd_node, WDT["P580"],
                       Literal(entry["start_iso"], datatype=XSD.dateTime)))
            if entry.get("end_iso"):
                g.add((wd_node, WDT["P582"],
                       Literal(entry["end_iso"], datatype=XSD.dateTime)))

    # ── 4. local follows / followed-by chain (10-year stride) ──────────────
    for start in starts_present:
        prev_start = start - 10
        next_start = start + 10
        if prev_start in decade_node:
            g.add((decade_node[start], WDT["P155"], decade_node[prev_start]))
        if next_start in decade_node:
            g.add((decade_node[start], WDT["P156"], decade_node[next_start]))

    # ── 5. wd:P155/P156 between Wikidata decade entities (when both known) ─
    #         redundant with the per-entry follows/followed links written in §3
    #         but keeps the chain consistent when SPARQL only returned a one-
    #         sided neighbour for a given decade.
    for start, entry in decade_qids.items():
        if not entry or not entry.get("qid"):
            continue
        wd_self = WD[entry["qid"]]
        prev = decade_qids.get(start - 10)
        nxt  = decade_qids.get(start + 10)
        if prev and prev.get("qid"):
            g.add((wd_self, WDT["P155"], WD[prev["qid"]]))
        if nxt and nxt.get("qid"):
            g.add((wd_self, WDT["P156"], WD[nxt["qid"]]))

    # ── 6. attach every track to its decade ────────────────────────────────
    if track_id_col in df.columns and year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce")
        for tid, year in zip(df[track_id_col], years):
            if pd.isna(year) or pd.isna(tid):
                continue
            d_start = decade_for_year(int(year))
            node = decade_node.get(d_start)
            if node is None:
                continue
            g.add((builder.track_uri(str(tid)), MRC["inDecade"], node))
            counts["track_links"] += 1

    if verbose:
        print(f"[decades] summary: {counts}")
    return counts


def collect_decade_qids_for_metadata(
    decade_qids: dict[int, Optional[dict]]
) -> list[str]:
    """Flatten resolve_decades output into the QID list for fetch_qid_metadata."""
    out: set[str] = set()
    for entry in decade_qids.values():
        if not entry:
            continue
        for key in ("qid", "century_qid", "follows_qid", "followed_qid"):
            v = entry.get(key)
            if v:
                out.add(v)
    return sorted(out)


__all__ = (
    "decade_for_year", "decade_label", "unique_decades_from_dataframe",
    "resolve_decades", "add_decades_to_graph",
    "collect_decade_qids_for_metadata",
    "WD_DECADE_TYPE",
)
