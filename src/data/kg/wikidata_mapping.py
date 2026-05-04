"""
Wikidata enrichment for KG instruments and genres.

Why
---
Our local KG mints `mo:Instrument` and `mrc:Genre` individuals from raw
labels (e.g. ``"Acoustic Grand Piano"``, ``"new wave"``).  Those nodes are
isolated leaves with only an `rdfs:label`.  Wikidata gives us a stable
QID, multilingual labels, and — most importantly — a `wdt:P279`
(*subclass of*) chain we can fold into the KG to build a real taxonomy
(e.g. *new wave* ⊂ *rock music* ⊂ *popular music* ⊂ *music genre*).

Pipeline
--------
1. **Resolve** each free-text label → best Wikidata QID, constrained to
   a *direct* ``wdt:P31`` instance of a *type root*:

   * instruments → must be ``wdt:P31 wd:Q110295396`` (type of musical
     instrument).  E.g. *piano* (``Q5994``).
   * genres      → must be ``wdt:P31 wd:Q188451``     (music genre).
     E.g. *new wave music* (``Q187760``).

   Resolution does an English exact-match SPARQL on
   ``rdfs:label``/``skos:altLabel`` first (precise, instant) and falls
   back to ``wbsearchentities`` full-text search for stubborn labels.
2. **Expand** each resolved QID into its ``wdt:P279*`` chain
   (subclass-of transitive closure), capturing one English label per
   QID — e.g. *piano* → *struck string instrument* → *string
   instrument* → *musical instrument*.
3. **Cache** every step to disk as JSON so subsequent runs are free.

Network etiquette
-----------------
We hit the public Wikidata APIs.  Throttle with ``sleep_s=0.05`` between
calls and set a descriptive ``User-Agent`` (Wikidata blocks requests
without one).  All resolution failures are recorded in the cache as
``None`` so we never re-query a known miss.

Outputs
-------
``resolve_labels`` returns a ``dict[label → qid|None]``.
``fetch_subclass_chains`` returns ``dict[qid → [(qid, label), …]]``
ordered from the entity itself up to (and including) the type root.

Both are persisted as JSON in ``data/interim/`` by the cached wrappers.

Use ``enrich_graph_with_wikidata`` to fold the resolved entities + their
parent chains back into a populated ``KGBuilder`` graph as
``owl:sameAs`` (entity ↔ Wikidata QID URI) and `rdfs:subClassOf` (or
``mrc:broader``) edges between QID nodes.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Iterable, Optional

import requests

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from .kg_builder import KGBuilder, MRC, MO


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WD             = Namespace("http://www.wikidata.org/entity/")
WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Type roots used for filtering search hits.
#
# We use the *meta-class* roots (P31 = "is a") rather than the broad
# "musical instrument" / "music genre" classes (which would force us to
# walk P279*).  This matches Wikidata's modelling, e.g.
#
#   piano (Q5994)            wdt:P31  wd:Q110295396  ("type of musical instrument")
#   new wave music (Q187760) wdt:P31  wd:Q188451     ("music genre")
INSTRUMENT_ROOT = "Q110295396"   # type of musical instrument
GENRE_ROOT      = "Q188451"      # music genre

DEFAULT_HEADERS = {
    "User-Agent": (
        "DL-KG-project/0.1 (academic; pfanyka MSc) "
        "python-requests/wikidata-mapping"
    ),
    "Accept": "application/json",
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _wbsearch(label: str, language: str = "en", limit: int = 7) -> list[dict]:
    """Full-text entity search; returns list of search hits (may be empty)."""
    params = {
        "action":   "wbsearchentities",
        "search":   label,
        "language": language,
        "uselang":  language,
        "type":     "item",
        "format":   "json",
        "limit":    limit,
    }
    r = requests.get(WIKIDATA_API, params=params, headers=DEFAULT_HEADERS,
                     timeout=15)
    r.raise_for_status()
    return r.json().get("search", [])


def _sparql_select(query: str) -> list[dict]:
    """Run a SPARQL SELECT and return its bindings list."""
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers=DEFAULT_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])


def _sparql_ask(query: str) -> bool:
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers=DEFAULT_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return bool(r.json().get("boolean", False))


def _is_instance_of_root(qid: str, root_qid: str) -> bool:
    """ASK whether ``qid`` is a *direct* P31 instance of ``root_qid``.

    We deliberately do NOT walk transitively here — the resolution
    contract says "the picked entity must itself be a *type of musical
    instrument* / *music genre*", which is exactly ``wdt:P31``.
    Hierarchical relations between the picked entities are captured by
    :func:`fetch_subclass_chain` (P279).
    """
    q = f"ASK {{ wd:{qid} wdt:P31 wd:{root_qid} . }}"
    return _sparql_ask(q)


# ─────────────────────────────────────────────────────────────────────────────
# Label → QID resolution
# ─────────────────────────────────────────────────────────────────────────────
import re as _re
_PAREN_RE = _re.compile(r"\s*\([^)]*\)")


def _label_variants(label: str) -> list[str]:
    """
    Generate progressively-relaxed search variants for stubborn labels.

    GM MIDI names like ``"Acoustic Grand Piano"`` or ``"Pad 3 (polysynth)"``
    don't exist verbatim on Wikidata, but their tail tokens do
    (``"grand piano"``, ``"piano"``).  We try, in order:

    1. the original label,
    2. the label with parenthesized qualifiers stripped,
    3. lowercase,
    4. the last 2 words,
    5. the last word.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if s and s.lower() not in seen:
            out.append(s); seen.add(s.lower())

    _add(label)
    no_parens = _PAREN_RE.sub("", label).strip()
    _add(no_parens)
    _add(no_parens.lower())
    toks = no_parens.split()
    if len(toks) >= 2:
        _add(" ".join(toks[-2:]).lower())
    if toks:
        _add(toks[-1].lower())
    return out


def _sparql_resolve_by_label(label: str, root_qid: str) -> Optional[str]:
    """
    SPARQL-only resolution: find an item whose English ``rdfs:label`` /
    ``skos:altLabel`` (case-insensitive) equals ``label`` AND is a
    direct ``wdt:P31`` of ``root_qid``.  Returns the QID of the first
    match, or ``None``.

    This is the *precise* path — no full-text fuzziness, no surprise
    homonyms.  Wins for canonical names like ``"piano"`` or ``"jazz"``.
    """
    # Escape backslashes / quotes for SPARQL string literal.
    safe = label.replace("\\", "\\\\").replace('"', '\\"')
    q = f"""
    SELECT ?item WHERE {{
      ?item wdt:P31 wd:{root_qid} .
      {{ ?item rdfs:label    ?l . FILTER(LANG(?l) = "en" && LCASE(STR(?l)) = LCASE("{safe}")) }}
      UNION
      {{ ?item skos:altLabel ?l . FILTER(LANG(?l) = "en" && LCASE(STR(?l)) = LCASE("{safe}")) }}
    }}
    LIMIT 1
    """
    try:
        rows = _sparql_select(q)
    except requests.RequestException:
        return None
    if not rows:
        return None
    uri = rows[0]["item"]["value"]
    return uri.rsplit("/", 1)[-1]


def resolve_label(
    label: str,
    type_root: str,
    sleep_s: float = 0.05,
) -> Optional[str]:
    """
    Resolve a free-text ``label`` to a Wikidata QID that is a direct
    ``wdt:P31`` of ``type_root``.

    Strategy (stop at first hit):

    1. **SPARQL exact match** on English ``rdfs:label`` / ``skos:altLabel``
       for each variant of ``label`` (handles canonical names instantly).
    2. **wbsearchentities full-text** fallback for stubborn labels (GM
       MIDI names, fuzzy genre tags) — each candidate is then verified
       with a strict ``wdt:P31 wd:{type_root}`` ASK.
    """
    variants = _label_variants(label)

    # ── 1. precise SPARQL pass ─────────────────────────────────────────
    for v in variants:
        qid = _sparql_resolve_by_label(v, type_root)
        if qid:
            return qid
        time.sleep(sleep_s)

    # ── 2. fuzzy fallback via wbsearchentities ─────────────────────────
    for v in variants:
        try:
            hits = _wbsearch(v)
        except requests.RequestException:
            continue
        for hit in hits:
            qid = hit.get("id")
            if not qid:
                continue
            time.sleep(sleep_s)
            try:
                if _is_instance_of_root(qid, type_root):
                    return qid
            except requests.RequestException:
                continue
    return None


def resolve_labels(
    labels: Iterable[str],
    type_root: str,
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    sleep_s: float = 0.05,
    verbose: bool = True,
) -> dict[str, Optional[str]]:
    """
    Resolve many labels, persisting the lookup table to JSON so future
    calls are O(1).  Existing entries are kept (including ``None``
    misses) unless ``force_refresh=True``.
    """
    cache: dict[str, Optional[str]] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        cache = json.loads(cache_path.read_text())
        if verbose:
            print(f"[wikidata] loaded {len(cache):,} cached label→QID entries "
                  f"from {cache_path.name}")

    todo = [lbl for lbl in dict.fromkeys(labels) if lbl not in cache]
    if verbose:
        print(f"[wikidata] resolving {len(todo):,} new labels "
              f"(type_root={type_root}) …")

    for i, lbl in enumerate(todo, 1):
        cache[lbl] = resolve_label(lbl, type_root, sleep_s=sleep_s)
        if verbose and (i % 25 == 0 or i == len(todo)):
            n_hits = sum(1 for v in cache.values() if v)
            print(f"  {i:>5}/{len(todo):,}  hits so far: {n_hits:,}")
        if cache_path and i % 50 == 0:    # checkpoint
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] cache → {cache_path}")

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Subclass-of chain expansion
# ─────────────────────────────────────────────────────────────────────────────
# Domain bounds for the subclass-of chain walk.  Without these, an
# unbounded ``wdt:P279*`` from any node eventually leaks into Wikidata's
# upper-ontology meta-classes ("entity", "continuant", "abstract
# entity") which are useless for our music-recommendation hierarchy.
#
# We bound the walk by requiring every ancestor to itself live within
# the music domain, expressed as a single SPARQL clause keyed on the
# type root passed to :func:`fetch_subclass_chain`.
#
# instruments → ancestor must satisfy  ``wdt:P279* wd:Q34379``
#               (be a kind of *musical instrument*)
# genres      → ancestor must satisfy  ``wdt:P31  wd:Q188451``
#               (be itself a *music genre*)
DOMAIN_BOUNDS: dict[str, str] = {
    INSTRUMENT_ROOT: "?node wdt:P279* wd:Q34379 .",   # musical instrument
    GENRE_ROOT:      "?node wdt:P31  wd:Q188451 .",   # music genre
}


def fetch_subclass_chain(qid: str, type_root: Optional[str] = None
                         ) -> list[tuple[str, str]]:
    """
    Walk ``wdt:P279*`` upward from ``qid`` and return every ancestor
    that still lives within the music domain (see :data:`DOMAIN_BOUNDS`),
    as ``(qid, en_label)`` tuples.

    Example for *piano* (Q5994):

        Q5994     piano
        Q1294979  struck string instrument
        Q1798603  string instrument
        Q34379    musical instrument

    Without ``type_root`` the walk is unbounded and you'll get BFO-level
    meta-classes (``entity``, ``continuant`` …) which we don't want in
    the KG.  ``type_root`` should be one of :data:`INSTRUMENT_ROOT` /
    :data:`GENRE_ROOT`.  The query is hard-capped at ``LIMIT 100``.
    """
    bound_clause = DOMAIN_BOUNDS.get(type_root or "", "")
    q = f"""
    SELECT ?node ?nodeLabel WHERE {{
      wd:{qid} wdt:P279* ?node .
      {bound_clause}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 100
    """
    try:
        rows = _sparql_select(q)
    except requests.RequestException:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        node_uri = r["node"]["value"]
        node_qid = node_uri.rsplit("/", 1)[-1]
        if node_qid in seen:
            continue
        seen.add(node_qid)
        label = r.get("nodeLabel", {}).get("value", node_qid)
        out.append((node_qid, label))
    return out


def fetch_subclass_chains(
    qids: Iterable[str],
    type_root: str,
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    sleep_s: float = 0.05,
    verbose: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    """Cached batch wrapper around :func:`fetch_subclass_chain`."""
    cache: dict[str, list[tuple[str, str]]] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        raw = json.loads(cache_path.read_text())
        cache = {k: [tuple(p) for p in v] for k, v in raw.items()}
        if verbose:
            print(f"[wikidata] loaded {len(cache):,} cached subclass chains "
                  f"from {cache_path.name}")

    todo = [q for q in dict.fromkeys(qids) if q and q not in cache]
    if verbose:
        print(f"[wikidata] expanding {len(todo):,} new QIDs (root={type_root}) …")

    for i, qid in enumerate(todo, 1):
        cache[qid] = fetch_subclass_chain(qid, type_root)
        time.sleep(sleep_s)
        if verbose and (i % 20 == 0 or i == len(todo)):
            print(f"  {i:>4}/{len(todo):,}")
        if cache_path and i % 25 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] cache → {cache_path}")

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Fold Wikidata into the KG
# ─────────────────────────────────────────────────────────────────────────────
def enrich_graph_with_wikidata(
    builder: KGBuilder,
    *,
    instrument_map: Optional[dict[str, Optional[str]]] = None,
    genre_map:      Optional[dict[str, Optional[str]]] = None,
    instrument_chains: Optional[dict[str, list[tuple[str, str]]]] = None,
    genre_chains:      Optional[dict[str, list[tuple[str, str]]]] = None,
    add_hierarchy: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Add ``owl:sameAs`` from each local instrument/genre node to its
    Wikidata QID, and (optionally) attach every QID returned by
    :func:`fetch_subclass_chain` as a direct ``rdfs:subClassOf``
    ancestor of the leaf QID.

    Each Wikidata QID node carries:

    * ``rdfs:label`` — the **canonical Wikidata English label**
      (from the chain query), *not* the local label that mapped to it,
      so e.g. ``wd:Q5994`` always shows as ``"piano"`` even though we
      reached it via ``"Acoustic Grand Piano"``.
    * ``rdf:type`` — ``mo:Instrument`` / ``mrc:Genre`` for the leaf
      QIDs we resolved, ``skos:Concept`` for ancestor-only QIDs.

    On hierarchy edges
    ------------------
    ``fetch_subclass_chain`` returns the full ancestor *set* in
    arbitrary SPARQL order — it does **not** preserve the actual P279
    DAG structure between ancestors.  If we wired a flat
    ``leaf → A → B → C`` chain we would invent edges that don't exist
    on Wikidata (e.g. ``valve horn → bass guitar``).

    We therefore add only the safe edges: ``leaf rdfs:subClassOf
    ancestor`` for every ancestor.  This loses the inter-ancestor
    structure but guarantees that ``?leaf rdfs:subClassOf* ?x`` returns
    exactly the bounded ancestor set that Wikidata's transitive P279
    closure gave us — no false positives.
    """
    g = builder.g
    counts = {
        "instrument_links": 0, "genre_links": 0,
        "qid_nodes": 0, "subclass_edges": 0,
    }
    qid_nodes_seen: set[URIRef] = set()

    def _ensure_qid_node(qid: str, wd_label: Optional[str],
                         leaf_type: Optional[URIRef]) -> URIRef:
        """Mint (or reuse) a Wikidata QID node.

        ``wd_label`` is the *canonical Wikidata* English label sourced
        from :func:`fetch_subclass_chain` (NOT the local label that
        mapped to the QID — that would corrupt the QID's identity).
        """
        node = WD[qid]
        if node not in qid_nodes_seen:
            if wd_label:
                g.add((node, RDFS.label, Literal(wd_label, lang="en")))
            # Leaves get the strong ontology type; ancestor-only nodes
            # are skos:Concept so we don't force every ancestor into
            # mo:Instrument / mrc:Genre.
            if leaf_type is not None:
                g.add((node, RDF.type, leaf_type))
            else:
                g.add((node, RDF.type, SKOS.Concept))
            qid_nodes_seen.add(node)
            counts["qid_nodes"] += 1
        return node

    def _link(local_uri_fn, label_to_qid, chains,
              leaf_type: URIRef, count_key: str) -> None:
        if not label_to_qid:
            return
        for label, qid in label_to_qid.items():
            if not qid:
                continue
            # Try to fish the canonical Wikidata label out of the chain
            # (the seed QID is always present in its own chain).
            wd_label_for_leaf = None
            if chains and qid in chains:
                for c_qid, c_lab in chains[qid]:
                    if c_qid == qid:
                        wd_label_for_leaf = c_lab
                        break

            local_uri = local_uri_fn(label)
            wd_node   = _ensure_qid_node(qid, wd_label_for_leaf, leaf_type)
            g.add((local_uri, OWL.sameAs, wd_node))
            counts[count_key] += 1

            if add_hierarchy and chains and qid in chains:
                for p_qid, p_lab in chains[qid]:
                    if p_qid == qid:
                        continue   # skip the seed itself
                    p_node = _ensure_qid_node(p_qid, p_lab, leaf_type=None)
                    g.add((wd_node, RDFS.subClassOf, p_node))
                    counts["subclass_edges"] += 1

    _link(builder.instrument_uri, instrument_map, instrument_chains,
          leaf_type=MO["Instrument"],
          count_key="instrument_links")
    _link(builder.genre_uri,      genre_map,      genre_chains,
          leaf_type=MRC["Genre"],
          count_key="genre_links")

    if verbose:
        print(f"[wikidata] enrichment summary: {counts}")
    return counts


__all__ = (
    "WD", "INSTRUMENT_ROOT", "GENRE_ROOT",
    "resolve_label", "resolve_labels",
    "fetch_subclass_chain", "fetch_subclass_chains",
    "enrich_graph_with_wikidata",
)
