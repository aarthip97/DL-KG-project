"""
Wikidata enrichment for KG instruments, genres, and decades.

Design
------
All Wikidata alignment uses pure OWL predicates — no SKOS:

* ``owl:sameAs``         — instance-to-instance: a local genre/instrument
  individual is the same entity as the corresponding Wikidata QID.
* ``rdfs:subClassOf``    — class hierarchy: child genre/instrument class
  is a subclass of its parent, propagating up to the domain roots
  (wd:Q188451 music genre, wd:Q34379 musical instrument).
* ``owl:equivalentClass`` — class-to-class: local domain classes
  (mrc:Genre, mo:Instrument) are declared equivalent to the Wikidata root
  class by ``KGBuilder.add_music_concept_hierarchy()``.

Performance
-----------
Three Wikidata HTTP endpoints are used and **all of them are pooled with
ThreadPoolExecutor**:

* WDQS SPARQL endpoint  -> exact-label resolution and P279* chain walks
* wbsearchentities      -> fuzzy label fallback
* wbgetentities (batched, <=50 IDs/call) -> English label, description,
  and aliases for every QID we mint into the KG

Caches
------
All network results persist as JSON in ``data/interim/``.
"""
from __future__ import annotations

import json
import pathlib
import re as _re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import requests
from tqdm.auto import tqdm

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .kg_builder import (
    KGBuilder, MRC, MO,
    WD_MUSICAL_CONCEPT, WD_ELEMENTS_OF_MUSIC,
    WD_MUSIC_GENRE, WD_MUSICAL_INSTRUMENT, WD_KEY_MUSIC,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WD              = Namespace("http://www.wikidata.org/entity/")
WDT             = Namespace("http://www.wikidata.org/prop/direct/")
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Resolution type-roots: items must be wdt:P31 of these.
INSTRUMENT_ROOT = "Q110295396"   # "type of musical instrument"
GENRE_ROOT      = "Q188451"      # "music genre"

# Domain filter for the **direct-parent** SPARQL query (``fetch_direct_parents``).
# Each value is a WHERE clause fragment that constrains ``?parent`` to stay
# within the music domain so we never walk into upper-ontology meta-classes.
_PARENT_DOMAIN_FILTER: dict[str, str] = {
    # parent must itself be a subclass-of musical instrument (Q34379)
    INSTRUMENT_ROOT: "?parent wdt:P279* wd:Q34379 .",
    # parent must be an instance-of music genre OR a subclass-of music genre
    GENRE_ROOT: """{ ?parent wdt:P31 wd:Q188451 . }
      UNION
      { ?parent wdt:P279+ wd:Q188451 . }""",
}

# Kept for backward compatibility with old call sites that pass DOMAIN_BOUNDS
# to fetch_subclass_chain (deprecated).
DOMAIN_BOUNDS: dict[str, str] = {
    INSTRUMENT_ROOT: "?node wdt:P279* wd:Q34379 .",
    GENRE_ROOT:      "?node wdt:P31  wd:Q188451 .",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "DL-KG-project/0.2 (academic; pfanyka MSc) "
        "python-requests/wikidata-mapping"
    ),
    "Accept": "application/json",
}

_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        _thread_local.session = s
    return s


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------
def _wbsearch(label: str, language: str = "en", limit: int = 7) -> list[dict]:
    params = {
        "action": "wbsearchentities", "search": label,
        "language": language, "uselang": language,
        "type": "item", "format": "json", "limit": limit,
    }
    r = _session().get(WIKIDATA_API, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("search", [])


def _sparql_select(query: str) -> list[dict]:
    r = _session().get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])


def _sparql_ask(query: str) -> bool:
    r = _session().get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    return bool(r.json().get("boolean", False))


def _is_instance_of_root(qid: str, root_qid: str) -> bool:
    return _sparql_ask(f"ASK {{ wd:{qid} wdt:P31 wd:{root_qid} . }}")


# ---------------------------------------------------------------------------
# Label -> QID resolution
# ---------------------------------------------------------------------------
_PAREN_RE = _re.compile(r"\s*\([^)]*\)")


def _label_variants(label: str) -> list[str]:
    """Progressively-relaxed search variants - handles GM MIDI names."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if s and s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())

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
    """SPARQL exact-match: english label/altLabel == label, P31 root_qid."""
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
    return rows[0]["item"]["value"].rsplit("/", 1)[-1]


def resolve_label(label: str, type_root: str,
                  max_fallback_variants: int = 2) -> Optional[str]:
    """Resolve one label to a QID (precise SPARQL -> fuzzy fallback).

    Parameters
    ----------
    max_fallback_variants : how many label variants to try in the *fuzzy*
        wbsearch fallback (not the initial SPARQL exact-match phase).
        Lowering this from the default ``len(_label_variants(label))`` (~5)
        dramatically cuts the worst-case request count for labels that will
        never resolve, at the cost of a tiny miss rate on unusual spellings.
        Set to ``0`` to disable the fuzzy fallback entirely.
    """
    variants = _label_variants(label)

    # Phase 1 — fast SPARQL exact-match (one request per variant, in parallel
    # across labels because each call happens inside a ThreadPoolExecutor
    # worker; within this function they are still sequential).
    for v in variants:
        qid = _sparql_resolve_by_label(v, type_root)
        if qid:
            return qid

    # Phase 2 — fuzzy fallback: wbsearch + P31 ASK check.
    # Capped at max_fallback_variants to avoid 40+ requests for hopeless labels.
    for v in variants[:max(0, max_fallback_variants)]:
        try:
            hits = _wbsearch(v)
        except requests.RequestException:
            continue
        for hit in hits:
            qid = hit.get("id")
            if not qid:
                continue
            try:
                if _is_instance_of_root(qid, type_root):
                    return qid
            except requests.RequestException:
                continue
    return None


def _parallel_dict_fill(items, worker, parallel, cache, cache_path,
                        verbose, label_for_log, checkpoint_every=100):
    """Run ``worker(item)`` over ``items`` in parallel, writing into ``cache``."""
    if not items:
        return
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(worker, it): it for it in items}
        bar = tqdm(as_completed(futures), total=len(futures),
                   desc=label_for_log, unit="item",
                   disable=not verbose, leave=True)
        for fut in bar:
            it = futures[fut]
            try:
                cache[it] = fut.result()
            except Exception:
                cache[it] = None
            with lock:
                done += 1
                if cache_path and done % checkpoint_every == 0:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(cache, indent=2, sort_keys=True, default=list))
        bar.close()


def resolve_labels(
    labels: Iterable[str],
    type_root: str,
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    parallel: int = 6,
    max_fallback_variants: int = 2,
    verbose: bool = True,
) -> dict[str, Optional[str]]:
    """Parallel label -> QID lookup with on-disk JSON cache.

    Parameters
    ----------
    max_fallback_variants : forwarded to :func:`resolve_label`; limits the
        number of label variants tried in the slow fuzzy-fallback phase
        (default 2 — enough to handle casing/punctuation variants while
        avoiding the ~40-request worst-case per unresolvable label).
    """
    cache: dict[str, Optional[str]] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        cache = json.loads(cache_path.read_text())
        if verbose:
            print(f"[wikidata] loaded {len(cache):,} cached label->QID entries "
                  f"from {cache_path.name}")

    todo = [lbl for lbl in dict.fromkeys(labels) if lbl not in cache]
    if verbose:
        print(f"[wikidata] resolving {len(todo):,} new labels "
              f"(type_root={type_root}, workers={parallel}) ...")

    _parallel_dict_fill(
        todo,
        lambda lbl: resolve_label(lbl, type_root,
                                   max_fallback_variants=max_fallback_variants),
        parallel, cache, cache_path, verbose, "labels",
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] cache -> {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Direct-parent expansion  (replaces the flat P279* ancestor dump)
# ---------------------------------------------------------------------------
def fetch_direct_parents(qid: str, type_root: str) -> list[tuple[str, str]]:
    """Fetch the *direct* ``wdt:P279`` parents of ``qid`` within the domain.

    Only one hop up — no transitive closure.  The caller (``build_parent_graph``)
    is responsible for recursing until the domain root is reached.

    Returns
    -------
    list of ``(parent_qid, parent_english_label)`` — empty on failure or
    when ``qid`` has no in-domain parents.
    """
    domain_clause = _PARENT_DOMAIN_FILTER.get(type_root, "")
    q = f"""
    SELECT DISTINCT ?parent ?parentLabel WHERE {{
      wd:{qid} wdt:P279 ?parent .
      {domain_clause}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 25
    """
    try:
        rows = _sparql_select(q)
    except requests.RequestException:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        p_qid = r["parent"]["value"].rsplit("/", 1)[-1]
        if p_qid in seen or p_qid == qid:
            continue
        seen.add(p_qid)
        label = r.get("parentLabel", {}).get("value", p_qid)
        out.append((p_qid, label))
    return out


def build_parent_graph(
    leaf_qids: Iterable[str],
    type_root: str,
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    parallel: int = 6,
    verbose: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    """Build a shared parent-graph via BFS from ``leaf_qids``.

    Starting from the leaf QIDs (direct Wikidata matches for our labels),
    fetches each item's **direct** ``wdt:P279`` parents that are still within
    the music domain, then fetches *their* parents, and so on until every
    reachable ancestor is in the graph.

    Cache format
    ------------
    A flat JSON dict: ``{qid: [[parent_qid, parent_label], …], …}``

    Each key maps to its **direct parents only** (no transitive ancestors).
    All intermediate nodes get their own top-level entry.  This is much more
    space-efficient than the old flat-ancestor dump *and* correctly represents
    multi-parent DAGs (e.g. EDM is broader than both trance and dance music).

    .. note::
        If the cache file was written by the old ``fetch_subclass_chains``
        function it will contain flat ancestor lists.  Pass
        ``force_refresh=True`` to regenerate in the new format.
    """
    graph: dict[str, list[tuple[str, str]]] = {}

    if cache_path and cache_path.exists() and not force_refresh:
        raw = json.loads(cache_path.read_text())
        graph = {k: [tuple(p) for p in v] for k, v in raw.items()}
        # Detect old flat-ancestor format: a non-empty entry that contains
        # the key QID itself as one of its "parents".
        _old_fmt = any(
            any(p[0] == k for p in v)
            for k, v in graph.items() if v
        )
        if _old_fmt:
            if verbose:
                print(
                    f"[wikidata] ⚠  {cache_path.name} uses the OLD flat-ancestor "
                    f"format — run with force_refresh=True to regenerate in the "
                    f"new direct-parent format."
                )
        else:
            if verbose:
                print(f"[wikidata] loaded {len(graph):,} parent-graph entries "
                      f"from {cache_path.name}")
            todo = [q for q in dict.fromkeys(leaf_qids) if q and q not in graph]
            if not todo:
                return graph
            # Fall through to fetch only the missing leaf entries

    queue: set[str] = {q for q in dict.fromkeys(leaf_qids) if q and q not in graph}
    round_n = 0
    while queue:
        round_n += 1
        if verbose:
            print(f"[wikidata] parent-graph BFS round {round_n}: "
                  f"fetching direct parents for {len(queue):,} QIDs ...")

        batch: dict[str, list[tuple[str, str]]] = {}
        _parallel_dict_fill(
            list(queue),
            lambda qid: fetch_direct_parents(qid, type_root),
            parallel, batch, None, verbose,
            f"direct-parents (r{round_n})",
            checkpoint_every=100,
        )
        graph.update(batch)

        # Checkpoint after each BFS round
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(graph, indent=2, sort_keys=True))

        # Next queue: parent QIDs discovered this round that have no entry yet
        queue = {
            p_qid
            for parents in batch.values()
            for p_qid, _ in parents
            if p_qid not in graph
        }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(graph, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] has parent graph!"
                  f"({len(graph):,} nodes, {round_n} BFS rounds)")
    return graph


# ---------------------------------------------------------------------------
# Deprecated: flat P279* ancestor dumps  (kept for backward compatibility)
# ---------------------------------------------------------------------------
def fetch_subclass_chain(qid: str, type_root: Optional[str] = None
                         ) -> list[tuple[str, str]]:
    """**Deprecated** — use :func:`fetch_direct_parents` instead.

    The old implementation dumps *all* P279* ancestors in a flat list,
    which (a) wastes space because shared ancestors are repeated per leaf,
    and (b) causes incorrect ``skos:broader`` edges (every ancestor is
    asserted as a direct broader concept of the leaf).
    """
    import warnings
    warnings.warn(
        "fetch_subclass_chain is deprecated; use fetch_direct_parents / "
        "build_parent_graph instead.",
        DeprecationWarning, stacklevel=2,
    )
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
        node_qid = r["node"]["value"].rsplit("/", 1)[-1]
        if node_qid in seen:
            continue
        seen.add(node_qid)
        out.append((node_qid, r.get("nodeLabel", {}).get("value", node_qid)))
    return out


def fetch_subclass_chains(
    qids: Iterable[str],
    type_root: str,
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    parallel: int = 6,
    verbose: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    """**Deprecated** — use :func:`build_parent_graph` instead."""
    import warnings
    warnings.warn(
        "fetch_subclass_chains is deprecated; use build_parent_graph instead.",
        DeprecationWarning, stacklevel=2,
    )
    return build_parent_graph(
        qids, type_root, cache_path, force_refresh, parallel, verbose,
    )


# ---------------------------------------------------------------------------
# QID metadata (label / description / aliases) via wbgetentities
# ---------------------------------------------------------------------------
def _wbgetentities_batch(qids: list[str], language: str = "en"
                         ) -> dict[str, dict]:
    """One wbgetentities call (<=50 IDs)."""
    if not qids:
        return {}
    params = {
        "action":    "wbgetentities",
        "ids":       "|".join(qids),
        "props":     "labels|descriptions|aliases",
        "languages": language,
        "format":    "json",
    }
    r = _session().get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json().get("entities", {})
    out: dict[str, dict] = {}
    for qid, ent in payload.items():
        labels  = ent.get("labels", {}).get(language, {})
        descs   = ent.get("descriptions", {}).get(language, {})
        aliases = ent.get("aliases", {}).get(language, []) or []
        out[qid] = {
            "label":       labels.get("value"),
            "description": descs.get("value"),
            "aliases":     [a.get("value") for a in aliases if a.get("value")],
        }
    return out


def fetch_qid_metadata(
    qids: Iterable[str],
    language: str = "en",
    cache_path: Optional[pathlib.Path] = None,
    force_refresh: bool = False,
    parallel: int = 4,
    batch_size: int = 50,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Fetch English label / description / aliases for a set of QIDs, using
    batched wbgetentities calls dispatched in parallel.
    """
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        cache = json.loads(cache_path.read_text())
        if verbose:
            print(f"[wikidata] loaded {len(cache):,} cached QID-metadata "
                  f"entries from {cache_path.name}")

    uniq = [q for q in dict.fromkeys(qids) if q and q not in cache]
    if verbose:
        print(f"[wikidata] fetching metadata for {len(uniq):,} new QIDs "
              f"(workers={parallel}, batch={batch_size}) ...")
    if not uniq:
        return cache

    batches = [uniq[i:i + batch_size] for i in range(0, len(uniq), batch_size)]
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_wbgetentities_batch, b, language): b
                   for b in batches}
        bar = tqdm(as_completed(futures), total=len(futures),
                   desc="qid-meta", unit="batch",
                   disable=not verbose, leave=True)
        for fut in bar:
            try:
                batch_out = fut.result()
            except Exception:                                  # noqa: BLE001
                batch_out = {}
            with lock:
                cache.update(batch_out)
        bar.close()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] cached!")
    return cache


# ---------------------------------------------------------------------------
# Fold Wikidata into the KG — pure OWL edition
# ---------------------------------------------------------------------------
def _attach_qid_metadata(g, node: URIRef, qid: str,
                         meta: dict, fallback_label: Optional[str]) -> None:
    """Stamp a Wikidata QID node with OWL annotations and type it as owl:Class.

    Emits ``rdfs:label`` (standard OWL annotation) and ``rdfs:comment``
    (description from Wikidata), always types the node as ``owl:Class`` so
    it appears in Protégé's class hierarchy and supports ``rdfs:subClassOf``.
    """
    label = meta.get("label") or fallback_label or qid
    g.add((node, RDF.type,   OWL.Class))
    g.add((node, RDFS.label, Literal(label, lang="en")))
    if meta.get("description"):
        g.add((node, RDFS.comment, Literal(meta["description"], lang="en")))


def enrich_graph_with_wikidata(
    builder: KGBuilder,
    *,
    instrument_map:    Optional[dict[str, Optional[str]]] = None,
    genre_map:         Optional[dict[str, Optional[str]]] = None,
    instrument_chains: Optional[dict[str, list[tuple[str, str]]]] = None,
    genre_chains:      Optional[dict[str, list[tuple[str, str]]]] = None,
    qid_metadata:      Optional[dict[str, dict]] = None,
    add_hierarchy:     bool = True,
    verbose:           bool = True,
) -> dict[str, int]:
    """Fold Wikidata results into ``builder.g`` using pure OWL predicates.

    Expects ``instrument_chains`` / ``genre_chains`` in the **direct-parent**
    format produced by :func:`build_parent_graph`:
    ``{qid: [(parent_qid, parent_label), …], …}``

    For each local genre/instrument individual the function:

    1. Asserts ``owl:sameAs`` to the matching Wikidata QID (instance equality);
    2. Asserts ``rdfs:subClassOf`` to the WD QID (class hierarchy — valid
       because local individuals are also declared ``owl:Class`` by KGBuilder);
    3. Recursively walks the parent graph upward, asserting
       ``wd:child rdfs:subClassOf wd:parent`` for each direct parent edge;
    4. Caps the walk at the domain roots (Q34379 / Q188451), which are
       connected to the upper Wikidata hierarchy by
       ``add_music_concept_hierarchy()``.
    """
    g = builder.g
    counts = {
        "instrument_links": 0, "genre_links":  0,
        "qid_concepts":     0, "subclass_edges": 0,
    }

    qid_seen: set[URIRef] = set()
    _hierarchy_visited: set[str] = set()
    metadata = qid_metadata or {}

    def _ensure_wd_class(qid: str, fallback_label: Optional[str] = None) -> URIRef:
        node = WD[qid]
        if node in qid_seen:
            return node
        _attach_qid_metadata(g, node, qid, metadata.get(qid, {}), fallback_label)
        qid_seen.add(node)
        counts["qid_concepts"] += 1
        return node

    def _walk_hierarchy(cur_qid: str, chains: dict) -> None:
        """Recursively assert rdfs:subClassOf edges upward (one hop per call)."""
        if cur_qid in _hierarchy_visited:
            return
        _hierarchy_visited.add(cur_qid)

        cur_node = WD[cur_qid]
        for p_qid, p_lab in chains.get(cur_qid) or []:
            p_node = _ensure_wd_class(p_qid, fallback_label=p_lab)
            g.add((cur_node, RDFS.subClassOf, p_node))
            counts["subclass_edges"] += 1
            _walk_hierarchy(p_qid, chains)

    def _link(local_uri_fn, label_to_qid, chains, count_key,
              domain_class: URIRef) -> None:
        if not label_to_qid:
            return
        for label, qid in label_to_qid.items():
            local = local_uri_fn(label)
            if not qid:
                # No Wikidata match — subclass directly under the domain root
                g.add((local, RDFS.subClassOf, domain_class))
                continue

            chain_label = metadata.get(qid, {}).get("label") or label
            wd_node = _ensure_wd_class(qid, fallback_label=chain_label)
            # Instance equality: local individual IS the WD item
            g.add((local, OWL.sameAs,        wd_node))
            # Class hierarchy: local class is a subclass of the WD class
            g.add((local, RDFS.subClassOf,   wd_node))
            counts[count_key] += 1

            if add_hierarchy and chains:
                _walk_hierarchy(qid, chains)

    _link(builder.instrument_uri, instrument_map, instrument_chains,
          "instrument_links", MO["Instrument"])
    _link(builder.genre_uri,      genre_map,      genre_chains,
          "genre_links",      MRC["Genre"])

    # ── Ensure upper Wikidata concept nodes are typed as owl:Class ────────────
    _upper = [
        (WD_MUSIC_GENRE,        WD_ELEMENTS_OF_MUSIC,  "music genre"),
        (WD_MUSICAL_INSTRUMENT, WD_ELEMENTS_OF_MUSIC,  "musical instrument"),
        (WD_ELEMENTS_OF_MUSIC,  WD_MUSICAL_CONCEPT,    "elements of music"),
        (WD_MUSICAL_CONCEPT,    None,                  "musical concept"),
    ]
    for node, parent, label in _upper:
        if (node, RDF.type, OWL.Class) not in g:
            g.add((node, RDF.type,   OWL.Class))
            g.add((node, RDFS.label, Literal(label, lang="en")))
        if parent is not None:
            g.add((node, RDFS.subClassOf, parent))

    if verbose:
        print(f"[wikidata] enrichment summary: {counts}")
    return counts


# ---------------------------------------------------------------------------
# Audit — answer "did we add any nodes that were not in the dataset?"
# ---------------------------------------------------------------------------
def audit_wikidata_enrichment(
    builder: KGBuilder,
    instrument_map: Optional[dict[str, Optional[str]]] = None,
    genre_map:      Optional[dict[str, Optional[str]]] = None,
    show_examples:  int = 5,
    verbose:        bool = True,
) -> dict[str, dict]:
    """Inspect the populated graph and report coverage per domain class.

    Reports per domain (instruments / genres):

      * ``leaves_total``    — local named individuals in the domain;
      * ``leaves_linked``   — locals with an ``owl:sameAs`` to Wikidata;
      * ``leaves_orphan``   — locals without a Wikidata match;
      * ``wd_anchors``      — WD QID nodes that are direct sameAs targets;
      * ``wd_ancestors_only`` — WD nodes added only as hierarchy ancestors;
      * ``subclass_edges``  — total ``rdfs:subClassOf`` edges in the domain.
    """
    g = builder.g
    WD_PREFIX = str(WD)

    DOMAINS = {
        "instruments": (MO["Instrument"],  instrument_map),
        "genres":      (MRC["Genre"],      genre_map),
    }

    report: dict[str, dict] = {}
    for name, (domain_class, label_map) in DOMAINS.items():
        # 1. Local individuals (rdf:type domain_class, not a WD URI)
        all_local = {
            s for s in g.subjects(RDF.type, domain_class)
            if not str(s).startswith(WD_PREFIX)
        }

        # 2. Linked vs. orphan locals
        linked = {loc for loc in all_local if any(g.objects(loc, OWL.sameAs))}
        orphan = all_local - linked

        # 3. WD anchors vs. ancestor-only WD nodes
        anchored_wd: set[URIRef] = set()
        for loc in linked:
            for o in g.objects(loc, OWL.sameAs):
                if isinstance(o, URIRef) and str(o).startswith(WD_PREFIX):
                    anchored_wd.add(o)

        all_wd_in_domain = {
            o for s in all_local
            for o in g.objects(s, RDFS.subClassOf)
            if isinstance(o, URIRef) and str(o).startswith(WD_PREFIX)
        }
        ancestor_only = all_wd_in_domain - anchored_wd

        # 4. rdfs:subClassOf edge count within the domain
        subclass_edges = sum(
            1 for s, _, o in g.triples((None, RDFS.subClassOf, None))
            if s in all_local or s in all_wd_in_domain
        )

        # 5. Sample ancestor labels
        examples: list[tuple[str, str]] = []
        for n in list(ancestor_only)[:show_examples]:
            lab = next(g.objects(n, RDFS.label), None)
            qid = str(n).rsplit("/", 1)[-1]
            examples.append((qid, str(lab) if lab else qid))

        scheme_report = {
            "leaves_total":      len(all_local),
            "leaves_linked":     len(linked),
            "leaves_orphan":     len(orphan),
            "wd_anchors":        len(anchored_wd),
            "wd_ancestors_only": len(ancestor_only),
            "subclass_edges":    subclass_edges,
            "ancestor_examples": examples,
        }

        if label_map is not None:
            scheme_report["raw_labels_total"]  = len(label_map)
            scheme_report["raw_labels_hit"]    = sum(1 for v in label_map.values() if v)
            scheme_report["raw_labels_missed"] = sum(1 for v in label_map.values() if not v)

        report[name] = scheme_report

    if verbose:
        for name, r in report.items():
            print(f"\n── {name.upper()} ──")
            for k, v in r.items():
                if k == "ancestor_examples":
                    print(f"  {k:<22} :")
                    for qid, lab in v:
                        print(f"      {qid:<10} {lab}")
                else:
                    print(f"  {k:<22} : {v}")

    return report


__all__ = (
    "WD", "WDT",
    "INSTRUMENT_ROOT", "GENRE_ROOT",
    "DOMAIN_BOUNDS", "_PARENT_DOMAIN_FILTER",
    "resolve_label", "resolve_labels",
    "fetch_direct_parents", "build_parent_graph",
    "fetch_subclass_chain", "fetch_subclass_chains",  # deprecated
    "fetch_qid_metadata",
    "_attach_qid_metadata",
    "enrich_graph_with_wikidata",
    "audit_wikidata_enrichment",
)
