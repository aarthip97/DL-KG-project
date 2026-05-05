"""
Wikidata enrichment for KG instruments, genres, and decades - SKOS edition.

Why SKOS
--------
The previous version asserted ``rdfs:subClassOf`` between ``rdf:type``-d
Wikidata QID nodes. That is conceptually awkward: a *genre* or an
*instrument family* is not a class of musical works, it is a *concept*
in a controlled vocabulary. The SKOS data model is the industry standard
for that:

* every node is a ``skos:Concept`` belonging to a ``skos:ConceptScheme``;
* hierarchy uses ``skos:broader`` / ``skos:narrower`` (concept-to-concept,
  no class subsumption implied);
* the link from our local label-derived concept to the canonical
  Wikidata entity uses ``skos:exactMatch`` (semantic equivalence).

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
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from .kg_builder import KGBuilder, MRC, MO


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

# Domain bounds for the P279* ancestor walk - without them, an unbounded
# walk eventually leaks into upper-ontology meta-classes.
DOMAIN_BOUNDS: dict[str, str] = {
    INSTRUMENT_ROOT: "?node wdt:P279* wd:Q34379 .",   # subset of musical instrument
    GENRE_ROOT:      "?node wdt:P31  wd:Q188451 .",   # is a music genre
}

# ConceptScheme URIs (minted under the mrc namespace).
INSTRUMENT_SCHEME = MRC["scheme/Instruments"]
GENRE_SCHEME      = MRC["scheme/Genres"]
DECADE_SCHEME     = MRC["scheme/Decades"]

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
# Subclass-of (P279*) chain expansion
# ---------------------------------------------------------------------------
def fetch_subclass_chain(qid: str, type_root: Optional[str] = None
                         ) -> list[tuple[str, str]]:
    """Walk wdt:P279* upward from ``qid``, bounded to the music domain."""
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
    """Parallel cached batch wrapper around fetch_subclass_chain."""
    cache: dict[str, list[tuple[str, str]]] = {}
    if cache_path and cache_path.exists() and not force_refresh:
        raw = json.loads(cache_path.read_text())
        cache = {k: [tuple(p) for p in v] for k, v in raw.items()}
        if verbose:
            print(f"[wikidata] loaded {len(cache):,} cached subclass chains "
                  f"from {cache_path.name}")

    todo = [q for q in dict.fromkeys(qids) if q and q not in cache]
    if verbose:
        print(f"[wikidata] expanding {len(todo):,} new QIDs "
              f"(root={type_root}, workers={parallel}) ...")

    _parallel_dict_fill(
        todo, lambda qid: fetch_subclass_chain(qid, type_root),
        parallel, cache, cache_path, verbose, "chains", checkpoint_every=50,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if verbose:
            print(f"[wikidata] cache -> {cache_path}")
    return cache


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
            print(f"[wikidata] cache -> {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Fold Wikidata into the KG - SKOS edition
# ---------------------------------------------------------------------------
def _ensure_scheme(g, scheme_uri: URIRef, label: str) -> None:
    if (scheme_uri, RDF.type, SKOS.ConceptScheme) in g:
        return
    g.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
    g.add((scheme_uri, SKOS.prefLabel, Literal(label, lang="en")))


def _attach_qid_metadata(g, node: URIRef, scheme: URIRef, qid: str,
                        meta: dict, fallback_label: Optional[str],
                        protege_friendly: bool = True) -> None:
    """Stamp a QID node with SKOS label/definition/altLabels.

    When ``protege_friendly`` is True the node is *also* typed as
    ``owl:Class`` (in addition to ``skos:Concept``) and given an
    ``rdfs:label`` mirroring ``skos:prefLabel`` — that makes Protégé
    render the node in the Class hierarchy tab (which only shows
    ``owl:Class`` + ``rdfs:subClassOf``, not SKOS).
    """
    label = meta.get("label") or fallback_label or qid
    g.add((node, RDF.type, SKOS.Concept))
    g.add((node, SKOS.inScheme, scheme))
    g.add((node, SKOS.prefLabel, Literal(label, lang="en")))
    if protege_friendly:
        g.add((node, RDF.type, OWL.Class))
        g.add((node, RDFS.label, Literal(label, lang="en")))
    if meta.get("description"):
        g.add((node, SKOS.definition, Literal(meta["description"], lang="en")))
        if protege_friendly:
            g.add((node, RDFS.comment, Literal(meta["description"], lang="en")))
    for alias in meta.get("aliases", []) or []:
        g.add((node, SKOS.altLabel, Literal(alias, lang="en")))


def enrich_graph_with_wikidata(
    builder: KGBuilder,
    *,
    instrument_map:    Optional[dict[str, Optional[str]]] = None,
    genre_map:         Optional[dict[str, Optional[str]]] = None,
    instrument_chains: Optional[dict[str, list[tuple[str, str]]]] = None,
    genre_chains:      Optional[dict[str, list[tuple[str, str]]]] = None,
    qid_metadata:      Optional[dict[str, dict]] = None,
    add_hierarchy:     bool = True,
    protege_friendly:  bool = True,
    verbose:           bool = True,
) -> dict[str, int]:
    """
    Fold Wikidata results into ``builder.g`` using the SKOS data model.

    Local nodes get ``a skos:Concept ; skos:inScheme <scheme> ;
    skos:prefLabel "<local>"@en ; skos:exactMatch wd:Q...``.

    Wikidata QIDs are minted (idempotently) as ``skos:Concept`` in the
    same scheme, carrying ``skos:prefLabel`` (canonical English label),
    ``skos:definition`` (English description), and one ``skos:altLabel``
    per English alias.

    Hierarchy edges are added as ``leaf skos:broader ancestor`` for each
    ancestor returned by fetch_subclass_chain (no inter-ancestor edges
    are invented - the chain query does not preserve the DAG topology).

    When ``protege_friendly=True`` (default) we *additionally* dual-emit
    OWL hierarchy triples so Protégé's *Class* hierarchy panel renders
    the structure (Protégé does not visualise ``skos:broader`` between
    SKOS individuals — only ``rdfs:subClassOf`` between ``owl:Class``
    nodes). Specifically:

      * each Wikidata anchor node is *also* typed ``owl:Class`` and
        given an ``rdfs:label`` mirroring its ``skos:prefLabel``;
      * each ``leaf skos:broader ancestor`` triple is mirrored as
        ``leaf rdfs:subClassOf ancestor``;
      * each local genre / instrument node is *also* typed ``owl:Class``
        and given ``rdfs:subClassOf <Wikidata-anchor>`` (in addition to
        the existing ``skos:exactMatch`` link), and rooted under
        ``mrc:Genre`` / ``mo:Instrument`` so it shows up under the
        domain class in Protégé.

    The SKOS triples are *not* removed — both views co-exist.
    """
    g = builder.g
    counts = {
        "instrument_links": 0, "genre_links":   0,
        "qid_concepts":     0, "broader_edges": 0,
        "subclass_edges":   0,
    }

    _ensure_scheme(g, INSTRUMENT_SCHEME, "Musical instruments")
    _ensure_scheme(g, GENRE_SCHEME,      "Music genres")

    qid_seen: set[URIRef] = set()
    metadata = qid_metadata or {}

    def _ensure_concept(qid: str, scheme: URIRef,
                        fallback_label: Optional[str] = None) -> URIRef:
        node = WD[qid]
        if node in qid_seen:
            return node
        _attach_qid_metadata(g, node, scheme, qid,
                             metadata.get(qid, {}), fallback_label,
                             protege_friendly=protege_friendly)
        qid_seen.add(node)
        counts["qid_concepts"] += 1
        return node

    def _link(local_uri_fn, label_to_qid, chains, scheme, count_key,
              domain_class: URIRef):
        if not label_to_qid:
            return
        for label, qid in label_to_qid.items():
            local = local_uri_fn(label)
            g.add((local, RDF.type, SKOS.Concept))
            g.add((local, SKOS.inScheme, scheme))
            g.add((local, SKOS.prefLabel, Literal(label, lang="en")))
            if protege_friendly:
                # Dual-type so Protégé renders the local node as a class.
                # Note: this is OWL Full (the same IRI is both an
                # individual of mrc:Genre / mo:Instrument *and* an
                # owl:Class) but Protégé tolerates it, and we keep the
                # original instance typing so SHACL / range constraints
                # on mrc:hasGenre still hold.
                g.add((local, RDF.type, OWL.Class))
                g.add((local, RDFS.subClassOf, domain_class))
            if not qid:
                continue

            chain_label = None
            if chains and qid in chains:
                for c_qid, c_lab in chains[qid]:
                    if c_qid == qid:
                        chain_label = c_lab
                        break
            wd_node = _ensure_concept(qid, scheme, fallback_label=chain_label)
            g.add((local, SKOS.exactMatch, wd_node))
            if protege_friendly:
                # The leaf is a (Wikidata-grounded) sub-kind of its anchor.
                g.add((local, RDFS.subClassOf, wd_node))
                counts["subclass_edges"] += 1
            counts[count_key] += 1

            if add_hierarchy and chains and qid in chains:
                for p_qid, p_lab in chains[qid]:
                    if p_qid == qid:
                        continue
                    p_node = _ensure_concept(p_qid, scheme,
                                             fallback_label=p_lab)
                    g.add((wd_node, SKOS.broader, p_node))
                    counts["broader_edges"] += 1
                    if protege_friendly:
                        g.add((wd_node, RDFS.subClassOf, p_node))
                        counts["subclass_edges"] += 1

    _link(builder.instrument_uri, instrument_map, instrument_chains,
          INSTRUMENT_SCHEME, "instrument_links", MO["Instrument"])
    _link(builder.genre_uri,      genre_map,      genre_chains,
          GENRE_SCHEME,      "genre_links",      MRC["Genre"])

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
    """
    Inspect the populated graph and report, per scheme:

      * ``leaves_total``     — local label-derived concepts in the scheme;
      * ``leaves_linked``    — leaves with a ``skos:exactMatch`` to Wikidata;
      * ``leaves_orphan``    — leaves *without* a Wikidata match (still in KG);
      * ``wd_anchors``       — Wikidata QID nodes that are direct matches of a leaf;
      * ``wd_ancestors``     — Wikidata QID nodes added *only* as ancestors
        (these are the "extra" hierarchy nodes the user worried about — they
        are never used as a leaf in the data, only as a parent class);
      * ``broader_edges``    — total ``skos:broader`` edges within the scheme;
      * ``subclass_edges``   — total ``rdfs:subClassOf`` edges (Protégé view);
      * ``ancestor_examples`` — sample of ancestor QIDs with their labels.

    Pass ``instrument_map`` / ``genre_map`` to also report dataset-side
    coverage (how many *raw* labels the resolver hit / missed).
    """
    g = builder.g
    report: dict[str, dict] = {}

    SCHEMES = {
        "instruments": (INSTRUMENT_SCHEME, instrument_map),
        "genres":      (GENRE_SCHEME,      genre_map),
    }

    for name, (scheme, label_map) in SCHEMES.items():
        # 1. all SKOS concepts in this scheme
        concepts = set(g.subjects(SKOS.inScheme, scheme))
        wd_in_scheme = {c for c in concepts if str(c).startswith(str(WD))}
        leaves       = concepts - wd_in_scheme  # local ex:.../ nodes

        # 2. leaves with / without a Wikidata exactMatch
        linked_leaves = {l for l in leaves
                         if any(g.objects(l, SKOS.exactMatch))}
        orphan_leaves = leaves - linked_leaves

        # 3. anchors  vs.  ancestors-only Wikidata nodes
        anchored_wd: set[URIRef] = set()
        for l in linked_leaves:
            for o in g.objects(l, SKOS.exactMatch):
                if isinstance(o, URIRef) and str(o).startswith(str(WD)):
                    anchored_wd.add(o)
        ancestor_only_wd = wd_in_scheme - anchored_wd

        # 4. hierarchy edge counts (only edges *within* the scheme)
        broader_edges  = sum(1 for s, _, o in g.triples((None, SKOS.broader, None))
                             if s in concepts and o in concepts)
        subclass_edges = sum(1 for s, _, o in g.triples((None, RDFS.subClassOf, None))
                             if s in concepts and o in concepts)

        # 5. ancestor examples (label them via skos:prefLabel)
        examples: list[tuple[str, str]] = []
        for n in list(ancestor_only_wd)[:show_examples]:
            lab = next(g.objects(n, SKOS.prefLabel), None)
            qid = str(n).rsplit("/", 1)[-1]
            examples.append((qid, str(lab) if lab else qid))

        scheme_report = {
            "leaves_total":       len(leaves),
            "leaves_linked":      len(linked_leaves),
            "leaves_orphan":      len(orphan_leaves),
            "wd_anchors":         len(anchored_wd),
            "wd_ancestors_only":  len(ancestor_only_wd),
            "broader_edges":      broader_edges,
            "subclass_edges":     subclass_edges,
            "ancestor_examples":  examples,
        }

        # 6. raw resolver coverage (if the label map was provided)
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
    "INSTRUMENT_SCHEME", "GENRE_SCHEME", "DECADE_SCHEME",
    "resolve_label", "resolve_labels",
    "fetch_subclass_chain", "fetch_subclass_chains",
    "fetch_qid_metadata",
    "_attach_qid_metadata",
    "enrich_graph_with_wikidata",
    "audit_wikidata_enrichment",
)
