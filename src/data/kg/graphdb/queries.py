"""Centralised SPARQL queries used by the KG export / EDA pipeline.

These were lifted out of `notebooks/04_DL_pipeline.ipynb` so the same
constants can be used both interactively (against an in-memory rdflib graph
via `data.eda.sparql_query`) and against a live GraphDB endpoint via
:class:`graphdb.GraphDBClient`.

Conventions
-----------
* Every query is **prefixed with** :data:`_SPARQL_PREFIXES`.
* Queries that need a live URI are exposed as **callables**
  (``f(track_uri=...)``) so the substitution is explicit instead of a
  positional hack.
* :data:`STATS_QUERIES` maps a short name → query string and is consumed by
  :func:`graphdb.repo.KGRepo.export_all` to produce one CSV per query.
"""

from __future__ import annotations

from typing import Dict


# ── Shared prefix block ────────────────────────────────────────────────
# .lstrip() is required: a leading newline before the first PREFIX makes
# rdflib's parser raise ParseException("found 'PREFIX' at char 1").
_SPARQL_PREFIXES = """
PREFIX mrc:    <http://purl.org/ontology/mrc/>
PREFIX mo:     <http://purl.org/ontology/mo/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf:   <http://xmlns.com/foaf/0.1/>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX wdt:    <http://www.wikidata.org/prop/direct/>
PREFIX wd:     <http://www.wikidata.org/entity/>
PREFIX scheme: <http://purl.org/ontology/mrc/scheme/>
PREFIX track:  <http://purl.org/ontology/mrc/resource/track/>
PREFIX artist: <http://purl.org/ontology/mrc/resource/artist/>
PREFIX user:   <http://purl.org/ontology/mrc/resource/user/>
PREFIX genre:  <http://purl.org/ontology/mrc/resource/genre/>
PREFIX inst:   <http://purl.org/ontology/mrc/resource/instrument/>
PREFIX decade: <http://purl.org/ontology/mrc/resource/decade/>
PREFIX tempo:  <http://purl.org/ontology/mrc/resource/tempo/>
PREFIX key:    <http://purl.org/ontology/mrc/resource/key/>
PREFIX mode:   <http://purl.org/ontology/mrc/resource/mode/>
PREFIX perf:   <http://purl.org/ontology/mrc/resource/performance/>
""".lstrip()


def _q(body: str) -> str:
    """Glue the shared prefix block in front of a query body."""
    return _SPARQL_PREFIXES + body


# ── Genre statistics ───────────────────────────────────────────────────
QUERY_GENRES_SIMPLE = _q("""
SELECT ?genreLabel (COUNT(DISTINCT ?artist) AS ?n_artists)
WHERE {
    ?artist mrc:hasGenre ?g .
    ?g      rdfs:label ?genreLabel .
    FILTER(LANG(?genreLabel) = "en")
}
GROUP BY ?genreLabel
ORDER BY DESC(?n_artists)
LIMIT 30
""")

QUERY_GENRES_RICH = _q("""
SELECT ?genreLabel
       (COUNT(DISTINCT ?artist)        AS ?n_artists)
       (ROUND(SUM(?w) * 1000) / 1000   AS ?total_weight)
       (ROUND(AVG(?w) * 1000) / 1000   AS ?avg_weight)
WHERE {
    ?artist mrc:hasGenreAssoc ?assoc .
    ?assoc  mrc:genre  ?g ;
            mrc:weight ?w .
    ?g      rdfs:label ?genreLabel .
    FILTER(LANG(?genreLabel) = "en")
}
GROUP BY ?genreLabel
ORDER BY DESC(?total_weight)
LIMIT 30
""")


# ── Discovery probes (return one URI to plug into the parametric queries)
QUERY_FIND_TRACK_SIMPLE = _q("""
SELECT ?title_URI ?title WHERE {
    ?title_URI a mrc:MSDTrack ;
               dct:title ?title .
} LIMIT 1
""")

QUERY_FIND_TRACK_RICH = _q("""
SELECT ?title_URI ?title WHERE {
    ?title_URI  a              mrc:MSDTrack ;
                dct:title      ?title .
    ?perf       mrc:hasTrack   ?title_URI ;
                mrc:hasKey     ?k ;
                mrc:hasMode    ?m .
} LIMIT 1
""")

QUERY_FIND_ARTIST_WITH_WEIGHTS = _q("""
SELECT ?artist_URI ?name WHERE {
    ?artist_URI foaf:name         ?name ;
                mrc:hasGenreAssoc ?assoc .
    ?assoc      mrc:weight        ?w .
} LIMIT 1
""")


# ── Parametric queries (inject a live URI before running) ──────────────
def query_key_mode_simple(track_uri: str | None) -> str:
    if track_uri is None:
        return _q("SELECT ?title ?keyLabel ?modeLabel WHERE { FILTER(false) }")
    return _q("""
SELECT ?title ?keyLabel ?modeLabel
WHERE {{
    BIND(<{track_uri}> AS ?t)
    ?t  dct:title   ?title ;
        mrc:hasKey  ?k ;
        mrc:hasMode ?m .
    OPTIONAL {{ ?k rdfs:label ?keyLabel  . FILTER(LANG(?keyLabel)  = "en") }}
    OPTIONAL {{ ?m rdfs:label ?modeLabel . FILTER(LANG(?modeLabel) = "en") }}
}}
""").format(track_uri=track_uri)


def query_key_mode_rich(track_uri: str | None) -> str:
    if track_uri is None:
        return _q("SELECT ?title ?keyLabel ?modeLabel ?keyConf ?modeConf ?tempo WHERE { FILTER(false) }")
    return _q("""
SELECT ?title ?keyLabel ?modeLabel ?keyConf ?modeConf ?tempo
WHERE {{
    BIND(<{track_uri}> AS ?t)
    ?t    dct:title      ?title .
    ?perf mrc:hasTrack   ?t ;
          mrc:hasKey     ?k ;
          mrc:hasMode    ?m .
    OPTIONAL {{ ?perf mo:tempo           ?tempo    }}
    OPTIONAL {{ ?perf mrc:keyConfidence  ?keyConf  }}
    OPTIONAL {{ ?perf mrc:modeConfidence ?modeConf }}
    OPTIONAL {{ ?k rdfs:label ?keyLabel  . FILTER(LANG(?keyLabel)  = "en") }}
    OPTIONAL {{ ?m rdfs:label ?modeLabel . FILTER(LANG(?modeLabel) = "en") }}
}}
""").format(track_uri=track_uri)


def query_artist_genre_weights(artist_uri: str | None) -> str:
    if artist_uri is None:
        return _q("SELECT ?artistName ?genreLabel ?weight WHERE { FILTER(false) }")
    return _q("""
SELECT DISTINCT ?artistName ?genreLabel ?weight
WHERE {{
    BIND(<{artist_uri}> AS ?artist)
    ?artist foaf:name         ?artistName ;
            mrc:hasGenreAssoc ?assoc .
    ?assoc  mrc:genre         ?g ;
            mrc:weight        ?weight .
    ?g      rdfs:label        ?genreLabel .
    FILTER(LANG(?genreLabel) = "en")
}}
ORDER BY DESC(?weight)
""").format(artist_uri=artist_uri)


# ── Confidence-filtered key statistics ─────────────────────────────────
QUERY_CONFIDENT_KEYS_SIMPLE = _q("""
SELECT ?keyLabel (COUNT(*) AS ?n)
WHERE {
    ?t a mrc:MSDTrack ;
       mrc:hasKey ?k .
    ?k rdfs:label ?keyLabel .
    FILTER(LANG(?keyLabel) = "en")
}
GROUP BY ?keyLabel
ORDER BY DESC(?n)
LIMIT 24
""")

QUERY_CONFIDENT_KEYS_RICH = _q("""
SELECT ?keyLabel (COUNT(*) AS ?n_high_confidence)
WHERE {
    ?perf a mo:Performance ;
          mrc:hasKey          ?k ;
          mrc:keyConfidence   ?kc .
    FILTER(?kc >= 0.8)
    ?k rdfs:label ?keyLabel .
    FILTER(LANG(?keyLabel) = "en")
}
GROUP BY ?keyLabel
ORDER BY DESC(?n_high_confidence)
LIMIT 24
""")


# ── Hierarchy traversal ────────────────────────────────────────────────
QUERY_GENRE_HIERARCHY = _q("""
SELECT DISTINCT ?parentLabel ?childLabel ?leafLabel
WHERE {
    ?leaf rdf:type mrc:Genre ;
          rdfs:label ?leafLabel .
    FILTER(LANG(?leafLabel) = "en")
    ?leaf    rdfs:subClassOf+  ?child .
    ?child   rdfs:subClassOf   ?parent .
    OPTIONAL { ?child  rdfs:label ?childLabel  . FILTER(LANG(?childLabel)  = "en") }
    OPTIONAL { ?parent rdfs:label ?parentLabel . FILTER(LANG(?parentLabel) = "en") }
}
ORDER BY ?leafLabel ?childLabel
LIMIT 200
""")

QUERY_INSTRUMENT_HIERARCHY = _q("""
SELECT DISTINCT ?parentLabel ?childLabel ?leafLabel
WHERE {
    ?leaf rdf:type mo:Instrument ;
          rdfs:label ?leafLabel .
    FILTER(LANG(?leafLabel) = "en")
    ?leaf   rdfs:subClassOf+  ?child .
    ?child  rdfs:subClassOf   ?parent .
    OPTIONAL { ?child  rdfs:label ?childLabel  . FILTER(LANG(?childLabel)  = "en") }
    OPTIONAL { ?parent rdfs:label ?parentLabel . FILTER(LANG(?parentLabel) = "en") }
}
ORDER BY ?leafLabel ?childLabel
LIMIT 200
""")

QUERY_DECADE_CHAIN = _q("""
SELECT ?prevLabel ?decadeLabel ?nextLabel ?centuryLabel
WHERE {
    ?decade a mrc:Decade ;
            rdfs:label ?decadeLabel .
    FILTER(LANG(?decadeLabel) = "en")
    OPTIONAL {
        ?decade rdfs:subClassOf ?century .
        OPTIONAL { ?century rdfs:label ?centuryLabel . FILTER(LANG(?centuryLabel) = "en") }
    }
    OPTIONAL {
        ?decade wdt:P155 ?prev .
        OPTIONAL { ?prev rdfs:label ?prevLabel . FILTER(LANG(?prevLabel) = "en") }
    }
    OPTIONAL {
        ?decade wdt:P156 ?next .
        OPTIONAL { ?next rdfs:label ?nextLabel . FILTER(LANG(?nextLabel) = "en") }
    }
}
ORDER BY ?decadeLabel
""")


# ── User listening behaviour ───────────────────────────────────────────
QUERY_USER_SONG_LOG = _q("""
SELECT ?userName ?trackTitle ?artistName ?listenCount
WHERE {
    {
        SELECT ?u WHERE {
            ?u a mrc:Listener ;
               mrc:hasListeningInteraction ?ev .
        } LIMIT 1
    }
    ?u foaf:name   ?userName ;
       mrc:hasListeningInteraction ?ev .
    ?ev mrc:onTrack     ?track ;
        mrc:listenCount ?listenCount .
    ?track dct:title    ?trackTitle .
    OPTIONAL {
        ?perf mrc:hasTrack  ?track ;
              mo:performer  ?artist .
        ?artist foaf:name   ?artistName .
    }
}
ORDER BY DESC(?listenCount)
""")

QUERY_TOP5_POPULAR_SONGS = _q("""
SELECT ?trackTitle ?artistName
       (COUNT(DISTINCT ?u) AS ?n_listeners)
       (ROUND(AVG(?cnt) * 100) / 100 AS ?avg_listens)
       (SUM(?cnt) AS ?total_listens)
WHERE {
    ?u  mrc:hasListeningInteraction ?ev .
    ?ev mrc:onTrack     ?track ;
        mrc:listenCount ?cnt .
    ?track dct:title    ?trackTitle .
    OPTIONAL {
        ?perf mrc:hasTrack ?track ;
              mo:performer ?artist .
        ?artist foaf:name  ?artistName .
    }
}
GROUP BY ?trackTitle ?artistName
ORDER BY DESC(?n_listeners)
LIMIT 50
""")


# ── Equivalence pairs (used to merge nodes before PyKEEN export) ──────
# Fetches owl:sameAs (instance-level) and owl:equivalentClass (class-level)
# pairs so the exporter can build a canonical-URI map via union-find.
# Run with infer=False — we want only the explicit assertions; we do the
# transitive closure ourselves so the canonical choice is deterministic.
QUERY_EQUIV_PAIRS = _q("""
SELECT ?a ?b WHERE {
    { ?a owl:sameAs ?b }
    UNION
    { ?a owl:equivalentClass ?b }
    FILTER(isIRI(?a) && isIRI(?b) && ?a != ?b)
}
""")


# ── Triple export for PyKEEN ───────────────────────────────────────────
# Run with infer=True so RDFS+ class memberships are included.
# owl:sameAs nodes are MERGED (not just dropped) by the exporter's
# union-find canonicalisation step; the predicate is excluded here only
# to avoid writing redundant self-loop triples after the merge.
# Other filters applied:
#   • reflexive edges     — no self-loops (also catches post-merge ones)
#   • non-IRI nodes       — no blank nodes from OWL restriction machinery
#   • OWL machinery preds — created by reasoner, have no embedding meaning
QUERY_PYKEEN_TRIPLES = _q("""
SELECT ?h ?r ?t
WHERE {
    ?h ?r ?t .
    FILTER(?h != ?t)
    FILTER(isIRI(?h) && isIRI(?r) && isIRI(?t))
    FILTER(?r NOT IN (
        owl:sameAs,
        owl:onProperty, owl:someValuesFrom, owl:allValuesFrom,
        owl:hasValue, owl:onClass, owl:onDataRange,
        owl:complementOf, owl:intersectionOf, owl:unionOf,
        rdf:first, rdf:rest
    ))
}
""")


# ── Entity / relation enumeration (used to build the node dict) ────────
# OWL/RDFS meta-types are excluded — they are schema vocabulary, not data
# nodes, and would create noise embedding dimensions.
_OWL_META_TYPES = """(
    owl:Class, owl:ObjectProperty, owl:DatatypeProperty,
    owl:AnnotationProperty, owl:NamedIndividual, owl:Ontology,
    owl:TransitiveProperty, owl:SymmetricProperty, owl:FunctionalProperty,
    owl:InverseFunctionalProperty, owl:Restriction,
    rdfs:Datatype, rdfs:Class, rdf:Property,
    foaf:Agent
)"""

QUERY_ALL_ENTITIES = _q(f"""
SELECT DISTINCT ?e ?type WHERE {{
    ?e a ?type .
    FILTER(isIRI(?e) && isIRI(?type))
    FILTER(?type NOT IN {_OWL_META_TYPES})
}}
ORDER BY ?type ?e
""")

QUERY_ALL_RELATIONS = _q("""
SELECT DISTINCT ?r (COUNT(*) AS ?n) WHERE {
    ?h ?r ?t .
    FILTER(isIRI(?r) && isIRI(?h) && isIRI(?t))
    FILTER(?r != owl:sameAs)
}
GROUP BY ?r
ORDER BY DESC(?n)
""")


# ── Edge-list queries for the PyG edge_dict (build_rich_hetero_graph) ──────
# These return canonicalisable IRI pairs (+ optional weight/count) per edge
# type, and TRAVERSE the bnode-mediated structures (listening events, genre
# associations) that the IRI-only PyKEEN TSV cannot express — so the edge_dict
# is complete on both the simple and the rich graph. Each query is intentionally
# tolerant (UNION + OPTIONAL) so the same SPARQL works against either variant:
# listening as ``mrc:listenedTo`` (simple) or via ``hasListeningInteraction``
# (rich, with ``listenCount``); genre as ``mrc:hasGenre`` (simple) or via
# ``hasGenreAssoc`` (rich, with ``weight``).
# Consumed by exports.export_edge_dict; column ORDER is the contract
# (subject, object[, weight/count]) — the exporter reads positionally.

QUERY_EDGE_USER_TRACK = _q("""
SELECT ?user ?track ?count WHERE {
    { ?user mrc:hasListeningInteraction ?ev .
      ?ev   mrc:onTrack ?track .
      OPTIONAL { ?ev mrc:listenCount ?count } }
    UNION
    { ?user mrc:listenedTo ?track . }
}
""")

QUERY_EDGE_TRACK_ARTIST = _q("""
SELECT DISTINCT ?track ?artist WHERE {
    ?perf mrc:hasTrack  ?track ;
          mo:performer  ?artist .
}
""")

QUERY_EDGE_TRACK_KEY = _q("""
SELECT DISTINCT ?track ?key WHERE {
    { ?track mrc:hasKey ?key .
      FILTER(STRSTARTS(STR(?track), "http://purl.org/ontology/mrc/resource/track/")) }
    UNION
    { ?perf mrc:hasTrack ?track ; mrc:hasKey ?key . }
}
""")

QUERY_EDGE_TRACK_MODE = _q("""
SELECT DISTINCT ?track ?mode WHERE {
    { ?track mrc:hasMode ?mode .
      FILTER(STRSTARTS(STR(?track), "http://purl.org/ontology/mrc/resource/track/")) }
    UNION
    { ?perf mrc:hasTrack ?track ; mrc:hasMode ?mode . }
}
""")

QUERY_EDGE_TRACK_TEMPO = _q("""
SELECT DISTINCT ?track ?tempo WHERE {
    ?perf mrc:hasTrack       ?track ;
          mrc:hasTempoClass  ?tempo .
}
""")

QUERY_EDGE_TRACK_INSTRUMENT = _q("""
SELECT DISTINCT ?track ?inst WHERE {
    ?perf mrc:hasTrack    ?track ;
          mo:instrument   ?inst .
}
""")

QUERY_EDGE_TRACK_DECADE = _q("""
SELECT DISTINCT ?track ?decade WHERE {
    ?track mrc:inDecade ?decade .
}
""")

QUERY_EDGE_ARTIST_GENRE = _q("""
SELECT DISTINCT ?artist ?genre ?weight WHERE {
    { ?artist mrc:hasGenre ?genre . }
    UNION
    { ?artist mrc:hasGenreAssoc ?assoc .
      ?assoc  mrc:genre  ?genre .
      OPTIONAL { ?assoc mrc:weight ?weight } }
}
""")

QUERY_EDGE_GENRE_PARENT = _q("""
SELECT DISTINCT ?child ?parent WHERE {
    ?child  owl:sameAs ?wd_child .
    ?parent owl:sameAs ?wd_parent .
    ?wd_child (rdfs:subClassOf)+ ?wd_parent .
    FILTER(STRSTARTS(STR(?child),  "http://purl.org/ontology/mrc/resource/genre/"))
    FILTER(STRSTARTS(STR(?parent), "http://purl.org/ontology/mrc/resource/genre/"))
    FILTER(?child != ?parent)
}
""")

QUERY_EDGE_INSTRUMENT_PARENT = _q("""
SELECT DISTINCT ?child ?parent WHERE {
    ?child  owl:sameAs ?wd_child .
    ?parent owl:sameAs ?wd_parent .
    ?wd_child (rdfs:subClassOf)+ ?wd_parent .
    FILTER(STRSTARTS(STR(?child),  "http://purl.org/ontology/mrc/resource/instrument/"))
    FILTER(STRSTARTS(STR(?parent), "http://purl.org/ontology/mrc/resource/instrument/"))
    FILTER(?child != ?parent)
}
""")


# ── Repo-wide health stats (cheap, run them all on every export) ───────
QUERY_TRIPLE_COUNT = _q("""
SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }
""")

QUERY_NODE_TYPE_HISTOGRAM = _q(f"""
SELECT ?type (COUNT(DISTINCT ?e) AS ?n)
WHERE {{
    ?e a ?type .
    FILTER(isIRI(?e) && isIRI(?type))
    FILTER(?type NOT IN {_OWL_META_TYPES})
    FILTER(!STRSTARTS(STR(?type), "http://www.wikidata.org/"))
}}
GROUP BY ?type
ORDER BY DESC(?n)
""")


# ── Named bundle the export driver iterates over ───────────────────────
# Each entry → one CSV in cfg.stats_dir / "<name>.csv"
STATS_QUERIES: Dict[str, str] = {
    "triple_count":           QUERY_TRIPLE_COUNT,
    "node_type_histogram":    QUERY_NODE_TYPE_HISTOGRAM,
    "relation_histogram":     QUERY_ALL_RELATIONS,
    "genres_simple":          QUERY_GENRES_SIMPLE,
    "genres_rich":            QUERY_GENRES_RICH,
    "confident_keys_simple":  QUERY_CONFIDENT_KEYS_SIMPLE,
    "confident_keys_rich":    QUERY_CONFIDENT_KEYS_RICH,
    "genre_hierarchy":        QUERY_GENRE_HIERARCHY,
    "instrument_hierarchy":   QUERY_INSTRUMENT_HIERARCHY,
    "decade_chain":           QUERY_DECADE_CHAIN,
    "top5_popular_songs":     QUERY_TOP5_POPULAR_SONGS,
}
