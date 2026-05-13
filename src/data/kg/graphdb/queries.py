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
    ?g      skos:prefLabel ?genreLabel .
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
    ?g      skos:prefLabel ?genreLabel .
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
               dct:title  ?title ;
               mrc:hasKey  ?k ;
               mrc:hasMode ?m .
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
def query_key_mode_simple(track_uri: str) -> str:
    return _q("""
SELECT ?title ?keyLabel ?modeLabel
WHERE {{
    BIND(<{track_uri}> AS ?t)
    ?t  dct:title   ?title ;
        mrc:hasKey  ?k ;
        mrc:hasMode ?m .
    OPTIONAL {{ ?k skos:prefLabel ?keyLabel  . FILTER(LANG(?keyLabel)  = "en") }}
    OPTIONAL {{ ?m skos:prefLabel ?modeLabel . FILTER(LANG(?modeLabel) = "en") }}
}}
""").format(track_uri=track_uri)


def query_key_mode_rich(track_uri: str) -> str:
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
    OPTIONAL {{ ?k skos:prefLabel ?keyLabel  . FILTER(LANG(?keyLabel)  = "en") }}
    OPTIONAL {{ ?m skos:prefLabel ?modeLabel . FILTER(LANG(?modeLabel) = "en") }}
}}
""").format(track_uri=track_uri)


def query_artist_genre_weights(artist_uri: str) -> str:
    return _q("""
SELECT ?artistName ?genreLabel ?weight
WHERE {{
    BIND(<{artist_uri}> AS ?artist)
    ?artist foaf:name         ?artistName ;
            mrc:hasGenreAssoc ?assoc .
    ?assoc  mrc:genre         ?g ;
            mrc:weight        ?weight .
    ?g      skos:prefLabel    ?genreLabel .
    FILTER(LANG(?genreLabel) = "en")
}}
ORDER BY DESC(?weight)
""").format(artist_uri=artist_uri)


# ── Confidence-filtered key statistics ─────────────────────────────────
QUERY_CONFIDENT_KEYS_SIMPLE = _q("""
SELECT ?keyLabel (COUNT(*) AS ?n)
WHERE {
    ?t a mrc:MSDTrack ;
       mrc:hasKey  ?k .
    ?k skos:prefLabel ?keyLabel .
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
    ?k skos:prefLabel ?keyLabel .
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
    ?leaf skos:inScheme mrc:GenreScheme ;
          rdfs:label    ?leafLabel .
    FILTER(LANG(?leafLabel) = "en")
    ?leaf    skos:broader+  ?child .
    ?child   skos:broader   ?parent .
    OPTIONAL { ?child  rdfs:label ?childLabel  . FILTER(LANG(?childLabel)  = "en") }
    OPTIONAL { ?parent rdfs:label ?parentLabel . FILTER(LANG(?parentLabel) = "en") }
}
ORDER BY ?leafLabel ?childLabel
LIMIT 200
""")

QUERY_INSTRUMENT_HIERARCHY = _q("""
SELECT DISTINCT ?parentLabel ?childLabel ?leafLabel
WHERE {
    ?leaf skos:inScheme mrc:InstrumentScheme ;
          rdfs:label    ?leafLabel .
    FILTER(LANG(?leafLabel) = "en")
    ?leaf   skos:broader+  ?child .
    ?child  skos:broader   ?parent .
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
            skos:prefLabel ?decadeLabel .
    FILTER(LANG(?decadeLabel) = "en")
    OPTIONAL {
        ?decade skos:broader ?century .
        OPTIONAL { ?century rdfs:label ?centuryLabel . FILTER(LANG(?centuryLabel) = "en") }
    }
    OPTIONAL {
        ?decade wdt:P155 ?prev .
        OPTIONAL { ?prev skos:prefLabel ?prevLabel . FILTER(LANG(?prevLabel) = "en") }
    }
    OPTIONAL {
        ?decade wdt:P156 ?next .
        OPTIONAL { ?next skos:prefLabel ?nextLabel . FILTER(LANG(?nextLabel) = "en") }
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


# ── Triple export for PyKEEN ───────────────────────────────────────────
# Drops owl:sameAs links (they're the high-degree noise that wrecks
# negative sampling) and any reflexive (?h ?r ?h) edges.
QUERY_PYKEEN_TRIPLES = _q("""
SELECT ?h ?r ?t
WHERE {
    ?h ?r ?t .
    FILTER(?r != owl:sameAs)
    FILTER(?h != ?t)
    FILTER(isIRI(?h) && isIRI(?t))
}
""")


# ── Entity / relation enumeration (used to build the node dict) ────────
QUERY_ALL_ENTITIES = _q("""
SELECT DISTINCT ?e ?type WHERE {
    { ?e a ?type . FILTER(isIRI(?e)) }
}
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


# ── Repo-wide health stats (cheap, run them all on every export) ───────
QUERY_TRIPLE_COUNT = _q("""
SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }
""")

QUERY_NODE_TYPE_HISTOGRAM = _q("""
SELECT ?type (COUNT(DISTINCT ?e) AS ?n)
WHERE { ?e a ?type . FILTER(isIRI(?e)) }
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
