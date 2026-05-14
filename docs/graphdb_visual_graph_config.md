# GraphDB Visual Graph Configuration
## `music_recsys` repository — `localhost:7200`

> **How to use this file**  
> Open `http://localhost:7200/graphs-visualizations/config/save?repositoryId=music_recsys`,
> paste each query into the matching field, save the config under a memorable name
> (e.g. *"MRC — full overview"*), and share the config name with colleagues so
> everyone loads the same view.

---

## Namespace prefixes (used in every query below)

```sparql
PREFIX mrc:    <http://purl.org/ontology/mrc/>
PREFIX mo:     <http://purl.org/ontology/mo/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf:   <http://xmlns.com/foaf/0.1/>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX wdt:    <http://www.wikidata.org/prop/direct/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX track:  <http://purl.org/ontology/mrc/resource/track/>
PREFIX artist: <http://purl.org/ontology/mrc/resource/artist/>
PREFIX genre:  <http://purl.org/ontology/mrc/resource/genre/>
PREFIX inst:   <http://purl.org/ontology/mrc/resource/instrument/>
PREFIX decade: <http://purl.org/ontology/mrc/resource/decade/>
PREFIX key:    <http://purl.org/ontology/mrc/resource/key/>
PREFIX mode:   <http://purl.org/ontology/mrc/resource/mode/>
```

---

## 1 · Start Graph Query (CONSTRUCT)

> **Field:** *"Start with graph query results"*  
> This seeds the canvas on load.

### Option A — T-Box overview (class + property skeleton)

Shows every class and the object-properties that connect them — useful for
understanding the schema without any data flooding the canvas.

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>

CONSTRUCT {
    ?cls  a              owl:Class .
    ?cls  rdfs:label     ?clsLabel .
    ?cls  rdfs:subClassOf ?parent .
    ?prop a              owl:ObjectProperty .
    ?prop rdfs:domain    ?domain .
    ?prop rdfs:range     ?range .
    ?prop rdfs:label     ?propLabel .
}
WHERE {
    { ?cls a owl:Class . OPTIONAL { ?cls rdfs:label ?clsLabel } }
    UNION
    { ?cls rdfs:subClassOf ?parent . FILTER(isIRI(?parent)) }
    UNION
    {
        ?prop a owl:ObjectProperty .
        OPTIONAL { ?prop rdfs:domain ?domain }
        OPTIONAL { ?prop rdfs:range  ?range  }
        OPTIONAL { ?prop rdfs:label  ?propLabel }
    }
}
```

### Option B — Mixed T-Box + A-Box sampler (recommended for first look)

Shows the full class hierarchy **plus** 3 example tracks with their artist,
genre, key, mode, and decade.  Gives an immediate feel for real data
alongside the schema.

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>

CONSTRUCT {
    # ── T-Box ────────────────────────────────────────────────
    ?cls  a owl:Class ; rdfs:label ?clsLabel .
    ?cls  rdfs:subClassOf ?parent .
    ?prop a owl:ObjectProperty ;
          rdfs:domain ?domain ; rdfs:range ?range ; rdfs:label ?propLabel .

    # ── A-Box sample ─────────────────────────────────────────
    ?track  a mrc:MSDTrack ; dct:title ?title .
    ?track  mrc:hasGenre  ?genre .
    ?genre  skos:prefLabel ?genreLabel .
    ?artist foaf:name     ?artistName .
    ?perf   a mo:Performance ;
            mrc:hasTrack  ?track ;
            mo:performer  ?artist ;
            mrc:hasKey    ?key ;
            mrc:hasMode   ?mode ;
            mrc:hasTempo  ?tempo .
    ?key    skos:prefLabel ?keyLabel .
    ?mode   skos:prefLabel ?modeLabel .
    ?decade a mrc:Decade ; skos:prefLabel ?decadeLabel .
    ?track  mrc:releasedInDecade ?decade .
}
WHERE {
    # T-Box
    { ?cls a owl:Class . OPTIONAL { ?cls rdfs:label ?clsLabel } }
    UNION { ?cls rdfs:subClassOf ?parent . FILTER(isIRI(?parent)) }
    UNION {
        ?prop a owl:ObjectProperty .
        OPTIONAL { ?prop rdfs:domain ?domain }
        OPTIONAL { ?prop rdfs:range  ?range  }
        OPTIONAL { ?prop rdfs:label  ?propLabel }
    }
    # A-Box sample — 3 tracks
    UNION {
        {
            SELECT ?track WHERE { ?track a mrc:MSDTrack } LIMIT 3
        }
        ?track dct:title ?title .
        OPTIONAL { ?track mrc:hasGenre ?genre . ?genre skos:prefLabel ?genreLabel }
        OPTIONAL {
            ?perf mrc:hasTrack ?track ;
                  mo:performer ?artist .
            ?artist foaf:name ?artistName .
            OPTIONAL { ?perf mrc:hasKey  ?key  . ?key  skos:prefLabel ?keyLabel  }
            OPTIONAL { ?perf mrc:hasMode ?mode . ?mode skos:prefLabel ?modeLabel }
            OPTIONAL { ?perf mo:tempo    ?tempo }
        }
        OPTIONAL {
            ?track mrc:releasedInDecade ?decade .
            ?decade skos:prefLabel ?decadeLabel
        }
    }
}
```

### Option C — Genre + instrument scheme hierarchy only

Good for inspecting the SKOS taxonomy.

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {
    ?concept a skos:Concept ;
             skos:prefLabel ?label ;
             skos:broader   ?broader .
    ?broader skos:prefLabel ?broaderLabel .
}
WHERE {
    ?concept skos:inScheme ?scheme .
    FILTER(?scheme IN (mrc:GenreScheme, mrc:InstrumentScheme))
    OPTIONAL { ?concept skos:prefLabel ?label . FILTER(LANG(?label) = "en") }
    OPTIONAL {
        ?concept skos:broader ?broader .
        OPTIONAL { ?broader skos:prefLabel ?broaderLabel . FILTER(LANG(?broaderLabel) = "en") }
    }
}
LIMIT 150
```

---

## 2 · Node Expansion Query (CONSTRUCT)

> **Field:** *"This is a CONSTRUCT query that determines which nodes and edges are
> added to the graph when the user expands an existing node."*  
> The variable `?node` is replaced with the IRI of the clicked node at runtime.

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>

CONSTRUCT {
    ?node ?outPred ?outObj .
    ?outObj rdfs:label ?outObjLabel .
    ?outObj skos:prefLabel ?outObjPrefLabel .
    ?inSubj ?inPred ?node .
    ?inSubj rdfs:label ?inSubjLabel .
    ?inSubj foaf:name  ?inSubjName .
    ?inSubj dct:title  ?inSubjTitle .
}
WHERE {
    # Outgoing edges (what this node points TO)
    {
        ?node ?outPred ?outObj .
        FILTER(isIRI(?outObj))
        OPTIONAL { ?outObj rdfs:label     ?outObjLabel      }
        OPTIONAL { ?outObj skos:prefLabel ?outObjPrefLabel  }
    }
    UNION
    # Incoming edges (what points TO this node) — limit to avoid star explosion
    {
        ?inSubj ?inPred ?node .
        FILTER(isIRI(?inSubj))
        OPTIONAL { ?inSubj rdfs:label ?inSubjLabel }
        OPTIONAL { ?inSubj foaf:name  ?inSubjName  }
        OPTIONAL { ?inSubj dct:title  ?inSubjTitle }
    }
}
LIMIT 30
```

---

## 3 · Node Basic Info Query (SELECT)

> **Field:** *"This SELECT query determines the basic information about a node
> (type, label, comment, rank)."*

Colour rules:
| `?type` local name | colour (auto-assigned by GraphDB) |
|---|---|
| `MSDTrack` | one colour |
| `Performance` | another |
| `MusicArtist` | another |
| `Listener` | another |
| `Concept` (genre/key/mode) | another |
| `Decade` | another |

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?type ?label ?comment ?rank
WHERE {
    BIND(<?>  AS ?node)          # replaced at runtime

    # Primary type (first one found wins — order matters)
    OPTIONAL { ?node a mrc:MSDTrack      . BIND(mrc:MSDTrack      AS ?t1) }
    OPTIONAL { ?node a mo:Performance    . BIND(mo:Performance     AS ?t2) }
    OPTIONAL { ?node a mo:MusicArtist    . BIND(mo:MusicArtist     AS ?t3) }
    OPTIONAL { ?node a mrc:Listener      . BIND(mrc:Listener       AS ?t4) }
    OPTIONAL { ?node a skos:Concept      . BIND(skos:Concept       AS ?t5) }
    OPTIONAL { ?node a mrc:Decade        . BIND(mrc:Decade         AS ?t6) }
    OPTIONAL { ?node a owl:Class         . BIND(owl:Class          AS ?t7) }
    BIND(COALESCE(?t1,?t2,?t3,?t4,?t5,?t6,?t7) AS ?type)

    # Best label (priority: title > name > prefLabel > rdfs:label > local IRI name)
    OPTIONAL { ?node dct:title       ?l1 }
    OPTIONAL { ?node foaf:name       ?l2 }
    OPTIONAL { ?node skos:prefLabel  ?l3 . FILTER(LANG(?l3) = "en") }
    OPTIONAL { ?node rdfs:label      ?l4 . FILTER(LANG(?l4) = "" || LANG(?l4) = "en") }
    BIND(COALESCE(?l1,?l2,?l3,?l4) AS ?label)

    # Comment / description
    OPTIONAL { ?node skos:definition ?c1 . FILTER(LANG(?c1) = "en") }
    OPTIONAL { ?node rdfs:comment    ?c2 . FILTER(LANG(?c2) = "en") }
    BIND(COALESCE(?c1,?c2) AS ?comment)

    # Rank — make popular tracks larger (normalised 0–1)
    OPTIONAL {
        SELECT ?node (IF(COUNT(DISTINCT ?u) / 50.0 > 1.0, 1.0, COUNT(DISTINCT ?u) / 50.0) AS ?listenRank) WHERE {
            ?u mrc:hasListeningInteraction ?ev .
            ?ev mrc:onTrack ?node .
        } GROUP BY ?node
    }
    BIND(COALESCE(?listenRank, 0.3) AS ?rank)
}
```

---

## 4 · Edge Label Query (SELECT)

> **Field:** *"This SELECT query determines the basic information about an edge."*

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?label
WHERE {
    BIND(<?>  AS ?edge)          # replaced at runtime by the edge IRI

    OPTIONAL { ?edge rdfs:label     ?l1 . FILTER(LANG(?l1) = "en") }
    OPTIONAL { ?edge skos:prefLabel ?l2 . FILTER(LANG(?l2) = "en") }
    BIND(COALESCE(?l1, ?l2) AS ?label)
}
```

---

## 5 · Node Extra Properties Query (SELECT)

> **Field:** *"This SELECT query determines the extra properties shown for a node
> when the info icon is clicked."*

```sparql
PREFIX mrc:  <http://purl.org/ontology/mrc/>
PREFIX mo:   <http://purl.org/ontology/mo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT ?property ?value
WHERE {
    BIND(<?>  AS ?node)          # replaced at runtime

    # All literal properties on the node
    ?node ?p ?v .
    FILTER(isLiteral(?v))
    BIND(REPLACE(STR(?p), "^.*[/#]", "") AS ?property)
    BIND(STR(?v) AS ?value)

    # Exclude noisy internals
    FILTER(?p NOT IN (
        <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
    ))
}
ORDER BY ?property
LIMIT 40
```

---

## Quick-reference: saved config names

| Config name | Start query | Best for |
|---|---|---|
| `MRC — T-Box only` | Option A | Explaining the schema to new collaborators |
| `MRC — mixed overview` | Option B | General exploration with real data |
| `MRC — genre/instrument schemes` | Option C | Inspecting SKOS taxonomies |

---

## Reproducing the same config on a new machine

1. Clone the repo and start GraphDB (`~/graphdb-server/bin/graphdb`).
2. Create the `music_recsys` repository (notebook **04**, cell 23 with `USE_GRAPHDB=True`).
3. Load the KG (notebook **04** runs `client.upload_rdf(KG_TTL)`).
4. Open `http://localhost:7200/graphs-visualizations/config/save?repositoryId=music_recsys`.
5. Paste each query from this file into the matching field.
6. Save with the config name shown in the table above.

> The visual graph configs are stored server-side in
> `~/graphdb-server/data/repositories/music_recsys/` and are **not** committed
> to git. This file is the single source of truth for re-creating them.
