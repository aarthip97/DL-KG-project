"""
src/data/kg
===========
KG construction + Wikidata enrichment for the MusicRecSys project.

The public surface stays compact - everything else (HTTP plumbing,
SPARQL builders, threading) is private to its own module.
"""
from .tempo_classes import (  # noqa: F401
    TEMPO_CLASSES,
    classify_tempo,
    add_tempo_class_column,
    tempo_class_table,
)
from .variable_selection import (  # noqa: F401
    DEFAULT_KG_COLUMNS,
    KG_RENAME_MAP,
    INTERIM_KG_FEATURES,
    select_kg_columns,
    load_music_features,
    merge_parquet_with_interim,
)
from .kg_builder import (  # noqa: F401
    MRC, MO, FOAF, EVENT, DCT,
    EX,
    TRACK_NS, ARTIST_NS, USER_NS, GENRE_NS, INSTRUMENT_NS,
    DECADE_NS, TEMPO_NS, KEY_NS, MODE_NS, PERFORMANCE_NS,
    KGBuilder,
)
from .user_data import (  # noqa: F401
    restrict_taste_profile_to_kg,
    load_or_build_kg_taste_profile,
)
from .wikidata_mapping import (  # noqa: F401
    WD, WDT,
    INSTRUMENT_ROOT, GENRE_ROOT,
    resolve_label, resolve_labels,
    fetch_direct_parents, build_parent_graph,
    fetch_subclass_chain, fetch_subclass_chains,
    fetch_qid_metadata,
    enrich_graph_with_wikidata,
    audit_wikidata_enrichment,
)
from .decades import (  # noqa: F401
    decade_for_year, decade_label, unique_decades_from_dataframe,
    resolve_decades, add_decades_to_graph,
    collect_decade_qids_for_metadata,
    WD_DECADE_TYPE,
)
from .listening import (  # noqa: F401
    user_uri,
    add_listening_schema,
    add_users_to_graph,
    stream_users_to_ntriples,
    ensure_listening_sidecar,
    merge_sidecar_into_graph,
)
