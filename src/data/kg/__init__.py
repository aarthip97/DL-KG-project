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
)
from .variable_selection import (  # noqa: F401
    DEFAULT_KG_COLUMNS,
    KG_RENAME_MAP,
    INTERIM_KG_FEATURES,
    select_kg_columns,
    load_interim_features,
    merge_parquet_with_interim,
)
from .kg_builder import (  # noqa: F401
    MRC, MO, FOAF, EVENT, DCT, EX,
    INSTRUMENT_SCHEME_URI, GENRE_SCHEME_URI, DECADE_SCHEME_URI,
    KGBuilder,
)
from .user_data import (  # noqa: F401
    restrict_taste_profile_to_kg,
    load_or_build_kg_taste_profile,
)
from .wikidata_mapping import (  # noqa: F401
    WD, WDT,
    INSTRUMENT_ROOT, GENRE_ROOT,
    INSTRUMENT_SCHEME, GENRE_SCHEME, DECADE_SCHEME,
    resolve_label, resolve_labels,
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
)
