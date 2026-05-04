"""
src/data/kg
===========
Helpers for populating the MusicRecSys knowledge graph from the
processed dataset (`lakh_msd_dataset.parquet`) and the jSymbolic
feature dump (`data/interim/interim.csv`).

Public API
----------
    from data.kg import (
        # tempo classes (Music Theory Academy categorical tempo markings)
        TEMPO_CLASSES, classify_tempo, add_tempo_class_column,

        # variable selection + interim merge
        DEFAULT_KG_COLUMNS, select_kg_columns,
        load_interim_features, merge_parquet_with_interim,

        # KG construction
        MRC, MO, FOAF, KGBuilder,
    )
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
    MRC,
    MO,
    FOAF,
    KGBuilder,
)
from .user_data import (  # noqa: F401
    restrict_taste_profile_to_kg,
    load_or_build_kg_taste_profile,
)
from .wikidata_mapping import (  # noqa: F401
    WD,
    INSTRUMENT_ROOT,
    GENRE_ROOT,
    resolve_labels,
    fetch_subclass_chains,
    enrich_graph_with_wikidata,
)
