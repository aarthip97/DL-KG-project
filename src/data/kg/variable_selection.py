"""
Variable selection + interim-features merge for KG construction.

Two responsibilities
--------------------
1. Pick a *small, KG-relevant* subset of columns from the processed
   parquet (``data/processed/lakh_msd_dataset.parquet``).  The full
   parquet has ~40 columns and we don't want every one of them as a
   datatype property — most acoustic noise stays in the parquet.

2. Optionally enrich each row with a *small* subset of jSymbolic
   features extracted by jSymbolic (``data/interim/interim.csv``,
   ~1496 columns).  We join on the MIDI md5 hash because:

      * the parquet stores ``midi_path`` as ``…/<md5>.mid``
      * the interim CSV's first column is a Windows path
        ``…\\midi_dir\\<md5>.mid``  (extracted on a different machine)

   ``Path(p).stem`` gives ``<md5>`` from both → reliable join key.
"""
from __future__ import annotations

import pathlib
import re
from typing import Iterable, Literal, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Parquet column selection
# ─────────────────────────────────────────────────────────────────────────────
# Keep only what the KG actually needs as nodes / properties / labels.
# Everything else stays in the parquet for downstream ML notebooks.
DEFAULT_KG_COLUMNS: tuple[str, ...] = (
    # identifiers / linkage
    "track_id",
    "midi_path",
    "song_id",
    "artist_id",     # MSD canonical artist key (used as the artist URI in the KG)
    "artist_mbid",   # MusicBrainz cross-reference
    # textual / display
    "artist_name",
    "title",
    "release",
    # tonality (we keep only the *named* key/mode — the integer codes
    # `key`/`mode` are dropped on purpose)
    "key_name", "key_confidence",
    "mode_name", "mode_confidence",
    # rhythm / scalar audio features
    "time_signature", "time_signature_confidence",
    "loudness",
    # year for biographical / temporal axis
    "year",
    # genre / tags  (artist_terms_weight pairs 1-to-1 with artist_terms and
    # carries the MSD-published weight per term; used for weighted
    # artist→genre edges in the rich KG variant)
    "primary_genre", "top3_genres",
    "artist_terms", "artist_terms_weight",
    # MIDI instrumentation
    "midi_n_instruments", "midi_instrument_names",
)


# Friendlier names used downstream (KG, notebooks).  Applied in
# :func:`select_kg_columns` after the column projection.
KG_RENAME_MAP: dict[str, str] = {
    "release":   "album_name",
    "key_name":  "key",
    "mode_name": "mode",
}


def select_kg_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_KG_COLUMNS,
    require_id: bool = True,
    rename: bool = True,
) -> pd.DataFrame:
    """
    Return ``df`` projected onto ``columns`` (silently skipping missing ones).

    Parameters
    ----------
    df : the full ``lakh_msd_dataset`` DataFrame.
    columns : iterable of column names to keep (defaults to
              :data:`DEFAULT_KG_COLUMNS`).
    require_id : if True, raise unless ``track_id`` survives the projection.
    rename : if True (default), apply :data:`KG_RENAME_MAP` so callers see
             ``album_name`` / ``key`` / ``mode`` instead of ``release`` /
             ``key_name`` / ``mode_name``.
    """
    keep = [c for c in columns if c in df.columns]
    out  = df[keep].copy()
    if require_id and "track_id" not in out.columns:
        raise KeyError(
            "track_id missing from selection — KG construction needs it as the "
            "primary linkage to mrc:MSDTrack instances."
        )
    if rename:
        out = out.rename(columns={k: v for k, v in KG_RENAME_MAP.items()
                                  if k in out.columns})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  jSymbolic interim features
# ─────────────────────────────────────────────────────────────────────────────
# A small, *interpretable* subset of the 1496 jSymbolic columns that map
# cleanly to ontology properties (mrc:hasRhythmFeature / mo:tempo / etc.)
# Extend this list as the ontology grows.
INTERIM_KG_FEATURES: tuple[str, ...] = (
    # rhythmic
    "Initial_Tempo", "Mean_Tempo", "Tempo_Variability",
    "Note_Density", "Note_Density_Variability",
    "Average_Note_Duration", "Variability_of_Note_Durations",
    "Amount_of_Staccato",
    "Rhythmic_Variability", "Polyrhythms",
    # melodic
    "Mean_Melodic_Interval",
    "Amount_of_Arpeggiation",
    "Repeated_Notes", "Chromatic_Motion", "Stepwise_Motion",
    "Average_Length_of_Melodic_Arcs",
    "Average_Interval_Spanned_by_Melodic_Arcs",
    "Melodic_Pitch_Variety",
    # texture / dynamics
    "Average_Number_of_Independent_Voices",
    "Variability_of_Number_of_Independent_Voices",
    "Voice_Equality_-_Number_of_Notes",
    "Variation_of_Dynamics",
    "Average_Note_to_Note_Change_in_Dynamics",
    # instrumentation prevalence (a few representative ones)
    "Acoustic_Guitar_Prevalence",
    "Electric_Guitar_Prevalence",
    "Brass_Prevalence",
    "Woodwinds_Prevalence",
    "String_Keyboard_Prevalence",
    # chordal
    "Chord_Duration",
    "Standard_Triads",
    "Dominant_Seventh_Chords",
)


_MD5_RE = re.compile(r"([0-9a-f]{32})", re.IGNORECASE)


def _md5_from_path(path: Optional[str]) -> Optional[str]:
    """
    Extract the 32-char hex MIDI md5 from a path string.

    Works for both POSIX (``…/<md5>.mid``) and Windows
    (``…\\midi_dir\\<md5>.mid``) layouts.
    """
    if path is None or not isinstance(path, str):
        return None
    # Try filename stem first (cheap, robust to any separator).
    stem = pathlib.PurePath(path.replace("\\", "/")).stem
    m = _MD5_RE.fullmatch(stem) or _MD5_RE.search(stem)
    if m:
        return m.group(1).lower()
    # Fallback: search the whole path.
    m = _MD5_RE.search(path)
    return m.group(1).lower() if m else None


def load_interim_features(
    interim_csv: str | pathlib.Path,
    feature_columns: Iterable[str] = INTERIM_KG_FEATURES,
    path_col: str = "Unnamed: 0",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load only the path column + a small feature subset from the interim CSV
    and add a clean ``midi_md5`` join key.

    Returns a DataFrame indexed by ``midi_md5`` (str, lowercase, 32 hex
    chars), with one column per requested feature.  Rows without a parsable
    md5 are dropped.
    """
    interim_csv = pathlib.Path(interim_csv)
    requested = [path_col, *feature_columns]
    df = pd.read_csv(interim_csv, usecols=lambda c: c in requested)

    # Some columns may be missing in older dumps — be permissive.
    missing = [c for c in feature_columns if c not in df.columns]
    if missing and verbose:
        print(f"[INFO] {len(missing)} requested interim features missing: "
              f"{missing[:5]}{'…' if len(missing) > 5 else ''}")

    df["midi_md5"] = df[path_col].map(_md5_from_path)
    n_bad = int(df["midi_md5"].isna().sum())
    if n_bad and verbose:
        print(f"[WARN] {n_bad} interim rows had no parseable MIDI md5 — dropped.")
    df = df.dropna(subset=["midi_md5"]).drop(columns=[path_col])
    df = df.set_index("midi_md5")

    if verbose:
        print(f"Loaded interim features: {df.shape[0]} rows × {df.shape[1]} cols.")
    return df


def merge_parquet_with_interim(
    parquet_df: pd.DataFrame,
    interim_df: pd.DataFrame,
    midi_path_col: str = "midi_path",
    how: Literal["left", "right", "outer", "inner"] = "left",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Left-join ``parquet_df`` (keyed by MIDI md5 derived from
    ``midi_path``) with the indexed ``interim_df`` returned by
    :func:`load_interim_features`.

    The resulting frame keeps every parquet row; interim columns are
    ``NaN`` when no jSymbolic features were extracted for that file.
    """
    if midi_path_col not in parquet_df.columns:
        raise KeyError(f"Parquet has no {midi_path_col!r} column to derive md5 from.")

    p = parquet_df.copy()
    p["midi_md5"] = p[midi_path_col].map(_md5_from_path)

    n_unmapped = int(p["midi_md5"].isna().sum())
    if n_unmapped and verbose:
        print(f"[WARN] {n_unmapped} parquet rows had no parseable MIDI md5.")

    merged = p.merge(
        interim_df,
        left_on="midi_md5",
        right_index=True,
        how=how,
    )
    if verbose:
        n_with_features = int(merged["Mean_Tempo"].notna().sum()) \
            if "Mean_Tempo" in merged.columns else 0
        print(f"Merged: {len(merged):,} rows total, "
              f"{n_with_features:,} with interim features "
              f"({100*n_with_features/max(len(merged),1):.1f}%).")
    return merged


__all__ = (
    "DEFAULT_KG_COLUMNS",
    "INTERIM_KG_FEATURES",
    "select_kg_columns",
    "load_interim_features",
    "merge_parquet_with_interim",
)
