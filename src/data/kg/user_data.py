"""
Restrict the Echo Nest Taste Profile to the songs that survived the KG
inner-join, then re-apply the cold-start min-plays filter.

Why
---
``00b_user_data_integration`` produced ``taste_profile_filtered.parquet``
with 7,227 songs and 299,156 users (each user has ≥ 5 interactions).
However, only 7,113 of those songs have a matched MIDI file *and* a
successful jSymbolic feature extraction — the rest are dropped by the KG
inner-join.

If we just drop the orphaned interactions, some users may fall below the
≥ 5-plays threshold.  This module re-filters those out so the user-level
distribution we care about (≥ 5 plays *of KG-known songs*) holds again.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd


def restrict_taste_profile_to_kg(
    taste: pd.DataFrame,
    kg_song_ids: set[str] | pd.Series,
    min_plays_per_user: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only interactions whose ``song_id`` appears in ``kg_song_ids``,
    then drop users with fewer than ``min_plays_per_user`` interactions
    *after* that restriction.

    Parameters
    ----------
    taste : DataFrame with columns ``user_id``, ``song_id``, ``play_count``.
    kg_song_ids : iterable / set / Series of song IDs that survived the
        KG inner-join (i.e. have both MSD metadata and jSymbolic features).
    min_plays_per_user : minimum number of (KG-restricted) interactions a
        user must have to be kept.  Default 5 mirrors notebook 00b.
    verbose : print before/after counts.

    Returns
    -------
    pd.DataFrame with the same schema as *taste*, indexed 0..n-1.
    """
    required = {"user_id", "song_id", "play_count"}
    missing = required - set(taste.columns)
    if missing:
        raise KeyError(f"taste profile is missing required columns: {missing}")

    if isinstance(kg_song_ids, pd.Series):
        kg_song_ids = set(kg_song_ids.dropna().astype(str).unique())
    else:
        kg_song_ids = set(kg_song_ids)

    n0       = len(taste)
    n_users0 = taste["user_id"].nunique()
    n_songs0 = taste["song_id"].nunique()

    # ── Step 1: restrict to KG songs ───────────────────────────────────────
    df = taste[taste["song_id"].isin(kg_song_ids)].copy()
    n1       = len(df)
    n_users1 = df["user_id"].nunique()
    n_songs1 = df["song_id"].nunique()
    if verbose:
        print(f"[user_data] KG-song restriction: "
              f"{n0:,} → {n1:,} rows  (−{n0 - n1:,})  |  "
              f"users {n_users0:,} → {n_users1:,}  |  "
              f"songs {n_songs0:,} → {n_songs1:,}")

    # ── Step 2: re-apply cold-start min-plays filter ────────────────────────
    if min_plays_per_user > 1:
        plays_per_user = df.groupby("user_id", observed=True).size()
        keep_users     = plays_per_user[plays_per_user >= min_plays_per_user].index
        df             = df[df["user_id"].isin(keep_users)].reset_index(drop=True)
        n2       = len(df)
        n_users2 = df["user_id"].nunique()
        n_songs2 = df["song_id"].nunique()
        if verbose:
            print(f"[user_data] cold-start (min={min_plays_per_user}): "
                  f"{n1:,} → {n2:,} rows  (−{n1 - n2:,}, "
                  f"−{n_users1 - n_users2:,} users)")
            print(f"[user_data] Final:  {n2:,} interactions  |  "
                  f"{n_users2:,} users  |  {n_songs2:,} songs")
    else:
        df = df.reset_index(drop=True)

    return df


def load_or_build_kg_taste_profile(
    cache_path: str | pathlib.Path,
    taste_source: str | pathlib.Path,
    kg_song_ids: Optional[set[str] | pd.Series] = None,
    min_plays_per_user: int = 5,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Cached wrapper around :func:`restrict_taste_profile_to_kg`.

    If ``cache_path`` exists and ``force_rebuild`` is False, just load it.
    Otherwise read ``taste_source`` (a parquet of pre-filtered triplets),
    restrict to ``kg_song_ids``, re-apply the min-plays filter, and write
    the result to ``cache_path``.
    """
    cache_path  = pathlib.Path(cache_path)
    taste_source = pathlib.Path(taste_source)

    if cache_path.exists() and not force_rebuild:
        if verbose:
            print(f"[user_data] loading cache from {cache_path}")
        return pd.read_parquet(cache_path)

    if kg_song_ids is None:
        raise ValueError(
            "kg_song_ids is required when building the cache from scratch."
        )

    if verbose:
        print(f"[user_data] building cache from {taste_source}")
    taste = pd.read_parquet(taste_source)
    out   = restrict_taste_profile_to_kg(
        taste, kg_song_ids,
        min_plays_per_user=min_plays_per_user,
        verbose=verbose,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    if verbose:
        print(f"[user_data] cached → {cache_path} "
              f"({cache_path.stat().st_size/1024:,.1f} KiB)")
    return out
