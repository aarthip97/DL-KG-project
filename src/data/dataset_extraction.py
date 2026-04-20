"""
src/data/dataset_extraction.py
================================
Helpers for building the aligned MIDI × MSD dataset.

Responsibilities
----------------
1. read_msd_metadata()     — reads one MSD .h5 file via h5py (no pytables needed)
2. LakhMSDLinker           — discovers all matched tracks, filters by DTW match
                             score, deduplicates, and returns a clean DataFrame
                             ready for feature extraction.

Expected directory layout (Lakh Matched Dataset)
-------------------------------------------------
lmd_matched/
  <X>/<Y>/<Z>/<TRXXXXX>/   ← one sub-dir per MSD track
      <md5hash>.mid          ← one or more MIDI candidates
lmd_matched_h5/
  <X>/<Y>/<Z>/<TRXXXXX>.h5  ← MSD summary features for the track

match_scores.json   ← {track_id: {md5_hash: float}}  (DTW alignment score)
                       lower = better fit (range ≈ 0.5 – 1.3)
"""

from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── MSD field mappings ────────────────────────────────────────────────────────
KEY_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MODE_NAMES = {0: 'minor', 1: 'major'}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Low-level HDF5 reader
# ─────────────────────────────────────────────────────────────────────────────

def read_msd_metadata(h5_path: str | pathlib.Path) -> dict:
    """
    Read **all** available metadata from a single MSD HDF5 summary file.

    Uses h5py directly (avoids the pytables / Python 3.10+ incompatibility).
    Covers every field exposed by the official MSongsDB getter functions
    (https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py).

    Parameters
    ----------
    h5_path : path to a ``<TRXXXXX>.h5`` file

    Returns
    -------
    dict — scalars are Python native types; array fields are summarised:

    Scalars (metadata group)
        artist_name, title, release, artist_id, song_id,
        artist_mbid, artist_playmeid, artist_7digitalid,
        release_7digitalid, track_7digitalid,
        artist_familiarity, artist_hotttnesss, song_hotttnesss,
        artist_latitude, artist_longitude, artist_location

    Scalars (analysis group)
        track_id, analysis_sample_rate, audio_md5,
        msd_key, key_name, key_confidence,
        msd_mode, mode_name, mode_confidence,
        msd_tempo, msd_time_sig, time_sig_confidence,
        msd_duration, msd_loudness, end_of_fade_in, start_of_fade_out,
        msd_danceability, msd_energy

    Scalars (musicbrainz group)
        year

    Array fields — stored as lists / numpy arrays
        artist_terms, artist_terms_freq, artist_terms_weight
        similar_artists
        mbtags, mbtags_count
        mean_chroma        (12-d mean pitch-class vector from segments_pitches)
        mean_timbre        (12-d mean timbre vector from segments_timbre)
        beats_count        (number of detected beats)
        bars_count         (number of detected bars)
        sections_count     (number of detected sections)
        tatums_count       (number of detected tatums)

    Derived
        primary_genre, top3_genres
    """
    def _str(val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode('utf-8', errors='replace').strip() or None
        return str(val).strip() or None

    def _arr_str(arr) -> list[str]:
        if arr is None:
            return []
        return [s for v in arr if (s := _str(v))]

    def _scalar(dataset: Any, row: int, cast: Any = float) -> Any:
        """Safely read a scalar field; return None on any error."""
        try:
            v = dataset[row]
            if cast is str:
                return _str(v)
            result = cast(v)
            # treat sentinel zeros as None for year
            if cast is int and result == 0 and str(getattr(dataset, 'name', '')).endswith('year'):
                return None
            return result
        except Exception:
            return None

    def _arr_slice(f, path: str, idx_path: str, row: int,
                   nrows: int) -> Optional[np.ndarray]:
        """
        Return the variable-length sub-array for *row* using the standard
        MSongsDB index pattern (idx_<field>[row] : idx_<field>[row+1]).
        """
        try:
            data = f[path]
            idx  = f[idx_path]
            start = int(idx[row])
            end   = int(idx[row + 1]) if row + 1 < nrows else len(data)
            return np.array(data[start:end])
        except Exception:
            return None

    with h5py.File(h5_path, 'r') as f:
        try:
            meta_songs  = f['metadata']['songs']
            anal_songs  = f['analysis']['songs']
            mb_songs    = f['musicbrainz']['songs']
        except KeyError:
            return {}

        row   = 0
        nrows = int(meta_songs.shape[0])

        # ── metadata.songs scalars ────────────────────────────────────────────
        artist_name         = _scalar(meta_songs['artist_name'],        row, str)
        title               = _scalar(meta_songs['title'],              row, str)
        release             = _scalar(meta_songs['release'],            row, str)
        artist_id           = _scalar(meta_songs['artist_id'],          row, str)
        song_id             = _scalar(meta_songs['song_id'],            row, str)
        artist_mbid         = _scalar(meta_songs['artist_mbid'],        row, str)
        artist_playmeid     = _scalar(meta_songs['artist_playmeid'],    row, int)
        artist_7digitalid   = _scalar(meta_songs['artist_7digitalid'],  row, int)
        release_7digitalid  = _scalar(meta_songs['release_7digitalid'], row, int)
        track_7digitalid    = _scalar(meta_songs['track_7digitalid'],   row, int)
        artist_familiarity  = _scalar(meta_songs['artist_familiarity'], row, float)
        artist_hotttnesss   = _scalar(meta_songs['artist_hotttnesss'],  row, float)
        song_hotttnesss     = _scalar(meta_songs['song_hotttnesss'],    row, float)
        artist_latitude     = _scalar(meta_songs['artist_latitude'],    row, float)
        artist_longitude    = _scalar(meta_songs['artist_longitude'],   row, float)
        artist_location     = _scalar(meta_songs['artist_location'],    row, str)

        # ── analysis.songs scalars ────────────────────────────────────────────
        track_id            = _scalar(anal_songs['track_id'],              row, str)
        analysis_sample_rate= _scalar(anal_songs['analysis_sample_rate'],  row, int)
        audio_md5           = _scalar(anal_songs['audio_md5'],             row, str)
        raw_key             = _scalar(anal_songs['key'],                   row, int)
        key_conf            = _scalar(anal_songs['key_confidence'],        row, float)
        raw_mode            = _scalar(anal_songs['mode'],                  row, int)
        mode_conf           = _scalar(anal_songs['mode_confidence'],       row, float)
        tempo               = _scalar(anal_songs['tempo'],                 row, float)
        time_sig            = _scalar(anal_songs['time_signature'],        row, int)
        time_sig_conf       = _scalar(anal_songs['time_signature_confidence'], row, float)
        duration            = _scalar(anal_songs['duration'],              row, float)
        loudness            = _scalar(anal_songs['loudness'],              row, float)
        end_fade_in         = _scalar(anal_songs['end_of_fade_in'],        row, float)
        start_fade_out      = _scalar(anal_songs['start_of_fade_out'],     row, float)
        danceability        = _scalar(anal_songs['danceability'],          row, float)
        energy              = _scalar(anal_songs['energy'],                row, float)

        key_name  = KEY_NAMES[int(raw_key or 0) % 12]
        mode_name = MODE_NAMES.get(int(raw_mode or 0), 'unknown')

        # ── musicbrainz.songs scalars ─────────────────────────────────────────
        raw_year = _scalar(mb_songs['year'], row, int)
        year     = raw_year if raw_year and raw_year != 0 else None

        # ── variable-length arrays (metadata) ─────────────────────────────────
        anal_nrows = int(anal_songs.shape[0])
        mb_nrows   = int(mb_songs.shape[0])

        try:
            terms        = _arr_str(f['metadata']['artist_terms'][:])
            terms_freq   = [float(v) for v in f['metadata']['artist_terms_freq'][:]]
            terms_weight = [float(v) for v in f['metadata']['artist_terms_weight'][:]]
        except Exception:
            terms, terms_freq, terms_weight = [], [], []

        try:
            similar_artists = _arr_str(
                _arr_slice(f, 'metadata/similar_artists',
                           'metadata/songs/idx_similar_artists', row, nrows)
            )
        except Exception:
            similar_artists = []

        # ── variable-length arrays (musicbrainz) ──────────────────────────────
        try:
            mbtags = _arr_str(
                _arr_slice(f, 'musicbrainz/artist_mbtags',
                           'musicbrainz/songs/idx_artist_mbtags', row, mb_nrows)
            )
            mbtags_count = [
                int(v) for v in (
                    _arr_slice(f, 'musicbrainz/artist_mbtags_count',
                               'musicbrainz/songs/idx_artist_mbtags', row, mb_nrows)
                    or []
                )
            ]
        except Exception:
            mbtags, mbtags_count = [], []

        # ── variable-length arrays (analysis) — aggregated ───────────────────
        #  segments_pitches  → mean chroma vector (12-d)
        try:
            pitches = _arr_slice(f, 'analysis/segments_pitches',
                                 'analysis/songs/idx_segments_pitches', row, anal_nrows)
            mean_chroma = pitches.mean(axis=0).tolist() if pitches is not None and pitches.ndim == 2 else None
        except Exception:
            mean_chroma = None

        #  segments_timbre   → mean timbre vector (12-d)
        try:
            timbre = _arr_slice(f, 'analysis/segments_timbre',
                                'analysis/songs/idx_segments_timbre', row, anal_nrows)
            mean_timbre = timbre.mean(axis=0).tolist() if timbre is not None and timbre.ndim == 2 else None
        except Exception:
            mean_timbre = None

        # structural counts
        def _count(path, idx_path):
            arr = _arr_slice(f, path, idx_path, row, anal_nrows)
            return int(len(arr)) if arr is not None else None

        beats_count    = _count('analysis/beats_start',    'analysis/songs/idx_beats_start')
        bars_count     = _count('analysis/bars_start',     'analysis/songs/idx_bars_start')
        sections_count = _count('analysis/sections_start', 'analysis/songs/idx_sections_start')
        tatums_count   = _count('analysis/tatums_start',   'analysis/songs/idx_tatums_start')

    # ── derived fields ────────────────────────────────────────────────────────
    primary_genre = None
    top3_genres   = []
    if terms and terms_freq:
        paired        = sorted(zip(terms_freq, terms), reverse=True)
        primary_genre = paired[0][1] if paired else None
        top3_genres   = [t for _, t in paired[:3]]

    return {
        # ── identifiers ───────────────────────────────────────────────────────
        'track_id':           track_id,
        'song_id':            song_id, # ECHO NEST song_id to be used when linking the track to user data 
        'artist_id':          artist_id,
        'artist_mbid':        artist_mbid,
        'artist_playmeid':    artist_playmeid,
        'artist_7digitalid':  artist_7digitalid,
        'release_7digitalid': release_7digitalid,
        'track_7digitalid':   track_7digitalid,
        'audio_md5':          audio_md5,
        # ── textual metadata ──────────────────────────────────────────────────
        'artist_name':     artist_name,
        'title':           title,
        'release':         release,
        'artist_location': artist_location,
        'artist_latitude': artist_latitude,
        'artist_longitude':artist_longitude,
        # ── popularity / social ───────────────────────────────────────────────
        'artist_familiarity': artist_familiarity,
        'artist_hotttnesss':  artist_hotttnesss,
        'song_hotttnesss':    song_hotttnesss,
        # ── tonal ─────────────────────────────────────────────────────────────
        'msd_key':          raw_key,
        'key_name':         key_name,
        'key_confidence':   key_conf,
        'msd_mode':         raw_mode,
        'mode_name':        mode_name,
        'mode_confidence':  mode_conf,
        # ── rhythm / time ─────────────────────────────────────────────────────
        'msd_tempo':            tempo,
        'msd_time_sig':         time_sig,
        'time_sig_confidence':  time_sig_conf,
        'msd_duration':         duration,
        'end_of_fade_in':       end_fade_in,
        'start_of_fade_out':    start_fade_out,
        'analysis_sample_rate': analysis_sample_rate,
        # ── audio characteristics ─────────────────────────────────────────────
        'msd_loudness':     loudness,
        'msd_danceability': danceability,
        'msd_energy':       energy,
        # ── temporal ──────────────────────────────────────────────────────────
        'year': year,
        # ── tags / genre ──────────────────────────────────────────────────────
        'artist_terms':        terms,
        'artist_terms_freq':   terms_freq,
        'artist_terms_weight': terms_weight,
        'primary_genre':       primary_genre,
        'top3_genres':         top3_genres,
        'similar_artists':     similar_artists,
        'mbtags':              mbtags,
        'mbtags_count':        mbtags_count,
        # ── aggregated audio analysis ─────────────────────────────────────────
        'mean_chroma':     mean_chroma,   # list[float] len=12
        'mean_timbre':     mean_timbre,   # list[float] len=12
        'beats_count':     beats_count,
        'bars_count':      bars_count,
        'sections_count':  sections_count,
        'tatums_count':    tatums_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Linker class
# ─────────────────────────────────────────────────────────────────────────────

class LakhMSDLinker:
    """
    Discovers, filters, and aligns Lakh MIDI files with MSD HDF5 metadata.

    Parameters
    ----------
    midi_root : str | Path
        Root of ``lmd_matched/`` (one sub-folder per MSD track ID).
    h5_root : str | Path
        Root of ``lmd_matched_h5/`` (one ``.h5`` file per MSD track ID).
    match_scores_path : str | Path | None
        Path to ``match_scores.json``.  If *None*, no score filtering is done.
    min_score : float
        DTW match score threshold.  Tracks with *all* candidates above this
        value are dropped.  Default ``0.70`` keeps ≈ 50 % of the dataset.
        **Lower score = better match.**
    pick_midi : ``'best'`` | ``'all'``
        ``'best'``  → one row per track (lowest score MIDI).
        ``'all'``   → one row per (track, MIDI candidate).
    """

    def __init__(
        self,
        midi_root: str | pathlib.Path,
        h5_root:   str | pathlib.Path,
        match_scores_path: Optional[str | pathlib.Path] = None,
        min_score: float = 0.70,
        pick_midi: str   = 'best',
    ):
        self.midi_root  = pathlib.Path(midi_root)
        self.h5_root    = pathlib.Path(h5_root)
        self.min_score  = min_score
        self.pick_midi  = pick_midi

        # load match scores
        self.match_scores: dict[str, dict[str, float]] = {}
        if match_scores_path is not None:
            with open(match_scores_path) as fh:
                self.match_scores = json.load(fh)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _h5_path(self, track_id: str) -> Optional[pathlib.Path]:
        """Return the HDF5 path for a track_id, or None if it doesn't exist."""
        p = self.h5_root / track_id[2] / track_id[3] / track_id[4] / f'{track_id}.h5'
        return p if p.exists() else None

    def _midi_dir(self, track_id: str) -> Optional[pathlib.Path]:
        """Return the MIDI directory for a track_id, or None if it doesn't exist."""
        d = self.midi_root / track_id[2] / track_id[3] / track_id[4] / track_id
        return d if d.exists() else None

    # ── public API ────────────────────────────────────────────────────────────

    def discover_tracks(
        self,
        max_tracks: Optional[int] = None,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Walk ``lmd_matched/`` and build a list of candidate track dicts.

        Each dict contains::

            {
              'track_id':   str,
              'h5_path':    Path | None,
              'midi_paths': [(midi_path, score), ...]   sorted best-first
            }

        Tracks whose best MIDI candidate exceeds ``self.min_score`` are
        excluded (when ``match_scores`` is loaded).
        """
        records   = []
        track_dirs = sorted(self.midi_root.rglob('TR*'))  # finds all TR* dirs

        if verbose:
            track_dirs = tqdm(track_dirs, desc='Discovering tracks')

        for track_dir in track_dirs:
            if not track_dir.is_dir():
                continue
            track_id = track_dir.name
            if not track_id.startswith('TR'):
                continue

            midi_files = sorted(track_dir.glob('*.mid'))
            if not midi_files:
                continue

            h5_path = self._h5_path(track_id)

            # attach match scores
            scores = self.match_scores.get(track_id, {})
            midi_with_scores = []
            for m in midi_files:
                md5 = m.stem
                sc  = scores.get(md5, float('inf'))
                midi_with_scores.append((m, sc))

            # sort by score ascending (lower = better)
            midi_with_scores.sort(key=lambda x: x[1])

            # filter: skip if even the best candidate is above threshold
            if self.match_scores and midi_with_scores[0][1] > self.min_score:
                continue

            records.append({
                'track_id':   track_id,
                'h5_path':    h5_path,
                'midi_paths': midi_with_scores,
            })

            if max_tracks and len(records) >= max_tracks:
                break

        return records

    def build_dataset(
        self,
        max_tracks: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build a flat, deduplicated DataFrame merging:
        - MIDI file paths (filtered + ranked by match score)
        - MSD HDF5 metadata

        Returns
        -------
        pd.DataFrame with columns:
            track_id, midi_path, match_score,
            artist_name, title, release, artist_id, song_id,
            msd_key, key_name, key_confidence,
            msd_mode, mode_name, mode_confidence,
            msd_tempo, msd_time_sig, msd_duration, msd_loudness,
            msd_danceability, msd_energy, year,
            primary_genre, top3_genres, artist_terms, mbtags
        """
        tracks = self.discover_tracks(max_tracks=max_tracks, verbose=verbose)
        rows   = []
        seen_hashes: set[str] = set()  # deduplicate by MIDI md5

        iter_tracks = tqdm(tracks, desc='Reading HDF5 metadata') if verbose else tracks

        for rec in iter_tracks:
            msd_meta = {}
            if rec['h5_path'] is not None:
                try:
                    msd_meta = read_msd_metadata(rec['h5_path'])
                except Exception as e:
                    if verbose:
                        print(f"[WARN] {rec['track_id']}: HDF5 read error — {e}")

            if self.pick_midi == 'best':
                candidates = rec['midi_paths'][:1]
            else:  # 'all'
                candidates = rec['midi_paths']

            for midi_path, score in candidates:
                md5 = midi_path.stem
                if md5 in seen_hashes:
                    continue  # exact duplicate file
                seen_hashes.add(md5)

                row = {
                    'track_id':    rec['track_id'],
                    'midi_path':   str(midi_path),
                    'match_score': score,
                }
                row.update(msd_meta)
                rows.append(row)

        df = pd.DataFrame(rows)

        # ── deduplication: keep one row per (artist_id, title) combo ─────────
        if 'artist_id' in df.columns and 'title' in df.columns:
            before = len(df)
            df = (
                df
                .sort_values('match_score')
                .drop_duplicates(subset=['artist_id', 'title'], keep='first')
                .reset_index(drop=True)
            )
            if verbose:
                print(f'Dedup: {before} → {len(df)} rows '
                      f'(removed {before - len(df)} duplicates by artist+title)')

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convenience: load a pre-built dataset from Parquet
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(parquet_path: str | pathlib.Path) -> pd.DataFrame:
    """Load the curated dataset Parquet produced by notebook 00."""
    return pd.read_parquet(parquet_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Echo Nest Taste Profile  (user × song interaction data)
# ─────────────────────────────────────────────────────────────────────────────

def load_taste_profile(
    triplets_path: str | pathlib.Path,
    chunksize: int = 500_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the Echo Nest Taste Profile triplets file into a DataFrame.

    The file format is tab-delimited with **no header**::

        <user_id>\\t<song_id>\\t<play_count>

    where ``song_id`` is the Echo Nest song ID (``SOXXXXXXXXXXXXXXXX``),
    matching the ``song_id`` field stored in each MSD HDF5 file.

    Parameters
    ----------
    triplets_path : path to ``train_triplets.txt`` (unzipped).
    chunksize : rows per chunk when reading (reduces peak RAM).
    verbose : print progress information.

    Returns
    -------
    pd.DataFrame with columns:
        ``user_id`` (str), ``song_id`` (str), ``play_count`` (int32)

    Notes
    -----
    The full file contains ~48 M rows (~1.5 GB).  Reading in chunks
    keeps peak memory below ~2 GB.
    """
    triplets_path = pathlib.Path(triplets_path)
    if not triplets_path.exists():
        raise FileNotFoundError(
            f"Triplets file not found: {triplets_path}\n"
            "Run:  python scripts/download_user_data.py"
        )

    if verbose:
        size_mb = triplets_path.stat().st_size / 1e6
        print(f"[load_taste_profile] Reading {triplets_path.name}  ({size_mb:.0f} MB) …")

    chunks = pd.read_csv(
        triplets_path,
        sep="\t",
        header=None,
        names=["user_id", "song_id", "play_count"],
        dtype={"user_id": "string", "song_id": "string", "play_count": "int32"},
        chunksize=chunksize,
        engine="c",
    )

    parts: list[pd.DataFrame] = []
    total = 0
    for chunk in chunks:
        parts.append(chunk)
        total += len(chunk)
        if verbose:
            print(f"\r[load_taste_profile]  {total:,} rows loaded …", end="", flush=True)

    df = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"\r[load_taste_profile]  {len(df):,} rows  ✓                    ")
    return df


def load_sid_mismatches(
    mismatches_path: str | pathlib.Path,
    verbose: bool = True,
) -> set[str]:
    """
    Parse the MSD / Echo Nest mismatch list and return the set of
    **Echo Nest song IDs** that are known to be incorrectly matched.

    File format (one mismatch per line)::

        ERROR: <SOUMNSI12AB0182807 TRMMGKQ128F9325E10> Artist - Title != ...

    The first token inside ``<…>`` is the Echo Nest ``song_id``;
    the second is the MSD ``track_id``.

    Parameters
    ----------
    mismatches_path : path to ``sid_mismatches.txt``.

    Returns
    -------
    set[str] — Echo Nest song IDs to exclude from the taste profile.
    """
    mismatches_path = pathlib.Path(mismatches_path)
    if not mismatches_path.exists():
        raise FileNotFoundError(
            f"Mismatches file not found: {mismatches_path}\n"
            "Run:  python scripts/download_user_data.py"
        )

    bad_song_ids: set[str] = set()
    import re
    pattern = re.compile(r"<(\S+)\s+(\S+)>")

    with open(mismatches_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                bad_song_ids.add(m.group(1))   # Echo Nest song_id

    if verbose:
        print(f"[load_sid_mismatches] {len(bad_song_ids):,} bad song IDs loaded "
              f"from {mismatches_path.name}")
    return bad_song_ids


def filter_taste_profile(
    triplets: pd.DataFrame,
    bad_song_ids: set[str],
    lmd_song_ids: Optional[set[str]] = None,
    min_plays_per_user: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply three sequential filters to the raw taste-profile triplets:

    1. **Mismatch removal** — drop rows whose ``song_id`` appears in
       *bad_song_ids* (from the Echo Nest / MSD mismatch list).
    2. **LMD intersection** — if *lmd_song_ids* is provided, keep only
       rows whose ``song_id`` appears in the Lakh MIDI Dataset
       (i.e. it has a matched MIDI file).
    3. **Cold-start user removal** — drop users who have fewer than
       *min_plays_per_user* interactions after the previous two filters.

    Parameters
    ----------
    triplets : output of :func:`load_taste_profile`.
    bad_song_ids : output of :func:`load_sid_mismatches`.
    lmd_song_ids : set of Echo Nest song IDs present in the LMD-matched
        Parquet (``dataset_df['song_id']``).  Pass ``None`` to skip this
        filter.
    min_plays_per_user : minimum number of interactions a user must have
        after filtering to be kept.  Default ``1`` removes only users with
        zero interactions.

    Returns
    -------
    pd.DataFrame — filtered triplets with the same columns as *triplets*.
    """
    stats: dict[str, int] = {}

    n0 = len(triplets)
    n_users_0  = triplets["user_id"].nunique()
    n_songs_0  = triplets["song_id"].nunique()
    stats["raw_rows"]  = n0
    stats["raw_users"] = n_users_0
    stats["raw_songs"] = n_songs_0

    # ── Step 1: remove mismatched songs ──────────────────────────────────────
    mask_bad = triplets["song_id"].isin(bad_song_ids)
    df = triplets[~mask_bad].copy()
    n1 = len(df)
    stats["after_mismatch_rows"]  = n1
    stats["after_mismatch_users"] = df["user_id"].nunique()
    stats["after_mismatch_songs"] = df["song_id"].nunique()
    stats["removed_by_mismatch"]  = n0 - n1

    if verbose:
        print(f"[filter] Step 1 — mismatch removal:  "
              f"{n0:,} → {n1:,} rows  (−{n0 - n1:,}  |  "
              f"{len(bad_song_ids):,} bad song IDs)")

    # ── Step 2: LMD intersection ──────────────────────────────────────────────
    if lmd_song_ids is not None:
        mask_lmd = df["song_id"].isin(lmd_song_ids)
        df = df[mask_lmd].copy()
        n2 = len(df)
        stats["after_lmd_rows"]  = n2
        stats["after_lmd_users"] = df["user_id"].nunique()
        stats["after_lmd_songs"] = df["song_id"].nunique()
        stats["removed_by_lmd"]  = n1 - n2
        if verbose:
            print(f"[filter] Step 2 — LMD intersection:  "
                  f"{n1:,} → {n2:,} rows  (−{n1 - n2:,}  |  "
                  f"{len(lmd_song_ids):,} LMD song IDs)")
    else:
        stats["after_lmd_rows"]  = n1
        stats["after_lmd_users"] = stats["after_mismatch_users"]
        stats["after_lmd_songs"] = stats["after_mismatch_songs"]
        stats["removed_by_lmd"]  = 0
        if verbose:
            print("[filter] Step 2 — LMD intersection: skipped (lmd_song_ids=None)")

    # ── Step 3: remove cold-start users ──────────────────────────────────────
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_plays_per_user].index
    df = df[df["user_id"].isin(active_users)].copy()
    n3 = len(df)
    stats["after_coldstart_rows"]  = n3
    stats["after_coldstart_users"] = df["user_id"].nunique()
    stats["after_coldstart_songs"] = df["song_id"].nunique()
    stats["removed_by_coldstart"]  = stats["after_lmd_rows"] - n3

    if verbose:
        print(f"[filter] Step 3 — cold-start removal (min={min_plays_per_user}):  "
              f"{stats['after_lmd_rows']:,} → {n3:,} rows  "
              f"(−{stats['removed_by_coldstart']:,} rows,  "
              f"−{stats['after_lmd_users'] - df['user_id'].nunique():,} users)")
        print(f"\n[filter] Final:  {n3:,} interactions  |  "
              f"{df['user_id'].nunique():,} users  |  "
              f"{df['song_id'].nunique():,} songs")

    df.attrs["filter_stats"] = stats
    return df.reset_index(drop=True)


def build_user_song_stats(
    triplets: pd.DataFrame,
    lmd_df: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute descriptive statistics about the (filtered) taste-profile triplets.

    Parameters
    ----------
    triplets : filtered taste-profile DataFrame
        (columns: ``user_id``, ``song_id``, ``play_count``).
    lmd_df : the Lakh-MSD dataset DataFrame (from notebook 00).
        If provided, adds metadata columns (artist, title, genre) to the
        per-song summary.

    Returns
    -------
    dict with keys:
        ``"per_user"``   — DataFrame indexed by user_id
        ``"per_song"``   — DataFrame indexed by song_id
        ``"play_count"`` — summary statistics of the play_count column
    """
    per_user = (
        triplets.groupby("user_id")
        .agg(
            n_songs=("song_id", "nunique"),
            total_plays=("play_count", "sum"),
            mean_plays=("play_count", "mean"),
            max_plays=("play_count", "max"),
        )
        .sort_values("total_plays", ascending=False)
    )

    per_song = (
        triplets.groupby("song_id")
        .agg(
            n_users=("user_id", "nunique"),
            total_plays=("play_count", "sum"),
            mean_plays=("play_count", "mean"),
            max_plays=("play_count", "max"),
        )
        .sort_values("total_plays", ascending=False)
    )

    if lmd_df is not None and "song_id" in lmd_df.columns:
        meta_cols = [c for c in ["song_id", "artist_name", "title", "primary_genre"]
                     if c in lmd_df.columns]
        meta = lmd_df[meta_cols].drop_duplicates(subset="song_id").set_index("song_id")  # type: ignore[call-overload]
        per_song = per_song.join(meta, how="left")

    play_count_stats: pd.DataFrame = triplets["play_count"].describe().rename("play_count").to_frame()

    return {
        "per_user": per_user,
        "per_song": per_song,
        "play_count": play_count_stats,
    }
