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

# Cap on every variable-length list field (artist_terms, similar_artists, mbtags…)
# Keeps the resulting DataFrame compact and avoids 100-element similar_artists rows.
MAX_LIST_ITEMS = 5


def read_msd_metadata(
    h5_path: str | pathlib.Path,
    include_acoustic: bool = False,
) -> dict:
    """
    Read the *useful* metadata from a single MSD HDF5 summary file.

    Field names mirror the official ``hdf5_getters.py`` from the Million Song
    Dataset distribution (Bertin-Mahieux, 2010): the dict key is exactly the
    name returned by the corresponding ``get_<field>`` function, with two
    derived helpers added — ``key_name`` and ``mode_name``.

    Parameters
    ----------
    h5_path : path to a ``<TRXXXXX>.h5`` file
    include_acoustic : bool, default False
        If True, also return the 12-d ``mean_segments_pitches`` (chroma) and
        ``mean_segments_timbre`` vectors plus structural counts
        (``n_beats``, ``n_bars``, ``n_sections``, ``n_tatums``).

    Returns
    -------
    dict — keys (named after the official MSD getters):

    Identifiers
        track_id, song_id, artist_id, artist_mbid, audio_md5
    Textual / location
        artist_name, title, release, artist_location,
        artist_latitude, artist_longitude
    Popularity
        artist_familiarity, artist_hotttnesss, song_hotttnesss
    Tonal / rhythmic scalars
        key, key_name, key_confidence,
        mode, mode_name, mode_confidence,
        tempo, time_signature, time_signature_confidence,
        duration, loudness, danceability, energy
    Temporal
        year
    Tags / genre  (each list capped at MAX_LIST_ITEMS)
        artist_terms, artist_terms_freq, artist_terms_weight,
        primary_genre, top3_genres,
        similar_artists, artist_mbtags, artist_mbtags_count
    (Optional, only if include_acoustic=True)
        mean_segments_pitches (12-d), mean_segments_timbre (12-d),
        n_beats, n_bars, n_sections, n_tatums
    """

    # ── tiny helpers ────────────────────────────────────────────────────────
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

    def _scalar(songs_ds, field: str, row: int, cast=float):
        """Read one scalar from a compound 'songs' dataset (pytables Table)."""
        try:
            v = songs_ds[field][row]
            if cast is str:
                return _str(v)
            return cast(v)
        except Exception:
            return None

    def _arr_slice(f, data_path: str, songs_ds, idx_field: str,
                   row: int, nrows: int) -> Optional[np.ndarray]:
        """
        Variable-length sub-array for ``row`` using the standard ``idx_<field>``
        pattern from the official ``hdf5_getters.py``::

            data[ idx_field[row] : idx_field[row+1] ]

        For the last (or only) song in the file we slice to the end of ``data``.
        ``idx_field`` lives **inside** the compound 'songs' dataset, so it must
        be accessed as ``songs_ds[idx_field]``, *not* as ``f[songs_path/idx_field]``
        (h5py cannot descend into a compound dataset by path).
        """
        try:
            data    = f[data_path]
            idx_col = songs_ds[idx_field]   # 1-D int array, length == nrows
            start   = int(idx_col[row])
            end     = int(idx_col[row + 1]) if row + 1 < nrows else int(data.shape[0])
            if end <= start:
                return np.array([])
            return np.array(data[start:end])
        except Exception:
            return None
        
    # ── normalise: empty lists → None so isnull() / isna() detect them ──────
    def _none_if_empty(v):
        """Return None for empty lists/pd.NA; pass everything else through."""
        if v is pd.NA:
            return None
        if isinstance(v, list) and len(v) == 0:
            return None
        return v

    # ── open file and grab the three 'songs' compound datasets ──────────────
    with h5py.File(h5_path, 'r') as f:
        try:
            meta_songs = f['metadata/songs']      # compound dataset (Table)
            anal_songs = f['analysis/songs']
            mb_songs   = f['musicbrainz/songs']
        except KeyError:
            return {}

        row = 0
        meta_nrows = int(meta_songs.shape[0])
        anal_nrows = int(anal_songs.shape[0])
        mb_nrows   = int(mb_songs.shape[0])

        # ── metadata.songs scalars (names match get_<field>) ────────────────
        artist_name        = _scalar(meta_songs, 'artist_name',        row, str)
        title              = _scalar(meta_songs, 'title',              row, str)
        release            = _scalar(meta_songs, 'release',            row, str)
        artist_id          = _scalar(meta_songs, 'artist_id',          row, str)
        song_id            = _scalar(meta_songs, 'song_id',            row, str)
        artist_mbid        = _scalar(meta_songs, 'artist_mbid',        row, str)
        artist_familiarity = _scalar(meta_songs, 'artist_familiarity', row, float)
        artist_hotttnesss  = _scalar(meta_songs, 'artist_hotttnesss',  row, float)
        song_hotttnesss    = _scalar(meta_songs, 'song_hotttnesss',    row, float)
        
        artist_latitude    = _scalar(meta_songs, 'artist_latitude',    row, float)
        artist_longitude   = _scalar(meta_songs, 'artist_longitude',   row, float)
        artist_location    = _scalar(meta_songs, 'artist_location',    row, str)

        # ── analysis.songs scalars ──────────────────────────────────────────
        track_id                  = _scalar(anal_songs, 'track_id',                  row, str)
        audio_md5                 = _scalar(anal_songs, 'audio_md5',                 row, str)
        key                       = _scalar(anal_songs, 'key',                       row, int)
        key_confidence            = _scalar(anal_songs, 'key_confidence',            row, float)
        mode                      = _scalar(anal_songs, 'mode',                      row, int)
        mode_confidence           = _scalar(anal_songs, 'mode_confidence',           row, float)
        tempo                     = _scalar(anal_songs, 'tempo',                     row, float)
        time_signature            = _scalar(anal_songs, 'time_signature',            row, int)
        time_signature_confidence = _scalar(anal_songs, 'time_signature_confidence', row, float)
        duration                  = _scalar(anal_songs, 'duration',                  row, float)
        loudness                  = _scalar(anal_songs, 'loudness',                  row, float)
        danceability              = _scalar(anal_songs, 'danceability',              row, float)
        energy                    = _scalar(anal_songs, 'energy',                    row, float)

        key_name  = KEY_NAMES[int(key or 0) % 12]
        mode_name = MODE_NAMES.get(int(mode or 0), 'unknown')

        # ── musicbrainz.songs scalars ───────────────────────────────────────
        raw_year = _scalar(mb_songs, 'year', row, int)
        year     = raw_year if raw_year and raw_year != 0 else None

        # ── variable-length: artist_terms / freq / weight (sorted, capped) ──
        terms_arr  = _arr_slice(f, 'metadata/artist_terms',
                                meta_songs, 'idx_artist_terms', row, meta_nrows)
        freq_arr   = _arr_slice(f, 'metadata/artist_terms_freq',
                                meta_songs, 'idx_artist_terms', row, meta_nrows)
        weight_arr = _arr_slice(f, 'metadata/artist_terms_weight',
                                meta_songs, 'idx_artist_terms', row, meta_nrows)

        terms_all = _arr_str(terms_arr)
        freqs   = [float(v) for v in (freq_arr   if freq_arr   is not None else [])]
        weights = [float(v) for v in (weight_arr if weight_arr is not None else [])]

        # pad to common length, then sort by frequency desc and cap
        n = min(len(terms_all), len(freqs), len(weights))
        if n > 0:
            paired = sorted(zip(freqs[:n], weights[:n], terms_all[:n]),
                            reverse=True)[:MAX_LIST_ITEMS]
            artist_terms        = [t for _, _, t in paired]
            artist_terms_freq   = [f for f, _, _ in paired]
            artist_terms_weight = [w for _, w, _ in paired]
        else:
            artist_terms, artist_terms_freq, artist_terms_weight = None, None, None

        # ── variable-length: similar_artists (capped) ───────────────────────
        sim_arr = _arr_slice(f, 'metadata/similar_artists',
                             meta_songs, 'idx_similar_artists', row, meta_nrows)
        # sim_arr is a numpy array — never use bare `if sim_arr` (ambiguous for len > 1)
        _sim_list = _arr_str(sim_arr)[:MAX_LIST_ITEMS] if (sim_arr is not None and len(sim_arr) > 0) else []
        similar_artists = _sim_list if _sim_list else None

        # ── variable-length: artist_mbtags + counts (sorted by count, capped) ─
        mbtags_arr  = _arr_slice(f, 'musicbrainz/artist_mbtags',
                                 mb_songs, 'idx_artist_mbtags', row, mb_nrows)
        mbcount_arr = _arr_slice(f, 'musicbrainz/artist_mbtags_count',
                                 mb_songs, 'idx_artist_mbtags', row, mb_nrows)
        mbtags_all  = _arr_str(mbtags_arr)
        mb_counts = [int(v) for v in (mbcount_arr if (mbcount_arr is not None and len(mbcount_arr) > 0) else [])]
        m = min(len(mbtags_all), len(mb_counts))
        if m > 0:
            mb_paired = sorted(zip(mb_counts[:m], mbtags_all[:m]),
                               reverse=True)[:MAX_LIST_ITEMS]
            artist_mbtags       = [t for _, t in mb_paired]
            artist_mbtags_count = [c for c, _ in mb_paired]
        else:
            artist_mbtags, artist_mbtags_count = None, None

    
    # ── derived: primary genre + top-3 genres from (already capped) terms ──
    primary_genre = artist_terms[0] if artist_terms else None
    top3_genres   = artist_terms[:3] if artist_terms else None

    return {
            # identifiers
            'track_id':    track_id,
            'song_id':     song_id,
            'artist_id':   artist_id,
            'artist_mbid': artist_mbid,
            'audio_md5':   audio_md5,
            # textual / location
            'artist_name':      artist_name,
            'title':            title,
            'release':          release,
            'artist_location':  artist_location,
            'artist_latitude':  artist_latitude,
            'artist_longitude': artist_longitude,
            # popularity
            'artist_familiarity': artist_familiarity,
            'artist_hotttnesss':  artist_hotttnesss,
            'song_hotttnesss':    song_hotttnesss,
            # tonal
            'key':             key,
            'key_name':        key_name,
            'key_confidence':  key_confidence,
            'mode':            mode,
            'mode_name':       mode_name,
            'mode_confidence': mode_confidence,
            # rhythm / time
            'tempo':                     tempo,
            'time_signature':            time_signature,
            'time_signature_confidence': time_signature_confidence,
            'duration':                  duration,
            # audio characteristics
            'loudness':     loudness,
            'danceability': danceability,
            'energy':       energy,
            # temporal
            'year': year,
            # tags / genre — empty list → None so isnull() counts them correctly
            'artist_terms':        _none_if_empty(artist_terms),
            'artist_terms_freq':   _none_if_empty(artist_terms_freq),
            'artist_terms_weight': _none_if_empty(artist_terms_weight),
            'primary_genre':       primary_genre,           # already None or str
            'top3_genres':         _none_if_empty(top3_genres),
            'similar_artists':     _none_if_empty(similar_artists),
            'artist_mbtags':       _none_if_empty(artist_mbtags),
            'artist_mbtags_count': _none_if_empty(artist_mbtags_count),
        }

# ─────────────────────────────────────────────────────────────────────────────
# 1b.  MIDI instrumentation reader (pretty_midi)  —  minimal version
# ─────────────────────────────────────────────────────────────────────────────

# Safety cap on the length of midi_instrument_names. The vast majority of LMD
# files have ≤ 15 distinct instruments; a few outliers go above 20.
MAX_INSTRUMENTS = 15


def read_midi_instrumentation(
    midi_path: str | pathlib.Path,
    max_items: int = MAX_INSTRUMENTS,
) -> dict:
    """
    Extract a minimal instrumentation summary from a MIDI file using
    ``pretty_midi``.

    Identification rules
    --------------------
    A MIDI file contains a list of *instrument tracks* (``pm.instruments``).
    Each track has:

      * ``program`` — General-MIDI program number, 0-127, set by a Program-Change
        message. ``pretty_midi.program_to_instrument_name(program)`` returns the
        official GM name (e.g. 0 → ``'Acoustic Grand Piano'``).
      * ``is_drum`` — True iff the track was authored on MIDI channel 10
        (the GM percussion channel). On that channel the *note number* selects
        the percussion sound (GM Percussion Map), **not** the program number;
        applying ``program_to_instrument_name`` to a drum track is meaningless.
        We therefore label every drum track simply as ``'Drums'``.

    The returned name list is **deduplicated** (so a file with 3 piano tracks
    contributes 'Acoustic Grand Piano' only once) but the count
    ``midi_n_instruments`` reflects all tracks (including drum kits).

    Returns
    -------
    dict with two keys (or ``{}`` on parsing failure):

        midi_n_instruments    : int        — number of instrument tracks
        midi_instrument_names : list[str]  — unique GM names (+ 'Drums' if any),
                                             capped at ``max_items``
    """
    try:
        import pretty_midi  # lazy import — keeps module importable without it
    except ImportError:
        return {}

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return {}

    try:
        names: list[str] = []
        seen: set[str] = set()
        has_drums = False

        for ins in pm.instruments:
            if ins.is_drum:
                has_drums = True
                continue  # named 'Drums' once at the end
            name = pretty_midi.program_to_instrument_name(int(ins.program))
            if name not in seen:
                seen.add(name)
                names.append(name)

        if has_drums:
            names.append('Drums')

        return {
            'midi_n_instruments':    len(pm.instruments),
            'midi_instrument_names': names[:max_items] if names else None,
        }
    except Exception:
        return {}


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
        DTW match score threshold.  Tracks with *all* candidates below this
        value are dropped.  Default ``0.55`` keeps ≈ 50 % of the dataset.
        **Higher score = better match.**
    pick_midi : ``'best'`` | ``'all'``
        ``'best'``  → one row per track (highest score MIDI).
        ``'all'``   → one row per (track, MIDI candidate).
    """

    def __init__(
        self,
        midi_root: str | pathlib.Path,
        h5_root:   str | pathlib.Path,
        match_scores_path: Optional[str | pathlib.Path] = None,
        min_score: float = 0.55,
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

        Tracks whose best MIDI candidate falls below ``self.min_score`` are
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

            # sort by score descending (higher = better)
            midi_with_scores.sort(key=lambda x: x[1], reverse=True)

            # filter: skip if even the best candidate is below threshold
            if self.match_scores and midi_with_scores[0][1] < self.min_score:
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
        include_midi: bool = False,
    ) -> pd.DataFrame:
        """
        Build a flat, deduplicated DataFrame merging:
        - MIDI file paths (filtered + ranked by match score)
        - MSD HDF5 metadata
        - (optional) MIDI instrumentation extracted with ``pretty_midi``

        Parameters
        ----------
        max_tracks : limit the number of MSD tracks processed (for quick runs).
        verbose : print progress / dedup info.
        include_midi : if True, parse each kept MIDI with ``pretty_midi`` and
            attach instrumentation columns (``midi_n_instruments``,
            ``midi_instrument_names``). Adds ~3-10 ms per file.

        Returns
        -------
        pd.DataFrame with columns (names match the official MSD ``hdf5_getters``):
            track_id, midi_path, match_score,
            artist_name, title, release, artist_id, song_id,
            key, key_name, key_confidence,
            mode, mode_name, mode_confidence,
            tempo, time_signature, time_signature_confidence,
            duration, loudness, danceability, energy, year,
            primary_genre, top3_genres,
            artist_terms, artist_terms_freq, artist_terms_weight,
            similar_artists, artist_mbtags, artist_mbtags_count
            [+ midi_n_instruments, midi_instrument_names if include_midi=True]
        """
        tracks = self.discover_tracks(max_tracks=max_tracks, verbose=verbose)
        rows   = []
        seen_hashes: set[str] = set()  # deduplicate by MIDI md5

        # Eagerly verify pretty_midi is importable so we fail loudly instead
        # of silently producing a DataFrame without midi_* columns.
        if include_midi:
            try:
                import pretty_midi 
            except ImportError as e:
                raise ImportError(
                    "include_midi=True requires `pretty_midi` "
                    "(pip install pretty_midi)."
                ) from e

        iter_tracks = tqdm(tracks, desc='Reading HDF5 metadata') if verbose else tracks

        midi_parse_failures = 0
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
                if include_midi:
                    midi_meta = read_midi_instrumentation(midi_path)
                    if not midi_meta:
                        midi_parse_failures += 1
                    # Always set both keys so the columns exist even if every
                    # parse failed — downstream code checks `'midi_n_instruments'
                    # in df.columns` to decide whether MIDI features are present.
                    row['midi_n_instruments']    = midi_meta.get('midi_n_instruments',    np.nan)
                    row['midi_instrument_names'] = midi_meta.get('midi_instrument_names', [])
                rows.append(row)

        if include_midi and verbose:
            ok = len(rows) - midi_parse_failures
            print(f'MIDI parsed OK for {ok}/{len(rows)} files '
                  f'({100*ok/max(len(rows),1):.1f}%)')

        df = pd.DataFrame(rows)

        # ── deduplication notes ──────────────────────────────────────────────
        # We deliberately do NOT collapse by (artist_id, title) here, because
        # legitimately distinct items in the MSD share that pair:
        #   - covers by the same artist (live vs. studio)
        #   - re-recordings / remasters / acoustic versions
        #   - remixes (often credited to the original artist)
        # Exact-file duplicates were already removed above via the MIDI md5
        # hash (`seen_hashes`); when pick_midi='best' we additionally keep at
        # most one row per MSD track_id by construction.
        # If you want a stricter, lossy collapse for downstream tasks, do it
        # in the analysis notebook — not here.
        if verbose:
            n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else len(df)
            n_songs  = df['song_id'].nunique()  if 'song_id'  in df.columns else None
            msg = f'Built dataset: {len(df)} rows ({n_tracks} unique track_ids'
            if n_songs is not None:
                msg += f', {n_songs} unique song_ids'
            print(msg + ')')

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
