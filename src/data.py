"""
src/data.py
===========
Dataset helpers for DL-KG-project.

Covers:
  - MSD HDF5 reader           (read_msd_metadata)
  - MIDI <-> HDF5 path utils  (_track_id_to_h5_path, _find_midi_for_track)
  - Match-score loading        (load_match_scores)
  - LakhMSDLinker              (discover_tracks, build_dataset)

Usage
-----
    from src.data import LakhMSDLinker, load_match_scores

    scores  = load_match_scores('data/raw/match_scores.json')
    linker  = LakhMSDLinker(
                  midi_root      = 'data/raw/lmd_matched',
                  h5_root        = 'data/raw/lmd_matched_h5',
                  match_scores   = scores,
              )
    tracks  = linker.discover_tracks(max_tracks=5000, min_score=0.70)
    df      = linker.build_dataset(tracks)
    df.to_parquet('data/interim/dataset.parquet', index=False)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

__all__ = [
    "read_msd_metadata",
    "load_match_scores",
    "LakhMSDLinker",
    "KEY_NAMES",
]

log = logging.getLogger(__name__)

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ---------------------------------------------------------------------------
# MSD HDF5 reader
# ---------------------------------------------------------------------------

def read_msd_metadata(h5_path: str | Path) -> Dict[str, Any]:
    """
    Read all useful metadata from a single MSD ``.h5`` file using h5py.
    Returns a flat dict ready to be merged as a DataFrame row.
    """
    import h5py

    info: Dict[str, Any] = {}
    with h5py.File(str(h5_path), "r") as h5:
        meta = h5["/metadata/songs"][0]
        info["artist_name"] = meta["artist_name"].decode("utf-8", errors="replace")
        info["title"]       = meta["title"].decode("utf-8", errors="replace")
        info["release"]     = meta["release"].decode("utf-8", errors="replace")
        info["artist_id"]   = meta["artist_id"].decode()
        info["song_id"]     = meta["song_id"].decode()

        ana = h5["/analysis/songs"][0]
        info["track_id"]         = ana["track_id"].decode()
        info["msd_key"]          = int(ana["key"])
        info["msd_key_name"]     = KEY_NAMES[int(ana["key"])]
        info["msd_key_conf"]     = float(ana["key_confidence"])
        info["msd_mode"]         = int(ana["mode"])
        info["msd_mode_name"]    = "major" if ana["mode"] == 1 else "minor"
        info["msd_mode_conf"]    = float(ana["mode_confidence"])
        info["msd_tempo"]        = float(ana["tempo"])
        info["msd_time_sig"]     = int(ana["time_signature"])
        info["msd_duration"]     = float(ana["duration"])
        info["msd_loudness"]     = float(ana["loudness"])
        info["msd_danceability"] = float(ana["danceability"])
        info["msd_energy"]       = float(ana["energy"])

        mb = h5["/musicbrainz/songs"][0]
        info["year"] = int(mb["year"])

        terms = [t.decode("utf-8", errors="replace") for t in h5["/metadata/artist_terms"][:]]
        freqs = list(h5["/metadata/artist_terms_freq"][:])
        if terms:
            paired = sorted(zip(freqs, terms), reverse=True)
            info["primary_genre"] = paired[0][1]
            info["top3_genres"]   = ";".join(t for _, t in paired[:3])
        else:
            info["primary_genre"] = ""
            info["top3_genres"]   = ""

        mbtags_raw = h5["/musicbrainz/artist_mbtags"][:]
        info["mbtags"] = ";".join(t.decode("utf-8", errors="replace") for t in mbtags_raw)

    return info


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _track_id_to_h5_path(track_id: str, h5_root: Path) -> Path:
    """TRAAAAV128F421A322  →  <h5_root>/A/A/V/TRAAAAV128F421A322.h5"""
    return h5_root / track_id[2] / track_id[3] / track_id[4] / f"{track_id}.h5"


def _find_midi_for_track(track_id: str, midi_root: Path) -> List[Path]:
    """Return every MIDI file inside  <midi_root>/<A>/<A>/<V>/<track_id>/"""
    subdir = midi_root / track_id[2] / track_id[3] / track_id[4] / track_id
    if not subdir.is_dir():
        return []
    return sorted(subdir.glob("*.mid")) + sorted(subdir.glob("*.midi"))


def _pick_best_midi(
    midi_files: Sequence[Path],
    match_scores: Optional[Dict[str, Any]],
    track_id: str,
) -> Tuple[Optional[Path], float]:
    """Return (best_midi_path, dtw_score) for a track's MIDI candidates."""
    if not midi_files:
        return None, 0.0
    scores: Dict[str, float] = {}
    if match_scores and track_id in match_scores:
        raw = match_scores[track_id]
        if isinstance(raw, dict):
            scores = {k: float(v) for k, v in raw.items()}

    best_path: Optional[Path] = None
    best_score = -1.0
    for mp in midi_files:
        sc = scores.get(mp.name, scores.get(str(mp), 0.0))
        if sc > best_score:
            best_score, best_path = sc, mp

    return (best_path or midi_files[0]), max(best_score, 0.0)


# ---------------------------------------------------------------------------
# Match-score loader
# ---------------------------------------------------------------------------

def load_match_scores(path: str | Path) -> Dict[str, Any]:
    """Load match_scores.json → {track_id: {md5.mid: score}}."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LakhMSDLinker
# ---------------------------------------------------------------------------

class LakhMSDLinker:
    """
    Discover and link Lakh MIDI ↔ MSD HDF5 files, then build a flat
    feature DataFrame.

    Parameters
    ----------
    midi_root : str | Path
        Root directory of ``lmd_matched`` (or ``lmd_aligned``).
    h5_root : str | Path
        Root directory of ``lmd_matched_h5``.
    match_scores : dict, optional
        Pre-loaded match_scores dict (from ``load_match_scores()``).
    """

    def __init__(
        self,
        midi_root: str | Path,
        h5_root: str | Path,
        match_scores: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.midi_root    = Path(midi_root)
        self.h5_root      = Path(h5_root)
        self.match_scores = match_scores or {}

    # ------------------------------------------------------------------

    def discover_tracks(
        self,
        max_tracks: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Walk ``h5_root`` to find all available track IDs, pair each with
        its best-scoring MIDI, and filter by ``min_score``.

        Returns a list of dicts: {track_id, h5_path, midi_path, match_score}.
        """
        tracks: List[Dict[str, Any]] = []

        for h5_file in sorted(self.h5_root.rglob("*.h5")):
            track_id = h5_file.stem
            if not track_id.startswith("TR"):
                continue

            midi_files = _find_midi_for_track(track_id, self.midi_root)
            if not midi_files:
                continue

            midi_path, score = _pick_best_midi(midi_files, self.match_scores, track_id)
            if score < min_score:
                continue

            tracks.append({
                "track_id":    track_id,
                "h5_path":     str(h5_file),
                "midi_path":   str(midi_path),
                "match_score": score,
            })

            if max_tracks and len(tracks) >= max_tracks:
                break

        log.info("Discovered %d tracks (min_score=%.2f)", len(tracks), min_score)
        return tracks

    # ------------------------------------------------------------------

    def build_dataset(
        self,
        tracks: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        For each discovered track, read MSD metadata from its HDF5 file
        and return a single merged DataFrame.

        MIDI feature extraction and jSymbolic2 are handled in notebook 01
        (feature extraction) — this method keeps the dataset building step
        lightweight and fast.
        """
        rows: List[Dict[str, Any]] = []
        n = len(tracks)

        for i, t in enumerate(tracks):
            if verbose and i % max(1, n // 20) == 0:
                print(f"  [{i+1}/{n}] {t['track_id']}")

            row: Dict[str, Any] = {
                "track_id":    t["track_id"],
                "midi_path":   t["midi_path"],
                "match_score": t["match_score"],
            }
            try:
                msd = read_msd_metadata(t["h5_path"])
                row.update(msd)
            except Exception as exc:
                log.debug("HDF5 read error %s: %s", t["track_id"], exc)
                row["msd_error"] = str(exc)[:80]

            rows.append(row)

        df = pd.DataFrame(rows)
        if verbose:
            print(f"\nDataset: {len(df):,} rows × {df.shape[1]} columns")
        return df
