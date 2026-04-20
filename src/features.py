"""
src/features.py
===============
Dual-branch music feature extraction for DL-KG-project.

Branch A — Statistical (jSymbolic2)
    Wraps the jSymbolic2 Java CLI to extract 200+ global statistical
    features (rhythm, pitch, chord change rate, …) from a MIDI file.
    Output: dict of ``jsym_*`` columns.

Branch B — Semantic (music21 / musif)
    Uses music21 to extract local, rule-based harmonic logic:
    Roman numerals, key modulations, chord density, melodic intervals.
    Output: dict of ``midi_*`` and ``sem_*`` columns.

Top-level helpers
-----------------
extract_all_features(midi_path, jsymbolic_jar)
    Run both branches and return a merged dict.

extract_midi_features(midi_path)
    Branch B only (no JAR required).

run_jsymbolic(midi_path, jsymbolic_jar)
    Branch A only.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Optional

__all__ = [
    "extract_all_features",
    "extract_midi_features",
    "run_jsymbolic",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Branch A — jSymbolic2
# ---------------------------------------------------------------------------

def run_jsymbolic(
    midi_path: str | Path,
    jsymbolic_jar: str | Path,
    java_bin: str = "java",
) -> Dict[str, Any]:
    """
    Run jSymbolic2 on a single MIDI file and return its features as a
    flat dict with ``jsym_`` prefix.

    jSymbolic2 is called as:
        java -jar jsymbolic2.jar -features <midi> <csv_out> <xml_config>

    Falls back gracefully if Java or the JAR is missing.
    """
    midi_path = Path(midi_path)
    jar_path  = Path(jsymbolic_jar)

    if not jar_path.exists():
        log.warning("jSymbolic2 JAR not found: %s", jar_path)
        return {"jsym_error": "jar_not_found"}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "features.csv"
        cmd = [
            java_bin, "-jar", str(jar_path),
            "-features_only",
            str(midi_path),
            str(out_csv),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return {"jsym_error": result.stderr[:120]}

            if not out_csv.exists():
                return {"jsym_error": "no_csv_output"}

            import csv
            feats: Dict[str, Any] = {}
            with open(out_csv, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            # jSymbolic CSV: row 0 = feature names, row 1 = values
            if len(rows) >= 2:
                for name, val in zip(rows[0], rows[1]):
                    try:
                        feats[f"jsym_{name}"] = float(val)
                    except ValueError:
                        feats[f"jsym_{name}"] = val
            return feats

        except FileNotFoundError:
            return {"jsym_error": "java_not_found"}
        except subprocess.TimeoutExpired:
            return {"jsym_error": "timeout"}


# ---------------------------------------------------------------------------
# Branch B — music21 semantic + statistical features
# ---------------------------------------------------------------------------

def extract_midi_features(midi_path: str | Path) -> Dict[str, Any]:
    """
    Extract a flat dict of music21-derived ``midi_*`` and ``sem_*``
    features from a single MIDI file.

    Returns a dict with a ``midi_parse_error`` key on failure so the
    caller can safely continue.
    """
    try:
        from music21 import converter
    except ImportError:
        return {"midi_parse_error": "music21_missing"}

    feats: Dict[str, Any] = {}
    try:
        sc = converter.parse(str(midi_path), quantizePost=False)
        feats.update(_rhythm_features(sc))
        feats.update(_pitch_features(sc))
        feats.update(_harmony_features(sc))
        feats.update(_structure_features(sc))
        feats.update(_semantic_features(sc))
    except Exception as exc:
        log.debug("music21 parse error %s: %s", midi_path, exc)
        feats["midi_parse_error"] = str(exc)[:120]

    return feats


# ---------------------------------------------------------------------------
# Branch A + B combined
# ---------------------------------------------------------------------------

def extract_all_features(
    midi_path: str | Path,
    jsymbolic_jar: Optional[str | Path] = None,
    java_bin: str = "java",
) -> Dict[str, Any]:
    """
    Run both branches and return a single merged feature dict.
    If ``jsymbolic_jar`` is None, only Branch B is run.
    """
    feats = extract_midi_features(midi_path)
    if jsymbolic_jar:
        feats.update(run_jsymbolic(midi_path, jsymbolic_jar, java_bin=java_bin))
    return feats


# ---------------------------------------------------------------------------
# music21 feature sub-functions
# ---------------------------------------------------------------------------

def _rhythm_features(sc) -> Dict[str, Any]:
    from music21 import tempo as tempo_mod

    feats: Dict[str, Any] = {}
    flat = sc.flatten()
    dur_ql = [
        n.duration.quarterLength
        for n in flat.notesAndRests
        if n.duration.quarterLength > 0
    ]
    if dur_ql:
        feats["midi_dur_mean_ql"]      = mean(dur_ql)
        feats["midi_dur_std_ql"]       = stdev(dur_ql) if len(dur_ql) > 1 else 0.0
        feats["midi_dur_min_ql"]       = min(dur_ql)
        feats["midi_dur_max_ql"]       = max(dur_ql)
        feats["midi_rhythmic_variety"] = len({round(d, 3) for d in dur_ql}) / len(dur_ql)
    else:
        for k in ("midi_dur_mean_ql", "midi_dur_std_ql", "midi_dur_min_ql",
                  "midi_dur_max_ql", "midi_rhythmic_variety"):
            feats[k] = 0.0

    tempos = list(flat.getElementsByClass(tempo_mod.MetronomeMark))
    bpm = [t.number for t in tempos if t.number and t.number > 0]
    feats["midi_tempo_mean"]         = mean(bpm) if bpm else 120.0
    feats["midi_tempo_std"]          = (stdev(bpm) if len(bpm) > 1 else 0.0)
    feats["midi_num_tempo_changes"]  = len(bpm)

    total_dur = sc.duration.quarterLength or 1.0
    notes_only = list(flat.notes)
    feats["midi_note_density"]       = len(notes_only) / total_dur
    feats["midi_total_duration_ql"]  = total_dur
    return feats


def _pitch_features(sc) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    flat   = sc.flatten()
    pitches = [n.pitch.midi for n in flat.notes if hasattr(n, "pitch")]
    if pitches:
        feats["midi_pitch_mean"]  = mean(pitches)
        feats["midi_pitch_std"]   = stdev(pitches) if len(pitches) > 1 else 0.0
        feats["midi_pitch_min"]   = min(pitches)
        feats["midi_pitch_max"]   = max(pitches)
        feats["midi_pitch_range"] = max(pitches) - min(pitches)
    else:
        for k in ("midi_pitch_mean", "midi_pitch_std", "midi_pitch_min",
                  "midi_pitch_max", "midi_pitch_range"):
            feats[k] = 0.0
    return feats


def _harmony_features(sc) -> Dict[str, Any]:
    import warnings
    feats: Dict[str, Any] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = sc.analyze("key")
        feats["midi_detected_key"]       = str(k)
        feats["midi_detected_mode"]      = k.mode
        feats["midi_key_correlate"]      = float(k.correlationCoefficient)
    except Exception:
        feats["midi_detected_key"]  = ""
        feats["midi_detected_mode"] = ""
        feats["midi_key_correlate"] = 0.0

    try:
        from music21 import chord as chord_mod
        flat   = sc.flatten()
        chords = list(flat.getElementsByClass(chord_mod.Chord))
        total  = sc.duration.quarterLength or 1.0
        feats["midi_chord_density"] = len(chords) / total
        feats["midi_num_chords"]    = len(chords)
    except Exception:
        feats["midi_chord_density"] = 0.0
        feats["midi_num_chords"]    = 0
    return feats


def _structure_features(sc) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    feats["midi_num_parts"] = len(sc.parts)

    # Polyphony index: mean simultaneous note count per beat
    try:
        chordified = sc.chordify()
        from music21 import chord as chord_mod
        chord_sizes = [
            len(c.pitches)
            for c in chordified.flatten().getElementsByClass(chord_mod.Chord)
        ]
        feats["midi_polyphony_index"] = mean(chord_sizes) if chord_sizes else 0.0
    except Exception:
        feats["midi_polyphony_index"] = 0.0

    return feats


def _semantic_features(sc) -> Dict[str, Any]:
    """
    Semantic Branch: Roman numeral analysis, key modulations.
    Output columns prefixed with ``sem_``.
    """
    import warnings
    feats: Dict[str, Any] = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detected_key = sc.analyze("key")

        chord_part = sc.chordify()
        from music21 import chord as chord_mod, roman

        rn_labels = []
        for c in chord_part.flatten().getElementsByClass(chord_mod.Chord):
            try:
                rn = roman.romanNumeralFromChord(c, detected_key)
                rn_labels.append(rn.figure)
            except Exception:
                pass

        unique_rn = set(rn_labels)
        feats["sem_num_unique_rn"]    = len(unique_rn)
        feats["sem_num_chords_total"] = len(rn_labels)

        # Transition entropy: how varied are the chord changes?
        if len(rn_labels) > 1:
            import math
            from collections import Counter
            bigrams = Counter(
                zip(rn_labels[:-1], rn_labels[1:])
            )
            total = sum(bigrams.values())
            entropy = -sum(
                (v / total) * math.log2(v / total)
                for v in bigrams.values()
            )
            feats["sem_transition_entropy"] = round(entropy, 4)
        else:
            feats["sem_transition_entropy"] = 0.0

        # Functional proportions  T / D / S / PD
        _FUNC = {"I": "T", "i": "T", "III": "T", "VI": "PD", "vi": "PD",
                 "ii": "PD", "IV": "S", "iv": "S", "V": "D", "V7": "D",
                 "vii": "D", "VII": "D"}
        func_counts: Dict[str, int] = {"T": 0, "D": 0, "S": 0, "PD": 0, "X": 0}
        for rn in rn_labels:
            root = rn.split("/")[0]
            func_counts[_FUNC.get(root, "X")] += 1

        total_rn = len(rn_labels) or 1
        for fn in ("T", "D", "S", "PD"):
            feats[f"sem_func_{fn.lower()}"] = func_counts[fn] / total_rn

        # Key modulations: count embedded Key objects
        from music21 import key as key_mod
        key_objects = list(sc.flatten().getElementsByClass(key_mod.Key))
        feats["sem_num_key_changes"] = max(0, len(key_objects) - 1)

    except Exception as exc:
        log.debug("Semantic feature error: %s", exc)
        for k in ("sem_num_unique_rn", "sem_num_chords_total",
                  "sem_transition_entropy", "sem_func_t", "sem_func_d",
                  "sem_func_s", "sem_func_pd", "sem_num_key_changes"):
            feats.setdefault(k, 0.0)

    return feats
