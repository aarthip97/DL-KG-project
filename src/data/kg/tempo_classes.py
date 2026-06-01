"""
Tempo classification per Music Theory Academy
(https://www.musictheoryacademy.com/how-to-read-sheet-music/tempo/).

Markings (low → high BPM):

    Larghissimo     <  20      Extremely slow
    Grave           [20,  40)  Slow and solemn
    Largo           [40,  60)  Very slow
    Adagio          [60,  76)  Slow and stately
    Andante         [76, 108)  Walking pace
    Moderato        [108,120)  Moderately
    Allegro         [120,168)  Fast
    Presto          [168,200)  Very fast
    Prestissimo     >= 200     Very very fast

We treat each interval as ``[lo, hi)`` (left-inclusive, right-exclusive)
so that a song at exactly 120 BPM is *Allegro*, exactly 168 BPM is
*Presto*, etc. The classifier returns ``None`` for ``NaN`` / ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math
import pandas as pd


@dataclass(frozen=True)
class TempoClass:
    name: str
    lo: float            # inclusive
    hi: float            # exclusive (math.inf for the top class)
    description: str


# Ordered low → high BPM. Order matters: classify_tempo iterates from low to high.
TEMPO_CLASSES: tuple[TempoClass, ...] = (
    TempoClass("Larghissimo", 0.0,    20.0,  "Extremely slow (< 20 BPM)"),
    TempoClass("Grave",       20.0,   40.0,  "Slow and solemn (20–40 BPM)"),
    TempoClass("Largo",       40.0,   60.0,  "Very slow (40–60 BPM)"),
    TempoClass("Adagio",      60.0,   76.0,  "Slow and stately (60–76 BPM)"),
    TempoClass("Andante",     76.0,  108.0,  "Walking pace (76–108 BPM)"),
    TempoClass("Moderato",   108.0,  120.0,  "Moderately (108–120 BPM)"),
    TempoClass("Allegro",    120.0,  168.0,  "Fast (120–168 BPM)"),
    TempoClass("Presto",     168.0,  200.0,  "Very fast (168–200 BPM)"),
    TempoClass("Prestissimo", 200.0, math.inf, "Very very fast (≥ 200 BPM)"),
)


def classify_tempo(bpm: Optional[float]) -> Optional[str]:
    """
    Map a single tempo value (BPM) to its Music Theory Academy class name.

    Returns ``None`` if ``bpm`` is ``None`` / ``NaN`` / non-positive.
    """
    if bpm is None:
        return None
    try:
        v = float(bpm)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or v <= 0:
        return None
    for tc in TEMPO_CLASSES:
        if tc.lo <= v < tc.hi:
            return tc.name
    # Should be unreachable since the last bucket extends to +inf.
    return TEMPO_CLASSES[-1].name


def add_tempo_class_column(
    df: pd.DataFrame,
    tempo_col: str = "Mean_Tempo",
    out_col: str = "tempo_class",
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with a new categorical column ``out_col``
    whose values are the tempo class names derived from ``df[tempo_col]``.

    The column is a pandas ``Categorical`` with the canonical low→high
    ordering of :data:`TEMPO_CLASSES`, so groupby/sort operations
    naturally produce slow-to-fast results.
    """
    if tempo_col not in df.columns:
        raise KeyError(
            f"Column {tempo_col!r} not found. Available: {list(df.columns)[:10]}…"
        )
    out = df.copy()
    out[out_col] = pd.Categorical(
        out[tempo_col].map(classify_tempo),
        categories=[tc.name for tc in TEMPO_CLASSES],
        ordered=True,
    )
    return out


def tempo_class_table() -> pd.DataFrame:
    """Return a small reference DataFrame describing every tempo class."""
    return pd.DataFrame(
        [
            {
                "tempo_class":   tc.name,
                "min_bpm":       tc.lo,
                "max_bpm":       tc.hi if tc.hi != math.inf else float("inf"),
                "description":   tc.description,
            }
            for tc in TEMPO_CLASSES
        ]
    )


__all__: Iterable[str] = (
    "TempoClass",
    "TEMPO_CLASSES",
    "classify_tempo",
    "add_tempo_class_column",
    "tempo_class_table",
)
