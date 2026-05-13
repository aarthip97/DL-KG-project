"""Plot helpers that read the CSVs produced by :mod:`graphdb.exports`.

The functions here are deliberately *file-driven* — they take a CSV path
and produce a PNG path — so the same code works whether you ran the
exports locally or pulled them down from Google Drive.

All figures are saved with ``bbox_inches='tight'`` so they drop straight
into a notebook ``IPython.display.Image`` call without further fiddling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


# ── small private helpers ──────────────────────────────────────────────
def _save(fig: Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure → %s", out_path)
    return out_path


def _bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    *,
    top_n: int = 20,
    horizontal: bool = True,
) -> Path:
    """Generic bar chart from a 2-column dataframe."""
    if label_col not in df.columns or value_col not in df.columns:
        raise KeyError(
            f"Expected columns '{label_col}' and '{value_col}' in "
            f"DataFrame; got {list(df.columns)}"
        )
    sub = df.nlargest(top_n, value_col).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(sub))))
    if horizontal:
        ax.barh(sub[label_col].astype(str), sub[value_col])
        ax.set_xlabel(value_col)
    else:
        ax.bar(sub[label_col].astype(str), sub[value_col])
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_path)


# ── public plot functions ──────────────────────────────────────────────
def plot_genre_distribution(csv_path: Path, out_path: Path,
                            top_n: int = 20) -> Path:
    df = pd.read_csv(csv_path)
    value = "total_weight" if "total_weight" in df.columns else "n_artists"
    return _bar(df, "genreLabel", value,
                f"Top {top_n} genres by {value}", out_path, top_n=top_n)


def plot_key_distribution(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    value = "n_high_confidence" if "n_high_confidence" in df.columns else "n"
    return _bar(df, "keyLabel", value,
                "Detected musical keys", out_path,
                top_n=24, horizontal=False)


def plot_node_type_histogram(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    type_col = "type" if "type" in df.columns else df.columns[0]
    n_col    = "n"    if "n"    in df.columns else df.columns[-1]
    # Strip the long URI prefix for readability — keep only the last path
    # segment / fragment.
    df = df.copy()
    df[type_col] = (
        df[type_col].astype(str)
                    .str.rsplit("#", n=1).str[-1]
                    .str.rsplit("/", n=1).str[-1]
    )
    return _bar(df, type_col, n_col,
                "KG node type distribution", out_path, top_n=30)


def plot_relation_histogram(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    rel_col = "r" if "r" in df.columns else df.columns[0]
    n_col   = "n" if "n" in df.columns else df.columns[-1]
    df = df.copy()
    df[rel_col] = (
        df[rel_col].astype(str)
                   .str.rsplit("#", n=1).str[-1]
                   .str.rsplit("/", n=1).str[-1]
    )
    return _bar(df, rel_col, n_col,
                "Predicate (relation) usage", out_path, top_n=30)


# ── batch driver ───────────────────────────────────────────────────────
# Maps each known stats CSV to the function that knows how to plot it.
# Anything not in this map is simply skipped (no error) — keeping the
# stats catalog and the plot catalog decoupled.
_PLOTTERS = {
    "genres_simple":         plot_genre_distribution,
    "genres_rich":           plot_genre_distribution,
    "confident_keys_simple": plot_key_distribution,
    "confident_keys_rich":   plot_key_distribution,
    "node_type_histogram":   plot_node_type_histogram,
    "relation_histogram":    plot_relation_histogram,
}


def plot_all(stats_dir: Path, plots_dir: Path,
             only: Optional[list[str]] = None) -> dict[str, Path]:
    """Render every stats CSV in ``stats_dir`` for which we have a plotter.

    Returns a dict of stats-name → output PNG path.
    """
    stats_dir = Path(stats_dir)
    plots_dir = Path(plots_dir)
    out: dict[str, Path] = {}
    for name, plotter in _PLOTTERS.items():
        if only is not None and name not in only:
            continue
        csv = stats_dir / f"{name}.csv"
        if not csv.is_file():
            log.debug("No CSV for %s; skipping plot", name)
            continue
        try:
            out[name] = plotter(csv, plots_dir / f"{name}.png")
        except (KeyError, ValueError) as e:
            log.warning("Could not plot %s: %s", name, e)
    return out
