"""
EDA helpers for the DL-KG pipeline notebooks.

Structure
─────────
  plot_lakh_overview(df, min_score)          → (Lakh/MSD dataset)
  plot_midi_instrumentation(df)              → (MIDI features)
  plot_user_funnel(stats)                    → (User data funnel)
  plot_user_diagnostics(filtered, per_user, per_song)  → (play-count)
  sparql_query(graph, query, **fmt_kwargs)   → (KG SPARQL)
  
  # Pre-built SPARQL query strings (import and pass to sparql_query)
  QUERY_TRACKS_PER_TEMPO_CLASS
  QUERY_TOP_GENRES
  QUERY_WD_LINK_COUNTS
  QUERY_PIANO_CHAIN
  QUERY_DECADE_WALK
  QUERY_KEY_MODE_PAIRS
"""
from __future__ import annotations

import collections
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Lakh × MSD dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_lakh_overview(
    df: pd.DataFrame,
    min_score: Optional[float] = None,
    show: bool = True,
) -> Figure:
    """
    2×3 overview of the Lakh/MSD per-track DataFrame.

    Parameters
    ----------
    df        : lakh_df from LakhMSDLinker / load_dataset
    min_score : if set, draws a vertical threshold line on the DTW score plot
    show      : call plt.show() before returning

    Returns
    -------
    matplotlib.Figure
    """
    num_cols = [
        "match_score", "tempo", "artist_familiarity", "artist_hotttnesss", "song_hotttnesss", 
        "duration", "loudness", "danceability", "energy", "key_confidence", "mode_confidence",
        "time_signature", "time_signature_confidence"
    ]
    present = [c for c in num_cols if c in df.columns]

    print("=== null summary ===")
    null_summary = (
        df.isnull()
        .sum()
        .rename("null_count")
        .to_frame()
        .assign(null_pct=lambda x: (x["null_count"] / len(df) * 100).round(2))
        .loc[lambda x: x["null_count"] > 0]
        .sort_values("null_count", ascending=False)
    )
    if not null_summary.empty:
        display(null_summary)
    else:
        print("  (no nulls)")

    print("\n=== dtypes ===")
    display(df.dtypes.value_counts().rename_axis("dtype").to_frame("count"))

    if present:
        print("\n=== numeric summary ===")
        display(df[present].describe().round(3))

    fig, axes = plt.subplots(4, 3, figsize=(24, 24))
    fig.suptitle("Lakh × MSD Dataset — Exploratory Overview", fontsize=14, fontweight="bold")

    # Match score
    ax = axes[0, 0]
    if "match_score" in df.columns:
        data = df["match_score"].dropna()
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", weights=weights)
        if min_score is not None:
            ax.axvline(min_score, color="red", linestyle="--", linewidth=2, label=f"threshold={min_score}")
            ax.legend()
    ax.set(title="Match Score (DTW)", xlabel="Score", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Tempo
    ax = axes[0, 1]
    if "tempo" in df.columns:
        data = df["tempo"].dropna()
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=60, color="coral", edgecolor="white", weights=weights)
    ax.set(title="Tempo", xlabel="BPM", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Key
    ax = axes[0, 2]
    if "key_name" in df.columns:
        key_proportion = df["key_name"].value_counts(normalize=True)
        ax.bar(key_proportion.index, key_proportion.values, color="mediumseagreen", edgecolor="white")
        ax.tick_params(axis="x", rotation=45)
    ax.set(title="Key Distribution", xlabel="Key", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Mode
    ax = axes[1, 0]
    if "mode_name" in df.columns:
        mode_proportion = df["mode_name"].value_counts(normalize=True)
        ax.pie(mode_proportion.values, labels=mode_proportion.index, autopct="%1.1f%%",
                colors=["coral", "steelblue"], startangle=90,
                wedgeprops={"edgecolor": "white"})
    ax.set_title("Mode Distribution")

    # Mode Confidence
    ax = axes[1, 1]
    if "mode_confidence" in df.columns:
        data = df["mode_confidence"].dropna()
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=50, color="orchid", edgecolor="white", weights=weights)
    ax.set(title="Mode Confidence", xlabel="Confidence", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Key Confidence
    ax = axes[1, 2]
    if "key_confidence" in df.columns:
        data = df["key_confidence"].dropna()
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=50, color="mediumpurple", edgecolor="white", weights=weights)
    ax.set(title="Key Confidence", xlabel="Confidence", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Top-20 Artists
    ax = axes[2, 0]
    if "artist_name" in df.columns:
        top_artists = df["artist_name"].dropna().value_counts().head(20)
        ax.barh(range(len(top_artists)), top_artists.values, color="teal", edgecolor="white")
        ax.set_yticks(range(len(top_artists)))
        ax.set_yticklabels(top_artists.index, fontsize=8)
        ax.invert_yaxis()
    ax.set(title="Top-20 Artists", xlabel="Count")

    # Top-20 Track Titles
    ax = axes[2, 1]
    if "title" in df.columns:
        top_titles = df["title"].dropna().value_counts().head(20)
        ax.barh(range(len(top_titles)), top_titles.values, color="slateblue", edgecolor="white")
        ax.set_yticks(range(len(top_titles)))
        ax.set_yticklabels([t[:30] for t in top_titles.index], fontsize=8)
        ax.invert_yaxis()
    ax.set(title="Top-20 Track Titles", xlabel="Count")

    # Top-20 Genres
    ax = axes[2, 2]
    genre_col = next((c for c in ("primary_genre", "genre", "artist_terms") if c in df.columns), None)
    if genre_col is not None:
        top_genres = df[genre_col].dropna().value_counts(normalize=True).head(20)
        ax.barh(range(len(top_genres)), top_genres.values, color="darksalmon", edgecolor="white")
        ax.set_yticks(range(len(top_genres)))
        ax.set_yticklabels([str(g)[:30] for g in top_genres.index], fontsize=8)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set(title="Top-20 Genres", xlabel="Proportion")

    # Duration
    ax = axes[3, 0]
    if "duration" in df.columns:
        duration_sec = df["duration"].dropna()
        data = duration_sec.clip(upper=duration_sec.quantile(0.99))
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=50, color="skyblue", edgecolor="white", weights=weights)
    ax.set(title="Duration (clipped at 99th percentile)", xlabel="Seconds", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Time Signature
    ax = axes[3, 1]
    if "time_signature" in df.columns:
        ts_proportion = df["time_signature"].value_counts(normalize=True).sort_index()
        ax.bar(ts_proportion.index.astype(str), ts_proportion.values, color="sandybrown", edgecolor="white")
    ax.set(title="Time Signature", xlabel="Signature", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Time Signature Confidence
    ax = axes[3, 2]
    if "time_signature_confidence" in df.columns:
        data = df["time_signature_confidence"].dropna()
        weights = np.ones(len(data)) / len(data)
        ax.hist(data, bins=50, color="lightcoral", edgecolor="white", weights=weights)
    ax.set(title="Time Signature Confidence", xlabel="Confidence", ylabel="Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # remove empty subplots
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if not axes[i, j].has_data():
                fig.delaxes(axes[i, j])

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_midi_instrumentation(df: pd.DataFrame, show: bool = False) -> Optional[Figure]:
    """
    Three-panel MIDI instrumentation plot + printed summary table.

    Parameters
    ----------
    df   : lakh_df — must have been built with include_midi=True
    show : call plt.show() before returning

    Returns
    -------
    matplotlib.Figure, or None if MIDI columns are absent
    """
    if "midi_n_instruments" not in df.columns:
        print("[INFO] No MIDI instrumentation columns — rerun build_dataset(..., include_midi=True).")
        return None

    midi_df   = df.dropna(subset=["midi_n_instruments"]).copy()
    all_names = [
        name
        for lst in midi_df["midi_instrument_names"]
        if lst is not None and len(lst) > 0
        for name in lst
    ]
    has_drums = midi_df["midi_instrument_names"].apply(
        lambda lst: "Drums" in lst if lst is not None and len(lst) > 0 else False
    )
    # ── printed summary ───────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "metric": [
            "# tracks parsed",
            "mean # instrument tracks / file",
            "median # instrument tracks / file",
            "% files containing drums",
            "# unique instrument names seen",
        ],
        "value": [
            f"{len(midi_df):,}  ({100*len(midi_df)/len(df):.1f}% of dataset)",
            f"{midi_df['midi_n_instruments'].mean():.2f}",
            f"{midi_df['midi_n_instruments'].median():.0f}",
            f"{100 * has_drums.mean():.1f}%",
            f"{len(set(all_names))}",
        ],
    })
    display(summary.set_index("metric"))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MIDI Instrumentation — Lakh × MSD subset", fontsize=14, fontweight="bold")

    ax = axes[0]
    n_inst = midi_df["midi_n_instruments"].clip(upper=20)
    ax.hist(n_inst, bins=range(0, 22), color="steelblue", edgecolor="white", align="left")
    ax.set(title="Number of instrument tracks per MIDI",
           xlabel="# tracks  (20 = 20+)", ylabel="# files")
    ax.set_xticks(range(0, 21, 2))

    ax = axes[1]
    top_names = collections.Counter(all_names).most_common(20)
    if top_names:
        labels, counts = zip(*top_names)
        total  = len(midi_df)
        freqs  = [c / total for c in counts]
        y_pos  = list(range(len(labels)))[::-1]
        ax.barh(y_pos, freqs[::-1], color="mediumpurple")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlim(0, max(freqs) * 1.15)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set(title="Top-20 instrument names (frequency across files)", xlabel="Proportion")

    ax = axes[2]
    dc = has_drums.value_counts().reindex([True, False], fill_value=0)
    ax.pie(dc.values, labels=["with drums", "no drums"],
            autopct="%1.1f%%", colors=["#4CAF50", "#FF7043"], startangle=90,
            wedgeprops={"edgecolor": "white"})
    ax.set_title("Files containing a drum track")

    plt.tight_layout()
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — User / taste-profile data
# ─────────────────────────────────────────────────────────────────────────────
def plot_user_funnel(stats: dict, show: bool = True) -> Figure:
    """
    Bar-chart funnel showing how many interactions / users / songs survive each
    filtering step.

    Parameters
    ----------
    stats : filter_stats dict produced by filter_taste_profile
            Expected keys: raw_rows, after_mismatch_rows, after_lmd_rows,
            after_coldstart_rows  (and *_users / *_songs variants)
    show  : call plt.show() before returning

    Returns
    -------
    matplotlib.Figure
    """
    steps = ["Raw", "−Mismatches", "∩ LMD", "−Cold-start"]
    rows_vals  = [stats.get("raw_rows", 0),  stats.get("after_mismatch_rows", 0),
                  stats.get("after_lmd_rows", 0),  stats.get("after_coldstart_rows", 0)]
    users_vals = [stats.get("raw_users", 0), stats.get("after_mismatch_users", 0),
                  stats.get("after_lmd_users", 0), stats.get("after_coldstart_users", 0)]
    songs_vals = [stats.get("raw_songs", 0), stats.get("after_mismatch_songs", 0),
                  stats.get("after_lmd_songs", 0), stats.get("after_coldstart_songs", 0)]

    summary_df = pd.DataFrame(
        {"Interactions": rows_vals, "Unique Users": users_vals, "Unique Songs": songs_vals},
        index=pd.Index(steps, name="Step"),
    )
    print(summary_df.to_string())

    def _fmt(x: float) -> str:
        if x >= 1_000_000:
            return f"{x/1e6:.1f}M"
        if x >= 1_000:
            return f"{x/1e3:.0f}K"
        return str(int(x))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, vals) in zip(axes, [
        ("Interactions", rows_vals),
        ("Unique Users", users_vals),
        ("Unique Songs", songs_vals),
    ]):
        bars = ax.bar(steps, vals, color=colors, edgecolor="white")
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))
        ax.tick_params(axis="x", labelsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    _fmt(val), ha="center", va="bottom", fontsize=8)

    plt.suptitle("Taste Profile — Filtering Funnel", fontsize=13, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_user_diagnostics(
    filtered: pd.DataFrame,
    per_user: pd.DataFrame,
    per_song: pd.DataFrame,
    show: bool = True,
) -> Figure:
    """
    2×2 diagnostic plots for the filtered taste profile.

    Parameters
    ----------
    filtered  : filtered triplets DataFrame (columns: user_id, song_id, play_count)
    per_user  : per-user stats DataFrame   (column: n_songs)
    per_song  : per-song stats DataFrame   (columns: n_users, total_plays)
    show      : call plt.show() before returning

    Returns
    -------
    matplotlib.Figure
    """
    def _fmt(x: float) -> str:
        if x >= 1_000_000:
            return f"{x/1e6:.1f}M"
        if x >= 1_000:
            return f"{x/1e3:.0f}K"
        return str(int(x))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.hist(filtered["play_count"], bins=100, log=True, color="#4C72B0", edgecolor="white")
    ax.set(xlabel="Play count", ylabel="# interactions (log scale)",
           title="Play-count distribution (log y)")

    ax = axes[0, 1]
    ax.hist(per_user["n_songs"], bins=80, log=True, color="#55A868", edgecolor="white")
    ax.set(xlabel="# songs played", ylabel="# users (log scale)", title="Songs per user")

    ax = axes[1, 0]
    ax.hist(per_song["n_users"], bins=80, log=True, color="#DD8452", edgecolor="white")
    ax.set(xlabel="# users who played song", ylabel="# songs (log scale)", title="Users per song")

    ax = axes[1, 1]
    top20 = per_song.nlargest(20, "total_plays").reset_index()
    label_col = "title" if "title" in top20.columns else "song_id"
    labels = top20[label_col].astype(str).str[:28].tolist()
    ax.barh(labels[::-1], top20["total_plays"].tolist()[::-1], color="#C44E52")
    ax.set(xlabel="Total plays", title="Top-20 most-played songs")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))

    plt.suptitle("Echo Nest Taste Profile — Filtered Dataset Diagnostics", fontsize=13, y=1.01)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Section 5+ — KG / SPARQL
# ─────────────────────────────────────────────────────────────────────────────

def sparql_query(graph, query: str, **fmt_kwargs) -> pd.DataFrame:
    """
    Run a SPARQL SELECT query against an rdflib Graph and return a DataFrame.

    Parameters
    ----------
    graph       : rdflib.Graph (e.g. builder.g)
    query       : SPARQL SELECT string (may contain {placeholders} for fmt_kwargs)
    **fmt_kwargs: keyword arguments forwarded to query.format(**fmt_kwargs)
                  e.g. sparql_query(g, QUERY_TOP_GENRES, limit=20)

    Returns
    -------
    pd.DataFrame  — one column per projected variable, one row per result
    """
    rendered = query.format(**fmt_kwargs) if fmt_kwargs else query
    results  = graph.query(rendered)
    if not results.vars:
        return pd.DataFrame()
    cols = [str(v) for v in results.vars]
    rows = [
        {c: (str(row[c]) if row[c] is not None else None) for c in cols}
        for row in results
    ]
    return pd.DataFrame(rows, columns=cols)