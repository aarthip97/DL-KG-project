"""Plotting helpers for qualitative & comparison results.

Kept separate from :mod:`qualitative` so that headless / W&B-only runs do not
need matplotlib in the import path.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# ─── Population-level plots (§6.8) ────────────────────────────────────────────

def plot_population_summary(
    pop_dfs: Mapping[str, pd.DataFrame],
    *,
    top_n: int,
    figsize=(16, 10),
):
    """One mosaic figure per model collection: JS violins + cosine hist + drift.

    ``pop_dfs`` is ``{model_name: per-user qualitative DataFrame}`` as returned
    by :func:`models.evaluation.qualitative.analyze_population`.
    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    models = list(pop_dfs)
    n_mod  = len(models)
    fig, axes = plt.subplots(3, n_mod, figsize=figsize, squeeze=False)

    plot_attr_pairs = [
        ("genre",       "js_genre",       "js_genre_test"),
        ("mode",        "js_mode",        "js_mode_test"),
        ("decade",      "js_decade",      "js_decade_test"),
        ("tempo_class", "js_tempo_class", None),
    ]

    for col, model in enumerate(models):
        df = pop_dfs[model]
        # ── Row 0: JS-divergence violin per attribute ──────────────────────
        ax = axes[0, col]
        data, labels = [], []
        for attr, rec_col, _ in plot_attr_pairs:
            if rec_col in df.columns:
                data.append(df[rec_col].dropna().values)
                labels.append(attr)
        if data:
            parts = ax.violinplot(data, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{model}\nJS(train→recs)", fontsize=10)

        # ── Row 1: cosine similarity histogram ─────────────────────────────
        ax = axes[1, col]
        if "cos_mean" in df.columns:
            ax.hist(df["cos_mean"].dropna(), bins=40, color="steelblue", alpha=0.8)
            ax.axvline(df["cos_mean"].mean(), color="tomato", linestyle="--",
                       label=f"mean={df['cos_mean'].mean():.3f}")
            ax.set_title(f"{model}\ncosine(rec→profile)", fontsize=10)
            ax.legend(fontsize=8)

        # ── Row 2: hit-rate vs cosine scatter ──────────────────────────────
        ax = axes[2, col]
        if {"n_hits", "cos_mean"}.issubset(df.columns):
            hit = df["n_hits"] / top_n
            ax.scatter(df["cos_mean"], hit, alpha=0.25, s=8, color="steelblue")
            ax.set_xlabel("cosine"); ax.set_ylabel(f"hits/{top_n}")
            ax.set_title(f"{model}\nhit-rate vs cosine", fontsize=10)

    fig.suptitle("Population-level qualitative comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ─── Per-user plot (§6.7) ─────────────────────────────────────────────────────

def plot_user_distribution_comparison(
    train_meta: pd.DataFrame,
    rec_meta:   pd.DataFrame,
    test_meta:  pd.DataFrame,
    *,
    attrs: Sequence[str] = ("genre", "mode", "tempo_class"),
    title: str = "Attribute distributions — train / recs / test",
    figsize=None,
):
    """Bar-chart triplet comparing attribute frequencies across the 3 sets."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (5.5 * len(attrs), 4.5)
    fig, axes = plt.subplots(1, len(attrs), figsize=figsize, squeeze=False)
    axes = axes[0]

    def _freq(df, col, label):
        if df.empty or col not in df.columns:
            return pd.Series(dtype=float).rename(label)
        return df[col].fillna("unk").value_counts(normalize=True).round(3).rename(label)

    for ax, attr in zip(axes, attrs):
        comp = (pd.concat([
            _freq(train_meta, attr, "train"),
            _freq(rec_meta,   attr, "recs"),
            _freq(test_meta,  attr, "test_gt"),
        ], axis=1)
        .fillna(0)
        .sort_values("train", ascending=False))
        comp.plot.bar(ax=ax, width=0.75, color=["steelblue", "darkorange", "seagreen"])
        ax.set_title(attr, fontsize=11)
        ax.set_ylabel("proportion")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig
