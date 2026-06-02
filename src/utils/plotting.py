"""Shared matplotlib plotting helpers."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def plot_radar_comparison(
    rows: Mapping[str, Mapping[str, float]],
    metric_keys: Sequence[str],
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "Radar — Test Metrics",
    figsize=(6, 5),
):
    """Radar/spider plot comparing several models on the same metric axes.

    Parameters
    ----------
    rows : ``{model_name: {metric: value}}``
    metric_keys : axes to display (in order, ≥ 3 for a meaningful polygon).
    colors : optional ``{model_name: matplotlib_color}``.
    """
    import matplotlib.pyplot as plt

    n = len(metric_keys)
    if n < 3:
        raise ValueError("Need at least 3 metric axes for a radar chart.")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    for label, row_d in rows.items():
        color = (colors or {}).get(label)
        vals = [row_d.get(k, 0) for k in metric_keys] + [row_d.get(metric_keys[0], 0)]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, size=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title(title, pad=15)
    fig.tight_layout()
    return fig


def plot_hgt_training_curves(history, *, save_path=None, figsize=(16, 4)):
    """Train vs val loss + val NDCG/Recall curves from a TrainResult history.

    Panel 1 overlays ``train/listwise_loss`` and (when present)
    ``val/listwise_loss`` — the same debiased listwise objective on train vs the
    held-out val positives, so a widening gap flags overfitting. Panels 2–3 show
    the held-out val NDCG and Recall trajectories (the selection signal).

    Parameters
    ----------
    history : list[dict] of per-epoch records (``TrainResult.history``); keys
        include ``epoch``, ``train/listwise_loss``, ``val/listwise_loss`` and
        ``val/ndcg@k`` / ``val/recall@k`` on eval epochs.
    save_path : optional path to write the figure (PNG, dpi 120).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    hist = pd.DataFrame(history).set_index("epoch")
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # panel 1: train vs val listwise loss
    axes[0].plot(hist.index, hist["train/listwise_loss"], color="tomato", label="train")
    if "val/listwise_loss" in hist.columns:
        s = hist["val/listwise_loss"].dropna()
        axes[0].plot(s.index, s.values, color="firebrick", ls="--",
                     marker="o", ms=3, label="val (held-out)")
        axes[0].legend(fontsize=8)
    axes[0].set_title("Listwise Loss (debiased) — train vs val")
    axes[0].set_xlabel("Epoch")

    # panels 2-3: held-out val ranking metrics
    for ax, key, color in ((axes[1], "ndcg", "darkorange"),
                           (axes[2], "recall", "steelblue")):
        cols = [c for c in hist.columns if c.startswith("val/") and key in c.lower()]
        if cols:
            s = hist[cols[0]].dropna()
            ax.plot(s.index, s.values, color=color, marker="o", ms=3)
            ax.set_title(f"Val {cols[0].split('/')[-1]}")
            ax.set_xlabel("Epoch")

    fig.suptitle("HGT Training (Debiased Listwise Loss)", fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_lambda_tradeoff(df, *, primary_k=10, best_lambda=None,
                         save_path=None, figsize=(7, 5)):
    """Accuracy vs popularity-bias trade-off across ``lambda_reg`` (λ ablation).

    Twin-axis plot: NDCG@k (accuracy, left, higher better) and PopularityBias@k
    (right, **lower better** — the quantity λ minimises) vs ``lambda_reg``; the
    chosen λ (max Overall_Score@k) is marked.

    Parameters
    ----------
    df : DataFrame with ``lambda_reg``, ``val_ndcg@{k}`` and ``val_pop_bias@{k}``
        columns (as produced by the λ ablation).
    best_lambda : optional λ to highlight with a vertical line.
    """
    import matplotlib.pyplot as plt

    d = df.sort_values("lambda_reg")
    x = d["lambda_reg"].to_numpy()
    ndcg = d[f"val_ndcg@{primary_k}"].to_numpy()
    popbias = d[f"val_pop_bias@{primary_k}"].to_numpy()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(x, ndcg, "o-", color="steelblue", label=f"NDCG@{primary_k} (accuracy ↑)")
    ax1.set_xlabel("lambda_reg (anti-popularity weight in the loss)")
    ax1.set_ylabel(f"NDCG@{primary_k}", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, popbias, "s--", color="purple",
             label=f"PopularityBias@{primary_k} (lower ↓ is better)")
    ax2.set_ylabel(f"PopularityBias@{primary_k}", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    if best_lambda is not None:
        ax1.axvline(best_lambda, color="crimson", ls=":", lw=1.5,
                    label=f"chosen λ={best_lambda:g} (max Overall@{primary_k})")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.set_title(f"λ trade-off: accuracy vs popularity bias (validation, K={primary_k})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
