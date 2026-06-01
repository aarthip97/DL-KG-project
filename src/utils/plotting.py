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
