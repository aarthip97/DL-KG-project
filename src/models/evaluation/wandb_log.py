"""W&B helpers for the evaluation package.

Functions log:
* per-model **multi-K metrics tables** (`evaluation/metrics_long`),
* per-model **population qualitative summaries** (`qualitative/<model>`),
* **pairwise significance** Wilcoxon p-value heatmaps (`comparison/pairwise/<metric>`),
* **Friedman + Nemenyi** results (`comparison/global/<metric>`).

All functions assume a W&B run is already active; they only call ``wandb.log``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import wandb


# ─── Multi-K metrics ──────────────────────────────────────────────────────────

def log_multi_k_table(
    metrics_long: pd.DataFrame, *, key: str = "evaluation/metrics_long",
) -> None:
    """Log the long-format per-model multi-K dataframe as a W&B Table."""
    wandb.log({key: wandb.Table(dataframe=metrics_long)})

    # Also log scalar leaderboard summaries: best K per (model, metric).
    pivot = metrics_long.pivot_table(index=["model", "metric"],
                                     columns="K", values="value")
    for (model, metric), row in pivot.iterrows():
        wandb.log({f"leaderboard/{model}/{metric}@best":
                   float(row.dropna().max())})


# ─── Population qualitative ───────────────────────────────────────────────────

def log_population_qualitative(
    pop_dfs: Mapping[str, pd.DataFrame], *, prefix: str = "qualitative",
) -> None:
    """Log per-model population qualitative DataFrames + scalar summaries."""
    for model, df in pop_dfs.items():
        if df.empty:
            continue
        wandb.log({f"{prefix}/{model}/per_user":
                   wandb.Table(dataframe=df)})
        # Scalar summaries that show up in the W&B leaderboard.
        summary = {
            "hit_rate_any":   float((df["n_hits"] > 0).mean()),
            "dom_genre_match": float(df.get("dom_genre_match", pd.Series(dtype=float)).mean()),
            "dom_mode_match":  float(df.get("dom_mode_match",  pd.Series(dtype=float)).mean()),
            "cos_mean":       float(df["cos_mean"].mean()),
            "js_genre_mean":  float(df["js_genre"].mean()),
        }
        for k, v in summary.items():
            wandb.summary[f"{prefix}/{model}/{k}"] = v


# ─── Pairwise comparison ──────────────────────────────────────────────────────

def log_pairwise_significance(
    pair_df: pd.DataFrame, *, key: str = "comparison/pairwise",
) -> None:
    """Log the pairwise table + per-metric Wilcoxon p-value heatmaps."""
    wandb.log({key: wandb.Table(dataframe=pair_df)})

    for metric, sub in pair_df.groupby("metric"):
        models = sorted(set(sub["model_A"]) | set(sub["model_B"]))
        mat = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
        for _, r in sub.iterrows():
            mat.at[r.model_A, r.model_B] = r.wilcoxon_p
            mat.at[r.model_B, r.model_A] = r.wilcoxon_p
        wandb.log({f"{key}/{metric}_wilcoxon_p":
                   wandb.Table(dataframe=mat.reset_index().rename(columns={"index": "model"}))})


# ─── Global comparison (Friedman + Nemenyi) ───────────────────────────────────

def log_global_comparison(
    global_results: Mapping[str, dict], *, key: str = "comparison/global",
) -> None:
    """Log Friedman omnibus + Nemenyi p-value matrices for each metric."""
    summary_rows = []
    for metric, res in global_results.items():
        if not res or pd.isna(res.get("friedman_p")):
            continue
        summary_rows.append({
            "metric":        metric,
            "n_users":       res["n_users"],
            "friedman_stat": res["friedman_stat"],
            "friedman_p":    res["friedman_p"],
            "best_model":    res["mean_ranks"].index[0],
            "best_mean_rank": float(res["mean_ranks"].iloc[0]),
        })
        wandb.log({f"{key}/{metric}/nemenyi_p":
                   wandb.Table(dataframe=res["nemenyi_p"]
                                          .reset_index()
                                          .rename(columns={"index": "model"}))})
        wandb.log({f"{key}/{metric}/mean_ranks":
                   wandb.Table(dataframe=res["mean_ranks"]
                                          .reset_index()
                                          .rename(columns={"index": "model"}))})

    if summary_rows:
        wandb.log({f"{key}/friedman_summary":
                   wandb.Table(dataframe=pd.DataFrame(summary_rows))})
