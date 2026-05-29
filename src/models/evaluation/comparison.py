"""Statistical comparison of multiple recommenders on per-user metric vectors.

We use the standard recipe for benchmarking recommender systems:

* **Pairwise** (two models at a time):
    - **Paired Wilcoxon signed-rank** — non-parametric, robust to non-normal
      per-user metric distributions; the default test.
    - **Paired *t*-test**             — for completeness when normality
      can be reasonably assumed.
    - **Cohen's** :math:`d_z`         — paired-difference effect size:

      .. math::

         d_z = \\dfrac{\\bar{x}_d}{s_d}

      where :math:`\\bar{x}_d` and :math:`s_d` are the mean and standard
      deviation of the per-user metric differences.

* **Global** (≥ 3 models on the same users):
    - **Friedman test** — non-parametric repeated-measures omnibus (the
      analogue of repeated-measures ANOVA).
    - **Nemenyi post-hoc** — pairwise critical-difference comparison once
      Friedman is significant.

All p-values are returned in dataframes that are ready to drop into a
W&B ``Table`` or a notebook display.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ─── Pairwise comparisons ─────────────────────────────────────────────────────

def _cohens_dz(diff: np.ndarray) -> float:
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def pairwise_significance(
    per_user_metrics: Mapping[str, pd.DataFrame],
    *,
    metrics: Iterable[str] = ("Recall@K", "NDCG@K", "MRR", "HitRate@K"),
    user_col: str = "u_idx",
) -> pd.DataFrame:
    """Pairwise paired tests over every (model_A, model_B) pair.

    Parameters
    ----------
    per_user_metrics : ``{model_name: per-user DataFrame}``
        Each DataFrame must contain ``user_col`` plus the requested metric
        columns. The intersection of users across the two models is used for
        each pair, so missing users are handled gracefully.
    metrics : metric column names to test.

    Returns a long DataFrame with ``[metric, model_A, model_B,
    n, mean_A, mean_B, mean_diff, cohens_dz, wilcoxon_p, ttest_p, winner]``.
    """
    metrics = list(metrics)
    rows: List[dict] = []
    for m in metrics:
        for a, b in combinations(per_user_metrics, 2):
            df_a = per_user_metrics[a].set_index(user_col)
            df_b = per_user_metrics[b].set_index(user_col)
            common = df_a.index.intersection(df_b.index)
            if m not in df_a.columns or m not in df_b.columns or len(common) < 5:
                continue
            xa = df_a.loc[common, m].astype(float).values
            xb = df_b.loc[common, m].astype(float).values
            diff = xa - xb

            try:
                w_stat, w_p = stats.wilcoxon(xa, xb, zero_method="wilcox",
                                             alternative="two-sided")
            except ValueError:
                w_stat, w_p = (np.nan, np.nan)
            try:
                t_stat, t_p = stats.ttest_rel(xa, xb, nan_policy="omit")
            except ValueError:
                t_stat, t_p = (np.nan, np.nan)

            rows.append({
                "metric":     m,
                "model_A":    a,
                "model_B":    b,
                "n":          int(len(common)),
                "mean_A":     float(np.nanmean(xa)),
                "mean_B":     float(np.nanmean(xb)),
                "mean_diff":  float(np.nanmean(diff)),
                "cohens_dz":  _cohens_dz(diff),
                "wilcoxon_p": float(w_p) if w_p is not None else np.nan,
                "ttest_p":    float(t_p) if t_p is not None else np.nan,
                "winner":     a if np.nanmean(diff) > 0 else b,
            })
    return pd.DataFrame(rows)


# ─── Global comparison: Friedman + Nemenyi ────────────────────────────────────

def _nemenyi(p: int, n: int, ranks: np.ndarray) -> pd.DataFrame:
    """Nemenyi post-hoc Q-statistic p-values matrix.

    ``ranks`` is the vector of mean ranks per model (length ``p``). Returns a
    p × p DataFrame (the diagonal is 1.0).
    """
    # Standard error of the mean-rank difference (Demšar 2006, eq. 5).
    se = np.sqrt(p * (p + 1) / (6.0 * n))
    out = np.ones((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            q = abs(ranks[i] - ranks[j]) / se
            # Studentised range distribution → use Tukey via scipy.
            out[i, j] = 1.0 - stats.studentized_range.cdf(q, p, np.inf)
    return out


def friedman_nemenyi(
    per_user_metrics: Mapping[str, pd.DataFrame],
    metric: str,
    *,
    user_col: str = "u_idx",
) -> dict:
    """Run Friedman + Nemenyi post-hoc on the chosen metric.

    Returns
    -------
    dict with keys
        ``friedman_stat``, ``friedman_p``, ``mean_ranks`` (Series),
        ``nemenyi_p`` (DataFrame), ``n_users``.
    """
    models = list(per_user_metrics)
    # Align users present for *every* model so Friedman sees a balanced design.
    common = None
    for name in models:
        idx = per_user_metrics[name].set_index(user_col).index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 5:
        return {"friedman_stat": np.nan, "friedman_p": np.nan,
                "mean_ranks": pd.Series(dtype=float),
                "nemenyi_p": pd.DataFrame(), "n_users": 0}

    mat = np.column_stack([
        per_user_metrics[name].set_index(user_col).loc[common, metric].astype(float).values
        for name in models
    ])  # (n_users, n_models)

    stat, p = stats.friedmanchisquare(*mat.T)

    # Mean ranks: per row, rank descending so higher metric → smaller rank
    # (i.e. rank 1 is best).  Ties → average rank.
    ranks = np.apply_along_axis(
        lambda r: stats.rankdata(-r, method="average"), axis=1, arr=mat,
    ).mean(axis=0)

    nem_p = _nemenyi(len(models), len(common), ranks)
    return {
        "friedman_stat": float(stat),
        "friedman_p":    float(p),
        "mean_ranks":    pd.Series(ranks, index=models, name="mean_rank")
                          .sort_values(),
        "nemenyi_p":     pd.DataFrame(nem_p, index=models, columns=models),
        "n_users":       int(len(common)),
    }


# ─── Convenience summary ──────────────────────────────────────────────────────

def summarise_comparison(
    per_user_metrics: Mapping[str, pd.DataFrame],
    *,
    metrics: Iterable[str] = ("Recall@K", "NDCG@K", "MRR", "HitRate@K"),
    user_col: str = "u_idx",
    alpha: float = 0.05,
) -> Dict[str, object]:
    """One-shot dict of every comparison artefact (handy for W&B)."""
    pair_df = pairwise_significance(per_user_metrics, metrics=metrics, user_col=user_col)
    pair_df["wilcoxon_significant"] = pair_df["wilcoxon_p"] < alpha
    pair_df["ttest_significant"]    = pair_df["ttest_p"]    < alpha

    global_results = {}
    if len(per_user_metrics) > 2:
        global_results: Dict[str, dict] = {
            m: friedman_nemenyi(per_user_metrics, m, user_col=user_col) for m in metrics
        }
    return {"pairwise": pair_df, "global": global_results, "alpha": alpha}
