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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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


# ─── Significance on Overall_Score@K and its components ───────────────────────

def overall_significance(
    bulk_recs: Mapping[str, Mapping[int, Sequence[int]]],
    ground_truth: Mapping[int, set],
    pop_norm: np.ndarray,
    n_songs: int,
    *,
    k: int = 10,
    weights: tuple = (0.60, 0.20, 0.20),
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise significance for ``Overall_Score@k`` and *only* its components.

    The selection criterion is
    ``0.6·NDCG@k + 0.2·Coverage@k + 0.2·(1−PopularityBias@k)`` — so we test those
    three components plus the composite itself, and nothing else (no MRR/HitRate
    etc., which do not feed the criterion).

    Two regimes, because the components live at different granularities:

    * **NDCG@k** and **AntiPopularity@k** (``1−PopularityBias@k``) are *per-user*
      → exact paired **Wilcoxon signed-rank** on the shared users (+ Cohen's dz).
    * **Coverage@k** is a *set-level* catalogue metric and **Overall_Score@k** is
      a composite that contains it, so neither is a per-user vector. For these we
      use a **paired user-level bootstrap**: resample the evaluated users with
      replacement ``n_boot`` times (the *same* resample for every model so the
      comparison is paired), recompute the metric on each resample, and read the
      two-sided p-value off the bootstrap distribution of the A−B difference
      together with a percentile CI.

    Returns a long DataFrame with columns ``[metric, model_A, model_B, n,
    mean_A, mean_B, mean_diff, effect, ci_low, ci_high, test, p_value, winner]``
    where ``effect`` is Cohen's dz for the Wilcoxon rows and NaN for bootstrap
    rows.
    """
    from .metrics import evaluate_recs_per_user

    w_ndcg, w_cov, w_pop = weights
    models = list(bulk_recs)

    # Per-user frames (NDCG@k, PopularityBias@k) on the users that have ground truth.
    per_user = {n: evaluate_recs_per_user(bulk_recs[n], ground_truth, pop_norm, k=k)
                for n in models}
    eval_users = {n: set(per_user[n]["u_idx"]) for n in models}
    common = sorted(set.intersection(*eval_users.values())) if eval_users else []

    # Aligned per-user matrices on the common users (rows = users, one col/model).
    def _col(n, metric):
        s = per_user[n].set_index("u_idx").loc[common, metric].astype(float)
        return s.values

    ndcg = {n: _col(n, "NDCG@K") for n in models}
    antipop = {n: 1.0 - _col(n, "PopularityBias") for n in models}

    # Per-user top-k item matrix (rows aligned to ``common``) for bootstrap coverage.
    sentinel = n_songs
    topk_mat = {}
    for n in models:
        rec = bulk_recs[n]
        mat = np.full((len(common), k), sentinel, dtype=np.int64)
        for i, u in enumerate(common):
            items = list(rec.get(u, []))[:k]
            mat[i, :len(items)] = items
        topk_mat[n] = mat

    rng = np.random.default_rng(seed)
    n_u = len(common)
    # Reusable presence buffer for the boolean-scatter coverage (cheaper than
    # ``np.unique`` and avoids a sort over millions of item ids per resample).
    _present = np.zeros(n_songs + 1, dtype=bool)

    def _coverage(mat, idx):
        _present[:] = False
        _present[mat[idx].ravel()] = True
        return float(_present[:n_songs].sum()) / n_songs

    # Full-sample point estimates (what the benchmark table reports).
    full_idx = np.arange(n_u)
    full_cov = {n: _coverage(topk_mat[n], full_idx) for n in models}
    full_overall = {n: w_ndcg * ndcg[n].mean() + w_cov * full_cov[n]
                    + w_pop * antipop[n].mean() for n in models}

    # Bootstrap distributions of Coverage@k and Overall_Score@k (paired resamples).
    # Indices are drawn one resample at a time — materialising all (n_boot × n_u)
    # at once would cost gigabytes on the full ~285k-user test set.
    boot_cov = {n: np.empty(n_boot) for n in models}
    boot_overall = {n: np.empty(n_boot) for n in models}
    for b in range(n_boot):
        idx = rng.integers(0, n_u, size=n_u) if n_u > 0 else np.empty(0, dtype=int)
        for n in models:
            cov_b = _coverage(topk_mat[n], idx)
            boot_cov[n][b] = cov_b
            boot_overall[n][b] = (w_ndcg * ndcg[n][idx].mean() + w_cov * cov_b
                                  + w_pop * antipop[n][idx].mean())

    def _boot_p(da):
        # Two-sided bootstrap p: 2·min(share ≤0, share ≥0), clamped to [1/n_boot, 1].
        if len(da) == 0:
            return np.nan
        p = 2.0 * min(np.mean(da <= 0), np.mean(da >= 0))
        return float(min(1.0, max(p, 1.0 / len(da))))

    rows: List[dict] = []
    for a, b in combinations(models, 2):
        # ── per-user Wilcoxon rows ────────────────────────────────────────────
        for metric, vec in (("NDCG@K", ndcg), (f"AntiPopularity@{k}", antipop)):
            xa, xb = vec[a], vec[b]
            diff = xa - xb
            try:
                _, w_p = stats.wilcoxon(xa, xb, zero_method="wilcox",
                                        alternative="two-sided")
            except ValueError:
                w_p = np.nan
            rows.append({
                "metric": metric, "model_A": a, "model_B": b, "n": n_u,
                "mean_A": float(xa.mean()), "mean_B": float(xb.mean()),
                "mean_diff": float(diff.mean()), "effect": _cohens_dz(diff),
                "ci_low": np.nan, "ci_high": np.nan, "test": "wilcoxon",
                "p_value": float(w_p) if w_p is not None else np.nan,
                "winner": a if diff.mean() > 0 else b,
            })
        # ── set-level / composite bootstrap rows ──────────────────────────────
        for metric, dist, full in ((f"Coverage@{k}", boot_cov, full_cov),
                                    (f"Overall_Score@{k}", boot_overall, full_overall)):
            da = dist[a] - dist[b]
            lo, hi = (np.percentile(da, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                      if len(da) else (np.nan, np.nan))
            rows.append({
                "metric": metric, "model_A": a, "model_B": b, "n": n_u,
                "mean_A": float(full[a]), "mean_B": float(full[b]),
                "mean_diff": float(full[a] - full[b]), "effect": np.nan,
                "ci_low": float(lo), "ci_high": float(hi),
                "test": f"bootstrap(B={n_boot})", "p_value": _boot_p(da),
                "winner": a if full[a] > full[b] else b,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant"] = out["p_value"] < alpha
    return out


def cosine_mean_comparison(
    pop_qual_dfs: Mapping[str, pd.DataFrame],
    *,
    ref_model: str = None,
    value_col: str = "cos_mean",
    user_col: str = "u_idx",
    alpha: float = 0.05,
    only_reference_miss: bool = False,
    hits_col: str = "n_hits",
) -> pd.DataFrame:
    """Is the HGT's recommendation–profile cosine higher than the other models'?

    Answers the qualitative question directly: for the *same* users, does the
    reference model (default: whichever model name contains "hgt") place
    recommendations with a higher per-user cosine-to-profile (``cos_mean`` from
    :func:`models.evaluation.qualitative.analyze_population`) than each other
    model? Because every model is scored on the same shared users this is a
    *paired* comparison — paired Wilcoxon signed-rank (two-sided **and** a
    one-sided *ref > other* test for the directional claim) + paired t-test,
    Cohen's dz, and ``pct_ref_higher`` (the share of users where the HGT's mean
    cosine is strictly higher, i.e. how often the tendency holds).

    Set ``only_reference_miss=True`` to restrict to the users where the reference
    retrieved **no** held-out track (``hits_col == 0``) — the "even when the HGT
    misses, is its cosine alignment still higher?" slice.

    Returns a long DataFrame ``[ref, other, scope, n, mean_ref, mean_other,
    mean_diff, pct_ref_higher, cohens_dz, wilcoxon_p, wilcoxon_greater_p,
    ttest_p, winner, significant]`` with one row per (ref, other) pair, sorted
    worst→best p-value.
    """
    models = list(pop_qual_dfs)
    if ref_model is None:
        ref_model = next((n for n in models if "hgt" in n.lower()), None)
    if ref_model is None or ref_model not in models:
        raise ValueError(f"reference model not found among {models}")

    frames = {n: pop_qual_dfs[n].set_index(user_col)[value_col].astype(float)
              for n in models if value_col in pop_qual_dfs[n].columns}
    # Optional "reference misses" slice: keep only users where the HGT scored 0 hits.
    ref_idx = frames[ref_model].index
    scope = "all"
    if only_reference_miss:
        rdf = pop_qual_dfs[ref_model].set_index(user_col)
        if hits_col not in rdf.columns:
            raise KeyError(f"only_reference_miss=True needs a '{hits_col}' column.")
        ref_idx = rdf.index[rdf[hits_col] == 0]
        scope = "reference_miss"

    rows: List[dict] = []
    ref_s = frames[ref_model]
    for other in models:
        if other == ref_model or other not in frames:
            continue
        oth_s = frames[other]
        common = ref_idx.intersection(ref_s.index).intersection(oth_s.index)
        xa, xb = ref_s.loc[common].values, oth_s.loc[common].values
        ok = ~(np.isnan(xa) | np.isnan(xb))
        xa, xb = xa[ok], xb[ok]
        if len(xa) < 5:
            continue
        diff = xa - xb
        try:
            _, w_p = stats.wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_p = np.nan
        try:
            _, w_pg = stats.wilcoxon(xa, xb, zero_method="wilcox", alternative="greater")
        except ValueError:
            w_pg = np.nan
        try:
            _, t_p = stats.ttest_rel(xa, xb, nan_policy="omit")
        except ValueError:
            t_p = np.nan
        rows.append({
            "ref": ref_model, "other": other, "scope": scope, "n": int(len(xa)),
            "mean_ref": float(xa.mean()), "mean_other": float(xb.mean()),
            "mean_diff": float(diff.mean()),
            "pct_ref_higher": float(np.mean(diff > 0)),
            "cohens_dz": _cohens_dz(diff),
            "wilcoxon_p": float(w_p) if w_p is not None else np.nan,
            "wilcoxon_greater_p": float(w_pg) if w_pg is not None else np.nan,
            "ttest_p": float(t_p) if t_p is not None else np.nan,
            "winner": ref_model if diff.mean() > 0 else other,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant"] = out["wilcoxon_p"] < alpha
        out = out.sort_values("wilcoxon_p", ascending=False, na_position="first")
    return out


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


# ─── benchmark visualisations ─────────────────────────────────────────────────
# Metrics where a LOWER value is better (so the heatmap colour flips). Anything
# named "Anti…Popularity" is higher-is-better and explicitly excluded.
_LOWER_BETTER = ("popularitybias", "pop_bias", "popbias")


def _lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return any(t in m for t in _LOWER_BETTER) and "anti" not in m


def _p_to_stars(p) -> str:
    """``*** <.001, ** <.01, * <.05, ns`` (``""`` when p is missing)."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def plot_benchmark_heatmaps(
    agg: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    ncols: int = 3,
    save_path=None,
    title: str = "Benchmark — metric × model × K",
):
    """Grid of per-metric heatmaps (rows = model, cols = K), annotated with the raw
    value.

    Colour is min-max normalised **within each metric** (green = better, flipped
    for lower-is-better metrics such as PopularityBias) so every panel is readable
    on its own scale. ``agg`` is ``run_benchmark()["agg"]`` — a ``(model, K) ×
    metric`` frame. Saved to ``save_path`` (PNG) when given.
    """
    import matplotlib.pyplot as plt

    df = agg.copy()
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        raise ValueError("agg must be indexed by (model, K) — pass run_benchmark()['agg']")
    df.index = df.index.set_names(["model", "K"])
    metrics = list(metrics or df.columns)
    models = list(dict.fromkeys(df.index.get_level_values("model")))
    ks = sorted(dict.fromkeys(df.index.get_level_values("K")))

    n = len(metrics)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.0 + 2.6 * ncols, 1.2 + 0.9 * nrows * max(len(models), 1)),
        squeeze=False)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, metrics):
        M = df[metric].unstack("K").reindex(index=models, columns=ks)
        arr = M.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        span = (hi - lo) or 1.0
        norm = (arr - lo) / span
        if _lower_is_better(metric):
            norm = 1.0 - norm
        ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"@{k}" for k in ks], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_title(metric + ("  ↓lower better" if _lower_is_better(metric) else ""),
                     fontsize=9)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="#111")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_significance_bars(
    pairwise: pd.DataFrame,
    *,
    metrics: Optional[Sequence[str]] = None,
    model_order: Optional[Sequence[str]] = None,
    ncols: int = 2,
    save_path=None,
    title: str = "Per-model means + pairwise significance",
):
    """Per-model bar chart for each significance metric, with pairwise brackets and
    significance stars (``*** <.001, ** <.01, * <.05``; ``ns`` pairs are skipped).

    ``pairwise`` is ``run_benchmark()["pairwise"]`` (from
    :func:`overall_significance`): each row is one model pair for one metric, with
    ``mean_A``/``mean_B`` and ``p_value``/``significant``. Saved to ``save_path``
    (PNG) when given.
    """
    import matplotlib.pyplot as plt

    df = pairwise.copy()
    metrics = list(metrics or dict.fromkeys(df["metric"]))
    n = len(metrics)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.0 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, metrics):
        sub = df[df["metric"] == metric]
        vals: Dict[str, float] = {}
        for _, r in sub.iterrows():
            vals.setdefault(str(r["model_A"]), float(r["mean_A"]))
            vals.setdefault(str(r["model_B"]), float(r["mean_B"]))
        models = [m for m in (model_order or list(vals)) if m in vals]
        y = [vals[m] for m in models]
        idx = {m: i for i, m in enumerate(models)}
        xpos = np.arange(len(models))
        bars = ax.bar(xpos, y, color="#6baed6", edgecolor="#3a6f93")
        ax.set_xticks(xpos)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(str(metric), fontsize=10)
        for b, v in zip(bars, y):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=7)

        # significant pairs only, shorter brackets first to reduce crossings.
        sig = []
        for _, r in sub.iterrows():
            a, b = str(r["model_A"]), str(r["model_B"])
            if a not in idx or b not in idx:
                continue
            stars = _p_to_stars(r.get("p_value"))
            keep = bool(r.get("significant", stars not in ("", "ns")))
            if keep and stars not in ("", "ns"):
                sig.append((abs(idx[a] - idx[b]), idx[a], idx[b], stars))
        sig.sort()
        ymax = max(y) if y else 1.0
        step = (ymax * 0.09) or 0.05
        ax.set_ylim(0, (ymax * 1.05 + step * (len(sig) + 1)) if ymax > 0 else 1.0)
        for level, (_, x0, x1, stars) in enumerate(sig):
            x0, x1 = sorted((x0, x1))
            yb = ymax * 1.04 + step * level
            ax.plot([x0, x0, x1, x1],
                    [yb, yb + step * 0.3, yb + step * 0.3, yb], lw=1.0, color="#444")
            ax.text((x0 + x1) / 2, yb + step * 0.32, stars, ha="center",
                    va="bottom", fontsize=8, color="#222")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(title + "   (*** p<.001  ** p<.01  * p<.05)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
