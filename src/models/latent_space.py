"""Latent-space analysis utilities for the trained HGT.

Pipeline
--------
1. :func:`extract_node_embeddings` — forward-pass the HGT once, return a
   dataframe with one row per node and its full embedding vector.
2. :func:`gmm_grid_search` — parallel Gaussian-Mixture-Model sweep over
   ``n_components × covariance_type``, selecting the model with the lowest
   **Bayesian Information Criterion**

   .. math::

      \\text{BIC} = -2\\,\\log L + p\\,\\log N

   (where :math:`p` is the number of free parameters and :math:`N` the sample
   size) — a parsimony-aware likelihood criterion.
3. :func:`umap_3d_project` — non-linear UMAP projection to 3-D using cosine
   distance for visualisation.
4. :func:`build_latent_plotly_figure` — interactive 3-D scatter coloured by
   node type, with hover information including the GMM ``Cluster_ID``.
5. :func:`plot_gmm_bic_curves` — matplotlib BIC curves for diagnostics.
"""
from __future__ import annotations

import itertools
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture


# ─── 1. Embedding extraction ──────────────────────────────────────────────────

def extract_node_embeddings(
    model: torch.nn.Module,
    data,
    device: str | torch.device,
) -> pd.DataFrame:
    """Run the trained HeteroGNN forward and return a tidy embedding table.

    Returns a DataFrame with columns ``Node_ID``, ``Node_Type``, ``Embedding``
    (a 1-D ``np.ndarray`` per row).
    """
    model.eval()
    with torch.no_grad():
        x_dict = {
            nt: data[nt].x.to(device) if data[nt].get("x") is not None else None
            for nt in data.node_types
        }
        edge_index_dict = {
            et: data[et].edge_index.to(device) for et in data.edge_types
        }
        emb_dict = model(x_dict, edge_index_dict)
        emb_dict = {k: v.detach().cpu().numpy() for k, v in emb_dict.items()}

    rows = [
        {"Node_ID": f"{nt}_{i}", "Node_Type": nt.capitalize(), "Embedding": emb}
        for nt, embs in emb_dict.items()
        for i, emb in enumerate(embs)
    ]
    return pd.DataFrame(rows)


def subsample_embeddings(
    df: pd.DataFrame, max_nodes: int, *, random_state: int = 42
) -> pd.DataFrame:
    """Uniformly down-sample if there are more rows than ``max_nodes``."""
    if len(df) <= max_nodes:
        return df
    rng = np.random.default_rng(random_state)
    keep = rng.choice(len(df), size=max_nodes, replace=False)
    return df.iloc[keep].reset_index(drop=True)


# ─── 2. GMM grid search ───────────────────────────────────────────────────────

def _fit_and_score_gmm(cv_type: str, n_components: int, X: np.ndarray):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cv_type,
        random_state=42,
        max_iter=200,
    )
    gmm.fit(X)
    return cv_type, n_components, float(gmm.bic(X)), gmm


def gmm_grid_search(
    X: np.ndarray,
    *,
    n_components_range: Iterable[int],
    cv_types: Sequence[str] = ("spherical", "tied", "diag", "full"),
    n_jobs: int = -1,
    verbose: int = 5,
) -> Tuple[GaussianMixture, dict, float, dict]:
    """Parallel BIC-driven sweep over (n_components, covariance_type).

    Returns
    -------
    best_gmm : fitted ``GaussianMixture`` with the lowest BIC.
    best_params : dict with keys ``n_components`` and ``covariance_type``.
    best_bic : float
    bic_matrix : dict ``{covariance_type: [bic for each n_components]}``.
    """
    n_components_range = list(n_components_range)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_and_score_gmm)(cv, n, X)
        for cv, n in itertools.product(cv_types, n_components_range)
    )

    bic_matrix = {cv: [] for cv in cv_types}
    best_bic, best_gmm, best_params = np.inf, None, {}
    for cv_type, n_components, bic, gmm in results:
        bic_matrix[cv_type].append(bic)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_params = {"n_components": n_components, "covariance_type": cv_type}
    return best_gmm, best_params, best_bic, bic_matrix


# ─── 3. UMAP projection ───────────────────────────────────────────────────────

def umap_3d_project(
    X: np.ndarray, *, random_state: int = 42, metric: str = "cosine"
) -> np.ndarray:
    """3-D UMAP projection (lazy import so umap-learn is optional)."""
    try:
        import umap
    except ImportError as e:
        raise ImportError("Install umap-learn:  pip install umap-learn") from e
    reducer = umap.UMAP(n_components=3, random_state=random_state, metric=metric)
    return reducer.fit_transform(X)


# ─── 4. Plotly visualisation ──────────────────────────────────────────────────

def build_latent_plotly_figure(
    df_latent: pd.DataFrame,
    best_params: Mapping[str, object],
    *,
    size_map: Mapping[str, int] | None = None,
    default_size: int = 5,
):
    """Interactive 3-D Plotly scatter coloured by ``Node_Type``."""
    import plotly.express as px

    size_map = dict(size_map or {
        "User": 3, "Track": 4, "Genre": 10,
        "Key": 10, "Mode": 10, "Instrument": 10,
    })

    fig = px.scatter_3d(
        df_latent,
        x="X", y="Y", z="Z",
        color="Node_Type",
        symbol="Node_Type",
        hover_name="Node_ID",
        hover_data=["Cluster_ID"],
        title=(f"Multimodal Latent Space — "
               f"{best_params['n_components']} clusters · "
               f"{best_params['covariance_type']} covariance"),
        opacity=0.8,
    )
    sizes = df_latent["Node_Type"].map(size_map).fillna(default_size).tolist()
    fig.update_traces(marker=dict(size=sizes, line=dict(width=0)))
    fig.update_layout(
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        legend_title="Node Type",
    )
    return fig


# ─── 5. BIC curve plot ────────────────────────────────────────────────────────

def plot_gmm_bic_curves(
    bic_matrix: Mapping[str, Sequence[float]],
    n_components_range: Sequence[int],
    best_params: Mapping[str, object],
    *,
    figsize: Tuple[int, int] = (10, 6),
    save_path=None,
):
    """Matplotlib BIC sweep figure (returns ``fig``)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for cv_type, bics in bic_matrix.items():
        ax.plot(n_components_range, bics, marker="o", linestyle="-",
                label=f"covariance: {cv_type}")
    ax.axvline(best_params["n_components"], color="k", linestyle="--",
               label=f"optimal K={best_params['n_components']}")
    ax.set_title("GMM hyper-parameter sweep — BIC (lower is better)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("BIC score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─── 6. Balanced subsampling (keep small node types visible) ──────────────────

def balanced_subsample(
    df: pd.DataFrame,
    *,
    max_per_type: Optional[int] = None,
    max_total: Optional[int] = None,
    type_col: str = "Node_Type",
    random_state: int = 42,
) -> pd.DataFrame:
    """Down-sample an embedding table without drowning rare node types.

    First caps every ``type_col`` group at ``max_per_type`` rows (so a handful of
    genre/key/mode nodes survive next to hundreds of thousands of users), then
    optionally caps the grand total at ``max_total``.  Both limits are optional.
    """
    rng = np.random.default_rng(random_state)
    out = df
    if max_per_type is not None:
        parts = []
        for _, grp in out.groupby(type_col, sort=False):
            if len(grp) > max_per_type:
                grp = grp.iloc[rng.choice(len(grp), max_per_type, replace=False)]
            parts.append(grp)
        out = pd.concat(parts)
    if max_total is not None and len(out) > max_total:
        out = out.iloc[rng.choice(len(out), max_total, replace=False)]
    return out.reset_index(drop=True)


# ─── 7. GMM with a Torch (GPU) backend + a unified BIC sweep ───────────────────

@dataclass
class GMMResult:
    """Outcome of :func:`fit_gmm_bic` — backend-agnostic.

    Attributes
    ----------
    best_params : ``{"n_components": k, "covariance_type": cv}`` of the lowest BIC.
    best_bic : the winning BIC.
    bic_matrix : ``{covariance_type: [bic per n_components]}`` for the BIC curves.
    labels : hard cluster assignment for every row of the fitted ``X``.
    means : ``(k, d)`` cluster centres **in the latent space** (UMAP-transformable).
    backend : human-readable note of which engine ran (torch-gpu / sklearn-cpu).
    """

    best_params: dict
    best_bic: float
    bic_matrix: dict
    labels: np.ndarray
    means: np.ndarray
    backend: str
    n_components_range: List[int] = field(default_factory=list)
    cv_types: Tuple[str, ...] = ()


def _gmm_n_params(k: int, d: int, cv_type: str) -> int:
    """Free-parameter count matching sklearn's ``GaussianMixture._n_parameters``."""
    if cv_type == "full":
        cov = k * d * (d + 1) // 2
    elif cv_type == "diag":
        cov = k * d
    elif cv_type == "spherical":
        cov = k
    else:
        raise ValueError(f"unsupported covariance_type {cv_type!r}")
    return int(cov + k * d + k - 1)        # cov + means + (weights - 1)


def _log_gaussian_torch(X, means, covs, cv_type, reg_covar):
    """Per-component log N(x | mu_k, Sigma_k) → ``[N, K]`` (loops K to save memory)."""
    n, d = X.shape
    k = means.shape[0]
    two_pi = d * math.log(2 * math.pi)
    if cv_type == "spherical":
        var = covs.clamp_min(reg_covar)                       # [K]
        sq = torch.cdist(X, means) ** 2                       # [N, K]
        log_det = d * torch.log(var)                          # [K]
        return -0.5 * (two_pi + log_det[None, :] + sq / var[None, :])
    out = X.new_empty(n, k)
    if cv_type == "diag":
        var = covs.clamp_min(reg_covar)                       # [K, D]
        log_det = torch.log(var).sum(1)                       # [K]
        for j in range(k):
            diff = X - means[j]
            out[:, j] = -0.5 * (two_pi + log_det[j]
                                + (diff * diff / var[j]).sum(1))
        return out
    # full
    eye = torch.eye(d, device=X.device, dtype=X.dtype)
    for j in range(k):
        cov = covs[j] + reg_covar * eye
        try:
            chol = torch.linalg.cholesky(cov)
        except Exception:                                     # noqa: BLE001
            chol = torch.linalg.cholesky(cov + 1e-3 * eye)
        diff = (X - means[j]).T                               # [D, N]
        sol = torch.linalg.solve_triangular(chol, diff, upper=False)
        maha = (sol * sol).sum(0)                             # [N]
        log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
        out[:, j] = -0.5 * (two_pi + log_det + maha)
    return out


def _kmeans_init_torch(X, k: int, gen, *, n_iter: int = 10):
    """k-means++ seeding + a few Lloyd iterations — matches sklearn's GMM init
    quality so the EM lands in a comparable optimum (random seeding does not)."""
    n = X.shape[0]
    first = torch.randint(0, n, (1,), generator=gen, device=X.device)
    means = X[first].clone()                                  # [1, D]
    d2 = ((X - means[0]) ** 2).sum(1)                         # [N]
    for _ in range(1, k):
        probs = d2 / d2.sum().clamp_min(1e-12)
        nxt = torch.multinomial(probs, 1, generator=gen)
        means = torch.cat([means, X[nxt]], dim=0)
        d2 = torch.minimum(d2, ((X - X[nxt][0]) ** 2).sum(1))
    for _ in range(n_iter):                                   # Lloyd refinement
        assign = torch.cdist(X, means).argmin(1)
        for j in range(k):
            sel = assign == j
            if bool(sel.any()):
                means[j] = X[sel].mean(0)
    return means


def _gmm_em_torch(
    X, k: int, cv_type: str, *, device, max_iter: int = 200,
    tol: float = 1e-3, reg_covar: float = 1e-6, seed: int = 42, n_init: int = 1,
):
    """A single (k, cv_type) GMM fit by EM on ``device``. Returns (bic, labels, means)."""
    n, d = X.shape
    best = None
    for init in range(n_init):
        gen = torch.Generator(device=device).manual_seed(seed + init)
        means = _kmeans_init_torch(X, k, gen)
        weights = torch.full((k,), 1.0 / k, device=device, dtype=X.dtype)
        if cv_type == "full":
            covs = torch.eye(d, device=device, dtype=X.dtype).expand(k, d, d).clone()
        elif cv_type == "diag":
            covs = torch.ones(k, d, device=device, dtype=X.dtype)
        else:
            covs = torch.ones(k, device=device, dtype=X.dtype)

        prev = None
        for _ in range(max_iter):
            log_prob = _log_gaussian_torch(X, means, covs, cv_type, reg_covar)
            lw = log_prob + torch.log(weights.clamp_min(1e-38))
            lnorm = torch.logsumexp(lw, dim=1, keepdim=True)
            resp = (lw - lnorm).exp()                         # [N, K]
            nk = resp.sum(0).clamp_min(1e-10)                 # [K]
            weights = nk / n
            means = (resp.t() @ X) / nk[:, None]
            if cv_type == "spherical":
                sq = torch.cdist(X, means) ** 2
                covs = ((resp * sq).sum(0) / (nk * d)).clamp_min(reg_covar)
            elif cv_type == "diag":
                covs = torch.empty(k, d, device=device, dtype=X.dtype)
                for j in range(k):
                    diff = X - means[j]
                    covs[j] = (resp[:, j:j + 1] * diff * diff).sum(0) / nk[j] + reg_covar
            else:
                covs = torch.empty(k, d, d, device=device, dtype=X.dtype)
                for j in range(k):
                    diff = X - means[j]
                    covs[j] = (resp[:, j:j + 1] * diff).t() @ diff / nk[j]
            ll = float(lnorm.sum())
            if prev is not None and abs(ll - prev) <= tol * (abs(prev) + 1e-12):
                break
            prev = ll

        log_prob = _log_gaussian_torch(X, means, covs, cv_type, reg_covar)
        lw = log_prob + torch.log(weights.clamp_min(1e-38))
        lnorm = torch.logsumexp(lw, dim=1, keepdim=True)
        ll = float(lnorm.sum())
        bic = -2.0 * ll + _gmm_n_params(k, d, cv_type) * math.log(n)
        labels = (lw - lnorm).argmax(1)
        if best is None or bic < best[0]:
            best = (bic, labels.detach().cpu().numpy(),
                    means.detach().cpu().numpy())
    return best


def fit_gmm_bic(
    X: np.ndarray,
    *,
    n_components_range: Iterable[int],
    cv_types: Sequence[str] = ("full", "diag", "spherical"),
    device: str = "auto",
    n_jobs: int = -1,
    max_iter: int = 200,
    reg_covar: float = 1e-6,
    seed: int = 42,
    verbose: int = 5,
) -> GMMResult:
    """BIC-driven GMM sweep over ``n_components × covariance_type``.

    Picks the engine automatically: a **GPU** Torch EM when a CUDA device is
    available (``device="auto"``) or when ``device`` names a CUDA device, else
    the **multi-worker** sklearn sweep (joblib, ``n_jobs``).  Both paths return
    the identical :class:`GMMResult`, so callers don't care which ran.

    The covariance geometries default to the three requested for this project —
    ``full`` (per-cluster ellipsoid), ``diag`` (axis-aligned) and ``spherical``
    (isotropic).
    """
    X = np.asarray(X, dtype=np.float32)
    n_components_range = list(n_components_range)
    cv_types = tuple(cv_types)
    resolved = (("cuda" if torch.cuda.is_available() else "cpu")
                if device == "auto" else str(device))

    bic_matrix: dict = {cv: [] for cv in cv_types}
    best = (np.inf, None, None, None)            # (bic, params, labels, means)

    if resolved.startswith("cuda"):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=resolved)
        for cv in cv_types:
            for n in n_components_range:
                bic, labels, means = _gmm_em_torch(
                    Xt, n, cv, device=resolved, max_iter=max_iter,
                    reg_covar=reg_covar, seed=seed)
                bic_matrix[cv].append(bic)
                if bic < best[0]:
                    best = (bic, {"n_components": n, "covariance_type": cv},
                            labels, means)
        backend = f"torch-gpu ({resolved})"
    else:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_fit_and_score_gmm)(cv, n, X)
            for cv in cv_types for n in n_components_range
        )
        for cv_type, n_components, bic, gmm in results:
            bic_matrix[cv_type].append(bic)
            if bic < best[0]:
                best = (bic, {"n_components": n_components, "covariance_type": cv_type},
                        gmm.predict(X), gmm.means_)
        backend = f"sklearn-cpu (n_jobs={n_jobs})"

    return GMMResult(
        best_params=best[1], best_bic=float(best[0]), bic_matrix=bic_matrix,
        labels=np.asarray(best[2]), means=np.asarray(best[3]), backend=backend,
        n_components_range=n_components_range, cv_types=cv_types,
    )


# ─── 8. UMAP projection (2-D/3-D) + 2-D cluster plot ──────────────────────────

def umap_project(
    X: np.ndarray,
    *,
    n_components: int = 2,
    return_reducer: bool = False,
    metric: str = "cosine",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """UMAP projection to ``n_components`` dims (lazy import, umap-learn optional).

    When ``return_reducer`` is set, also returns the fitted reducer so cluster
    centres (or any new points) can be placed in the *same* space via
    ``reducer.transform(...)``.
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("Install umap-learn:  pip install umap-learn") from e
    reducer = umap.UMAP(n_components=n_components, metric=metric,
                        n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=random_state)
    coords = reducer.fit_transform(X)
    return (coords, reducer) if return_reducer else coords


def plot_latent_2d(
    coords: np.ndarray,
    node_types: Sequence[str],
    *,
    labels: Optional[np.ndarray] = None,
    centers_2d: Optional[np.ndarray] = None,
    best_params: Optional[Mapping[str, object]] = None,
    figsize: Tuple[int, int] = (15, 6),
    point_size: int = 5,
    alpha: float = 0.5,
):
    """2-D latent scatter: one panel coloured by node type, one by GMM cluster.

    ``centers_2d`` (the GMM means passed through the UMAP reducer) are drawn as
    labelled black crosses so each cluster's location is explicit.
    """
    import matplotlib.pyplot as plt

    node_types = np.asarray(node_types)
    ncols = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    axes = axes[0]

    ax = axes[0]
    for nt in pd.unique(node_types):
        m = node_types == nt
        ax.scatter(coords[m, 0], coords[m, 1], s=point_size, alpha=alpha, label=str(nt))
    ax.set_title("Latent space — by node type")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8, loc="best")

    if labels is not None:
        ax = axes[1]
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=point_size,
                   alpha=alpha, cmap="tab20")
        if centers_2d is not None:
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X",
                       s=140, edgecolor="white", linewidths=1.0, zorder=5)
            for i, (cx, cy) in enumerate(centers_2d):
                ax.annotate(str(i), (cx, cy), fontsize=8, fontweight="bold",
                            ha="center", va="center", color="white", zorder=6)
        title = "Latent space — GMM clusters"
        if best_params:
            title += (f"  (K={best_params['n_components']}, "
                      f"{best_params['covariance_type']})")
        ax.set_title(title)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    fig.tight_layout()
    return fig


# ─── 9. Persisting the analysis (so a restart needs no forward pass) ───────────

def assign_nearest_cluster(emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Hard-assign every embedding to its nearest GMM centroid (squared L2).

    A cheap MAP proxy for *all* nodes — the GMM is fit on a balanced subsample,
    so this extends its cluster identification to the full node set in the same
    (Euclidean) geometry the mixture was estimated in.
    """
    emb = np.asarray(emb, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)
    d2 = (emb ** 2).sum(1, keepdims=True) - 2.0 * emb @ C.T + (C ** 2).sum(1)[None, :]
    return d2.argmin(1).astype(np.int64)


def save_latent_analysis(
    out_dir,
    *,
    emb_df: pd.DataFrame,
    gmm: "GMMResult",
    centers_2d: Optional[np.ndarray] = None,
    subsample: Optional[pd.DataFrame] = None,
    composition: Optional[pd.DataFrame] = None,
    k_range: Optional[Iterable[int]] = None,
    gmm_users: Optional["GMMResult"] = None,
    user_composition: Optional[pd.DataFrame] = None,
    user_subsample: Optional[pd.DataFrame] = None,
    user_centers_2d: Optional[np.ndarray] = None,
) -> Path:
    """Persist a full latent-space analysis under ``out_dir``.

    Writes, all reload-able without the model or a forward pass:

    * ``node_embeddings.npz`` — every node's 64-d embedding + ``Node_ID`` /
      ``Node_Type`` (the heavy, reusable artefact).
    * ``gmm.pkl`` — the :class:`GMMResult` (params, BIC metrics, centroids, hard
      labels) plus the UMAP-projected ``centers_2d`` and the swept ``k_range``.
    * ``gmm_bic.csv`` — tidy ``(covariance_type, n_components, bic)`` metrics.
    * ``nodes_clustered.parquet`` — every node's nearest-centroid ``Cluster``
      (cluster identification for the full node set).
    * ``subsample_umap.parquet`` — the plotted subsample with its exact GMM
      ``Cluster_ID`` + 2-D UMAP ``X``/``Y``.
    * ``cluster_composition.csv`` — the ``cluster × Node_Type`` count table.
    * ``gmm_users.pkl`` — *(optional)* a second :class:`GMMResult` from a
      **users-only** sweep (listener-archetype centroids) plus its node-type
      composition and the UMAP-projected ``centers_2d``, written only when
      ``gmm_users`` is supplied.
    * ``users_umap.parquet`` — *(optional)* the plotted users-only subsample with
      its exact GMM ``Cluster_ID`` + 2-D UMAP ``X``/``Y`` (so the archetype
      scatter redraws on reload without a forward pass or a fresh projection).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.vstack(emb_df["Embedding"].to_numpy()).astype(np.float32)
    np.savez_compressed(
        out_dir / "node_embeddings.npz",
        node_id=emb_df["Node_ID"].to_numpy().astype(str),
        node_type=emb_df["Node_Type"].to_numpy().astype(str),
        emb=emb,
    )

    krange = list(k_range) if k_range is not None else list(gmm.n_components_range)
    with open(out_dir / "gmm.pkl", "wb") as f:
        pickle.dump({"gmm": gmm,
                     "centers_2d": None if centers_2d is None else np.asarray(centers_2d),
                     "k_range": krange}, f)

    bic_rows = [
        {"covariance_type": cv, "n_components": k, "bic": float(b)}
        for cv, bics in gmm.bic_matrix.items()
        for k, b in zip(krange, bics)
    ]
    pd.DataFrame(bic_rows).to_csv(out_dir / "gmm_bic.csv", index=False)

    pd.DataFrame({
        "Node_ID": emb_df["Node_ID"].to_numpy(),
        "Node_Type": emb_df["Node_Type"].to_numpy(),
        "Cluster": assign_nearest_cluster(emb, gmm.means),
    }).to_parquet(out_dir / "nodes_clustered.parquet", index=False)

    if subsample is not None:
        cols = [c for c in ("Node_ID", "Node_Type", "Cluster_ID", "X", "Y")
                if c in subsample.columns]
        subsample[cols].to_parquet(out_dir / "subsample_umap.parquet", index=False)
    if composition is not None:
        composition.to_csv(out_dir / "cluster_composition.csv")
    if gmm_users is not None:
        with open(out_dir / "gmm_users.pkl", "wb") as f:
            pickle.dump({"gmm": gmm_users,
                         "composition": (None if user_composition is None
                                         else user_composition),
                         "centers_2d": (None if user_centers_2d is None
                                        else np.asarray(user_centers_2d))}, f)
    if user_subsample is not None:
        cols = [c for c in ("Node_ID", "Node_Type", "Cluster_ID", "X", "Y")
                if c in user_subsample.columns]
        user_subsample[cols].to_parquet(out_dir / "users_umap.parquet", index=False)
    return out_dir


def load_latent_analysis(out_dir) -> dict:
    """Reload what :func:`save_latent_analysis` wrote.

    Returns a dict with ``emb_df`` (full embeddings), ``gmm`` (:class:`GMMResult`),
    ``centers_2d``, ``k_range`` and — when present — ``subsample``,
    ``composition``, ``nodes_clustered``, and the users-only ``gmm_users`` +
    ``user_composition`` + ``user_centers_2d`` + ``user_subsample``.
    """
    out_dir = Path(out_dir)
    npz = np.load(out_dir / "node_embeddings.npz", allow_pickle=False)
    emb_df = pd.DataFrame({
        "Node_ID": npz["node_id"].astype(str),
        "Node_Type": npz["node_type"].astype(str),
        "Embedding": list(npz["emb"].astype(np.float32)),
    })
    with open(out_dir / "gmm.pkl", "rb") as f:
        blob = pickle.load(f)
    out = {"emb_df": emb_df, "gmm": blob["gmm"],
           "centers_2d": blob.get("centers_2d"), "k_range": blob.get("k_range")}
    for key, fname, kw in (
        ("subsample", "subsample_umap.parquet", {}),
        ("nodes_clustered", "nodes_clustered.parquet", {}),
    ):
        p = out_dir / fname
        if p.exists():
            out[key] = pd.read_parquet(p, **kw)
    p = out_dir / "cluster_composition.csv"
    if p.exists():
        out["composition"] = pd.read_csv(p, index_col=0)
    p = out_dir / "gmm_users.pkl"
    if p.exists():
        with open(p, "rb") as f:
            blob_u = pickle.load(f)
        out["gmm_users"] = blob_u["gmm"]
        out["user_composition"] = blob_u.get("composition")
        out["user_centers_2d"] = blob_u.get("centers_2d")
    p = out_dir / "users_umap.parquet"
    if p.exists():
        out["user_subsample"] = pd.read_parquet(p)
    return out


# ─── high-level orchestration (used by the §13 notebook cell) ─────────────────

@dataclass
class LatentAnalysis:
    """A full latent-space analysis: all-type GMM + users-only GMM + 2-D layouts.

    Bundles everything the §13 cell produces / reloads so the notebook can
    compute, persist, reload and redraw it through one object.
    """
    emb_df: pd.DataFrame
    gmm: "GMMResult"
    centers_2d: Optional[np.ndarray] = None
    subsample: Optional[pd.DataFrame] = None
    composition: Optional[pd.DataFrame] = None
    k_range: Optional[Iterable[int]] = None
    gmm_users: Optional["GMMResult"] = None
    user_composition: Optional[pd.DataFrame] = None
    user_subsample: Optional[pd.DataFrame] = None
    user_centers_2d: Optional[np.ndarray] = None

    def save(self, out_dir) -> Path:
        """Persist via :func:`save_latent_analysis` (reloadable without a model)."""
        return save_latent_analysis(
            out_dir, emb_df=self.emb_df, gmm=self.gmm, centers_2d=self.centers_2d,
            subsample=self.subsample, composition=self.composition,
            k_range=self.k_range, gmm_users=self.gmm_users,
            user_composition=self.user_composition,
            user_subsample=self.user_subsample, user_centers_2d=self.user_centers_2d)

    @classmethod
    def load(cls, out_dir) -> "LatentAnalysis":
        """Reload a persisted analysis (no model, no forward pass)."""
        a = load_latent_analysis(out_dir)
        return cls(
            emb_df=a["emb_df"], gmm=a["gmm"], centers_2d=a.get("centers_2d"),
            subsample=a.get("subsample"), composition=a.get("composition"),
            k_range=a.get("k_range"), gmm_users=a.get("gmm_users"),
            user_composition=a.get("user_composition"),
            user_subsample=a.get("user_subsample"),
            user_centers_2d=a.get("user_centers_2d"))


def compute_latent_analysis(
    model,
    data,
    *,
    device: str = "cpu",
    k_range: Iterable[int] = range(2, 21),
    cv_types: Sequence[str] = ("full", "diag", "spherical"),
    max_per_type: int = 5000,
    max_total: int = 40000,
    verbose: bool = True,
) -> LatentAnalysis:
    """Fit the all-type and users-only GMMs on the FULL node set + 2-D UMAP views.

    The GMMs are fit on every node (no subsampling); the subsamples are only a
    readable view for the UMAP scatter, coloured by the full-fit labels. Uses the
    training-consistent undirected forward pass — pure inference, no retraining.
    """
    import torch_geometric.transforms as T

    log = print if verbose else (lambda *_: None)
    krange = list(k_range)

    data_u = T.ToUndirected(merge=False)(data)
    emb_df = extract_node_embeddings(model, data_u, device)
    log("Nodes per type:")
    log(emb_df["Node_Type"].value_counts().to_string())

    X = np.vstack(emb_df["Embedding"].values).astype(np.float32)
    log(f"\nClustering ALL {X.shape[0]:,} nodes x {X.shape[1]} dims (full set, no subsampling)")
    gmm = fit_gmm_bic(X, n_components_range=krange, cv_types=cv_types,
                      device="auto", n_jobs=-1)
    log(f"[GMM] backend={gmm.backend}  best={gmm.best_params}  BIC={gmm.best_bic:,.0f}")
    emb_df = emb_df.assign(Cluster_ID=gmm.labels)

    emb_s = balanced_subsample(emb_df, max_per_type=max_per_type, max_total=max_total)
    coords, reducer = umap_project(
        np.vstack(emb_s["Embedding"].values).astype(np.float32),
        n_components=2, return_reducer=True)
    centers_2d = reducer.transform(gmm.means)
    emb_s = emb_s.assign(X=coords[:, 0], Y=coords[:, 1])
    comp = pd.crosstab(emb_df["Cluster_ID"], emb_df["Node_Type"])

    # users-only sweep → listener-archetype centroids (drive the §14 cold-start)
    u_all = emb_df[emb_df["Node_Type"].str.lower() == "user"].copy()
    Xu = np.vstack(u_all["Embedding"].values).astype(np.float32)
    log(f"\nClustering ALL {Xu.shape[0]:,} USER nodes x {Xu.shape[1]} dims (full users-only set)")
    gmm_users = fit_gmm_bic(Xu, n_components_range=krange, cv_types=cv_types,
                            device="auto", n_jobs=-1)
    log(f"[GMM-users] backend={gmm_users.backend}  best={gmm_users.best_params}  "
        f"BIC={gmm_users.best_bic:,.0f}")
    u_all = u_all.assign(Cluster_ID=gmm_users.labels)
    user_comp = pd.crosstab(u_all["Cluster_ID"], u_all["Node_Type"])
    u_s = subsample_embeddings(u_all, max_total)
    ucoords, ureducer = umap_project(
        np.vstack(u_s["Embedding"].values).astype(np.float32),
        n_components=2, return_reducer=True)
    ucenters_2d = ureducer.transform(gmm_users.means)
    u_s = u_s.assign(X=ucoords[:, 0], Y=ucoords[:, 1])

    return LatentAnalysis(
        emb_df=emb_df, gmm=gmm, centers_2d=centers_2d, subsample=emb_s,
        composition=comp, k_range=krange, gmm_users=gmm_users,
        user_composition=user_comp, user_subsample=u_s, user_centers_2d=ucenters_2d)


def resolve_latent_analysis(
    *,
    latent_dir,
    model=None,
    data=None,
    device: str = "cpu",
    force: bool = False,
    in_memory: Optional[LatentAnalysis] = None,
    persist: bool = True,
    verbose: bool = True,
    **compute_kwargs,
) -> Optional[LatentAnalysis]:
    """Get the latent analysis via the no-recompute ladder: memory → compute → disk.

    Priority mirrors the §13 cell: reuse an in-memory :class:`LatentAnalysis`
    (re-persisting it); else compute from the live model when ``force`` or no
    cache exists; else reload the cached analysis; else compute if a model is
    available. Returns ``None`` when neither a model nor a cache is available.
    """
    latent_dir = Path(latent_dir)
    disk = ((latent_dir / "node_embeddings.npz").exists()
            and (latent_dir / "gmm.pkl").exists())
    log = print if verbose else (lambda *_: None)
    have_model = model is not None and data is not None

    def _compute_and_persist():
        a = compute_latent_analysis(model, data, device=device, verbose=verbose,
                                    **compute_kwargs)
        if persist:
            a.save(latent_dir)
            log(f"[latent] persisted → {latent_dir}/")
        return a

    if not force and isinstance(in_memory, LatentAnalysis):
        if persist:
            in_memory.save(latent_dir)
            log(f"[latent] persisted in-memory results → {latent_dir}/")
        return in_memory
    if force and have_model:
        return _compute_and_persist()
    if disk:
        a = LatentAnalysis.load(latent_dir)
        log(f"[latent] reloaded ← {latent_dir}/ (best={a.gmm.best_params}; "
            "no forward pass)")
        return a
    if have_model:
        return _compute_and_persist()
    return None


def show_latent_clusters(analysis: LatentAnalysis, *, bic_save_path=None) -> None:
    """Draw the all-type BIC curves + 2-D UMAP scatter + composition table."""
    import matplotlib.pyplot as plt

    g = analysis.gmm
    krange = list(analysis.k_range or g.n_components_range)
    plot_gmm_bic_curves(g.bic_matrix, krange, g.best_params, save_path=bic_save_path)
    plt.show()
    s = analysis.subsample
    if s is not None and {"X", "Y", "Cluster_ID"} <= set(s.columns):
        plot_latent_2d(s[["X", "Y"]].to_numpy(), s["Node_Type"].values,
                       labels=s["Cluster_ID"].to_numpy(),
                       centers_2d=analysis.centers_2d, best_params=g.best_params)
        plt.show()
    if analysis.composition is not None:
        print("\nCluster x node-type composition (all nodes):")
        print(analysis.composition.to_string())


def show_user_archetypes(analysis: LatentAnalysis) -> None:
    """Draw the users-only BIC curves + archetype UMAP scatter + per-cluster counts."""
    import matplotlib.pyplot as plt

    g = analysis.gmm_users
    if g is None:
        return
    krange = list(analysis.k_range or g.n_components_range)
    plot_gmm_bic_curves(g.bic_matrix, krange, g.best_params)
    plt.show()
    s = analysis.user_subsample
    if s is not None and {"X", "Y", "Cluster_ID"} <= set(s.columns):
        plot_latent_2d(s[["X", "Y"]].to_numpy(), s["Node_Type"].values,
                       labels=s["Cluster_ID"].to_numpy(),
                       centers_2d=analysis.user_centers_2d, best_params=g.best_params)
        plt.show()
    if analysis.user_composition is not None:
        print("\nListener archetypes (users per cluster):")
        print(analysis.user_composition.to_string())


def plotly_3d_figure(analysis: LatentAnalysis):
    """Interactive 3-D UMAP scatter of the subsample, or ``None`` if unavailable.

    Needs the subsample's embeddings (present after a fresh compute, dropped on
    a disk reload), so it silently returns ``None`` when they are absent.
    """
    s = analysis.subsample
    if s is None or "Embedding" not in s.columns:
        return None
    X_sub = np.vstack(s["Embedding"].values).astype(np.float32)
    c3 = umap_project(X_sub, n_components=3)
    df3 = s.assign(X=c3[:, 0], Y=c3[:, 1], Z=c3[:, 2])
    return build_latent_plotly_figure(df3, analysis.gmm.best_params)
