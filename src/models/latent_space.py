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
from dataclasses import dataclass, field
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
