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
from typing import Iterable, Mapping, Sequence, Tuple

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
