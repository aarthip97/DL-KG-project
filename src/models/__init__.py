"""Deep-learning stack for the music recommender capstone.

Public API
----------
KG → graph
    :func:`extract_dl_artifacts`        — RDF graph → PyKEEN TSV + edge dict.
    :func:`build_rich_hetero_graph`     — edge dict + embeddings → HeteroData.

Audio autoencoder
    :class:`jSymbolicAutoencoder`       — dense jSymbolic feature compressor.
    :func:`train_autoencoder`           — AE training loop.
    :func:`extract_embeddings`          — encode a feature matrix with a trained AE.

KG embeddings
    :class:`KGEResult`                  — RotatE/ComplEx result container.
    :func:`train_kge`                   — train KGE model via PyKEEN.
    :func:`load_kge_checkpoint`         — reload embeddings from a saved checkpoint.

HGT model
    :class:`RecommenderHGT`             — Heterogeneous Graph Transformer.
    :class:`TrainResult`                — training result container.
    :func:`train_hgt`                   — HGT training loop with optional W&B.

Loss & ranking
    :func:`compute_log_pop_prior`       — log-popularity prior vector.
    :func:`debiased_listwise_loss`      — popularity-debiased cross-entropy loss.
    :func:`evaluate_top_k`             — Recall / NDCG / Hit-rate @ K.

KNN baseline
    :func:`run_knn_sweep`               — k-sweep + test eval (PyTorch KNN).
    :func:`_matrix_to_tensor`           — sparse/dense matrix → float32 tensor.
    :func:`_find_neighbors_torch`       — batched matmul cosine KNN.
    :func:`_build_recs_torch`           — gather+sum neighbour scoring.

Data splits
    :func:`make_stratified_splits`      — train/val/test with stratum balance.
    :func:`build_song_strata`           — song-level stratum labels.
    :func:`user_level_stratified_split` — per-user interaction split.
    :func:`save_splits` / :func:`load_splits` — parquet I/O for splits.
    :func:`compute_split_distributions` — distribution audit.
    :func:`plot_split_distributions`    — distribution audit plots.

Latent space analysis
    :func:`extract_node_embeddings`     — pull HGT node embeddings.
    :func:`subsample_embeddings`        — random subsample for plotting.
    :func:`gmm_grid_search`             — BIC-optimal GMM fit.
    :func:`umap_3d_project`             — 3-D UMAP projection.
    :func:`build_latent_plotly_figure`  — interactive Plotly scatter.
    :func:`plot_gmm_bic_curves`         — BIC / AIC curve plot.
"""

# ── KG → HeteroData ──────────────────────────────────────────────────────────
from .kg_to_hetero import (
    extract_dl_artifacts,
    build_rich_hetero_graph,
)

# ── Audio autoencoder ─────────────────────────────────────────────────────────
from .autoencoder import (
    jSymbolicAutoencoder,
    train_autoencoder,
    extract_embeddings,
)

# ── KG embeddings (RotatE / ComplEx via PyKEEN) ───────────────────────────────
from .kg_embeddings import (
    KGEResult,
    train_kge,
    load_kge_checkpoint,
)

# ── HGT model ─────────────────────────────────────────────────────────────────
from .hgt import RecommenderHGT
from .train_DL import train_hgt, TrainResult

# ── Loss & ranking utilities ──────────────────────────────────────────────────
from .loss import (
    compute_log_pop_prior,
    debiased_listwise_loss,
    evaluate_top_k,
)

# ── KNN collaborative-filtering baseline ─────────────────────────────────────
from .knn_cf import (
    run_knn_sweep,
    _matrix_to_tensor,
    _find_neighbors_torch,
    _build_recs_torch,
)

# ── Train / val / test splits ─────────────────────────────────────────────────
from .train_val_test_split import (
    make_stratified_splits,
    build_song_strata,
    user_level_stratified_split,
    save_splits,
    load_splits,
    compute_split_distributions,
    plot_split_distributions,
)

# ── Latent-space analysis ─────────────────────────────────────────────────────
from .latent_space import (
    extract_node_embeddings,
    subsample_embeddings,
    gmm_grid_search,
    umap_3d_project,
    build_latent_plotly_figure,
    plot_gmm_bic_curves,
)

__all__ = [
    # KG → graph
    "extract_dl_artifacts",
    "build_rich_hetero_graph",
    # Autoencoder
    "jSymbolicAutoencoder",
    "train_autoencoder",
    "extract_embeddings",
    # KG embeddings
    "KGEResult",
    "train_kge",
    "load_kge_checkpoint",
    # HGT
    "RecommenderHGT",
    "train_hgt",
    "TrainResult",
    # Loss & ranking
    "compute_log_pop_prior",
    "debiased_listwise_loss",
    "evaluate_top_k",
    # KNN baseline
    "run_knn_sweep",
    "_matrix_to_tensor",
    "_find_neighbors_torch",
    "_build_recs_torch",
    # Splits
    "make_stratified_splits",
    "build_song_strata",
    "user_level_stratified_split",
    "save_splits",
    "load_splits",
    "compute_split_distributions",
    "plot_split_distributions",
    # Latent space
    "extract_node_embeddings",
    "subsample_embeddings",
    "gmm_grid_search",
    "umap_3d_project",
    "build_latent_plotly_figure",
    "plot_gmm_bic_curves",
]
