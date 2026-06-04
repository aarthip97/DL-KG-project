"""Reusable evaluation toolkit for recommender systems.

Sub-modules
-----------
* :mod:`metrics`       — top-K accuracy, ranking, coverage and popularity-bias
  metrics + multi-K table builder.
* :mod:`recommenders`  — uniform :class:`Recommender` interface wrapping the
  Popularity, KNN-CF and HGT models.
* :mod:`qualitative`   — model-agnostic per-user and population qualitative
  analysis (genre / mode / tempo / decade alignment, JS-divergence …).
* :mod:`comparison`    — pairwise (paired t-test, Wilcoxon) + global
  (Friedman + Nemenyi post-hoc) statistical tests across models.
* :mod:`wandb_log`     — Weights & Biases helpers for logging the artefacts
  produced by the modules above.
"""
from __future__ import annotations

from .metrics import (
    dcg,
    evaluate_recs,
    evaluate_recs_per_user,
    overall_score,
    precision_at_k,
    multi_k_evaluation,
)
from .recommenders import (
    Recommender,
    PopularityRecommender,
    KNNRecommender,
    HGTRecommender,
    XGBHybridRecommender,
)
from .qualitative import analyze_user, analyze_population, AttributeArrays
from .comparison import (
    pairwise_significance,
    overall_significance,
    cosine_mean_comparison,
    friedman_nemenyi,
    summarise_comparison,
    plot_benchmark_heatmaps,
    plot_significance_bars,
)
from .explainability import (
    HGTExplainer,
    Explanation,
    Reason,
    EdgeAttention,
    capture_hgt_attention,
    FaithfulAttribution,
    faithful_attribution,
    faithful_attribution_ig,
    plot_edge_type_importance,
    plot_attention_vs_faithful,
    build_attribution_panels,
    plot_explanation_graphs,
)
from .training_panel import plot_hgt_panel, overall_at_k
from .rehydrate import (
    rebuild_baselines_from_disk,
    rebuild_hgt_recommender_from_disk,
    load_index_bridges_from_disk,
    load_song_meta,
    load_eval_ground_truth,
)

__all__ = [
    # metrics
    "dcg", "evaluate_recs", "evaluate_recs_per_user",
    "overall_score", "precision_at_k", "multi_k_evaluation",
    # recommenders
    "Recommender", "PopularityRecommender", "KNNRecommender", "HGTRecommender",
    "XGBHybridRecommender",
    # qualitative
    "analyze_user", "analyze_population", "AttributeArrays",
    # comparison
    "pairwise_significance", "overall_significance", "cosine_mean_comparison",
    "friedman_nemenyi", "summarise_comparison",
    "plot_benchmark_heatmaps", "plot_significance_bars",
    # explainability
    "HGTExplainer", "Explanation", "Reason", "EdgeAttention",
    "capture_hgt_attention",
    "FaithfulAttribution", "faithful_attribution", "faithful_attribution_ig",
    "plot_edge_type_importance", "plot_attention_vs_faithful",
    "build_attribution_panels", "plot_explanation_graphs",
    "plot_hgt_panel", "overall_at_k",
    # rehydration
    "rebuild_baselines_from_disk", "rebuild_hgt_recommender_from_disk",
    "load_index_bridges_from_disk", "load_song_meta", "load_eval_ground_truth",
]
