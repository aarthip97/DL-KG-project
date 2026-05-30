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
    friedman_nemenyi,
    summarise_comparison,
)
from .explainability import (
    HGTExplainer,
    Explanation,
    Reason,
    EdgeAttention,
    capture_hgt_attention,
)
from .rehydrate import rebuild_baselines_from_disk

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
    "pairwise_significance", "friedman_nemenyi", "summarise_comparison",
    # explainability
    "HGTExplainer", "Explanation", "Reason", "EdgeAttention",
    "capture_hgt_attention",
    # rehydration
    "rebuild_baselines_from_disk",
]
