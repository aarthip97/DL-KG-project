"""Notebook orchestration glue for the KG→DL pipeline (§8–14).

Keeps the pipeline notebook's later sections thin: the rehydration ("self-heal")
ladders, checkpoint save/load, ablation harnesses, benchmark assembly and
qualitative case selection live here instead of inline. Everything is pure
orchestration over the core model code in :mod:`models` and
:mod:`models.evaluation` — no retraining, no metric changes.
"""
from __future__ import annotations

from .checkpoints import (
    SELECTION_CRITERION,
    resolve_best_params,
    cache_pickle, load_pickle,
    save_hgt_run, load_hgt_run,
    save_best_params, load_best_params,
)
from .ablation import (
    default_ablation_axes,
    run_phase1_ablation, run_init_ablation, run_direction_ablation,
)
from .benchmark import (
    EvalContext, RecommenderSet, to_kg_seen,
    load_eval_context, build_hgt_recommender,
    assemble_recommenders, run_benchmark,
)
from .qualitative_cases import ensure_qual_arrays, select_contrastive_cases
from .context import HGTContext, ensure_hgt_context
from .explain import UserExplainer, build_track_label_fn, build_user_explainer
from .personas import (
    assemble_persona_pack, profile_user_archetypes, build_cold_start)
from .training_inputs import HGTTrainingInputs, ensure_hgt_training_inputs
from .gpu import gpu_free_gb, suggest_user_batch_size

__all__ = [
    # checkpoints
    "SELECTION_CRITERION", "resolve_best_params",
    "cache_pickle", "load_pickle", "save_hgt_run", "load_hgt_run",
    "save_best_params", "load_best_params",
    # ablation
    "default_ablation_axes",
    "run_phase1_ablation", "run_init_ablation", "run_direction_ablation",
    # benchmark
    "EvalContext", "RecommenderSet", "to_kg_seen",
    "load_eval_context", "build_hgt_recommender",
    "assemble_recommenders", "run_benchmark",
    # qualitative
    "ensure_qual_arrays", "select_contrastive_cases",
    # context + explanation
    "HGTContext", "ensure_hgt_context",
    "UserExplainer", "build_track_label_fn", "build_user_explainer",
    # personas
    "assemble_persona_pack", "profile_user_archetypes", "build_cold_start",
    # training inputs (§8 standalone)
    "HGTTrainingInputs", "ensure_hgt_training_inputs",
    # gpu helpers
    "gpu_free_gb", "suggest_user_batch_size",
]
