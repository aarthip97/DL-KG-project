"""Checkpoint I/O and config resolution for the pipeline notebook (§8).

Lifts the inline save/load + best-param resolution out of the HGT training cells
so they read as ``resolve config → train → persist``. Pure orchestration glue —
no behaviour change, no retraining.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

#: The single model-selection criterion, recorded in every run's metadata.
SELECTION_CRITERION = (
    "Overall_Score@10 = 0.6*NDCG + 0.2*Coverage + 0.2*(1-PopularityBias)"
)


def resolve_best_params(
    defaults: Mapping[str, Any],
    *,
    in_memory: Optional[Mapping[str, Any]] = None,
    json_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Resolve the HGT config via the standard ladder: memory → JSON → defaults.

    Args:
        defaults: The documented default config (always the base layer).
        in_memory: ``BEST_HGT_PARAMS`` from a Phase-1 sweep run this session, if any.
        json_path: Path to ``hgt_best_params.json`` written by a previous sweep.
        verbose: Print which tier supplied the config.

    Returns:
        ``{**defaults, **best}`` with the highest-priority source that exists.
    """
    log = print if verbose else (lambda *_: None)
    cfg = dict(defaults)
    if in_memory:
        cfg.update(in_memory)
        log(f"[hgt] Phase-1 best config (in memory): {dict(in_memory)}")
        return cfg
    if json_path is not None and Path(json_path).exists():
        loaded = json.load(open(json_path))
        cfg.update(loaded)
        log(f"[hgt] Phase-1 best config ← {Path(json_path).name}")
        return cfg
    log("[hgt] no Phase-1 result found — using documented defaults")
    return cfg


def cache_pickle(obj: Any, path, *, force: bool = False,
                 verbose: bool = True, label: str = "object") -> Path:
    """Pickle ``obj`` to ``path`` unless it already exists (``force`` overrides)."""
    path = Path(path)
    if force or not path.exists():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        if verbose:
            print(f"[cache] {label} → {path.name}")
    return path


def load_pickle(path) -> Any:
    """Unpickle ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_hgt_run(
    result,
    *,
    model_path,
    result_path,
    history_csv,
    config: Mapping[str, Any],
    epochs_requested: int,
    verbose: bool = True,
) -> dict:
    """Persist a finished HGT run: weights + result + history CSV + metadata JSON.

    Mirrors the §8 final-run cell: ``model.pt`` (state_dict), ``<result>.pkl``
    (the full :class:`TrainResult`), ``<history>.csv`` (per-epoch), and a
    ``<history>_meta.json`` sidecar (config + best-val + test metrics + final/best
    loss). No experiment tracker.

    Returns the four written paths.
    """
    import torch

    model_path, result_path = Path(model_path), Path(result_path)
    history_csv = Path(history_csv)

    torch.save(result.model.state_dict(), model_path)
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    hist = pd.DataFrame(result.history)
    hist.to_csv(history_csv, index=False)

    meta: dict = {
        "model": "RecommenderHGT",
        "config": dict(config),
        "epochs_requested": int(epochs_requested),
        "n_epochs_recorded": len(result.history),
        "selection_criterion": SELECTION_CRITERION,
        "best_val": result.best_val,
        "test_metrics": result.test_metrics,
    }
    loss_cols = [c for c in hist.columns if "loss" in c.lower()]
    if loss_cols:
        lc = loss_cols[0]
        s = hist[lc].dropna()
        if len(s):
            meta["loss_column"] = lc
            meta["final_train_loss"] = float(s.iloc[-1])
            meta["best_train_loss"] = float(s.min())
    meta_path = history_csv.with_name(history_csv.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    if verbose:
        print(f"[SAVED] HGT → {model_path.name} + {result_path.name} "
              f"+ {history_csv.name} + {meta_path.name}")
    return {"model_path": model_path, "result_path": result_path,
            "history_csv": history_csv, "meta_path": meta_path}


def load_hgt_run(result_path):
    """Reload a saved :class:`TrainResult` pickle (``<result>.pkl``)."""
    return load_pickle(result_path)


def save_best_params(params: Mapping[str, Any], json_path, *, verbose: bool = True) -> Path:
    """Write the Phase-1 winning config to ``hgt_best_params.json``."""
    json_path = Path(json_path)
    with open(json_path, "w") as f:
        json.dump(dict(params), f, indent=2)
    if verbose:
        print(f"[SAVED] best params → {json_path.name}")
    return json_path


def load_best_params(json_path) -> Optional[dict]:
    """Read ``hgt_best_params.json`` if present, else ``None``."""
    json_path = Path(json_path)
    return json.load(open(json_path)) if json_path.exists() else None


__all__ = [
    "SELECTION_CRITERION",
    "resolve_best_params",
    "cache_pickle", "load_pickle",
    "save_hgt_run", "load_hgt_run",
    "save_best_params", "load_best_params",
]
