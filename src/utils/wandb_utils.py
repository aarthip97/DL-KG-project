"""Weights & Biases convenience helpers used by the pipeline notebook."""
from __future__ import annotations

from typing import Mapping, Optional

import wandb


def log_baseline_run(
    model_name: str,
    metrics: Mapping[str, float],
    *,
    project: str,
    group: str,
    entity: Optional[str] = None,
    n_users: int,
    n_songs: int,
    top_n: int,
    extra_config: Optional[Mapping[str, object]] = None,
) -> str:
    """Create a tiny standalone W&B run logging the test metrics of a baseline.

    The baseline appears as a separate row in the project leaderboard, so its
    scalar metrics line up next to the deep-learning runs in the auto-generated
    bar charts. Returns the URL of the created run.

    Parameters
    ----------
    model_name : str
        Display name of the baseline (e.g. ``"MostPopular"``, ``"KNN-CF"``).
    metrics : mapping
        Final test metrics to log under the ``test/`` prefix.
    project, group, entity : str
        W&B project / group / entity (entity may be ``None``).
    n_users, n_songs, top_n : int
        Dataset descriptors recorded in ``run.config``.
    extra_config : mapping, optional
        Additional config entries (e.g. ``{"best_k": 25}``).
    """
    cfg = {
        "model_type": "baseline",
        "model_name": model_name,
        "n_users": n_users,
        "n_songs": n_songs,
        "top_n": top_n,
    }
    if extra_config:
        cfg.update(dict(extra_config))

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type="baseline",
        name=f"baseline_{model_name.lower()}",
        config=cfg,
        tags=["baseline", model_name.lower()],
        reinit=True,
    )
    wandb.log({f"test/{k}": float(v) for k, v in metrics.items()})
    for k, v in metrics.items():
        wandb.summary[f"test/{k}"] = float(v)
    wandb.summary["model_name"] = model_name
    wandb.finish()
    return run.url
