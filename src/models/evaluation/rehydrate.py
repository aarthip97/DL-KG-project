"""Reconstruct the baseline recommenders from their on-disk training artefacts.

After a kernel restart the in-memory recommender objects are gone, but the
*trained* artefacts each baseline produced are still on disk:

* **MostPopular** — needs only ``song_popularity`` + ``train_seen`` (both cheap
  to rebuild from the cached stratified splits).
* **KNN-CF** — the neighbour table (``torch.save`` cache with ``nbrs_tensor`` +
  ``all_query``) and the chosen ``best_k`` (in the KNN/pop summary JSON).
* **XGBoost-Hybrid** — the trained booster (``.ubj``), the AE embedding matrix
  (parquet) and the per-track categorical features, which are *deterministically*
  rebuilt from ``kg_input.parquet`` + the train split (``te_seed`` is fixed) — so
  no model is retrained.

:func:`rebuild_baselines_from_disk` assembles whichever of these it can, so the
final-benchmark cell can run standalone without re-executing the baseline
training cells.  Anything whose artefacts are missing is skipped with a note
rather than raising, so a partial set still lets the benchmark proceed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping, Optional, Set

import numpy as np

from .recommenders import KNNRecommender, PopularityRecommender, XGBHybridRecommender


def _exists(p) -> bool:
    return p is not None and Path(p).exists()


def rebuild_baselines_from_disk(
    *,
    train_seen: Mapping[int, Set[int]],
    song_popularity: np.ndarray,
    train_matrix_norm,
    n_songs: int,
    splits_dir,
    knn_nbrs_cache=None,
    knn_summary_json=None,
    xgb_model_cache=None,
    ae_embeddings_path=None,
    kg_input_path=None,
    xgb_n_candidates: int = 200,
    top_instruments: int = 20,
    verbose: bool = True,
) -> List:
    """Rebuild ``[MostPopular, KNN-CF, XGBoost-Hybrid]`` from cached artefacts.

    Only the recommenders whose artefacts are present are returned; none of them
    triggers training.  ``song_popularity`` / ``train_matrix_norm`` / ``train_seen``
    are expected to already be rebuilt from the cached splits by the caller.

    Returns the list of successfully reconstructed recommenders (MostPopular is
    always included as it has no separate artefact).
    """
    log = print if verbose else (lambda *_: None)
    recs: List = [PopularityRecommender(song_popularity, train_seen)]
    log("[rehydrate] MostPopular ← splits")

    # ── KNN-CF ────────────────────────────────────────────────────────────────
    if _exists(knn_nbrs_cache) and _exists(knn_summary_json):
        try:
            import torch  # noqa: PLC0415
            saved = torch.load(knn_nbrs_cache, map_location="cpu", weights_only=True)
            all_nbrs = saved["nbrs_tensor"].cpu().numpy()
            qrow = {int(u): i for i, u in enumerate(saved["all_query"].tolist())}
            best_k = int(json.loads(Path(knn_summary_json).read_text())["best_k"])
            recs.append(KNNRecommender(train_matrix_norm, train_seen,
                                       all_nbrs, qrow, best_k=best_k))
            log(f"[rehydrate] KNN-CF ← {Path(knn_nbrs_cache).name} (best_k={best_k})")
        except Exception as e:  # noqa: BLE001
            log(f"[rehydrate] KNN-CF skipped ({e.__class__.__name__}: {e})")
    else:
        log("[rehydrate] KNN-CF skipped (neighbours cache / summary JSON missing)")

    # ── XGBoost-Hybrid (booster reloaded, features rebuilt deterministically) ──
    if _exists(xgb_model_cache) and _exists(ae_embeddings_path) and _exists(kg_input_path):
        try:
            import pandas as pd  # noqa: PLC0415
            import xgboost as xgb  # noqa: PLC0415
            from ..xgb_hybrid import (  # noqa: PLC0415
                _load_ae_matrix, build_track_categorical_features,
            )
            splits_dir = Path(splits_dir)
            sid2sidx = (
                pd.concat([
                    pd.read_parquet(splits_dir / f"{s}.parquet", columns=["song_id", "s_idx"])
                    for s in ("train", "val", "test")
                ])
                .drop_duplicates("song_id")
                .set_index("song_id")["s_idx"]
                .to_dict()
            )
            train_df = pd.read_parquet(splits_dir / "train.parquet")
            merged = pd.read_parquet(kg_input_path)
            track_extra, _ = build_track_categorical_features(
                track_meta=merged, train_df=train_df, song_id_to_sidx=sid2sidx,
                n_songs=n_songs, top_instruments=top_instruments)
            ae_matrix, _ = _load_ae_matrix(ae_embeddings_path, sid2sidx, n_songs)
            booster = xgb.Booster()
            booster.load_model(str(xgb_model_cache))
            recs.append(XGBHybridRecommender(
                model=booster, ae_matrix=ae_matrix, train_seen=train_seen,
                n_candidates=xgb_n_candidates, track_extra=track_extra))
            log(f"[rehydrate] XGBoost-Hybrid ← {Path(xgb_model_cache).name} (no retrain)")
        except Exception as e:  # noqa: BLE001
            log(f"[rehydrate] XGBoost-Hybrid skipped ({e.__class__.__name__}: {e})")
    else:
        log("[rehydrate] XGBoost-Hybrid skipped (booster / AE / kg_input cache missing)")

    return recs


__all__ = ("rebuild_baselines_from_disk",)
