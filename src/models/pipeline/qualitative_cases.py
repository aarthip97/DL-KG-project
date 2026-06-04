"""Qualitative-analysis glue for the pipeline notebook (§10, §11).

Two helpers lifted from the case-study cells:

* :func:`ensure_qual_arrays` — rebuild the attribute arrays / song-vector matrix
  used by the qualitative analysers when the §6.4 globals were wiped by a restart.
* :func:`select_contrastive_cases` — the model-CONTRASTIVE per-user case picker
  (hgt-win / hgt-edge / baseline-win / all-success / all-fail) that answers
  "what does the HGT add over a strong baseline?".
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def ensure_qual_arrays(
    train_matrix_norm,
    *,
    kg_input_path,
    splits_dir,
    song_meta=None,
    idx2song: Optional[Mapping[int, str]] = None,
    verbose: bool = True,
):
    """Rebuild ``(attrs, song_vectors, song_norms, idx2song, song_meta)`` from disk.

    ``idx2song`` / ``song_meta`` are reused if passed (warm kernel), else rebuilt
    from the cached splits + ``kg_input.parquet``. ``song_vectors`` is the dense
    train matrix; ``song_norms`` its per-column L2 norms (zeros clamped to 1).
    """
    from ..evaluation import AttributeArrays, load_song_meta

    if idx2song is None:
        sp = pd.concat([
            pd.read_parquet(Path(splits_dir) / f"{s}.parquet",
                            columns=["song_id", "s_idx"])
            for s in ("train", "val", "test")
        ]).drop_duplicates("s_idx").set_index("s_idx")["song_id"].to_dict()
        idx2song = {int(k): str(v) for k, v in sp.items()}
    if song_meta is None:
        song_meta = load_song_meta(kg_input_path)

    attrs = AttributeArrays.from_song_meta(song_meta, idx2song)
    song_vectors = train_matrix_norm.toarray()
    song_norms = np.linalg.norm(song_vectors, axis=0, keepdims=True)
    song_norms[song_norms == 0] = 1.0
    if verbose:
        print("[rehydrate] attrs / song_vectors / song_norms ← disk")
    return attrs, song_vectors, song_norms, idx2song, song_meta


def select_contrastive_cases(
    pop_qual_dfs: Mapping[str, pd.DataFrame],
    *,
    hgt_name: Optional[str] = None,
    recs_hgt_name: Optional[str] = None,
    min_train: int = 5,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    """Pick model-contrastive qualitative cases from the §10 per-user hit table.

    Combines each model's per-user hit counts on the shared user set, buckets
    users by training-profile size (light/medium/heavy), then selects:
    ``hgt-win-{profile}`` (HGT hits where every baseline misses, spread across
    profiles), ``hgt-edge`` (largest HGT margin where baselines also hit),
    ``baseline-win`` (honest reverse), and ``all-success`` / ``all-fail``.

    Returns ``(cases, overview_df, hits_all, hgt_name)``.
    """
    hits_all, hit_cols, cos_cols = None, [], []
    for name, dfq in pop_qual_dfs.items():
        col = f"hits_{name}"
        hit_cols.append(col)
        ren = {"n_hits": col}
        keep = [col]
        # Carry each model's per-user cosine-to-profile so the case table can
        # show it even when a model (e.g. the HGT) retrieved no held-out track.
        if "cos_mean" in dfq.columns:
            ccol = f"cos_{name}"
            ren["cos_mean"] = ccol
            cos_cols.append(ccol)
            keep.append(ccol)
        h = dfq.set_index("u_idx").rename(columns=ren)
        hits_all = (h[["n_train"] + keep].copy() if hits_all is None
                    else hits_all.join(h[keep], how="inner"))
    n_models = len(hit_cols)
    hits_all["n_models_hit"] = (hits_all[hit_cols] > 0).sum(axis=1)

    q = hits_all["n_train"].quantile([0.34, 0.67])
    def _bucket(n):
        return ("light" if n <= q.iloc[0]
                else "medium" if n <= q.iloc[1] else "heavy")
    hits_all["profile"] = hits_all["n_train"].map(_bucket)

    if hgt_name is None:
        hgt_name = next((n for n in pop_qual_dfs if "hgt" in n.lower()), recs_hgt_name)
    hgt_col = f"hits_{hgt_name}"
    base_cols = [c for c in hit_cols if c != hgt_col]
    hits_all["hgt_hits"] = hits_all[hgt_col] if hgt_col in hits_all.columns else 0
    hits_all["base_best"] = hits_all[base_cols].max(axis=1) if base_cols else 0
    hits_all["hgt_margin"] = hits_all["hgt_hits"] - hits_all["base_best"]

    def _pick(mask, by="n_train", n=1, asc=False):
        sub = hits_all[mask]
        if sub.empty:
            return []
        cols = list(dict.fromkeys(([by] if isinstance(by, str) else list(by)) + ["n_train"]))
        return sub.sort_values(cols, ascending=asc).head(n).index.tolist()

    valid = hits_all["n_train"] >= min_train
    hgt_only = valid & (hits_all["hgt_hits"] > 0) & (hits_all["base_best"] == 0)
    hgt_better = valid & (hits_all["hgt_margin"] > 0) & (hits_all["base_best"] > 0)
    base_only = valid & (hits_all["base_best"] > 0) & (hits_all["hgt_hits"] == 0)
    all_hit = hits_all["n_models_hit"] == n_models
    all_miss = valid & (hits_all["n_models_hit"] == 0)

    cases: dict = {}
    seen_prof: set = set()
    for u in _pick(hgt_only, by="hgt_hits", n=8):
        p = hits_all.at[u, "profile"]
        if p in seen_prof:
            continue
        seen_prof.add(p)
        cases[f"hgt-win-{p}"] = u
        if len(seen_prof) >= 2:
            break
    edge = _pick(hgt_better, by="hgt_margin", n=1)
    if edge:
        cases["hgt-edge"] = edge[0]
    if not any(k.startswith("hgt-") for k in cases):       # guarantee an HGT case
        for j, u in enumerate(_pick(hgt_better, by="hgt_margin", n=2)):
            cases[f"hgt-edge-{j + 1}"] = u
    bw = _pick(base_only, by="base_best", n=1)
    if bw:
        cases["baseline-win"] = bw[0]
    a_s = _pick(all_hit, by="n_train", n=1)
    if a_s:
        cases["all-success"] = a_s[0]
    af = _pick(all_miss, by="n_train", n=1)
    if af:
        cases["all-fail"] = af[0]

    def _winner(u):
        row = hits_all.loc[u, hit_cols]
        return row.idxmax().replace("hits_", "") if row.max() > 0 else "(none)"

    def _cos_leader(u):
        if not cos_cols:
            return "(n/a)"
        row = hits_all.loc[u, cos_cols].astype(float)
        return row.idxmax().replace("cos_", "") if row.notna().any() else "(n/a)"

    overview = []
    for label, u in cases.items():
        row = {"case": label, "u_idx": int(u),
               "n_train": int(hits_all.at[u, "n_train"]),
               "profile": hits_all.at[u, "profile"], "winner": _winner(u)}
        for c in hit_cols:
            row[c.replace("hits_", "")] = int(hits_all.at[u, c])
        # Per-model cosine-to-profile (shown even where hits=0) + the cosine leader.
        for c in cos_cols:
            row[c] = round(float(hits_all.at[u, c]), 4)
        if cos_cols:
            row["cos_leader"] = _cos_leader(u)
        overview.append(row)
    overview_df = pd.DataFrame(overview).set_index("case")
    return cases, overview_df, hits_all, hgt_name


__all__ = ["ensure_qual_arrays", "select_contrastive_cases"]
