"""Per-user stratified train / val / test split for the KG-based recommender.

Stratification is performed jointly over song-level attributes so that the train,
validation and test subsets all see approximately the same musical
distribution.  By default we stratify on::

    primary_genre × decade × key × mode × tempo_class

Why those attributes?
    * **genre** and **decade** are the two strongest editorial axes a user
      perceives;
    * **key** and **mode** make the splits musically meaningful (a model that
      only sees major-mode rock in training will struggle on minor-mode jazz);
    * **tempo_class** (Adagio / Allegro / …) summarises rhythmic feel and is a
      cheap categorical view of `Mean_Tempo`.

Why **not** stratify by `artist_name`?
    Artists are highly correlated with genre and have an extremely long tail
    (≈ 3.8 k unique artists for ≈ 7.1 k songs ⇒ ~1.9 songs/artist on average,
    most of them singletons).  Adding artist to the stratum key would explode
    the number of strata and force almost every per-user sub-group into the
    "less-than-3-interactions ⇒ train-only" fallback, defeating the purpose of
    stratification.  Genre + decade already captures most of the editorial
    signal carried by the artist.

The split is performed **per user** so that every user keeps a personal
70 / 10 / 20 ratio and the evaluation never has to recommend songs for a user
that was unseen during training.

Inputs are taken from the canonical KG artefacts produced by the KG-filtering
step (``data/interim/kg_input.parquet`` and
``data/interim/kg_taste_profile.parquet``).  Outputs are written to
``data/final/splits/`` as a dictionary of parquet files plus a small JSON
metadata side-car.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_STRATA_ATTRS: Tuple[str, ...] = (
    "primary_genre",
    "decade",
    "key",
    "mode",
    "tempo_class",
)

# Slimmer alternative when the default produces too many strata (the
# `primary_genre` field has ~272 levels and is the main driver of fragmentation).
# Use this with ``attrs=COARSE_STRATA_ATTRS`` if you need to recover a tighter
# 70/10/20 ratio without changing the splitter algorithm.
COARSE_STRATA_ATTRS: Tuple[str, ...] = (
    "decade",
    "mode",
    "tempo_class",
)

# --------------------------------------------------------------------------- #
# Stratum construction
# --------------------------------------------------------------------------- #

def _norm_str(s: pd.Series, fill: str = "unk") -> pd.Series:
    """Lower-case, trim, replace whitespace runs with `_` and fill NaNs."""
    return (
        s.fillna(fill)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )


def _bucket_genre(s: pd.Series, top_n: int) -> pd.Series:
    """Keep the ``top_n`` most frequent genre labels; collapse the rest to ``'other'``.

    The bucketing is derived entirely from the distribution within ``s`` (i.e.
    from the songs present in ``merged_df``) so it is reproducible without any
    external lookup table.
    """
    normed = _norm_str(s)
    top_genres = set(normed.value_counts().head(top_n).index)
    return normed.where(normed.isin(top_genres), other="other")


def build_song_strata(
    merged_df: pd.DataFrame,
    attrs: Sequence[str] = DEFAULT_STRATA_ATTRS,
    top_n_genres: int | None = None,
) -> pd.Series:
    """Return a Series mapping ``song_id`` → composite stratum string.

    The stratum is a ``"|"``-joined concatenation of the (normalised) values of
    ``attrs`` for each song.  Two special transformations are applied
    automatically:

    * ``"decade"`` — the ``year`` column is floor-divided to the nearest decade
      (e.g. 1997 → ``"1990"``).
    * ``"primary_genre"`` (when ``top_n_genres`` is set) — any genre outside the
      ``top_n_genres`` most common labels is replaced with ``"other"``.  This
      reduces cardinality from ~272 → ``top_n_genres + 1`` levels and leads to
      much better-populated strata without discarding the genre axis entirely.

    The output index is guaranteed to be unique (one row per song) so the
    series can be used safely with :py:meth:`pandas.Series.map`.

    Parameters
    ----------
    merged_df:
        The ``kg_input.parquet`` DataFrame (or any frame with a ``song_id``
        column and the columns listed in ``attrs``).
    attrs:
        Ordered sequence of column names to join into the stratum key.
    top_n_genres:
        If not ``None``, only the *top-N* most frequent ``primary_genre``
        values are kept verbatim; all others become ``"other"``.
        Has no effect when ``"primary_genre"`` is not in ``attrs``.
    """

    tmp = merged_df.drop_duplicates("song_id").copy()
    parts: list[np.ndarray] = []
    for attr in attrs:
        if attr == "decade":
            year_num = pd.to_numeric(tmp.get("year"), errors="coerce").fillna(0).astype(int)
            parts.append((year_num // 10 * 10).astype(str).values)
        elif attr == "primary_genre" and top_n_genres is not None:
            if attr not in tmp.columns:
                raise KeyError(
                    f"Column '{attr}' not found in merged dataframe. "
                    f"Available: {list(tmp.columns)[:20]}…"
                )
            parts.append(_bucket_genre(tmp[attr], top_n=top_n_genres).values)
        else:
            if attr not in tmp.columns:
                raise KeyError(
                    f"Column '{attr}' not found in merged dataframe. "
                    f"Available: {list(tmp.columns)[:20]}…"
                )
            parts.append(_norm_str(tmp[attr]).values)

    stratum = parts[0]
    for p in parts[1:]:
        stratum = np.char.add(np.char.add(stratum.astype(str), "|"), p.astype(str))
    return pd.Series(stratum, index=tmp["song_id"].values, name="stratum")


# --------------------------------------------------------------------------- #
# Splitter
# --------------------------------------------------------------------------- #

def user_level_stratified_split(
    interactions_df: pd.DataFrame,
    song_strata: pd.Series,
    val_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = 42,
    user_col: str = "u_idx",
    song_col: str = "song_id",
    min_stratum_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user, per-stratum train / val / test split.

    For each user we group their interactions by the song's stratum.  Within
    each (user, stratum) sub-group of size ``n >= min_stratum_size`` we send
    ``floor(n*test_size)`` rows (≥ 1) to test, ``floor(n*val_size)`` rows
    (≥ 1) to val and the rest to train.

    Sub-groups smaller than ``min_stratum_size`` are **pooled per user** and
    then randomly split with the same target ratios — this avoids the
    pathological behaviour of "every tiny stratum → train" which heavily
    skewed the global ratios when the stratum cardinality is high (e.g. the
    full 5-attribute key produces ~4.4 k strata for ~7 k songs).

    For users whose total interactions are below ``min_stratum_size`` we
    deterministically allocate at least one row to test (and one to val if
    n ≥ 2) so cold users still appear in evaluation.
    """

    rng = np.random.default_rng(seed)
    df = interactions_df.copy()
    df["stratum"] = df[song_col].map(song_strata).fillna("_misc_")

    train_p = 1.0 - val_size - test_size

    train_rows: list = []
    val_rows: list = []
    test_rows: list = []

    def _alloc_pool(idxs: np.ndarray):
        """Random 70/10/20 assignment that *guarantees* the expected ratios
        on the pool by using a fixed permutation + integer quotas."""
        n = len(idxs)
        if n == 0:
            return
        if n == 1:
            # single interaction → always to train; a user with nothing in
            # training has no embedding and cannot be meaningfully evaluated
            train_rows.append(int(idxs[0]))
            return
        if n == 2:
            # one to train, the other to val or test (val gets it ~33%, test ~67%)
            shuf = rng.permutation(idxs)
            train_rows.append(int(shuf[0]))
            other = rng.choice([1, 2], p=[val_size / (val_size + test_size),
                                          test_size / (val_size + test_size)])
            (val_rows if other == 1 else test_rows).append(int(shuf[1]))
            return
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        # protect train from going below 1
        if n_test + n_val >= n:
            n_test = max(1, n - 2)
            n_val = 1
        shuf = rng.permutation(idxs)
        test_rows.extend(int(i) for i in shuf[:n_test])
        val_rows.extend(int(i) for i in shuf[n_test : n_test + n_val])
        train_rows.extend(int(i) for i in shuf[n_test + n_val :])

    for _, grp in df.groupby(user_col, sort=False):
        leftovers: list[int] = []
        for _, sgrp in grp.groupby("stratum", sort=False):
            idxs = sgrp.index.to_numpy()
            n = len(idxs)
            if n < min_stratum_size:
                leftovers.extend(int(i) for i in idxs)
                continue
            n_test = max(1, int(np.floor(n * test_size)))
            n_val = max(1, int(np.floor(n * val_size)))
            if n_test + n_val >= n:  # safety for very small n
                n_test = max(1, n - 2); n_val = 1
            shuffled = rng.permutation(idxs)
            test_rows.extend(int(i) for i in shuffled[:n_test])
            val_rows.extend(int(i) for i in shuffled[n_test : n_test + n_val])
            train_rows.extend(int(i) for i in shuffled[n_test + n_val :])

        # Pool the remaining (small-stratum) rows for this user and split
        # them with the target ratios — this is what restores ~70/10/20.
        if leftovers:
            _alloc_pool(np.asarray(leftovers))

    return df.loc[train_rows].copy(), df.loc[val_rows].copy(), df.loc[test_rows].copy()


# --------------------------------------------------------------------------- #
# Distribution diagnostics
# --------------------------------------------------------------------------- #

def _attr_dist(split_df: pd.DataFrame, song_strata: pd.Series, attr_names: Sequence[str]):
    s = split_df["song_id"].map(song_strata).dropna().str.split("|", expand=True)
    s.columns = list(attr_names)
    return {c: s[c].value_counts(normalize=True) for c in s.columns}


def _attr_counts(split_df: pd.DataFrame, song_strata: pd.Series, attr_names: Sequence[str]):
    s = split_df["song_id"].map(song_strata).dropna().str.split("|", expand=True)
    s.columns = list(attr_names)
    return {c: s[c].value_counts() for c in s.columns}


def compute_split_distributions(
    splits: Mapping[str, pd.DataFrame],
    song_strata: pd.Series,
    attr_names: Sequence[str] = DEFAULT_STRATA_ATTRS,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return ``{"proportion": {attr: df}, "count": {attr: df}, "js_div": df}``.

    * ``proportion`` — one DataFrame per attribute with one column per split,
      values are the in-split proportion of each level.
    * ``count`` — same shape but with raw interaction counts.
    * ``js_div`` — Jensen-Shannon divergence of each non-train split against
      train, for every attribute (lower = better balance).
    """

    from scipy.spatial.distance import jensenshannon  # lazy import

    # decade is derived from year and lives at a fixed position in the stratum
    attr_names = tuple(attr_names)

    prop: Dict[str, pd.DataFrame] = {}
    cnt: Dict[str, pd.DataFrame] = {}
    js_rows: list[dict] = []

    dists = {name: _attr_dist(d_, song_strata, attr_names) for name, d_ in splits.items()}
    counts = {name: _attr_counts(d_, song_strata, attr_names) for name, d_ in splits.items()}

    split_names = list(splits.keys())
    ref = "train" if "train" in split_names else split_names[0]
    other_splits = [s for s in split_names if s != ref]

    for attr in attr_names:
        prop_df = (
            pd.DataFrame({s: dists[s][attr] for s in split_names})
            .fillna(0)
            .sort_values(ref, ascending=False)
        )
        cnt_df = (
            pd.DataFrame({s: counts[s][attr] for s in split_names})
            .fillna(0)
            .astype(int)
            .loc[prop_df.index]
        )
        prop[attr] = prop_df.round(4)
        cnt[attr] = cnt_df

        row = {"attribute": attr}
        for s in other_splits:
            row[f"JS({s}||{ref})"] = float(jensenshannon(prop_df[s].values, prop_df[ref].values))
        js_rows.append(row)

    return {
        "proportion": prop,
        "count": cnt,
        "js_div": pd.DataFrame(js_rows).set_index("attribute").round(4),
    }


def plot_split_distributions(
    distributions: Mapping[str, Mapping[str, pd.DataFrame]],
    top_k: int = 12,
    figsize_per_attr: Tuple[float, float] = (4.4, 4.5),
):
    """Return a ``matplotlib`` figure with a 2 × N grid (proportion / count)."""
    import matplotlib.pyplot as plt

    prop = distributions["proportion"]
    cnt = distributions["count"]
    attrs = list(prop.keys())
    n = len(attrs)
    fig, axes = plt.subplots(
        2, n, figsize=(figsize_per_attr[0] * n, figsize_per_attr[1] * 2),
        squeeze=False,
    )
    for col_idx, attr in enumerate(attrs):
        prop[attr].head(top_k).plot.bar(ax=axes[0, col_idx], width=0.8)
        axes[0, col_idx].set_title(f"{attr} — proportion", fontsize=10)
        axes[0, col_idx].set_ylabel("proportion")
        axes[0, col_idx].tick_params(axis="x", rotation=45, labelsize=7)
        axes[0, col_idx].legend(fontsize=7)

        cnt[attr].head(top_k).plot.bar(ax=axes[1, col_idx], width=0.8)
        axes[1, col_idx].set_title(f"{attr} — count", fontsize=10)
        axes[1, col_idx].set_ylabel("# interactions")
        axes[1, col_idx].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1, col_idx].legend(fontsize=7)

    fig.suptitle(
        f"Stratified split — attribute distributions (top-{top_k} per attribute)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_splits(
    splits: Mapping[str, pd.DataFrame],
    out_dir: Path,
    fmt: str = "parquet",
    metadata: dict | None = None,
) -> Dict[str, Path]:
    """Persist a ``{"train": df, "val": df, "test": df}`` mapping.

    Each split goes to its own file (``train.parquet`` / ``val.parquet`` / …).
    A ``metadata.json`` side-car is written next to them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt not in ("parquet", "csv", "pickle"):
        raise ValueError(f"Unsupported fmt={fmt!r}; use 'parquet', 'csv' or 'pickle'.")

    written: Dict[str, Path] = {}
    for name, df in splits.items():
        if fmt == "parquet":
            path = out_dir / f"{name}.parquet"
            df.reset_index(drop=True).to_parquet(path, index=False)
        elif fmt == "csv":
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        else:  # pickle
            path = out_dir / f"{name}.pkl"
            df.to_pickle(path)
        written[name] = path

    meta = {
        "format": fmt,
        "splits": {k: {"rows": int(len(v)), "path": written[k].name} for k, v in splits.items()},
        **(metadata or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return written


def load_splits(in_dir: Path, fmt: str = "parquet") -> Dict[str, pd.DataFrame]:
    """Reverse of :func:`save_splits`."""
    in_dir = Path(in_dir)
    fmt = fmt.lower()
    suffix = {"parquet": ".parquet", "csv": ".csv", "pickle": ".pkl"}[fmt]
    out: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = in_dir / f"{name}{suffix}"
        if not path.exists():
            raise FileNotFoundError(path)
        if fmt == "parquet":
            out[name] = pd.read_parquet(path)
        elif fmt == "csv":
            out[name] = pd.read_csv(path)
        else:
            out[name] = pd.read_pickle(path)
    return out


# --------------------------------------------------------------------------- #
# High-level orchestrator
# --------------------------------------------------------------------------- #

def make_stratified_splits(
    kg_input_path: Path,
    kg_taste_path: Path,
    out_dir: Path | None = None,
    attrs: Sequence[str] = DEFAULT_STRATA_ATTRS,
    val_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = 42,
    min_users_per_song: int = 3,
    min_songs_per_user: int = 5,
    save_format: str = "parquet",
) -> dict:
    """End-to-end: read interim parquets → encode IDs → stratify → save.

    Returns a dict with keys ``train``, ``val``, ``test`` (DataFrames),
    ``song_strata``, ``user2idx``, ``song2idx`` and ``saved_paths``.
    """
    merged = pd.read_parquet(kg_input_path)
    taste = pd.read_parquet(kg_taste_path)

    # ── ID encoding (cold-start filter) ─────────────────────────────────────
    df = taste.copy()
    song_counts = df.groupby("song_id")["user_id"].nunique()
    df = df[df["song_id"].isin(song_counts[song_counts >= min_users_per_song].index)]
    user_counts = df.groupby("user_id")["song_id"].nunique()
    df = df[df["user_id"].isin(user_counts[user_counts >= min_songs_per_user].index)]

    user_ids = sorted(df["user_id"].unique())
    song_ids = sorted(df["song_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    song2idx = {s: i for i, s in enumerate(song_ids)}
    df["u_idx"] = df["user_id"].map(user2idx)
    df["s_idx"] = df["song_id"].map(song2idx)

    # ── Stratify ────────────────────────────────────────────────────────────
    song_strata = build_song_strata(merged, attrs=attrs)
    train_df, val_df, test_df = user_level_stratified_split(
        df, song_strata, val_size=val_size, test_size=test_size, seed=seed,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    saved: Dict[str, Path] = {}
    if out_dir is not None:
        saved = save_splits(
            splits, out_dir, fmt=save_format,
            metadata={
                "attrs": list(attrs),
                "val_size": val_size,
                "test_size": test_size,
                "seed": seed,
                "n_users": len(user_ids),
                "n_songs": len(song_ids),
                "n_strata": int(song_strata.nunique()),
                "min_users_per_song": min_users_per_song,
                "min_songs_per_user": min_songs_per_user,
                "kg_input_path": str(kg_input_path),
                "kg_taste_path": str(kg_taste_path),
            },
        )

    return {
        **splits,
        "song_strata": song_strata,
        "user2idx": user2idx,
        "song2idx": song2idx,
        "interactions": df,
        "saved_paths": saved,
    }
