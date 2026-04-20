"""
src/data/music_feature_extraction.py
======================================
Symbolic music feature extraction.

Two back-ends are provided:

1. **DIDONE pipeline** (recommended for large datasets) — wraps
   ``vendor/music_symbolic_features`` (DIDONEproject) which runs musif,
   music21 and jSymbolic in an isolated Python 3.10 environment.

   Workflow::

       bash scripts/setup_didone.sh       # one-time setup
       bash scripts/extract_features.sh   # extracts CSVs
       df = load_didone_features(...)      # load results here

2. **Direct musif** (fallback / small batches) — calls musif in the
   current Python environment.

   References
   ----------
   - https://musif.didone.eu/Tutorial.html
   - https://musif.didone.eu/Configuration.html
   - pip install musif

Public API
----------
load_didone_features(output_dir, ...)    → pd.DataFrame  (DIDONE CSV loader)
merge_didone_features(output_dir, ...)   → pd.DataFrame  (merge all extractors)
extract_musif_features(midi_dir, ...)    → pd.DataFrame  (direct musif, 900+ cols)
postprocess_features(df, ...)            → pd.DataFrame  (cleaned / typed)
batch_extract(midi_paths, out_parquet)   → pd.DataFrame  (chunked, cached)
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
from typing import Optional

import pandas as pd

# ── default feature modules ────────────────────────────────────────────────────
# "lyrics" is excluded: no lyric data in MIDI files
DEFAULT_FEATURES: list[str] = [
    "core",
    "ambitus",
    "melody",
    "tempo",
    "density",
    "texture",
    "scale",
    "key",
    "dynamics",
    "rhythm",
]

# ── DIDONE extractor names ─────────────────────────────────────────────────────
DIDONE_EXTRACTORS: tuple[str, ...] = ("musif", "music21", "jsymbolic")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  DIDONE pipeline loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_didone_features(
    output_dir: str | pathlib.Path,
    extractor: str = "musif",
    dataset_name: str = "lmd_matched",
    extension: str = "mid",
    encoding: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a single-extractor CSV produced by ``scripts/extract_features.sh``.

    The DIDONE pipeline writes one CSV per extractor per dataset, named::

        <output_dir>/<dataset_name>/<extractor>-<extension>.csv
        e.g.  data/interim/didone_features/lmd_matched/musif-mid.csv

    Parameters
    ----------
    output_dir : root output directory set in ``settings.py`` (OUTPUT).
        Typically ``data/interim/didone_features/``.
    extractor : one of ``"musif"``, ``"music21"``, ``"jsymbolic"``.
    dataset_name : name of the dataset directory under *output_dir*.
    extension : file extension without dot used during extraction.
        Default ``"mid"``.
    encoding : CSV encoding.  ``None`` = auto-detect via *chardet*.

    Returns
    -------
    pd.DataFrame — one row per MIDI file.

    Raises
    ------
    FileNotFoundError if the expected CSV does not exist.
    """
    if extractor not in DIDONE_EXTRACTORS:
        raise ValueError(f"extractor must be one of {DIDONE_EXTRACTORS}, got {extractor!r}")

    output_dir = pathlib.Path(output_dir)
    csv_path = output_dir / dataset_name / f"{extractor}-{extension}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"DIDONE output CSV not found: {csv_path}\n"
            f"Run:  bash scripts/extract_features.sh --only {extractor}"
        )

    if encoding is None:
        try:
            import chardet
            raw = csv_path.read_bytes()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
        except ImportError:
            encoding = "utf-8"

    df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)

    # normalise the filename column to a consistent name
    if "file_name" in df.columns:
        df = df.rename(columns={"file_name": "FileName"})
    if "FileName" not in df.columns:
        # jSymbolic uses a different column name
        for candidate in ("filename", "Filename", "file", "File"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "FileName"})
                break

    # tag with extractor name to avoid column collisions when merging
    meta_cols = {"FileName", "midi_path"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df.rename(columns={c: f"{extractor}__{c}" for c in feature_cols})

    if verbose:
        print(f"[load_didone] {extractor:10s} → {len(df):6d} rows × "
              f"{len(feature_cols):5d} feature cols  ({csv_path})")

    return df.reset_index(drop=True)


def merge_didone_features(
    output_dir: str | pathlib.Path,
    dataset_name: str = "lmd_matched",
    extractors: tuple[str, ...] = DIDONE_EXTRACTORS,
    extension: str = "mid",
    how: str = "outer",  # type: ignore[assignment]
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and merge CSV outputs from all three DIDONE extractors.

    Each extractor's feature columns are prefixed with ``<extractor>__``
    to avoid name collisions.  The DataFrames are joined on ``FileName``.

    Parameters
    ----------
    output_dir : see :func:`load_didone_features`.
    dataset_name : see :func:`load_didone_features`.
    extractors : subset of extractors to load.
        Default: all three (``("musif", "music21", "jsymbolic")``).
    extension : see :func:`load_didone_features`.
    how : pandas ``merge`` strategy (``"outer"`` keeps all files even if
        one extractor failed; use ``"inner"`` for intersection only).

    Returns
    -------
    pd.DataFrame — merged feature table.
    """
    dfs: list[pd.DataFrame] = []

    for extractor in extractors:
        try:
            df = load_didone_features(
                output_dir=output_dir,
                extractor=extractor,
                dataset_name=dataset_name,
                extension=extension,
                verbose=verbose,
            )
            dfs.append(df)
        except FileNotFoundError as exc:
            if verbose:
                print(f"[merge_didone] SKIP {extractor}: {exc}")

    if not dfs:
        if verbose:
            print("[merge_didone] No extractor outputs found.")
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="FileName", how=how, suffixes=("", "_dup"))
        # drop accidental duplicate columns
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

    if verbose:
        print(f"\n[merge_didone] Final shape: {merged.shape[0]} rows × {merged.shape[1]} cols")

    return merged.reset_index(drop=True)



# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_musif_features(
    midi_dir: str | pathlib.Path,
    features: Optional[list[str]] = None,
    basic_modules: Optional[list[str]] = None,
    parallel: int = -1,
    cache_dir: Optional[str | pathlib.Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract symbolic features from all MIDI files in *midi_dir*.

    Parameters
    ----------
    midi_dir : directory containing ``.mid`` / ``.midi`` files
    features : list of musif feature-module names.
        Defaults to :data:`DEFAULT_FEATURES`.
    basic_modules : musif basic_modules list.
        Defaults to ``["scoring"]``.
    parallel : number of parallel workers (-1 = all CPUs).
    cache_dir : if given, musif will store intermediate cache here.
    verbose : print progress / warnings.

    Returns
    -------
    pd.DataFrame  (one row per file, 900+ feature columns)
    """
    try:
        from musif.config import ExtractConfiguration
        from musif.extract.extract import FeaturesExtractor
    except ImportError as exc:
        raise ImportError(
            "musif is not installed.  Run:  pip install musif"
        ) from exc

    midi_dir   = pathlib.Path(midi_dir)
    features   = features or DEFAULT_FEATURES
    basic_mods = basic_modules or ["scoring"]

    config_kwargs: dict = {
        "data_dir":      str(midi_dir),
        "basic_modules": basic_mods,
        "features":      features,
        "parallel":      parallel,
    }
    if cache_dir is not None:
        config_kwargs["cache_dir"] = str(cache_dir)

    config    = ExtractConfiguration(None, **config_kwargs)
    extractor = FeaturesExtractor(config)

    if verbose:
        print(f"[musif] Extracting from {midi_dir}  ({len(list(midi_dir.glob('*.mid')))} files) …")

    df = extractor.extract()

    if verbose:
        print(f"[musif] Done — {len(df)} rows × {len(df.columns)} columns")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_features(
    df: pd.DataFrame,
    config_yaml: Optional[str | pathlib.Path] = None,
    drop_threshold: float = 0.8,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean and post-process the raw musif feature DataFrame.

    Steps
    -----
    1. Run ``musif.DataProcessor`` (normalises / imputes).
    2. Drop columns with > *drop_threshold* fraction of NaN.
    3. Infer better dtypes (int/float/category).

    Parameters
    ----------
    df : output from :func:`extract_musif_features`
    config_yaml : path to a musif ``config.yml`` (optional).
        Pass ``None`` to use musif's internal defaults.
    drop_threshold : drop columns that are > this fraction NaN.

    Returns
    -------
    Cleaned pd.DataFrame
    """
    try:
        from musif.process.processor import DataProcessor
    except ImportError as exc:
        raise ImportError("musif is not installed.  Run:  pip install musif") from exc

    yaml_path = str(config_yaml) if config_yaml is not None else None
    processed = DataProcessor(df, yaml_path).process().data

    # drop nearly-empty columns
    before = processed.shape[1]
    thresh = int(len(processed) * (1 - drop_threshold))
    processed = processed.dropna(axis=1, thresh=max(thresh, 1))

    if verbose:
        print(f"[postprocess] {before} → {processed.shape[1]} columns "
              f"(dropped {before - processed.shape[1]} high-NaN columns)")

    # convert obvious numeric columns
    processed = processed.infer_objects()

    return processed.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Batch extraction (chunked, with Parquet caching)
# ─────────────────────────────────────────────────────────────────────────────

def batch_extract(
    midi_paths: list[str | pathlib.Path],
    out_parquet: str | pathlib.Path,
    chunk_size: int = 500,
    features: Optional[list[str]] = None,
    parallel: int = -1,
    overwrite: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features from an arbitrary list of MIDI files in chunks,
    caching intermediate results to Parquet shards.

    musif's ``FeaturesExtractor`` expects a directory, so each chunk is
    copied into a temp directory, extracted, then the temp dir is removed.

    Parameters
    ----------
    midi_paths : flat list of .mid file paths
    out_parquet : destination for the final merged Parquet file
    chunk_size : number of files to process per musif call
    features : see :func:`extract_musif_features`
    parallel : parallel workers
    overwrite : if True, re-extract even if *out_parquet* already exists

    Returns
    -------
    Combined pd.DataFrame (also saved to *out_parquet*)
    """
    out_parquet = pathlib.Path(out_parquet)

    if out_parquet.exists() and not overwrite:
        if verbose:
            print(f"[batch_extract] Loading cached result from {out_parquet}")
        return pd.read_parquet(out_parquet)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    midi_paths = [pathlib.Path(p) for p in midi_paths]

    chunks       = [midi_paths[i:i + chunk_size] for i in range(0, len(midi_paths), chunk_size)]
    shard_dir    = out_parquet.parent / (out_parquet.stem + '_shards')
    shard_dir.mkdir(exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks):
        shard_path = shard_dir / f'shard_{idx:04d}.parquet'

        if shard_path.exists() and not overwrite:
            if verbose:
                print(f"[batch_extract] Shard {idx+1}/{len(chunks)} — loading cache")
            all_dfs.append(pd.read_parquet(shard_path))
            continue

        if verbose:
            print(f"[batch_extract] Shard {idx+1}/{len(chunks)}  ({len(chunk)} files)")

        with tempfile.TemporaryDirectory(prefix='musif_chunk_') as tmp:
            tmp_dir = pathlib.Path(tmp)
            for p in chunk:
                dest = tmp_dir / p.name
                shutil.copy2(p, dest)

            try:
                df_chunk = extract_musif_features(
                    tmp_dir,
                    features=features,
                    parallel=parallel,
                    verbose=False,
                )
                # tag with original paths (musif uses filename as key)
                name_to_path = {p.name: str(p) for p in chunk}
                if 'FileName' in df_chunk.columns:
                    df_chunk['midi_path'] = df_chunk['FileName'].map(name_to_path)

                df_chunk.to_parquet(shard_path, index=False)
                all_dfs.append(df_chunk)
            except Exception as exc:
                print(f"[WARN] Shard {idx} failed: {exc}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(out_parquet, index=False)

    if verbose:
        print(f"[batch_extract] Saved {len(combined)} rows → {out_parquet}")

    return combined
