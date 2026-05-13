"""
nb_env.py — Notebook environment & path configuration
======================================================
Imported by every notebook immediately after ROOT and ON_COLAB are set.

Usage (notebook cell 4):
    from nb_env import setup
    globals().update(setup(ROOT, ON_COLAB))

`setup()` returns a plain dict whose keys become notebook globals:
    ROOT, ON_COLAB, DEVICE, SEED, USE_GRAPHDB
    _DATA, RAW, INTERIM, PROCESSED, FINAL, ONTOLOGY
    LAKH_PQ, PER_SONG_CSV, PER_USER_CSV, TASTE_PQ
    ONTO_BASE, ONTO_OUT, ONTO_OUT_SIMPLE, LISTENING_NT, LISTENING_NT_SIM
    INTERIM_CSV, KG_INPUT_PQ, KG_TASTE_PQ, SPLIT_PQ, KFOLD_CSV
    KG_GRAPHDB_DIR, KG_STATS_DIR, KG_PLOTS_DIR
    FINAL_SPLITS_DIR, FINAL_MODELS_DIR
    KNN_VAL_CSV, KNN_TEST_CSV, KNN_VAL_PLOT_PNG
    KNN_POP_CSV, KNN_POP_JSON, HGT_RESULT_PATH, AE_EMBEDDINGS_PQ
    MODELS_DIR, AE_WEIGHTS_DIR, HGT_WEIGHTS_DIR, KNN_CACHE_DIR
    KNN_NBRS_CACHE, HGT_MODEL_PATH, AE_MODEL_PATH
    KNN_RESULTS_DIR, HGT_RESULTS_DIR, AE_RESULTS_DIR
    JSYMBOLIC_JAR, GDRIVE_DATA_ROOT
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


# ── .env / secret helpers ─────────────────────────────────────────────────────

def _load_first_existing_dotenv(candidates: list[Path | None]) -> Path | None:
    """Load the first .env file that exists from *candidates*."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("  [INFO] python-dotenv not installed — .env loading skipped.")
        return None
    for p in candidates:
        if p and Path(p).exists():
            from dotenv import load_dotenv  # noqa: F811
            load_dotenv(dotenv_path=p, override=False)
            print(f"  [.env] loaded {p}")
            return Path(p)
    print(f"  [.env] none of the {len(candidates)} candidate paths found")
    return None


def _colab_secret(key: str) -> str | None:
    """Return a Colab Secret value, or None (safe outside Colab)."""
    try:
        from google.colab import userdata  # type: ignore[import]
        v = userdata.get(key)
        return v if v else None
    except Exception:
        return None


def _resolve_secret(key: str, default: str | None = None) -> str | None:
    """3-level fallback: Colab Secret → env var → *default*."""
    v = _colab_secret(key)
    if v:
        return v
    for k in (key, key.upper(), key.lower()):
        v = os.getenv(k)
        if v:
            return v
    return default


# ── Colab symlink helper ───────────────────────────────────────────────────────

def _ensure_symlink(target: Path, link: Path) -> None:
    """Point *link* → *target*, handling stale links and non-empty dirs safely."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if not any(link.iterdir()):
            link.rmdir()
        else:
            _bak = link.with_name(link.name + ".repo_backup")
            _i = 1
            while _bak.exists():
                _bak = link.with_name(f"{link.name}.repo_backup.{_i}")
                _i += 1
            link.rename(_bak)
            print(f"  [MOVE] {link.name}  →  {_bak.name}/  (repo files preserved)")
    link.symlink_to(target, target_is_directory=True)
    print(f"  [LINK] {link.name}  →  {target}")


# ── Display helper ────────────────────────────────────────────────────────────

def _short(p: Path | None, base: Path) -> str:
    """Display path relative to *base* when possible, else absolute."""
    if p is None:
        return "None"
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


# ── Main entry point ──────────────────────────────────────────────────────────

def setup(ROOT: Path, ON_COLAB: bool) -> dict[str, Any]:  # noqa: N803
    """
    Full environment setup.  Must be called after ROOT is determined.
    Returns a dict suitable for ``globals().update(…)``.

    On Colab it also:
      - searches Drive for a .env file
      - sets up data/ and jSymbolic/ symlinks inside the cloned repo
      - authenticates W&B if WANDB_API_KEY is available
    """
    ROOT = Path(ROOT)

    # ── .env loading ──────────────────────────────────────────────────────────
    gdrive_proj: Path | None = None

    if ON_COLAB:
        # gdrive_proj_path may already be in env from Colab Secret (set before
        # Drive mounted by the minimal bootstrap cell) — resolve it now that
        # Drive is up so we can probe for a .env inside it.
        _gdrive_str = _resolve_secret(
            "gdrive_proj_path",
            default="/content/drive/MyDrive/DL-KG-project",
        )
        gdrive_proj = Path(_gdrive_str)

        print("Searching for .env (Colab):")
        _dotenv_candidates: list[Path | None] = []
        _explicit = _colab_secret("dotenv_path")
        if _explicit:
            _dotenv_candidates.append(Path(_explicit))
        _dotenv_candidates += [
            Path("/content/.env"),                  # drag-drop target
            Path("/content/drive/MyDrive/.env"),
            gdrive_proj / ".env",
            ROOT / ".env",
        ]
        _load_first_existing_dotenv(_dotenv_candidates)

        # Re-resolve now that .env may have added more vars
        _gdrive_str = _resolve_secret("gdrive_proj_path", default=_gdrive_str)
        gdrive_proj = Path(_gdrive_str)

        if not gdrive_proj.exists():
            print(f"  [WARN] gdrive_proj_path not found on this Drive: {gdrive_proj}")
        else:
            print(f"  [OK]   gdrive_proj_path = {gdrive_proj}")

        # ── Drive symlinks ────────────────────────────────────────────────────
        _gdrive_data = gdrive_proj / "data"
        _gdrive_jsym = gdrive_proj / "jSymbolic"
        if _gdrive_data.exists():
            _ensure_symlink(_gdrive_data, ROOT / "data")
        if _gdrive_jsym.exists():
            _ensure_symlink(_gdrive_jsym, ROOT / "jSymbolic")

        # ── W&B auth ──────────────────────────────────────────────────────────
        _wandb_key = _resolve_secret("WANDB_API_KEY")
        if _wandb_key:
            try:
                import wandb  # type: ignore[import]
                wandb.login(key=_wandb_key, relogin=True)
                print("W&B authenticated.")
            except Exception as e:
                print(f"  [WARN] W&B login failed: {e}")
        else:
            print("  [INFO] WANDB_API_KEY not set — W&B logging disabled.")

    else:
        print("Searching for .env (local):")
        _load_first_existing_dotenv([ROOT / ".env", Path.cwd() / ".env"])

    # ── Path constants ────────────────────────────────────────────────────────
    _DATA    = ROOT / "data"
    RAW      = _DATA / "raw"
    INTERIM  = _DATA / "interim"
    PROCESSED = _DATA / "processed"
    FINAL    = _DATA / "final"
    ONTOLOGY = _DATA / "ontology"

    # Source CSV / Parquet inputs
    LAKH_PQ       = PROCESSED / "lakh_msd_dataset.parquet"
    PER_SONG_CSV  = PROCESSED / "lakh_msd_per_song.csv"
    PER_USER_CSV  = PROCESSED / "lakh_msd_per_user.csv"
    TASTE_PQ      = PROCESSED / "user_song_taste.parquet"

    # KG construction artefacts
    ONTO_BASE        = ONTOLOGY / "knowledge_graph_full.ttl"
    ONTO_OUT         = ONTOLOGY / "knowledge_graph_full_with_users.ttl"
    ONTO_OUT_SIMPLE  = ONTOLOGY / "knowledge_graph_simple.ttl"
    LISTENING_NT     = ONTOLOGY / "listening_triples.nt"
    LISTENING_NT_SIM = ONTOLOGY / "listening_triples_simple.nt"

    # DL pipeline artefacts
    INTERIM_CSV  = INTERIM   / "music_features_interim.csv"
    KG_INPUT_PQ  = PROCESSED / "kg_input.parquet"
    KG_TASTE_PQ  = PROCESSED / "kg_input_with_taste.parquet"
    SPLIT_PQ     = PROCESSED / "kg_input_with_taste_split.parquet"
    KFOLD_CSV    = PROCESSED / "kg_input_kfold.csv"

    # GraphDB / KG export artefacts
    KG_GRAPHDB_DIR = PROCESSED / "graphdb"
    KG_STATS_DIR   = FINAL / "kg_stats"
    KG_PLOTS_DIR   = FINAL / "kg_plots"

    # Final results & cached splits
    FINAL_SPLITS_DIR = FINAL / "splits"
    KNN_VAL_CSV      = FINAL / "knn_validation.csv"
    KNN_TEST_CSV     = FINAL / "knn_test.csv"
    KNN_VAL_PLOT_PNG = FINAL / "knn_val_curve.png"
    KNN_POP_CSV      = FINAL / "knn_pop_baseline.csv"
    KNN_POP_JSON     = FINAL / "knn_pop_baseline_summary.json"
    HGT_RESULT_PATH  = FINAL / "hgt_results.json"
    AE_EMBEDDINGS_PQ = FINAL / "ae_embeddings.parquet"

    # Model directories (lives in ROOT/models/ — gitignored)
    FINAL_MODELS_DIR = FINAL / "models"
    KNN_RESULTS_DIR  = FINAL_MODELS_DIR / "knn"
    HGT_RESULTS_DIR  = FINAL_MODELS_DIR / "hgt"
    AE_RESULTS_DIR   = FINAL_MODELS_DIR / "autoencoder"

    MODELS_DIR      = ROOT / "models"
    AE_WEIGHTS_DIR  = MODELS_DIR / "autoencoder"
    HGT_WEIGHTS_DIR = MODELS_DIR / "hgt"
    KNN_CACHE_DIR   = MODELS_DIR / "knn"
    KNN_NBRS_CACHE  = KNN_CACHE_DIR / "neighbours.npz"
    HGT_MODEL_PATH  = HGT_WEIGHTS_DIR / "model.pt"
    AE_MODEL_PATH   = AE_WEIGHTS_DIR / "model.pt"

    # jSymbolic
    JSYMBOLIC_JAR = Path(
        _resolve_secret(
            "jSymbolic2_path",
            default=str(ROOT / "jSymbolic" / "jSymbolic2.jar"),
        )
    )
    GDRIVE_DATA_ROOT = (ROOT / "data") if ON_COLAB else None

    # ── Create directories ────────────────────────────────────────────────────
    for _p in [
        INTERIM, PROCESSED, FINAL,
        KG_GRAPHDB_DIR, KG_STATS_DIR, KG_PLOTS_DIR,
        FINAL_SPLITS_DIR, FINAL_MODELS_DIR,
        KNN_RESULTS_DIR, HGT_RESULTS_DIR, AE_RESULTS_DIR,
        MODELS_DIR, AE_WEIGHTS_DIR, HGT_WEIGHTS_DIR, KNN_CACHE_DIR,
    ]:
        _p.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────────
    import numpy as np  # noqa: PLC0415
    import torch        # noqa: PLC0415

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── GraphDB flag ──────────────────────────────────────────────────────────
    USE_GRAPHDB: bool = (not ON_COLAB) and bool(os.environ.get("GRAPHDB_URL"))

    # ── Sanity print ──────────────────────────────────────────────────────────
    print(f"ROOT             : {_short(ROOT, ROOT.parent)}")
    print(f"data             : {_short(_DATA, ROOT)}"
          + (f"  →  {_DATA.resolve()}" if _DATA.is_symlink() else ""))
    print(f"models           : {_short(MODELS_DIR, ROOT)}")
    print(f"jSymbolic        : {_short(JSYMBOLIC_JAR, ROOT)}"
          f"  ({'found' if JSYMBOLIC_JAR.exists() else 'NOT FOUND'})")
    print(f"device           : {DEVICE}")
    print(f"USE_GRAPHDB      : {USE_GRAPHDB}")

    # Return every name that downstream cells need as a global
    return dict(
        ROOT=ROOT, ON_COLAB=ON_COLAB, SEED=SEED, DEVICE=DEVICE,
        USE_GRAPHDB=USE_GRAPHDB,
        _DATA=_DATA, RAW=RAW, INTERIM=INTERIM, PROCESSED=PROCESSED,
        FINAL=FINAL, ONTOLOGY=ONTOLOGY,
        LAKH_PQ=LAKH_PQ, PER_SONG_CSV=PER_SONG_CSV,
        PER_USER_CSV=PER_USER_CSV, TASTE_PQ=TASTE_PQ,
        ONTO_BASE=ONTO_BASE, ONTO_OUT=ONTO_OUT,
        ONTO_OUT_SIMPLE=ONTO_OUT_SIMPLE,
        LISTENING_NT=LISTENING_NT, LISTENING_NT_SIM=LISTENING_NT_SIM,
        INTERIM_CSV=INTERIM_CSV, KG_INPUT_PQ=KG_INPUT_PQ,
        KG_TASTE_PQ=KG_TASTE_PQ, SPLIT_PQ=SPLIT_PQ, KFOLD_CSV=KFOLD_CSV,
        KG_GRAPHDB_DIR=KG_GRAPHDB_DIR, KG_STATS_DIR=KG_STATS_DIR,
        KG_PLOTS_DIR=KG_PLOTS_DIR,
        FINAL_SPLITS_DIR=FINAL_SPLITS_DIR, FINAL_MODELS_DIR=FINAL_MODELS_DIR,
        KNN_VAL_CSV=KNN_VAL_CSV, KNN_TEST_CSV=KNN_TEST_CSV,
        KNN_VAL_PLOT_PNG=KNN_VAL_PLOT_PNG,
        KNN_POP_CSV=KNN_POP_CSV, KNN_POP_JSON=KNN_POP_JSON,
        HGT_RESULT_PATH=HGT_RESULT_PATH, AE_EMBEDDINGS_PQ=AE_EMBEDDINGS_PQ,
        KNN_RESULTS_DIR=KNN_RESULTS_DIR, HGT_RESULTS_DIR=HGT_RESULTS_DIR,
        AE_RESULTS_DIR=AE_RESULTS_DIR,
        MODELS_DIR=MODELS_DIR, AE_WEIGHTS_DIR=AE_WEIGHTS_DIR,
        HGT_WEIGHTS_DIR=HGT_WEIGHTS_DIR, KNN_CACHE_DIR=KNN_CACHE_DIR,
        KNN_NBRS_CACHE=KNN_NBRS_CACHE,
        HGT_MODEL_PATH=HGT_MODEL_PATH, AE_MODEL_PATH=AE_MODEL_PATH,
        JSYMBOLIC_JAR=JSYMBOLIC_JAR, GDRIVE_DATA_ROOT=GDRIVE_DATA_ROOT,
        # expose helpers so other cells can use them if needed
        _short=_short, _resolve_secret=_resolve_secret,
    )
