"""
nb_env.py — Notebook environment & path configuration
======================================================
Imported by every notebook immediately after ROOT and ON_COLAB are set.

Usage (notebook cell 4):
    from nb_env import setup
    globals().update(setup(ROOT, ON_COLAB))

`setup()` returns a plain dict whose keys become notebook globals:
    ROOT, ON_COLAB, DEVICE, SEED, USE_GRAPHDB, CAPACITY
    _DATA, RAW, INTERIM, PROCESSED, FINAL, ONTOLOGY
    LAKH_PQ, PER_SONG_CSV, PER_USER_CSV, TASTE_PQ
    ONTO_BASE, ONTO_OUT, ONTO_OUT_SIMPLE, LISTENING_NT, LISTENING_NT_SIM
    INTERIM_CSV, KG_INPUT_PQ, KG_TASTE_PQ, SPLIT_PQ, KFOLD_CSV
    KG_GRAPHDB_DIR, KG_STATS_DIR, KG_PLOTS_DIR
    FINAL_SPLITS_DIR
    KNN_VAL_CSV, KNN_TEST_CSV, KNN_VAL_PLOT_PNG
    KNN_POP_CSV, KNN_POP_JSON, HGT_RESULT_PATH, AE_EMBEDDINGS_PQ
    MODELS_DIR, AE_WEIGHTS_DIR, HGT_WEIGHTS_DIR, KNN_CACHE_DIR, KGE_WEIGHTS_DIR
    KNN_NBRS_CACHE, HGT_MODEL_PATH, AE_MODEL_PATH
    KGE_ROTATE_PATH, KGE_COMPLEX_PATH
    QUALITATIVE_DIR
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
            return Path(p)
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
        _gdrive_str: str = _resolve_secret(
            "gdrive_proj_path",
            default="/content/drive/MyDrive/DL-KG-project",
        ) or "/content/drive/MyDrive/DL-KG-project"
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
        _dotenv_loaded_colab = _load_first_existing_dotenv(_dotenv_candidates)
        if not _dotenv_loaded_colab:
            print("  [.env] not found in Drive or repo — continuing without it")

        # Re-resolve now that .env may have added more vars
        _gdrive_str = _resolve_secret("gdrive_proj_path", default=_gdrive_str) or _gdrive_str
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

        # Persist trained model weights to Drive as well. data/ already lives on
        # Drive via the symlink above, but models/ did not — so on Colab the VM
        # recycling silently dropped every trained weight file. Mirroring it to
        # Drive makes Colab behave like a local checkout (weights survive runs).
        if gdrive_proj.exists():
            _gdrive_models = gdrive_proj / "models"
            _gdrive_models.mkdir(parents=True, exist_ok=True)
            _ensure_symlink(_gdrive_models, ROOT / "models")

        # ── W&B auth ──────────────────────────────────────────────────────────
        _wandb_key = _resolve_secret("WANDB_API_KEY")
        if _wandb_key:
            try:
                import wandb  # type: ignore[import]
                #Temporary hack to prevent colab from hanging
                # sys.modules["google.colab2"] = sys.modules["google.colab"]
                # del sys.modules["google.colab"]
                wandb.login(key=_wandb_key, relogin=True)
                # sys.modules["google.colab"] = sys.modules["google.colab2"]
                print("W&B authenticated.")
            except Exception as e:
                print(f"  [WARN] W&B login failed: {e}")
        else:
            print("  [INFO] WANDB_API_KEY not set — W&B logging disabled.")

    else:
        _dotenv_loaded = _load_first_existing_dotenv([ROOT / ".env", Path.cwd() / ".env"])
        print(f"  [.env] {'found' if _dotenv_loaded else 'not found'}")

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
    ONTO_BASE        = ONTOLOGY / "knowledge_graph_original.ttl"
    ONTO_OUT         = ONTOLOGY / "knowledge_graph_rich.ttl"
    ONTO_OUT_SIMPLE  = ONTOLOGY / "knowledge_graph_simple.ttl"
    LISTENING_NT     = ONTOLOGY / "listening_triples_rich.nt"
    LISTENING_NT_SIM = ONTOLOGY / "listening_triples_simple.nt"

    # DL pipeline artefacts
    JSYMB_CSV    = INTERIM   / "jsymbolic_features.csv"
    M21_CSV      = INTERIM   / "music21_features.csv"
    FEATURES_PQ  = PROCESSED / 'music_features.parquet'
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
    XGB_RESULTS_CSV  = FINAL / "xgb_hybrid_results.csv"
    XGB_MODEL_CACHE  = FINAL / "xgb_hybrid.ubj"
    # Training/validation loss histories — CSV + PNG duplicates of the W&B curves,
    # persisted to Drive (data/final is symlinked) so they survive without W&B and
    # are trivially extractable (pandas/Excel), no unpickling required.
    AE_HISTORY_CSV   = FINAL / "ae_loss_history.csv"
    AE_LOSS_PNG      = FINAL / "ae_loss_curve.png"
    HGT_HISTORY_CSV  = FINAL / "hgt_loss_history.csv"
    HGT_CURVES_PNG   = FINAL / "hgt_training_curves.png"
    KGE_HISTORY_CSV  = FINAL / "kge_loss_history.csv"
    KGE_LOSS_PNG     = FINAL / "kge_loss_curve.png"

    # Model weight directories (lives in ROOT/models/ — gitignored)
    MODELS_DIR       = ROOT / "models"
    AE_WEIGHTS_DIR   = MODELS_DIR / "autoencoder"
    HGT_WEIGHTS_DIR  = MODELS_DIR / "hgt"
    KNN_CACHE_DIR    = MODELS_DIR / "knn"
    KGE_WEIGHTS_DIR  = MODELS_DIR / "kge"
    KNN_NBRS_CACHE   = KNN_CACHE_DIR / "neighbours.npz"
    HGT_MODEL_PATH   = HGT_WEIGHTS_DIR / "model.pt"
    AE_MODEL_PATH    = AE_WEIGHTS_DIR / "model.pt"
    KGE_ROTATE_PATH  = KGE_WEIGHTS_DIR / "kge_rotate_embeddings.pt"
    KGE_COMPLEX_PATH = KGE_WEIGHTS_DIR / "kge_complex_embeddings.pt"

    # Qualitative analysis outputs (population-level CSVs + plots)
    QUALITATIVE_DIR = FINAL / "qualitative"

    # Wikidata enrichment artefacts
    WD_INSTR_PQ        = INTERIM / "wikidata_instruments.json"
    WD_INSTR_CHAINS_PQ = INTERIM / "wikidata_instrument_chains.json"
    WD_GENRE_PQ        = INTERIM / "wikidata_genres.json"
    WD_GENRE_CHAINS_PQ = INTERIM / "wikidata_genre_chains.json"
    WD_QID_META_PQ     = INTERIM / "wikidata_qid_metadata.json"
    WD_DECADES_PQ      = INTERIM / "wikidata_decades.json"

    # jSymbolic — local: always repo-local first; Colab: Drive path via secret
    _jsym_default = str(ROOT / "jSymbolic" / "jSymbolic2.jar")
    if ON_COLAB:
        JSYMBOLIC_JAR = Path(
            _resolve_secret("jSymbolic2_path", default=_jsym_default) or _jsym_default
        )
    else:
        # On local, ignore any Drive path that may have leaked into the env;
        # only honour jSymbolic2_path if it points to something that actually exists.
        _jsym_override = _resolve_secret("jSymbolic2_path")
        if _jsym_override and Path(_jsym_override).exists():
            JSYMBOLIC_JAR = Path(_jsym_override)
        else:
            JSYMBOLIC_JAR = Path(_jsym_default)
    JSYMB_CONFIG = str(ROOT / "jSymbolic" / "jSymbolicConfig.txt")
    GDRIVE_DATA_ROOT = (ROOT / "data") if ON_COLAB else None

    # ── Create directories ────────────────────────────────────────────────────
    for _p in [
        INTERIM, PROCESSED, FINAL,
        KG_GRAPHDB_DIR, KG_STATS_DIR, KG_PLOTS_DIR,
        FINAL_SPLITS_DIR, QUALITATIVE_DIR,
        MODELS_DIR, AE_WEIGHTS_DIR, HGT_WEIGHTS_DIR, KNN_CACHE_DIR, KGE_WEIGHTS_DIR,
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

    # ── Hardware-aware capacity ───────────────────────────────────────────────
    # Detect VRAM / RAM and recommend batch sizes + workload counts so the same
    # notebook scales from a Colab T4 up to a large GPU server automatically.
    # Exposed as the CAPACITY global; training cells read it with the small-tier
    # values as a safe baseline and may always override.
    try:
        from utils.resources import recommend_capacity  # noqa: PLC0415
        CAPACITY = recommend_capacity(verbose=True)
    except Exception as _cap_err:  # noqa: BLE001
        print(f"  [INFO] capacity auto-sizing unavailable ({_cap_err}); using defaults.")
        CAPACITY = {}

    # ── GraphDB flag ──────────────────────────────────────────────────────────
    USE_GRAPHDB: bool = (not ON_COLAB) and bool(os.environ.get("GRAPHDB_URL"))

    # ── W&B run config ──────────────────────────────────────────────────────────
    # Resolved from Colab Secret → env var → project default, so every notebook
    # cell can use WANDB_PROJECT/ENTITY/GROUP as globals without each contributor
    # hardcoding (or forgetting) them. The API *key* is never read here — it stays
    # a Colab Secret / netrc entry and is consumed by the wandb.login() above.
    WANDB_PROJECT = _resolve_secret("WANDB_PROJECT", default="kgdl2526musicrecs")
    WANDB_ENTITY  = _resolve_secret("WANDB_ENTITY",  default="kgdlmusicrecs-fcul")
    WANDB_GROUP   = _resolve_secret("WANDB_GROUP",   default="devel")

    # ── Sanity print ──────────────────────────────────────────────────────────
    def _rel(p: Path) -> str:
        """Show path relative to ROOT, or just the name if it escapes ROOT."""
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return p.name  # never expose absolute system paths

    _jsym_status = "✓ found" if JSYMBOLIC_JAR.exists() else "✗ not found"
    _data_suffix = f"  → {_DATA.resolve().name}" if _DATA.is_symlink() else ""
    print(f"  root       : {ROOT.name}/")
    print(f"  data/      : {_rel(_DATA)}/{_data_suffix}")
    print(f"  models/    : {_rel(MODELS_DIR)}/")
    print(f"  jSymbolic  : {_rel(JSYMBOLIC_JAR)}  ({_jsym_status})")
    print(f"  device     : {DEVICE}")
    print(f"  USE_GRAPHDB: {USE_GRAPHDB}")

    # Return every name that downstream cells need as a global
    return {
        "ROOT": ROOT, "ON_COLAB": ON_COLAB, "SEED": SEED, "DEVICE": DEVICE,
        "USE_GRAPHDB": USE_GRAPHDB, "CAPACITY": CAPACITY,
        "WANDB_PROJECT": WANDB_PROJECT, "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_GROUP": WANDB_GROUP,
        "_DATA": _DATA, "RAW": RAW, "INTERIM": INTERIM, "PROCESSED": PROCESSED,
        "FINAL": FINAL, "ONTOLOGY": ONTOLOGY,
        "LAKH_PQ": LAKH_PQ, "PER_SONG_CSV": PER_SONG_CSV,
        "PER_USER_CSV": PER_USER_CSV, "TASTE_PQ": TASTE_PQ,
        "ONTO_BASE": ONTO_BASE, "ONTO_OUT": ONTO_OUT,
        "ONTO_OUT_SIMPLE": ONTO_OUT_SIMPLE,
        "LISTENING_NT": LISTENING_NT, "LISTENING_NT_SIM": LISTENING_NT_SIM,
        "JSYMB_CSV": JSYMB_CSV, "M21_CSV": M21_CSV,
        "JSYMB_CONFIG": JSYMB_CONFIG, "KG_INPUT_PQ": KG_INPUT_PQ,
        "FEATURES_PQ": FEATURES_PQ,
        "KG_TASTE_PQ": KG_TASTE_PQ, "SPLIT_PQ": SPLIT_PQ, "KFOLD_CSV": KFOLD_CSV,
        "KG_GRAPHDB_DIR": KG_GRAPHDB_DIR, "KG_STATS_DIR": KG_STATS_DIR,
        "KG_PLOTS_DIR": KG_PLOTS_DIR,
        "FINAL_SPLITS_DIR": FINAL_SPLITS_DIR,
        "KNN_VAL_CSV": KNN_VAL_CSV, "KNN_TEST_CSV": KNN_TEST_CSV,
        "KNN_VAL_PLOT_PNG": KNN_VAL_PLOT_PNG,
        "KNN_POP_CSV": KNN_POP_CSV, "KNN_POP_JSON": KNN_POP_JSON,
        "HGT_RESULT_PATH": HGT_RESULT_PATH, "AE_EMBEDDINGS_PQ": AE_EMBEDDINGS_PQ,
        "XGB_RESULTS_CSV": XGB_RESULTS_CSV, "XGB_MODEL_CACHE": XGB_MODEL_CACHE,
        "AE_HISTORY_CSV": AE_HISTORY_CSV, "AE_LOSS_PNG": AE_LOSS_PNG,
        "HGT_HISTORY_CSV": HGT_HISTORY_CSV, "HGT_CURVES_PNG": HGT_CURVES_PNG,
        "KGE_HISTORY_CSV": KGE_HISTORY_CSV, "KGE_LOSS_PNG": KGE_LOSS_PNG,
        "MODELS_DIR": MODELS_DIR, "AE_WEIGHTS_DIR": AE_WEIGHTS_DIR,
        "HGT_WEIGHTS_DIR": HGT_WEIGHTS_DIR, "KNN_CACHE_DIR": KNN_CACHE_DIR,
        "KGE_WEIGHTS_DIR": KGE_WEIGHTS_DIR,
        "KNN_NBRS_CACHE": KNN_NBRS_CACHE,
        "HGT_MODEL_PATH": HGT_MODEL_PATH, "AE_MODEL_PATH": AE_MODEL_PATH,
        "KGE_ROTATE_PATH": KGE_ROTATE_PATH, "KGE_COMPLEX_PATH": KGE_COMPLEX_PATH,
        "QUALITATIVE_DIR": QUALITATIVE_DIR,
        "JSYMBOLIC_JAR": JSYMBOLIC_JAR, "GDRIVE_DATA_ROOT": GDRIVE_DATA_ROOT,
        "WD_INSTR_PQ": WD_INSTR_PQ, "WD_INSTR_CHAINS_PQ": WD_INSTR_CHAINS_PQ,
        "WD_GENRE_PQ": WD_GENRE_PQ, "WD_GENRE_CHAINS_PQ": WD_GENRE_CHAINS_PQ,
        "WD_QID_META_PQ": WD_QID_META_PQ, "WD_DECADES_PQ": WD_DECADES_PQ,
        # expose helpers so other cells can use them if needed
        "_short": _short, "_resolve_secret": _resolve_secret,
    }
