#!/usr/bin/env python
"""Headless end-to-end DL pipeline runner.

This script reproduces the Section "Autoencoder -> KGE -> HGT" stages of
notebooks/04_DL_pipeline.ipynb in a strictly command-line workflow that is
suitable for Slurm or any SSH-only environment.  It assumes the data
artefacts up to the populated TTL already exist on disk (i.e. that the
notebook stages 1-4 have been completed at least once).

Usage examples
--------------
Train all three stages from scratch:
    python scripts/run_pipeline.py all \
        --data-root data \
        --kg-ttl data/final/MusicRecSyst_populated.ttl \
        --epochs-ae 30 --epochs-kge 200 --epochs-hgt 100 \
        --device cuda --wandb-project music-recommender-system

Train only the autoencoder:
    python scripts/run_pipeline.py autoencoder --data-root data --epochs-ae 30

Train only the KGE step (assumes triple TSV already exists):
    python scripts/run_pipeline.py kge \
        --kg-triples data/interim/kg_triples.tsv \
        --epochs-kge 500 --device cuda

Train only the HGT step (assumes embeddings + edge_dict on disk):
    python scripts/run_pipeline.py hgt \
        --data-root data --epochs-hgt 100 --device cuda

Outputs
-------
- data/interim/kg_triples.tsv    PyKEEN triple file
- data/interim/edge_dict.json    HeteroData edge index dict
- models/autoencoder/ae_model.pt
- models/kge/kge_checkpoint.pt
- models/hgt/hgt_model.pt
- models/hgt/hgt_results.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make src/ importable when run from the repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def stage_extract_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    """Parse the populated TTL and emit (kg_triples.tsv, edge_dict.json)."""
    import rdflib
    from models.kg_to_hetero import extract_dl_artifacts

    ttl_path = Path(args.kg_ttl)
    interim  = Path(args.data_root) / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    tsv_out  = interim / "kg_triples.tsv"
    dict_out = interim / "edge_dict.json"

    if tsv_out.exists() and dict_out.exists() and not args.force:
        log.info("Skipping artefact extraction (cache hit). Use --force to rebuild.")
        return tsv_out, dict_out

    log.info("Loading TTL from %s", ttl_path)
    g = rdflib.Graph()
    g.parse(ttl_path)
    log.info("Parsed graph with %d triples", len(g))

    extract_dl_artifacts(g, str(tsv_out), str(dict_out))
    log.info("Wrote %s and %s", tsv_out, dict_out)
    return tsv_out, dict_out


def stage_autoencoder(args: argparse.Namespace) -> Path:
    """Train the jSymbolic autoencoder, save model + per-track embeddings."""
    from models.autoencoder import (
        jSymbolicAutoencoder, train_autoencoder, extract_embeddings,
    )

    interim    = Path(args.data_root) / "interim"
    weights    = Path(args.data_root).parent / "models" / "autoencoder"
    weights.mkdir(parents=True, exist_ok=True)
    model_path = weights / "ae_model.pt"
    emb_path   = interim / "ae_embeddings.parquet"

    feat_csv = Path(args.feature_csv) if args.feature_csv else interim / "interim.csv"
    if not feat_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feat_csv}")

    log.info("Loading features from %s", feat_csv)
    df = pd.read_csv(feat_csv)
    track_ids = df["song_id"] if "song_id" in df.columns else df.index
    feat_cols = [c for c in df.columns if c not in {"song_id"}]
    X = df[feat_cols].select_dtypes(include="number").to_numpy(dtype=np.float32)
    log.info("Feature matrix shape: %s", X.shape)

    model = jSymbolicAutoencoder(input_dim=X.shape[1], bottleneck=args.ae_bottleneck)

    model, history = train_autoencoder(
        model, X,
        epochs=args.epochs_ae,
        batch_size=args.batch_ae,
        lr=args.lr_ae,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=f"ae_dim{args.ae_bottleneck}_ep{args.epochs_ae}",
    )

    log.info("Final train loss: %.6f", history["train_loss"][-1])
    torch.save(model.state_dict(), model_path)
    log.info("Saved autoencoder weights -> %s", model_path)

    emb = extract_embeddings(model, X, device=args.device, as_dataframe=True, index=track_ids)
    emb.to_parquet(emb_path)
    log.info("Saved %d-dim embeddings (%d tracks) -> %s",
             emb.shape[1], emb.shape[0], emb_path)
    return model_path


def stage_kge(args: argparse.Namespace) -> Path:
    """Train RotatE/ComplEx via PyKEEN, save embedding checkpoint."""
    from models.kg_embeddings import train_kge

    triples_path = Path(args.kg_triples) if args.kg_triples \
        else Path(args.data_root) / "interim" / "kg_triples.tsv"
    if not triples_path.exists():
        raise FileNotFoundError(
            f"Triple TSV not found: {triples_path}.  Run the 'extract' "
            f"stage or the full pipeline first."
        )

    weights = Path(args.data_root).parent / "models" / "kge"
    weights.mkdir(parents=True, exist_ok=True)
    cp_path = weights / "kge_checkpoint.pt"

    train_kge(
        triples_path,
        model=args.kge_model,
        entity_dim=args.kge_dim,
        epochs=args.epochs_kge,
        batch_size=args.batch_kge,
        lr=args.lr_kge,
        device=args.device,
        checkpoint_path=cp_path,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{args.kge_model}_dim{args.kge_dim}_ep{args.epochs_kge}",
    )
    log.info("KGE checkpoint saved -> %s", cp_path)
    return cp_path


def stage_hgt(args: argparse.Namespace) -> Path:
    """Build HeteroData, train HGT recommender, save model + results."""
    from models.kg_embeddings import load_kge_checkpoint
    from models.kg_to_hetero import build_rich_hetero_graph
    from models.train_DL import train_hgt

    interim   = Path(args.data_root) / "interim"
    weights   = Path(args.data_root).parent / "models" / "hgt"
    weights.mkdir(parents=True, exist_ok=True)
    model_path  = weights / "hgt_model.pt"
    result_path = weights / "hgt_results.pkl"

    edge_dict_path = interim / "edge_dict.json"
    kge_cp_path    = Path(args.data_root).parent / "models" / "kge" / "kge_checkpoint.pt"
    ae_emb_path    = interim / "ae_embeddings.parquet"

    for p in (edge_dict_path, kge_cp_path, ae_emb_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artefact missing: {p}")

    log.info("Loading edge_dict from %s", edge_dict_path)
    with edge_dict_path.open() as f:
        edge_dict = json.load(f)

    log.info("Loading KGE embeddings from %s", kge_cp_path)
    rotate_emb = load_kge_checkpoint(kge_cp_path)

    log.info("Loading audio embeddings from %s", ae_emb_path)
    ae_df = pd.read_parquet(ae_emb_path)
    audio_feats = {str(idx): row.to_numpy(dtype=np.float32) for idx, row in ae_df.iterrows()}

    log.info("Building HeteroData graph")
    data = build_rich_hetero_graph(
        edge_dict=edge_dict,
        rotate_embeddings=rotate_emb,
        track_audio_features=audio_feats,
    )
    log.info("HeteroData: %s", data)

    log.info("Training HGT for %d epochs on %s", args.epochs_hgt, args.device)
    result = train_hgt(
        data,
        epochs=args.epochs_hgt,
        batch_size=args.batch_hgt,
        lr=args.lr_hgt,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=f"hgt_h{args.hgt_hidden}_ep{args.epochs_hgt}",
    )

    torch.save(result.model.state_dict(), model_path)
    with result_path.open("wb") as f:
        pickle.dump({"history": result.history, "metrics": result.metrics}, f)
    log.info("Saved HGT model -> %s", model_path)
    log.info("Saved HGT results -> %s", result_path)
    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["all", "extract", "autoencoder", "kge", "hgt"],
                   help="Which pipeline stage to run.")

    # Paths
    p.add_argument("--data-root", default="data", help="Path to data/ directory.")
    p.add_argument("--kg-ttl",
                   default="data/final/MusicRecSyst_populated.ttl",
                   help="Populated TTL file (used by 'extract' / 'all').")
    p.add_argument("--kg-triples",
                   help="Pre-built triple TSV (skips extract stage when provided).")
    p.add_argument("--feature-csv",
                   help="Per-track feature CSV (defaults to data/interim/interim.csv).")
    p.add_argument("--force", action="store_true",
                   help="Force re-running stages even if cached outputs exist.")

    # General
    p.add_argument("--device", default=None,
                   help="'cuda' / 'cpu' / None (auto-detect).")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name (omit to disable W&B).")

    # Autoencoder hyper-params
    p.add_argument("--epochs-ae", type=int, default=30)
    p.add_argument("--batch-ae",  type=int, default=256)
    p.add_argument("--lr-ae",     type=float, default=1e-3)
    p.add_argument("--ae-bottleneck", type=int, default=128)

    # KGE hyper-params
    p.add_argument("--kge-model", choices=["RotatE", "ComplEx"], default="RotatE")
    p.add_argument("--kge-dim",   type=int, default=128,
                   help="Complex entity dim (output is 2*kge-dim wide).")
    p.add_argument("--epochs-kge", type=int, default=500)
    p.add_argument("--batch-kge",  type=int, default=512)
    p.add_argument("--lr-kge",     type=float, default=1e-3)

    # HGT hyper-params
    p.add_argument("--epochs-hgt", type=int, default=100)
    p.add_argument("--batch-hgt",  type=int, default=4096)
    p.add_argument("--lr-hgt",     type=float, default=1e-3)
    p.add_argument("--hgt-hidden", type=int, default=128)

    return p


def main() -> None:
    args = build_parser().parse_args()
    log.info("Stage = %s | device = %s", args.stage, args.device or "auto")

    if args.stage in {"all", "extract"}:
        stage_extract_artifacts(args)

    if args.stage in {"all", "autoencoder"}:
        stage_autoencoder(args)

    if args.stage in {"all", "kge"}:
        stage_kge(args)

    if args.stage in {"all", "hgt"}:
        stage_hgt(args)

    log.info("Done.")


if __name__ == "__main__":
    main()
