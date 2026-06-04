#!/usr/bin/env python
"""Local preview / rebuild for the §13 latent-space (GMM) figures.

Loads the persisted analysis under ``data/final/latent/`` (no model, no forward
pass) and rebuilds the cluster scatter plots — all node types and users-only —
plus the BIC curves, saving them as PNGs so you can iterate on the plotting
locally without Colab. Mirrors what §13 draws.

Usage:
    .venv/bin/python scripts/preview_latent_plots.py [--latent-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.latent_space import (LatentAnalysis, plot_latent_2d,
                                  plot_gmm_bic_curves)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-dir", default=str(ROOT / "data" / "final" / "latent"))
    args = ap.parse_args()
    latent = Path(args.latent_dir)

    a = LatentAnalysis.load(latent)

    def _scatter(sub, centers, params, name):
        if sub is None:
            print(f"[skip]    {name}: no subsample on disk")
            return
        fig = plot_latent_2d(
            sub[["X", "Y"]].to_numpy(), sub["Node_Type"].values,
            labels=sub["Cluster_ID"].to_numpy(), centers_2d=centers,
            best_params=params)
        out = latent / name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[rebuilt] {out}")

    _scatter(a.subsample, a.centers_2d, a.gmm.best_params,
             "latent_clusters_all.png")
    if a.gmm_users is not None:
        _scatter(a.user_subsample, a.user_centers_2d, a.gmm_users.best_params,
                 "latent_clusters_users.png")

    krange = list(a.k_range or a.gmm.n_components_range)
    fig = plot_gmm_bic_curves(a.gmm.bic_matrix, krange, a.gmm.best_params,
                              save_path=latent / "latent_gmm_bic.png")
    plt.close(fig)
    print(f"[rebuilt] {latent / 'latent_gmm_bic.png'}")


if __name__ == "__main__":
    main()
