#!/usr/bin/env python
"""Local preview / load-or-rebuild for the §9 benchmark figures.

Reads the CSVs in ``data/final/benchmark/`` and, for each figure (multi-K
heatmaps, significance bars), **keeps** the PNG on disk if it is already there,
otherwise **rebuilds** it from the stored CSV with the current plotting code and
saves it. So you can inspect/iterate on the plots locally — no Colab, no
re-running the benchmark: delete a PNG you do not like and re-run this to
regenerate it (mirrors the notebook's load-or-rebuild cell).

Only needs numpy / pandas / scipy / matplotlib — it loads ``comparison.py``
standalone (via importlib), so it does NOT import torch / torch_geometric.

Usage:
    .venv/bin/python scripts/preview_benchmark_plots.py [--force] [--bench-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: just write PNGs
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_comparison():
    """Import comparison.py directly, bypassing the heavy package __init__."""
    path = ROOT / "src" / "models" / "evaluation" / "comparison.py"
    spec = importlib.util.spec_from_file_location("benchmark_comparison", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default=str(ROOT / "data" / "final" / "benchmark"))
    ap.add_argument("--force", action="store_true",
                    help="rebuild every figure even if its PNG already exists")
    args = ap.parse_args()

    bench = Path(args.bench_dir)
    cmp = _load_comparison()

    agg = pd.read_csv(bench / "benchmark_multiK.csv", index_col=["model", "K"])
    pair = pd.read_csv(bench / "benchmark_significance.csv")
    order = None
    sel = bench / "benchmark_selection.csv"
    if sel.exists():
        order = list(pd.read_csv(sel, index_col=0).index)
    pss_p = bench / "benchmark_peruser_sample.csv"          # per-user violins
    psamp = pd.read_csv(pss_p) if pss_p.exists() else None

    figures = {
        "heatmaps": (
            bench / "benchmark_multiK_heatmaps.png",
            lambda p: cmp.plot_benchmark_heatmaps(agg, save_path=p)),
        "significance": (
            bench / "benchmark_significance_bars.png",
            lambda p: cmp.plot_significance_bars(pair, per_user_sample=psamp,
                                                 model_order=order, save_path=p)),
    }
    for name, (png, build) in figures.items():
        if png.exists() and not args.force:
            print(f"[keep]    {name:12s} -> {png}   (delete it to rebuild)")
            continue
        fig = build(png)
        plt.close(fig)
        print(f"[rebuilt] {name:12s} -> {png}")


if __name__ == "__main__":
    main()
