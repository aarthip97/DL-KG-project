"""Compact, report/poster-ready training+selection+ablation panel for the HGT.

Everything is rebuilt from the artefacts written under ``data/final/`` by the
pipeline — no model in memory, no retraining — so it runs standalone in Colab and
saves a single PNG you can drop into the report/poster.

Sub-panels (pick any via ``include``):
  * ``"loss"``      — train vs validation listwise loss per epoch.
  * ``"selection"`` — validation Overall_Score@10 and its three components
                      (NDCG@10, Coverage@10, 1−PopBias@10) per epoch.
  * ``"lambda"``    — λ_reg ablation: how the regulariser trades accuracy for
                      coverage (and lifts Overall@10).
  * ``"init"``      — RotatE-KGE vs random node initialisation (test metrics).
  * ``"sweep"``     — phase-1 one-factor-at-a-time hyper-parameter sweep.

Selection criterion (matches the trainer):
    Overall_Score@10 = 0.6·NDCG + 0.2·Coverage + 0.2·(1 − PopularityBias)
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# consistent colours across panels
_C_OVERALL = "#222222"
_C_NDCG = "#1f77b4"
_C_COV = "#2ca02c"
_C_POP = "#d62728"        # used for (1 − PopBias)
_C_TRAIN = "#1f77b4"
_C_VAL = "#ff7f0e"
_C_ROTATE = "#2E9B9B"
_C_RANDOM = "#9aa0aa"


def overall_at_k(ndcg: float, coverage: float, pop_bias: float) -> float:
    """The trainer's selection score: 0.6·NDCG + 0.2·Cov + 0.2·(1−PopBias)."""
    return 0.6 * ndcg + 0.2 * coverage + 0.2 * (1.0 - pop_bias)


def _load_cpu(path: Path):
    """Unpickle a TrainResult saved with CUDA tensors, mapping storages to CPU."""
    import torch

    class _CPU(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
            return super().find_class(module, name)

    try:
        with open(path, "rb") as fh:
            return _CPU(fh).load()
    except Exception:                       # noqa: BLE001
        return None


# ── individual sub-panels ────────────────────────────────────────────────────
def _panel_loss(ax, loss_df, best_epoch):
    ax.plot(loss_df["epoch"], loss_df["train/listwise_loss"],
            color=_C_TRAIN, lw=1.8, label="train")
    if "val/listwise_loss" in loss_df:
        v = loss_df.dropna(subset=["val/listwise_loss"])
        if len(v):
            ax.plot(v["epoch"], v["val/listwise_loss"], "o-", color=_C_VAL,
                    ms=3, lw=1.3, label="validation")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#888", ls="--", lw=1.0)
    ax.set_xlabel("epoch"); ax.set_ylabel("listwise loss")
    ax.set_title("a) Training vs validation loss", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)


def _panel_selection(ax, loss_df, best_epoch):
    v = loss_df.dropna(subset=["val/ndcg@10"])
    e = v["epoch"]
    overall = (v["val/monitor_score"] if "val/monitor_score" in v
               else overall_at_k(v["val/ndcg@10"], v["val/coverage@10"],
                                 v["val/pop_bias@10"]))
    ax.plot(e, overall, color=_C_OVERALL, lw=2.2, label="Overall@10", zorder=5)
    ax.plot(e, v["val/ndcg@10"], color=_C_NDCG, lw=1.4, label="NDCG@10")
    ax.plot(e, v["val/coverage@10"], color=_C_COV, lw=1.4, label="Coverage@10")
    ax.plot(e, 1.0 - v["val/pop_bias@10"], color=_C_POP, lw=1.4,
            label="1 − PopBias@10")
    if best_epoch is not None:
        bo = float(overall.loc[v["epoch"] == best_epoch].iloc[0])
        ax.axvline(best_epoch, color="#888", ls="--", lw=1.0)
        ax.scatter([best_epoch], [bo], color=_C_OVERALL, zorder=6, s=30)
        ax.annotate(f"best {bo:.3f} (ep {int(best_epoch)})",
                    (best_epoch, bo), textcoords="offset points", xytext=(-8, -14),
                    fontsize=8, ha="right", va="top", color=_C_OVERALL)
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("b) Validation Overall@10 & components", fontsize=10, loc="left")
    # mid-right gap (between Overall ~0.37 and Coverage ~0.69) is clear of curves
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, ncol=2, loc="center right")


def _panel_lambda(ax, lam_df, chosen=None):
    d = lam_df.sort_values("lambda_reg")
    x = d["lambda_reg"]
    ax.plot(x, d["overall@10"], "o-", color=_C_OVERALL, lw=2.0, label="Overall@10",
            zorder=5)
    ax.plot(x, d["val_ndcg@10"], "s--", color=_C_NDCG, lw=1.2, ms=4, label="NDCG@10")
    ax.plot(x, d["val_coverage@10"], "^--", color=_C_COV, lw=1.2, ms=4,
            label="Coverage@10")
    ax.plot(x, 1.0 - d["val_pop_bias@10"], "v--", color=_C_POP, lw=1.2, ms=4,
            label="1 − PopBias@10")
    if chosen is not None:
        ax.axvline(chosen, color="#888", ls=":", lw=1.2)
        _row = d[d["lambda_reg"] == chosen]
        y_at = float(_row["overall@10"].iloc[0]) if len(_row) else 0.25
        ax.annotate(f"chosen λ={chosen:g}", (chosen, y_at),
                    textcoords="offset points", xytext=(-6, -12),
                    fontsize=8, ha="right", va="top", color="#555",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.8))
    ax.set_xlabel("λ_reg  (coverage regulariser)"); ax.set_ylabel("validation score")
    ax.set_ylim(0, 1.0)
    ax.set_title("c) λ_reg ablation — accuracy ↔ coverage trade-off",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, ncol=2, loc="center right")


def _panel_init(ax, rotate_tm, random_tm):
    names = ["Overall@10", "NDCG@10", "Coverage@10", "1−PopBias@10"]

    def _vals(tm):
        return [overall_at_k(tm["ndcg@10"], tm["coverage@10"], tm["pop_bias@10"]),
                tm["ndcg@10"], tm["coverage@10"], 1.0 - tm["pop_bias@10"]]

    r, q = _vals(rotate_tm), _vals(random_tm)
    x = np.arange(len(names)); w = 0.38
    b1 = ax.bar(x - w / 2, r, w, color=_C_ROTATE, label="RotatE KGE")
    b2 = ax.bar(x + w / 2, q, w, color=_C_RANDOM, label="random init")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel("test score")
    ax.set_title("d) Node init: RotatE KGE vs random", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)


def _panel_sweep(ax, sweep_df):
    d = sweep_df.sort_values("overall@10")
    axes_col = d["axis"] if "axis" in d else pd.Series(["?"] * len(d))
    cats = list(dict.fromkeys(axes_col))
    cmap = {c: col for c, col in zip(cats, plt_color_cycle(len(cats)))}
    colors = [cmap[a] for a in axes_col]
    y = np.arange(len(d))
    ax.barh(y, d["overall@10"], color=colors)
    labels = [f"{a}: h{int(r.hidden_channels)} L{int(r.num_layers)} "
              f"H{int(r.num_heads)} d{r.dropout:g} lr{r.lr:g} τ{r.temperature:g}"
              for a, (_, r) in zip(axes_col, d.iterrows())]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Overall@10 (validation)")
    ax.set_title("Phase-1 hyper-parameter sweep", fontsize=10, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap[c], label=c) for c in cats],
              fontsize=7, frameon=False, ncol=2)


def plt_color_cycle(n):
    import matplotlib.pyplot as plt
    base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [base[i % len(base)] for i in range(n)]


# ── orchestrator ─────────────────────────────────────────────────────────────
def plot_hgt_panel(
    final_dir,
    *,
    include: Sequence[str] = ("loss", "selection", "lambda", "init"),
    save_path: Optional[str] = None,
    title: Optional[str] = "HGT — training, model selection & ablations",
    ncols: int = 2,
    panel_size=(5.5, 3.6),
    dpi: int = 150,
    verbose: bool = True,
):
    """Build (and optionally save) the HGT training/selection/ablation panel.

    Reads only the on-disk artefacts in ``final_dir`` — ``hgt_loss_history.csv``,
    ``hgt_loss_history_meta.json``, ``hgt_lambda_ablation.csv``,
    ``hgt_ablation_phase1.csv``, ``hgt_results.pkl`` / ``hgt_results_random.pkl``.
    Panels whose artefacts are missing are skipped with a note (so it degrades
    gracefully if you only ran part of the pipeline).

    Returns the matplotlib ``Figure`` (also written to ``save_path`` when given).
    """
    import matplotlib.pyplot as plt

    fdir = Path(final_dir)
    loss_csv = fdir / "hgt_loss_history.csv"
    meta_json = fdir / "hgt_loss_history_meta.json"
    lam_csv = fdir / "hgt_lambda_ablation.csv"
    sweep_csv = fdir / "hgt_ablation_phase1.csv"
    main_pkl = fdir / "hgt_results.pkl"
    rand_pkl = fdir / "hgt_results_random.pkl"

    loss_df = pd.read_csv(loss_csv) if loss_csv.exists() else None
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}

    best_epoch = None
    if loss_df is not None and "val/monitor_score" in loss_df:
        v = loss_df.dropna(subset=["val/monitor_score"])
        if len(v):
            best_epoch = int(v.loc[v["val/monitor_score"].idxmax(), "epoch"])

    # decide which panels we can actually draw
    ready = []
    for p in include:
        if p in ("loss", "selection") and loss_df is None:
            if verbose:
                print(f"[skip] '{p}': {loss_csv.name} not found")
            continue
        if p == "lambda" and not lam_csv.exists():
            if verbose:
                print(f"[skip] 'lambda': {lam_csv.name} not found")
            continue
        if p == "init" and not (main_pkl.exists() and rand_pkl.exists()):
            if verbose:
                print("[skip] 'init': need hgt_results.pkl + hgt_results_random.pkl")
            continue
        if p == "sweep" and not sweep_csv.exists():
            if verbose:
                print(f"[skip] 'sweep': {sweep_csv.name} not found")
            continue
        ready.append(p)
    if not ready:
        raise RuntimeError("No panels could be built — check artefacts in "
                           f"{fdir}")

    n = len(ready)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]

    chosen_lambda = (meta.get("config", {}) or {}).get("lambda_reg")
    for ax, p in zip(flat, ready):
        if p == "loss":
            _panel_loss(ax, loss_df, best_epoch)
        elif p == "selection":
            _panel_selection(ax, loss_df, best_epoch)
        elif p == "lambda":
            _panel_lambda(ax, pd.read_csv(lam_csv), chosen=chosen_lambda)
        elif p == "init":
            main = meta.get("test_metrics") or getattr(_load_cpu(main_pkl),
                                                        "test_metrics", None)
            rand = getattr(_load_cpu(rand_pkl), "test_metrics", None)
            if main and rand:
                _panel_init(ax, main, rand)
            else:
                ax.set_axis_off()
        elif p == "sweep":
            _panel_sweep(ax, pd.read_csv(sweep_csv))
    for ax in flat[n:]:
        ax.set_axis_off()

    if title:
        cfg = meta.get("config", {})
        sub = (f"  (hidden {cfg.get('hidden_channels','?')} · heads "
               f"{cfg.get('num_heads','?')} · layers {cfg.get('num_layers','?')} · "
               f"λ_reg {cfg.get('lambda_reg','?')} · τ {cfg.get('temperature','?')})"
               if cfg else "")
        fig.suptitle(title + sub, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"[saved] {save_path}")
    return fig


__all__ = ["plot_hgt_panel", "overall_at_k"]
