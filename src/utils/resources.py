"""Hardware-aware resource detection for dynamic batch / workload sizing.

Detects available GPU VRAM and system RAM, then maps them to a small set of
training / inference hyper-parameters (batch sizes, candidate counts,
user-chunk sizes). The same notebook can then run efficiently on a Colab T4
(16 GB) and scale up automatically on a large GPU server (e.g. an
RTX PRO 6000, 80--100 GB VRAM, 100+ GB RAM) without manual edits.

Two axes are detected independently because the workloads bottleneck on
different resources:

* **VRAM-bound** -- the autoencoder mini-batches and the HGT user-chunk size
  (the full graph is forwarded in VRAM; bigger chunks mean fewer Python-loop
  iterations per epoch).
* **RAM-bound** -- the XGBoost candidate count, number of LTR training users,
  and inference batch size, which build large *host* feature tables
  (``n_train_users x n_candidates x 2 x emb_dim x 4`` bytes).

Typical usage (notebook, via :func:`nb_env.setup` which calls this and exposes
the result as the ``CAPACITY`` global)::

    from utils.resources import detect_hardware, recommend_capacity
    cap = recommend_capacity()
    ae_batch = cap["ae_batch_size"]
    ...

Every returned value is a *recommendation*; callers may always override it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    """Detected compute resources.

    Attributes
    ----------
    has_cuda:
        Whether a CUDA device is visible to PyTorch.
    gpu_name:
        Marketing name of the first CUDA device (``""`` on CPU).
    vram_gb:
        Total VRAM of the first CUDA device in GiB (``0.0`` on CPU).
    ram_gb:
        Total system RAM in GiB.
    """

    has_cuda: bool
    gpu_name: str
    vram_gb: float
    ram_gb: float


def _total_ram_gb() -> float:
    """Total system RAM in GiB. Stdlib first, psutil as a fallback."""
    # Linux (incl. Colab): sysconf is dependency-free and always present.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import psutil  # type: ignore[import]
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


def detect_hardware() -> Hardware:
    """Probe the current machine for CUDA VRAM and system RAM.

    Never raises: if PyTorch is missing or no GPU is present, the CUDA fields
    degrade gracefully to a CPU profile.
    """
    has_cuda = False
    gpu_name = ""
    vram_gb = 0.0
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            has_cuda = True
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = props.total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001  (torch missing, driver error, etc.)
        pass
    return Hardware(has_cuda=has_cuda, gpu_name=gpu_name,
                    vram_gb=round(vram_gb, 1), ram_gb=round(_total_ram_gb(), 1))


# ---------------------------------------------------------------------------
# Capacity recommendation
# ---------------------------------------------------------------------------

# VRAM tier -> (ae_batch_size, ae_infer_batch, hgt_user_batch_size).
# Tiers, by total VRAM: cpu (no GPU); t4 (<=20 GB, Colab T4); mid (<=48 GB,
# L4/A100-40); big (>48 GB, A100-80/H100/RTX PRO 6000).
# hgt_user_batch_size: with the two-stage backward in train_DL.py, larger
# values only affect stage-1 (cheap float32 loss loop) — they no longer
# keep the HGT graph alive across batches — so we can use large values safely.
_VRAM_TIERS = (
    #  max_gb,  label,  ae_batch, ae_infer, hgt_user_batch
    (0.0,   "cpu",   128,   256,    512),
    (20.0,  "t4",    256,   512,   1024),
    (48.0,  "mid",  1024,  2048,   8192),
    (1e9,   "big",  2048,  4096,  65536),
)

# RAM tier -> (xgb_n_candidates, xgb_n_train_users, xgb_infer_batch_users).
# Tiers, by total RAM: low (<=20 GB, Colab default); mid (<=64 GB);
# high (>64 GB, large server).
_RAM_TIERS = (
    #  max_gb,  label,  n_candidates, n_train_users, infer_batch_users
    (20.0,  "low",   200,   3_000,  1_000),
    (64.0,  "mid",   300,   8_000,  2_000),
    (1e9,   "high",  500,  20_000,  5_000),
)


def _pick(tiers, value):
    """Return the first tier row whose ``max_gb`` threshold covers ``value``."""
    for row in tiers:
        if value <= row[0]:
            return row
    return tiers[-1]


def recommend_capacity(
    hw: Optional[Hardware] = None,
    *,
    verbose: bool = True,
) -> dict:
    """Map detected hardware to recommended batch sizes and workload counts.

    Parameters
    ----------
    hw:
        A :class:`Hardware` instance; auto-detected via :func:`detect_hardware`
        when ``None``.
    verbose:
        Print a one-line summary of the detected tiers and chosen values.

    Returns
    -------
    dict
        Keys consumed by the training cells:
        ``ae_batch_size``, ``ae_infer_batch``, ``hgt_user_batch_size``,
        ``xgb_n_candidates``, ``xgb_n_train_users``, ``xgb_infer_batch_users``,
        ``num_workers`` -- plus the detected ``hardware`` and the resolved
        ``vram_tier`` / ``ram_tier`` labels for transparency.
    """
    hw = hw or detect_hardware()

    # GPU-bound workloads key off VRAM; with no CUDA, fall back to the cpu row.
    _, vram_label, ae_batch, ae_infer, hgt_user_batch = (
        _pick(_VRAM_TIERS, hw.vram_gb) if hw.has_cuda else _VRAM_TIERS[0]
    )
    # RAM-bound workloads (XGBoost host feature tables) key off system RAM.
    _, ram_label, xgb_cands, xgb_users, xgb_infer = _pick(_RAM_TIERS, hw.ram_gb)

    # DataLoader workers: a few per available CPU, capped to avoid oversubscribe.
    num_workers = min(8, max(0, (os.cpu_count() or 2) - 1))

    cap = {
        "ae_batch_size":          ae_batch,
        "ae_infer_batch":         ae_infer,
        "hgt_user_batch_size":    hgt_user_batch,
        "xgb_n_candidates":       xgb_cands,
        "xgb_n_train_users":      xgb_users,
        "xgb_infer_batch_users":  xgb_infer,
        "num_workers":            num_workers,
        "vram_tier":              vram_label,
        "ram_tier":               ram_label,
        "hardware":               hw,
    }

    if verbose:
        _dev = f"{hw.gpu_name} ({hw.vram_gb:g} GB)" if hw.has_cuda else "CPU (no CUDA)"
        print(
            f"  capacity   : {_dev}, {hw.ram_gb:g} GB RAM "
            f"-> vram tier '{vram_label}', ram tier '{ram_label}'\n"
            f"               ae_batch={ae_batch}  hgt_user_batch={hgt_user_batch}  "
            f"xgb[cands={xgb_cands}, train_users={xgb_users:,}, "
            f"infer_batch={xgb_infer:,}]"
        )

    return cap


__all__ = ("Hardware", "detect_hardware", "recommend_capacity")
