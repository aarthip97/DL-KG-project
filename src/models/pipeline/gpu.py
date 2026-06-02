"""Small GPU-memory helpers for the pipeline notebook.

Note on parallelism: the §8 ablation runs experiments **sequentially**. A single
full-graph HGT forward already saturates the GPU with large matmuls and is the
dominant VRAM consumer, so running several experiments at once on one GPU mostly
contends / risks OOM rather than going faster. The genuine lever for "more VRAM →
faster" is a larger ``user_batch_size`` (fewer loss-accumulation steps per epoch);
its effect is modest because the forward pass dominates, but :func:`suggest_user_batch_size`
scales it to the free memory so bigger GPUs do a bit more per step. Real
cross-experiment parallelism only helps with *multiple* GPUs (one run per device).
"""
from __future__ import annotations

from typing import Optional


def gpu_free_gb(device: int = 0) -> float:
    """Free VRAM in GiB on ``device`` (0.0 when CUDA is unavailable)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info(device)
        return free / (1024 ** 3)
    except Exception:           # noqa: BLE001
        return 0.0


def suggest_user_batch_size(
    *,
    base: int = 1024,
    gb_per_unit: float = 6.0,
    min_bs: int = 512,
    max_bs: int = 8192,
    device: int = 0,
    fallback: Optional[int] = None,
) -> int:
    """Heuristic ``user_batch_size`` scaled to free VRAM (clamped to a safe range).

    Returns ``base × max(1, round(free_gb / gb_per_unit))`` clamped to
    ``[min_bs, max_bs]``; falls back to ``fallback or base`` when no GPU is seen.
    Effect on speed is modest (the full-graph forward dominates VRAM) — use it as
    a convenience, not a substitute for picking ``user_batch_size`` deliberately.
    """
    free = gpu_free_gb(device)
    if free <= 0:
        return fallback if fallback is not None else base
    bs = int(base * max(1, round(free / gb_per_unit)))
    return max(min_bs, min(max_bs, bs))


__all__ = ["gpu_free_gb", "suggest_user_batch_size"]
