"""End-to-end training orchestration for the HGT recommender.

Features
--------
* **RandomLinkSplit** — clean train / val / test split of user→track edges.
* **BPR loss** with **popularity-weighted negative sampling** — hard negatives
  from high-popularity items improve ranking quality at negligible extra cost.
* **AMP (FP16/BF16)** — ~1.5–2× speedup on CUDA with near-zero accuracy loss.
* **Cosine LR schedule with warm restarts** — fast convergence, avoids
  getting stuck in a sharp minimum.
* **Gradient clipping** — stabilises training with deep HGT stacks.
* **Embedding cache** — full-graph forward pass is computed *once per eval
  epoch*, not once per mini-batch.
* **Weights & Biases** — optional; guarded import, full metric logging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.cuda.amp as amp
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from .hgt import RecommenderHGT
from .bpr import bpr_loss, evaluate_top_k, sample_negative_items

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model: RecommenderHGT
    history: list[dict] = field(default_factory=list)
    best_val: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)


# ── Negative sampling helpers ─────────────────────────────────────────────────

def _popularity_neg_items(
    pos_item_idx: torch.Tensor,
    num_items: int,
    item_counts: torch.Tensor | None,
    *,
    mix: float = 0.5,
) -> torch.Tensor:
    """Mixed uniform + popularity-proportional negative sampling.

    ``mix=0.5`` draws half of the negatives proportional to item popularity
    (harder negatives) and half uniformly at random.  Set ``mix=0.0`` to
    revert to pure uniform sampling.

    Parameters
    ----------
    item_counts : 1-D float tensor of interaction counts per item, length
        ``num_items``.  If ``None``, falls back to uniform sampling.
    mix : float in [0, 1]
        Fraction of negatives drawn from the popularity distribution.
    """
    if item_counts is None or mix == 0.0:
        return sample_negative_items(pos_item_idx, num_items)

    n = pos_item_idx.size(0)
    device = pos_item_idx.device
    n_pop = int(n * mix)
    n_uni = n - n_pop

    # Uniform part
    uni_neg = torch.randint(0, num_items, (n_uni,), device=device)

    # Popularity-proportional part (multinomial sample from item frequencies)
    counts_cpu = item_counts.cpu().float()
    probs = counts_cpu / counts_cpu.sum()
    pop_neg = torch.multinomial(probs, n_pop, replacement=True).to(device)

    neg = torch.cat([uni_neg, pop_neg])
    # Shuffle so the two halves don't form a systematic pattern
    return neg[torch.randperm(n, device=device)]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_hgt(
    data: HeteroData,
    *,
    edge_type: tuple[str, str, str] = ("user", "listened_to", "track"),
    rev_edge_type: tuple[str, str, str] = ("track", "rev_listened_to", "user"),
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8192,
    eval_every: int = 10,
    k_list: Iterable[int] = (10, 20),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    disjoint_train_ratio: float = 0.3,
    # Negative sampling
    neg_mix: float = 0.5,
    # LR schedule
    lr_t0: int = 20,            # epochs per cosine half-cycle
    lr_eta_min: float = 1e-5,
    # Gradient clipping
    clip_grad_norm: float = 1.0,
    # AMP
    use_amp: bool = True,
    # W&B
    device: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "music-recommender-hgt",
    wandb_config: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """Train an HGT recommender on ``data`` and return the fitted model + metrics.

    Parameters
    ----------
    data : HeteroData
        Populated heterogeneous graph (output of ``build_rich_hetero_graph``).
    hidden_channels : int
        Width of HGTConv layers.  128 is a good default.
    out_channels : int
        Embedding dimension used for BPR dot-product scoring.  32–64 works well.
    dropout : float
        Dropout between HGT layers and inside the projection head.
    neg_mix : float
        Fraction of negatives drawn from the popularity distribution (0 = pure
        uniform, 1 = pure popularity-weighted).  0.5 is a sensible default.
    lr_t0 : int
        Epoch period of the cosine warm restart cycle.
    clip_grad_norm : float
        Max L2 norm for gradient clipping.
    use_amp : bool
        Enable automatic mixed precision (FP16/BF16) on CUDA — typically 1.5–2×
        faster with negligible accuracy loss.
    """
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and (dev == "cuda" or str(dev).startswith("cuda"))

    # ── Split edges ───────────────────────────────────────────────────────────
    if rev_edge_type not in data.edge_types:
        data = T.ToUndirected(merge=False)(data)

    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=[edge_type],
        rev_edge_types=[rev_edge_type],
    )
    train_data, val_data, test_data = transform(data)
    train_data = train_data.to(dev)
    val_data   = val_data.to(dev)
    test_data  = test_data.to(dev)

    # ── Build item popularity weights for hard-negative sampling ──────────────
    src_t, _, dst_t = edge_type
    n_items = data[dst_t].num_nodes
    item_counts: torch.Tensor | None = None
    if neg_mix > 0.0:
        counts = torch.zeros(n_items, dtype=torch.float32)
        train_pos_all = train_data[edge_type].edge_label_index
        for item_idx in train_pos_all[1].cpu().tolist():
            counts[item_idx] += 1
        item_counts = counts.to(dev)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RecommenderHGT(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    # Materialise lazy Linear(-1, h) weights with a dry forward pass.
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)

    # ── Optimiser + scheduler + AMP scaler ───────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=lr_t0, eta_min=lr_eta_min
    )
    scaler: amp.GradScaler | None = amp.GradScaler() if use_amp else None

    # ── W&B initialisation ────────────────────────────────────────────────────
    if use_wandb and _WANDB:
        wandb.init(
            project=wandb_project,
            config={
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "neg_mix": neg_mix,
                "lr_t0": lr_t0,
                "use_amp": use_amp,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                **(wandb_config or {}),
            },
        )
        if _WANDB:
            wandb.watch(model, log="gradients", log_freq=100)

    # ── Training state ────────────────────────────────────────────────────────
    train_pos = train_data[edge_type].edge_label_index   # supervision edges
    history: list[dict] = []
    best_val_recall = -1.0
    best_state: dict | None = None
    best_val: dict = {}

    k_list = sorted(set(int(k) for k in k_list))
    primary_k = k_list[0]

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_pos.size(1), device=dev)
        epoch_loss = 0.0
        n_seen = 0

        for s in range(0, perm.numel(), batch_size):
            batch_idx = perm[s: s + batch_size]
            u      = train_pos[0, batch_idx]
            i_pos  = train_pos[1, batch_idx]
            i_neg  = _popularity_neg_items(
                i_pos, n_items, item_counts, mix=neg_mix
            )

            opt.zero_grad(set_to_none=True)

            # Forward + loss under optional AMP context
            with amp.autocast(enabled=(scaler is not None)):
                out = model(train_data.x_dict, train_data.edge_index_dict)
                loss = bpr_loss(out[src_t][u], out[dst_t][i_pos], out[dst_t][i_neg])

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                opt.step()

            epoch_loss += loss.item() * u.numel()
            n_seen += u.numel()

        scheduler.step()
        epoch_loss /= max(n_seen, 1)
        current_lr = scheduler.get_last_lr()[0]

        log = {
            "epoch":          ep,
            "train/bpr_loss": epoch_loss,
            "train/lr":       current_lr,
        }

        # ── Evaluation ────────────────────────────────────────────────────────
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            # Compute embeddings ONCE and reuse for both masking and scoring.
            with torch.no_grad():
                out = model(train_data.x_dict, train_data.edge_index_dict)

            val_eli = val_data[edge_type].edge_label_index
            val_metrics = evaluate_top_k(
                user_emb=out[src_t],
                item_emb=out[dst_t],
                eval_user_idx=val_eli[0],
                eval_item_idx=val_eli[1],
                train_edge_index=train_pos,
                k_list=k_list,
            )
            log.update({f"val/{k}": v for k, v in val_metrics.items()})

            if val_metrics[f"recall@{primary_k}"] > best_val_recall:
                best_val_recall = val_metrics[f"recall@{primary_k}"]
                best_state = copy.deepcopy(model.state_dict())
                best_val = dict(val_metrics)

        history.append(log)

        if verbose and (ep == 1 or ep % eval_every == 0 or ep == epochs):
            extra = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log.items()
                if k != "epoch"
            )
            print(f"[hgt ep={ep:04d}]  {extra}")

        if use_wandb and _WANDB:
            wandb.log(log, step=ep)

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(test_data.x_dict, test_data.edge_index_dict)

    test_eli = test_data[edge_type].edge_label_index
    test_metrics = evaluate_top_k(
        user_emb=out[src_t],
        item_emb=out[dst_t],
        eval_user_idx=test_eli[0],
        eval_item_idx=test_eli[1],
        train_edge_index=train_pos,
        k_list=k_list,
    )

    if verbose:
        print("[hgt test]", "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    if use_wandb and _WANDB:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.summary.update({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    return TrainResult(
        model=model,
        history=history,
        best_val=best_val,
        test_metrics=test_metrics,
    )


__all__ = ("train_hgt", "TrainResult")