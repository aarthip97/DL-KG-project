"""End-to-end training orchestration for the HGT recommender:

* split user→track edges with ``RandomLinkSplit``
* train with BPR loss + uniform negative sampling
* evaluate Recall@K and NDCG@K on the held-out positives, ranking against
  the **full** track catalogue with training-edge masking
* optional Weights & Biases logging (guarded import)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from .bpr import bpr_loss, evaluate_top_k, sample_negative_items
from .hgt import RecommenderHGT

try:                                  # optional dependency
    import wandb
    _WANDB = True
except ImportError:                   # pragma: no cover
    _WANDB = False


# ── Result container ──────────────────────────────────────────────────────
@dataclass
class TrainResult:
    model: RecommenderHGT
    history: list[dict] = field(default_factory=list)
    best_val: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)


# ── Training loop ─────────────────────────────────────────────────────────
def train_hgt(
    data: HeteroData,
    *,
    edge_type: tuple[str, str, str] = ("user", "listened_to", "track"),
    rev_edge_type: tuple[str, str, str] = ("track", "rev_listened_to", "user"),
    hidden_channels: int = 64,
    out_channels: int = 32,
    num_heads: int = 2,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8192,
    eval_every: int = 10,
    k_list: Iterable[int] = (10, 20),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    disjoint_train_ratio: float = 0.3,
    device: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "music-recommender-hgt",
    wandb_config: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """Train an HGT recommender on ``data`` and return the fitted model + metrics."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure the message-passing graph is bidirectional.
    data = T.ToUndirected(merge=False)(data) if rev_edge_type not in data.edge_types else data

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
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    model = RecommenderHGT(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    # Lazy ``Linear(-1, h)`` modules need one dry forward pass to materialise.
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_wandb and _WANDB:
        wandb.init(project=wandb_project, config={
            "hidden_channels": hidden_channels, "out_channels": out_channels,
            "num_heads": num_heads, "num_layers": num_layers,
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "val_ratio": val_ratio, "test_ratio": test_ratio,
            **(wandb_config or {}),
        })

    src_t, _, dst_t = edge_type
    train_pos = train_data[edge_type].edge_label_index   # supervision edges only
    n_items = data[dst_t].num_nodes
    history: list[dict] = []
    best_val_recall = -1.0
    best_state: Optional[dict] = None
    best_val: dict = {}

    for ep in range(1, epochs + 1):
        model.train()
        # Shuffle supervision edges and iterate in mini-batches
        perm = torch.randperm(train_pos.size(1), device=device)
        epoch_loss = 0.0
        n_seen = 0
        for s in range(0, perm.numel(), batch_size):
            batch_idx = perm[s:s + batch_size]
            u = train_pos[0, batch_idx]
            i_pos = train_pos[1, batch_idx]
            i_neg = sample_negative_items(i_pos, n_items)

            opt.zero_grad()
            out = model(train_data.x_dict, train_data.edge_index_dict)
            user_emb = out[src_t][u]
            pos_emb  = out[dst_t][i_pos]
            neg_emb  = out[dst_t][i_neg]
            loss = bpr_loss(user_emb, pos_emb, neg_emb)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * u.numel()
            n_seen += u.numel()
        epoch_loss /= max(n_seen, 1)

        log = {"epoch": ep, "train/bpr_loss": epoch_loss}

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
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

            primary = next(iter(k_list))
            if val_metrics[f"recall@{primary}"] > best_val_recall:
                best_val_recall = val_metrics[f"recall@{primary}"]
                best_state = copy.deepcopy(model.state_dict())
                best_val = val_metrics

        history.append(log)
        if verbose and (ep == 1 or ep % eval_every == 0 or ep == epochs):
            extra = " ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "epoch")
            print(f"[hgt] {extra}")
        if use_wandb and _WANDB:
            wandb.log(log)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        out = model(train_data.x_dict, train_data.edge_index_dict)
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
        print("[hgt] test:", test_metrics)
    if use_wandb and _WANDB:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    return TrainResult(model=model, history=history,
                       best_val=best_val, test_metrics=test_metrics)


__all__ = ("train_hgt", "TrainResult")
