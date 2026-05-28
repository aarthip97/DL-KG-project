"""End-to-end training orchestration for the HGT recommender.

Training strategy: Full-Graph Forward with Mini-Batch Listwise Loss.

The HGT graph is forwarded in full once per epoch. Every node therefore
receives an embedding that reflects its complete neighbourhood context, with
no sampling noise from subgraph methods. The listwise loss is then accumulated
in small user chunks (user_batch_size users at a time) so that the full
(num_users x num_tracks) logit matrix is never materialised in GPU memory.
A single backward pass over the accumulated scalar closes the epoch.

Loss: Intensity-Weighted Listwise cross-entropy with Logit Adjustment
(see bpr.debiased_listwise_loss). Raw listen counts serve as graded relevance
and a log-popularity penalty discourages the model from concentrating
recommendations on already-popular tracks.

Evaluation: full-ranking Recall@K and NDCG@K on held-out edges produced by
RandomLinkSplit, computed once every eval_every epochs and at the final epoch.

Extras: cosine LR warm restarts, gradient clipping, optional W&B logging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from .hgt import RecommenderHGT
from .loss import compute_log_pop_prior, debiased_listwise_loss, evaluate_top_k

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


@dataclass
class TrainResult:
    model: RecommenderHGT
    history: list[dict] = field(default_factory=list)
    best_val: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)

def train_hgt(
    data: HeteroData,
    user_interaction_matrix: torch.Tensor,
    track_listen_counts: torch.Tensor,
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
    user_batch_size: int = 1024,
    eval_every: int = 10,
    k_list: Iterable[int] = (10, 20),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    disjoint_train_ratio: float = 0.3,
    lambda_reg: float = 0.2,
    temperature: float = 0.1,
    lr_t0: int = 20,
    lr_eta_min: float = 1e-5,
    clip_grad_norm: float = 1.0,
    device: Optional[str] = None,
    use_amp: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "music-recommender-hgt",
    wandb_config: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """Train an HGT recommender with full-graph forward and mini-batch listwise loss.

    The full heterogeneous graph is forwarded through the HGT stack once per epoch
    to obtain globally-consistent node embeddings. The listwise loss is then
    accumulated in user_batch_size chunks so the full (U x I) logit matrix is never
    held in memory at once. A single backward pass closes the epoch, preserving the
    global gradient signal of the listwise objective.

    Parameters are:
      data                    -- populated HeteroData graph from build_rich_hetero_graph.
      user_interaction_matrix -- (U, I) tensor of raw listen counts per user per track.
                                 May live on CPU; rows are indexed by HGT user node ids.
      track_listen_counts     -- (I,) tensor of total listen counts per track, used to
                                 compute the log-popularity prior for logit adjustment.
      edge_type               -- relation triple for user-to-track supervision edges.
      rev_edge_type           -- reverse relation triple; added automatically if missing.
      hidden_channels         -- width of HGT message-passing layers.
      out_channels            -- dimension of final L2-normalised embeddings.
      num_heads               -- attention heads per HGTConv layer.
      num_layers              -- number of message-passing hops.
      dropout                 -- dropout probability in GNN layers and projection head.
      epochs                  -- total training epochs.
      lr                      -- initial AdamW learning rate.
      weight_decay            -- L2 regularisation coefficient.
      user_batch_size         -- users per mini-batch during loss accumulation.
      eval_every              -- evaluate validation metrics every this many epochs.
      k_list                  -- Recall@K and NDCG@K cut-offs.
      val_ratio               -- fraction of edges held out for validation.
      test_ratio              -- fraction of edges held out for testing.
      disjoint_train_ratio    -- fraction of training edges used as supervision labels.
      lambda_reg              -- popularity penalty scale in the listwise loss.
      temperature             -- softmax temperature (lower = sharper distribution).
      lr_t0                   -- epoch period for cosine warm restart cycle.
      lr_eta_min              -- minimum learning rate at the cosine trough.
      clip_grad_norm          -- maximum L2 norm for gradient clipping.
      device                  -- target device; defaults to cuda when available.
      use_wandb               -- enable Weights and Biases logging.
      wandb_project           -- W&B project name.
      wandb_config            -- extra key-value pairs merged into the W&B run config.
      seed                    -- random seed for reproducibility.
      verbose                 -- print one-line summary after each evaluation epoch.

    Returns a TrainResult with the best-val-checkpoint model, full epoch history,
    best validation metrics, and final test metrics.
    """
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Keep the (U, I) interaction matrix on CPU — debiased_listwise_loss calls
    # .to(device) per batch, so it never needs to live in VRAM all at once.
    user_interaction_matrix = user_interaction_matrix.cpu()

    _amp_enabled = use_amp and dev.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)

    # Edge splits for held-out evaluation
    if rev_edge_type not in data.edge_types:
        data = T.ToUndirected(merge=False)(data)
        if rev_edge_type not in data.edge_types:
            raise ValueError(
                f"ToUndirected() did not create the expected reverse edge type "
                f"{rev_edge_type!r}. Available edge types: {data.edge_types}"
            )

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

    src_t, _, dst_t = edge_type
    n_users = data[src_t].num_nodes

    # Pre-compute log-popularity prior once; reused every epoch
    log_track_pop = compute_log_pop_prior(track_listen_counts, device=dev)

    # DataLoader over user indices for mini-batch loss accumulation
    user_loader = DataLoader(
        range(n_users), batch_size=user_batch_size, shuffle=True
    )

    # Model
    model = RecommenderHGT(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    # Materialise lazy Linear(-1, h) weights with a dry forward pass
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=lr_t0, eta_min=lr_eta_min
    )

    if use_wandb and _WANDB:
        wandb.init(
            project=wandb_project,
            config={
                "hidden_channels": hidden_channels,
                "out_channels":    out_channels,
                "num_heads":       num_heads,
                "num_layers":      num_layers,
                "dropout":         dropout,
                "epochs":          epochs,
                "lr":              lr,
                "weight_decay":    weight_decay,
                "user_batch_size": user_batch_size,
                "lambda_reg":      lambda_reg,
                "temperature":     temperature,
                "lr_t0":           lr_t0,
                "val_ratio":       val_ratio,
                "test_ratio":      test_ratio,
                **(wandb_config or {}),
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    train_pos = train_data[edge_type].edge_label_index
    history: list[dict] = []
    best_val_recall = -1.0
    best_state: dict | None = None
    best_val: dict = {}

    k_list = sorted(set(int(k) for k in k_list))
    primary_k = k_list[0]

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # Full-graph forward + listwise loss — wrapped in autocast when AMP is on.
        # AMP halves VRAM for the HGT forward pass and speeds up matmuls on
        # Tensor-core GPUs (T4, A100).  Sensitive ops (softmax, log) are
        # automatically upcast to float32 by PyTorch inside the context.
        with torch.cuda.amp.autocast(enabled=_amp_enabled):
            out = model(train_data.x_dict, train_data.edge_index_dict)
            all_track_embs = out[dst_t]  # (I, D)

            # Accumulate listwise loss across user mini-batches, then back-prop once
            total_loss: torch.Tensor | None = None
            for user_batch_indices in user_loader:
                user_batch_indices = user_batch_indices.to(dev)
                batch_u_embs = out[src_t][user_batch_indices]
                batch_loss = debiased_listwise_loss(
                    batch_u_embs,
                    all_track_embs,
                    user_batch_indices,
                    user_interaction_matrix,
                    log_track_pop,
                    lambda_reg=lambda_reg,
                    temperature=temperature,
                )
                # Scale so accumulated loss equals the full-batch mean
                scaled = batch_loss * (user_batch_indices.numel() / n_users)
                total_loss = scaled if total_loss is None else total_loss + scaled

        if total_loss is not None:
            epoch_loss_scalar = total_loss.item()
            scaler.scale(total_loss).backward()
            # Unscale before clipping so the norm threshold is in the original scale
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            epoch_loss_scalar = 0.0
            opt.step()

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        log: dict = {
            "epoch":               ep,
            "train/listwise_loss": epoch_loss_scalar,
            "train/lr":            current_lr,
        }

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out_eval = model(train_data.x_dict, train_data.edge_index_dict)

            val_eli = val_data[edge_type].edge_label_index
            val_metrics = evaluate_top_k(
                user_emb=out_eval[src_t],
                item_emb=out_eval[dst_t],
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

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_test = model(test_data.x_dict, test_data.edge_index_dict)

    test_eli = test_data[edge_type].edge_label_index
    test_metrics = evaluate_top_k(
        user_emb=out_test[src_t],
        item_emb=out_test[dst_t],
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