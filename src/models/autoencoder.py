"""jSymbolic / audio-feature autoencoder used to compress per-track features
into a 128-dim embedding before injecting them as ``track`` node features.

The architecture is fixed (input → 512 → 256 → 128 → 256 → 512 → input)
to match the capstone spec; only ``input_dim`` adapts to the upstream
feature table.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# W&B is optional -- guard every call with _WANDB
try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class jSymbolicAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = [512, 256], bottleneck: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        result = self.encoder(x)
        self.train(was_training)
        return result


def _to_tensor(X) -> torch.Tensor:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.float32)
    elif isinstance(X, np.ndarray):
        X = X.astype(np.float32, copy=False)
    elif isinstance(X, torch.Tensor):
        return X.float()
    else:
        X = np.asarray(X, dtype=np.float32)
    return torch.from_numpy(X)


def train_autoencoder(
    model: jSymbolicAutoencoder,
    X,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    device: Optional[str] = None,
    verbose: bool = True,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config_extra: Optional[dict] = None,
) -> tuple[jSymbolicAutoencoder, dict[str, list[float]]]:
    """Standard reconstruction training loop.

    Parameters
    ----------
    model:
        An already-instantiated :class:`jSymbolicAutoencoder`.
    X:
        Feature matrix (numpy array, DataFrame, or Tensor).  Shape ``(N, input_dim)``.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    weight_decay:
        L2 regularisation coefficient for Adam.
    val_split:
        Fraction of data held out as a validation set (0 -> no validation).
    device:
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    verbose:
        Print epoch logs when ``True``.
    wandb_project:
        Weights and Biases project name. If ``None`` or wandb is not
        installed, no W&B logging is performed.
    wandb_run_name:
        Optional W&B run name. Defaults to ``ae_dim{bottleneck}_ep{epochs}``.
    wandb_config_extra:
        Optional extra key/value pairs merged into the W&B config dict
        (e.g. dataset name, feature source).

    Returns
    -------
    model:
        Trained model (moved to ``device``).
    history:
        ``{"train_loss": [...], "val_loss": [...]}`` -- ``val_loss`` is only
        present when ``val_split > 0``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    X_t    = _to_tensor(X).to(device)

    # -- W&B initialisation --------------------------------------------------
    # If the caller already started a run (e.g. in the notebook), reuse it
    # so we don't create nested/conflicting runs.  Otherwise start a new
    # run only if wandb_project is given.
    _run = None
    _owns_run = False
    if _WANDB:
        if _wandb.run is not None:
            _run = _wandb.run
        elif wandb_project is not None:
            bottleneck = model.encoder[-1].out_features
            run_name = wandb_run_name or f"ae_dim{bottleneck}_ep{epochs}"
            cfg = {
                "model":        "jSymbolicAutoencoder",
                "input_dim":    int(X_t.shape[1]),
                "bottleneck":   bottleneck,
                "epochs":       epochs,
                "batch_size":   batch_size,
                "lr":           lr,
                "weight_decay": weight_decay,
                "val_split":    val_split,
                "device":       device,
                "n_samples":    int(X_t.shape[0]),
            }
            if wandb_config_extra:
                cfg.update(wandb_config_extra)
            _run = _wandb.init(project=wandb_project, name=run_name, config=cfg)
            _owns_run = True

        if _run is not None:
            _wandb.watch(model, log="gradients", log_freq=100)

    # -- optional train / val split ------------------------------------------
    n_total = X_t.size(0)
    if val_split > 0:
        n_val   = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        perm    = torch.randperm(n_total, device=device)
        X_train = X_t[perm[:n_train]]
        X_val   = X_t[perm[n_train:]]
    else:
        X_train = X_t
        X_val   = None

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True, num_workers=0)
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": []}
    if X_val is not None:
        history["val_loss"] = []

    for ep in range(1, epochs + 1):
        # -- train ------------------------------------------------------------
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            opt.zero_grad()
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
        train_loss = running / X_train.size(0)
        history["train_loss"].append(train_loss)

        # -- validation -------------------------------------------------------
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), X_val).item()
            history["val_loss"].append(val_loss)
        else:
            val_loss = None

        # -- W&B per-epoch log ------------------------------------------------
        if _run is not None:
            log = {"train/loss": train_loss, "epoch": ep}
            if val_loss is not None:
                log["val/loss"] = val_loss
            _wandb.log(log, step=ep)

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            msg = f"[ae] epoch {ep:3d}/{epochs}  train_mse={train_loss:.6f}"
            if val_loss is not None:
                msg += f"  val_mse={val_loss:.6f}"
            print(msg)

    if _run is not None:
        _wandb.summary.update({
            "final_train_loss": history["train_loss"][-1],
            **({"final_val_loss": history["val_loss"][-1]} if X_val is not None else {}),
        })
        # Only finish the run if we created it ourselves; otherwise leave
        # it open for the caller to continue logging into.
        if _owns_run:
            _wandb.finish()

    return model, history


@torch.no_grad()
def extract_embeddings(
    model: jSymbolicAutoencoder,
    X,
    *,
    device: Optional[str] = None,
    batch_size: int = 512,
    as_dataframe: bool = False,
    index=None,
) -> torch.Tensor | pd.DataFrame:
    """Extract bottleneck embeddings, processing in batches to avoid OOM."""
    _dev: torch.device = (
        torch.device(device) if device
        else next(model.parameters()).device
    )
    model.eval()
    X_t     = _to_tensor(X)
    n       = X_t.size(0)
    chunks  = [X_t[i : i + batch_size].to(_dev) for i in range(0, n, batch_size)]
    emb     = torch.cat([model.embed(c) for c in chunks], dim=0).cpu()
    if as_dataframe:
        cols = [f"ae_{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb.numpy(), index=index, columns=cols)
    return emb


__all__ = ("jSymbolicAutoencoder", "train_autoencoder", "extract_embeddings")
