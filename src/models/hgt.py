"""Heterogeneous Graph Transformer (HGT) recommender.

Architecture overview
---------------------
* **Lazy input projections** — ``Linear(-1, hidden_channels)`` adapts to any
  node-feature width (384-D fused tracks, 256-D metadata, etc.) without
  requiring an explicit ``in_channels_dict`` at construction time.
* **Deep HGT layers** — each layer: ``HGTConv`` → ``LayerNorm`` → ``ReLU``
  → ``Dropout`` → **residual add** (prevents over-smoothing).
* **Output projection head** — a two-layer MLP maps ``hidden_channels →
  out_channels``, giving a compact dot-product space that generalises better
  than scoring directly in the wide hidden space.
* **L2-normalised outputs** — cosine-similarity scoring is more stable than
  raw inner product when node representations vary in norm.
* **``recommend()`` method** — wraps the forward pass into the
  ``Recommender`` protocol consumed by the evaluation package.

Speed knobs
-----------
* Lazy ``Linear(-1, ...)`` weights are materialised on the first forward
  call — no manual shape counting needed.
* Keeping ``hidden_channels ≤ 128`` and ``num_heads ≤ 4`` covers most
  datasets without blowing up HGTConv's quadratic head-split cost.
* L2 normalisation is fused into the forward pass (single ``F.normalize``
  call), so scoring at inference is a plain batched matmul.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class RecommenderHGT(nn.Module):
    """Heterogeneous Graph Transformer for top-N recommendation.

    Parameters
    ----------
    metadata : tuple
        ``(node_types, edge_types)`` as returned by ``HeteroData.metadata()``.
    hidden_channels : int
        Width of every HGTConv layer and the first MLP layer of the head.
    out_channels : int
        Width of the final L2-normalised embedding used for dot-product
        scoring.  Smaller values (32–64) generalise better.
    num_heads : int
        Multi-head attention heads inside each HGTConv layer.
        Must divide ``hidden_channels``.
    num_layers : int
        Number of message-passing hops.  3–4 is usually the sweet spot;
        more layers may over-smooth unless dropout is increased accordingly.
    dropout : float
        Dropout probability applied between layers and inside the head MLP.
    use_checkpoint : bool
        When ``True``, each HGTConv layer uses
        ``torch.utils.checkpoint`` to recompute activations during the
        backward pass instead of storing them.  Cuts activation VRAM
        by ~40–60 % at the cost of ~20 % more compute per epoch.  Safe
        to enable on any GPU but most beneficial on T4 / V100 where
        activation memory is the binding constraint.
    """

    def __init__(
        self,
        metadata,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        node_types: list[str] = list(metadata[0])

        # ── 1. Lazy type-specific input projections ──────────────────────────
        # Linear(-1, h) lets PyG infer the input dimension from the first
        # forward pass, so we never need to hard-code in_channels_dict.
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden_channels) for nt in node_types
        })

        # ── 2. HGT message-passing layers ────────────────────────────────────
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                group="sum",
            )
            for _ in range(num_layers)
        ])

        # One LayerNorm per node type per layer (independent normalisation).
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(p=dropout)

        # ── 3. Output projection head ─────────────────────────────────────────
        # Two-layer MLP per node type → compact dot-product embedding space.
        # The intermediate ReLU + Dropout add non-linearity and regularisation.
        self.head_dict = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, out_channels),
            )
            for nt in node_types
        })

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the full HGT stack and return L2-normalised embeddings.

        Returns
        -------
        dict mapping node-type string → ``(N, out_channels)`` float tensor,
        each row being a unit-norm embedding ready for cosine dot-product.
        """
        # Initial type-specific projection + ReLU
        out: dict[str, torch.Tensor] = {
            nt: F.relu(self.lin_dict[nt](x), inplace=True)
            for nt, x in x_dict.items()
        }

        # Multi-hop message passing: norm → activation → dropout → residual
        for i, conv in enumerate(self.convs):
            skip = {nt: h for nt, h in out.items()}          # keep reference
            # use_reentrant=False lets checkpoint handle dict inputs without
            # requiring every argument to be a Tensor (PyTorch >= 1.13).
            if self.use_checkpoint:
                out = checkpoint(conv, out, edge_index_dict, use_reentrant=False)
            else:
                out = conv(out, edge_index_dict)
            layer_norms: nn.ModuleDict = self.norms[i]  # type: ignore[assignment]
            for nt in out:
                out[nt] = layer_norms[nt](out[nt])
                out[nt] = F.relu(out[nt], inplace=True)
                out[nt] = self.dropout(out[nt])
                out[nt] = out[nt] + skip[nt]                 # residual

        # Project to out_channels and L2-normalise for cosine scoring
        return {
            nt: F.normalize(self.head_dict[nt](h), p=2, dim=-1)
            for nt, h in out.items()
        }

    # ── Embedding extraction (eval-time convenience) ──────────────────────────

    @torch.no_grad()
    def encode(
        self,
        data: HeteroData,
        *,
        device: Optional[torch.device | str] = None,
    ) -> dict[str, torch.Tensor]:
        """Run a single forward pass and return all node embeddings.

        Handy for caching embeddings once and reusing them for many queries.
        """
        self.eval()
        if device is not None:
            data = data.to(device)
        return self(data.x_dict, data.edge_index_dict)

    # ── Recommender interface ─────────────────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        data: HeteroData,
        user_ids: list[int] | torch.Tensor,
        seen_dict: dict[int, set[int]] | None = None,
        *,
        top_n: int = 20,
        user_type: str = "user",
        item_type: str = "track",
        query_batch: int = 512,
        precomputed_emb: dict[str, torch.Tensor] | None = None,
    ) -> dict[int, list[int]]:
        """Top-N recommendation for a batch of users.

        Parameters
        ----------
        data : HeteroData
            The full graph (moved to model's device automatically).
        user_ids : list[int] | Tensor
            KG node indices of users to generate recommendations for.
        seen_dict : {user_kg_idx: {item_kg_idx, ...}}, optional
            Training interactions to mask before ranking.
        top_n : int
            Number of items to return per user.
        user_type, item_type : str
            Node-type keys for users and items in ``data``.
        query_batch : int
            Process this many users per scoring chunk to control peak memory.
        precomputed_emb : dict, optional
            Pass already-computed embeddings (output of ``encode()``) to skip
            the forward pass — useful when recommending for many users at once.

        Returns
        -------
        dict mapping user_kg_idx → ordered list of item_kg_idx (best first).
        """
        self.eval()
        device = next(self.parameters()).device

        emb = (
            precomputed_emb
            if precomputed_emb is not None
            else self.encode(data, device=device)
        )
        u_emb = emb[user_type]    # (U, D)
        i_emb = emb[item_type]    # (I, D)

        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.tolist()

        recs: dict[int, list[int]] = {}
        for start in range(0, len(user_ids), query_batch):
            batch_uids: list[int] = user_ids[start: start + query_batch]
            u_idx = torch.tensor(batch_uids, device=device)
            scores = u_emb[u_idx] @ i_emb.T              # (B, I)

            # Mask training-seen items
            if seen_dict is not None:
                for bi, uid in enumerate(batch_uids):
                    seen = seen_dict.get(int(uid))
                    if seen:
                        scores[bi, list(seen)] = -torch.inf

            k = min(top_n, i_emb.size(0))
            _, top_idx = torch.topk(scores, k=k, dim=-1)
            for bi, uid in enumerate(batch_uids):
                recs[int(uid)] = top_idx[bi].tolist()

        return recs


__all__ = ("RecommenderHGT",)