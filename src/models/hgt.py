"""Heterogeneous Graph Transformer recommender.

Wraps ``torch_geometric.nn.HGTConv`` with per-node-type input projections
so node types of different feature dimensions can coexist.
"""
from __future__ import annotations

import torch
from torch import nn

try:                                  # pragma: no cover
    from torch_geometric.nn import HGTConv, Linear as PyGLinear
except ImportError as exc:           # pragma: no cover
    raise ImportError(
        "torch_geometric is required for the HGT model. "
        "Install via `pip install torch_geometric`."
    ) from exc


class RecommenderHGT(nn.Module):
    def __init__(
        self,
        metadata: tuple,                          # (node_types, edge_types)
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        node_types, _ = metadata
        # ``Linear(-1, h)`` lazily infers the in_features per node type.
        self.lin_in = nn.ModuleDict({
            nt: PyGLinear(-1, hidden_channels) for nt in node_types
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])
        self.lin_out = nn.ModuleDict({
            nt: nn.Linear(hidden_channels, out_channels) for nt in node_types
        })

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: t.relu() for nt, t in h.items()}
        return {nt: self.lin_out[nt](t) for nt, t in h.items()}


__all__ = ("RecommenderHGT",)
