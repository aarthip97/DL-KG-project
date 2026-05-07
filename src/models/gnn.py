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
from torch_geometric.nn import LightGCN # Or SAGEConv/GATConv

class RecommenderGNN(nn.Module):
    def __init__(self, audio_features, kge_features, freeze_strategy, hidden_channels=64):
        super().__init__()
        
        # 1. Initialize Audio Embeddings (128 dimensions)
        # Note: Non-track nodes will have zeros here.
        self.audio_emb = nn.Parameter(torch.tensor(audio_features, dtype=torch.float))
        
        # 2. Initialize KGE Embeddings (256 dimensions)
        self.kge_emb = nn.Parameter(torch.tensor(kge_features, dtype=torch.float))
        
        # 3. Apply the Freezing Logic based on the experiment
        if freeze_strategy == 'freeze_all':
            self.audio_emb.requires_grad = False
            self.kge_emb.requires_grad = False
            print("Status: ALL embeddings frozen. Only updating GNN weights.")
            
        elif freeze_strategy == 'freeze_audio':
            self.audio_emb.requires_grad = False
            self.kge_emb.requires_grad = True # The model can shift the structure
            print("Status: Audio frozen. KGE embeddings will be updated.")
            
        elif freeze_strategy == 'freeze_none':
            self.audio_emb.requires_grad = True
            self.kge_emb.requires_grad = True
            print("Status: NO embeddings frozen. Full representation learning.")
            
        else:
            raise ValueError("Unknown freeze strategy!")

        # 4. Define the GNN Layers
        # The input will be exactly 384 dimensions (128 + 256)
        self.conv1 = LightGCN(384, hidden_channels)
        self.conv2 = LightGCN(hidden_channels, hidden_channels)

    def forward(self, edge_index):
        # Dynamically concatenate the vectors on the forward pass!
        # This yields an (N, 384) tensor where gradients only flow back
        # to the halves that have requires_grad=True
        x = torch.cat([self.audio_emb, self.kge_emb], dim=-1)
        
        # Message Passing
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

__all__ = ("RecommenderGNN",)
