"""Deep-learning stack for the music recommender capstone.

Public API:

* :func:`prune_rdf_graph`           — strip OWL/RDF admin triples from a TSV.
* :func:`load_kg_as_hetero`         — TTL (+ listening sidecar) -> ``HeteroData``.
* :class:`jSymbolicAutoencoder`     — dense audio-feature compressor.
* :func:`train_autoencoder`         — convenience training loop for the AE.
* :class:`RecommenderHGT`           — Heterogeneous Graph Transformer model.
* :func:`bpr_loss` / :func:`evaluate_top_k` — BPR objective + ranking metrics.
* :func:`train_hgt`                 — HGT training loop with optional W&B.
"""

from .pruning import prune_rdf_graph
from .kg_to_hetero import load_kg_as_hetero, KGEncoding
from .autoencoder import jSymbolicAutoencoder, train_autoencoder
from .bpr import bpr_loss, evaluate_top_k
from .hgt import RecommenderHGT
from .train import train_hgt

__all__ = [
    "prune_rdf_graph",
    "load_kg_as_hetero",
    "KGEncoding",
    "jSymbolicAutoencoder",
    "train_autoencoder",
    "RecommenderHGT",
    "bpr_loss",
    "evaluate_top_k",
    "train_hgt",
]
