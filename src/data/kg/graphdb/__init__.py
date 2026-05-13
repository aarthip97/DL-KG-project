"""GraphDB client + workflow helpers.

This package replaces the old single-file ``graphdb_client.py``.  It exposes
a small, composable API:

    from data.kg.graphdb import GraphDBClient, GraphDBConfig, KGRepo, queries

    cfg    = GraphDBConfig.from_env()
    client = GraphDBClient(cfg)
    repo   = KGRepo(client, cfg)

    repo.bootstrap()                          # admin user + repository
    repo.upload_kg([ttl_file, nt_file])       # idempotent — checksum-aware
    repo.export_all(out_dir)                  # writes pykeen TSV + edges + stats

The notebook is meant to **read** the artefacts produced by ``repo.export_all``,
not to talk to GraphDB directly — that way Colab stays self-contained while the
heavy SPARQL work happens locally / on the cluster.
"""

from .config import GraphDBConfig
from .client import GraphDBClient, GraphDBError
from .repo import KGRepo
from . import queries
from . import exports
from . import viz

__all__ = [
    "GraphDBConfig",
    "GraphDBClient",
    "GraphDBError",
    "KGRepo",
    "queries",
    "exports",
    "viz",
]
