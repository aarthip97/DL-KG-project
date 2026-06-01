"""Back-compat shim around the new ``data.kg.graphdb`` package.

The original module exposed four module-level functions
(``setup_admin_user``, ``create_repository``, ``upload_rdf_file``,
``export_pykeen_tsv``) that walked a private ``requests.Session`` keyed off
of dotenv globals.  All real logic now lives in the package; this file
keeps the old import path working for existing callers.

Prefer the new API for any new code::

    from data.kg.graphdb import GraphDBConfig, GraphDBClient, KGRepo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .graphdb import GraphDBClient, GraphDBConfig, KGRepo
from .graphdb.exports import export_pykeen_tsv as _export_pykeen_tsv

__all__ = [
    "GraphDBConfig", "GraphDBClient", "KGRepo",
    # Legacy function names:
    "setup_admin_user", "create_repository",
    "upload_rdf_file", "export_pykeen_tsv",
]


def _client(cfg: Optional[GraphDBConfig] = None) -> GraphDBClient:
    return GraphDBClient(cfg or GraphDBConfig.from_env())


# ── Legacy functional API ───────────────────────────────────────────────
def setup_admin_user(cfg: Optional[GraphDBConfig] = None) -> None:
    _client(cfg).ensure_user()


def create_repository(cfg: Optional[GraphDBConfig] = None) -> None:
    _client(cfg).ensure_repository()


def upload_rdf_file(path: str | Path,
                    cfg: Optional[GraphDBConfig] = None,
                    *, replace: bool = False) -> None:
    _client(cfg).upload_rdf(Path(path), replace=replace)


def export_pykeen_tsv(out_path: str | Path,
                      cfg: Optional[GraphDBConfig] = None) -> Path:
    cli = _client(cfg)
    try:
        return _export_pykeen_tsv(cli, Path(out_path))
    finally:
        cli.close()
