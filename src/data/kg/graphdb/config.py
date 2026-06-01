"""Configuration for the GraphDB workflow.

A single dataclass that knows how to assemble itself from environment
variables (typically loaded from .env via python-dotenv).  Keeping all
GraphDB knobs in one place means the notebook + scripts + tests all share
the same source of truth, and switching from a local server to a cluster
deployment is a one-line edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GraphDBConfig:
    """All knobs needed to talk to a GraphDB instance and run the export.

    Resolution order for every field is:
        1. explicit constructor argument
        2. environment variable (case-insensitive — see ``_env``)
        3. hard-coded default
    """

    # ── Server connection ──
    url: str = "http://localhost:7200"
    repo_id: str = "music_recsys"
    username: Optional[str] = None
    password: Optional[str] = None

    # ── Repository creation ──
    ruleset: str = "rdfsplus-optimized"   # SAIL ruleset, see GraphDB docs
    repo_title: str = "Music RecSys Knowledge Graph"
    disable_same_as: bool = False         # owl:sameAs handling

    # ── HTTP behaviour ──
    timeout: float = 300.0                # seconds; uploads can be slow
    verify_ssl: bool = True

    # ── Where artefacts land ──
    # Defaults relative to repo root; the loader script (or the notebook)
    # passes its absolute paths via the constructor.
    out_dir: Path = field(default_factory=lambda: Path("data/processed/graphdb"))
    stats_dir: Path = field(default_factory=lambda: Path("data/final/kg_stats"))
    plots_dir: Path = field(default_factory=lambda: Path("data/final/kg_plots"))

    # ── Bookkeeping ──
    # File written next to ``out_dir`` mapping uploaded-file SHA256 → mtime,
    # so we can skip a re-upload on subsequent runs.
    upload_state_file: str = ".graphdb_uploaded.json"

    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, **overrides) -> "GraphDBConfig":
        """Build from environment variables.

        Recognised keys (case-insensitive, dotenv-friendly):
            GRAPHDB_URL, GRAPHDB_REPO, GRAPHDB_USERNAME, GRAPHDB_PASSWORD,
            GRAPHDB_RULESET, GRAPHDB_TIMEOUT, GRAPHDB_OUT_DIR,
            GRAPHDB_STATS_DIR, GRAPHDB_PLOTS_DIR.

        Any keyword passed explicitly wins over the env value.
        """
        kwargs: Dict[str, Any] = {
            "url":      _env("GRAPHDB_URL", "http://localhost:7200"),
            "repo_id":  _env("GRAPHDB_REPO", "music_recsys"),
            "username": _env("GRAPHDB_USERNAME"),
            "password": _env("GRAPHDB_PASSWORD"),
            "ruleset":  _env("GRAPHDB_RULESET", "rdfsplus-optimized"),
            "timeout":  float(_env("GRAPHDB_TIMEOUT", "300") or 300),
        }
        for k in ("out_dir", "stats_dir", "plots_dir"):
            v = _env("GRAPHDB_" + k.upper())
            if v:
                kwargs[k] = Path(v)
        kwargs.update(overrides)
        return cls(**kwargs)

    # ────────────────────────────────────────────────────────────────────
    @property
    def has_credentials(self) -> bool:
        return bool(self.username and self.password)

    @property
    def repo_endpoint(self) -> str:
        return f"{self.url.rstrip('/')}/repositories/{self.repo_id}"

    @property
    def statements_endpoint(self) -> str:
        return f"{self.repo_endpoint}/statements"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Case-insensitive os.getenv with a single fallback default."""
    for k in (key, key.upper(), key.lower()):
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return default
