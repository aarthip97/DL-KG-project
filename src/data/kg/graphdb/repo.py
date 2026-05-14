"""High-level repo orchestration: bootstrap, upload, and full export.

This is the only object the notebooks and CLI scripts need to know
about — it composes :class:`GraphDBClient`, :mod:`queries`,
:mod:`exports`, and :mod:`viz` into a single coherent API.

Typical use::

    from data.kg.graphdb import GraphDBConfig, GraphDBClient, KGRepo

    cfg = GraphDBConfig.from_env()
    with GraphDBClient(cfg) as cli:
        repo = KGRepo(cli, cfg)
        repo.bootstrap()
        repo.upload_kg([Path("data/processed/kg_rich.ttl"),
                        Path("data/processed/listening.nt")])
        repo.export_all()
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from . import exports, queries, viz
from .client import GraphDBClient
from .config import GraphDBConfig

log = logging.getLogger(__name__)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class KGRepo:
    """Composes a client + config and exposes idempotent pipeline steps."""

    def __init__(self, client: GraphDBClient, cfg: Optional[GraphDBConfig] = None):
        self.client = client
        self.cfg = cfg or client.cfg

    # ── bootstrap ────────────────────────────────────────────────────
    def bootstrap(self) -> None:
        """Create user (if needed) and the repository (if missing)."""
        if not self.client.ping():
            raise RuntimeError(
                f"GraphDB at {self.cfg.url} is not reachable — start it with "
                f"`scripts/setup_graphdb.sh` (local) or "
                f"`scripts/setup_graphdb_cluster.sh` (SLURM)."
            )
        self.client.ensure_user()
        self.client.ensure_repository()

    # ── upload (with content-hash dedup) ─────────────────────────────
    def _state_path(self) -> Path:
        return self.cfg.out_dir / self.cfg.upload_state_file

    def _load_state(self) -> Dict[str, str]:
        p = self._state_path()
        return json.loads(p.read_text()) if p.is_file() else {}

    def _save_state(self, state: Dict[str, str]) -> None:
        p = self._state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))

    def upload_kg(self, paths: Iterable[Path],
                  *, force: bool = False,
                  replace_first: bool = False) -> List[Path]:
        """Upload RDF files; skip those whose SHA-256 hasn't changed.

        Returns the list of files actually sent to the server.
        """
        state = {} if force else self._load_state()
        sent: List[Path] = []
        for i, raw in enumerate(paths):
            p = Path(raw)
            digest = _sha256(p)
            if state.get(str(p)) == digest:
                log.info("Skipping %s (already uploaded, hash matches)", p.name)
                continue
            self.client.upload_rdf(p, replace=replace_first and i == 0 and not sent)
            state[str(p)] = digest
            sent.append(p)
        if sent:
            self._save_state(state)
        return sent

    # ── full export ──────────────────────────────────────────────────
    def export_all(self,
                   *,
                   skip_pykeen: bool = False,
                   skip_stats: bool = False,
                   skip_plots: bool = False) -> Dict[str, Path]:
        """Run every artefact-producing step.

        Output paths are taken from :class:`GraphDBConfig` so a caller can
        steer them via env vars (``GRAPHDB_OUT_DIR`` etc.).  Returns a
        dict of artefact-name → path for downstream wiring.
        """
        cfg = self.cfg
        out: Dict[str, Path] = {}

        # 1. Stats CSVs
        if not skip_stats:
            cfg.stats_dir.mkdir(parents=True, exist_ok=True)
            for name, sparql in queries.STATS_QUERIES.items():
                out_csv = cfg.stats_dir / f"{name}.csv"
                try:
                    # infer=True: stats reflect the full logical graph
                    # (including RDFS+ inferences), not just explicit triples.
                    # This makes triple_count.csv show the actual total the
                    # embedder will see, which is > /size (explicit only).
                    df = self.client.select_df(sparql, infer=True)
                    df.to_csv(out_csv, index=False)
                    out[f"stats:{name}"] = out_csv
                    log.info("Stats[%s] → %d rows", name, len(df))
                except Exception as e:
                    log.warning("Stats[%s] failed: %s", name, e)

        # 2. PyKEEN triples + node dict + hetero edges
        if not skip_pykeen:
            triples = exports.export_pykeen_tsv(
                self.client, cfg.out_dir / "pykeen_triples.tsv"
            )
            ndict = exports.export_node_dict(
                self.client, cfg.out_dir / "node_dict.json"
            )
            edges = exports.export_hetero_edges(
                triples, ndict, cfg.out_dir / "hetero_edges.parquet"
            )
            out["pykeen_tsv"]    = triples
            out["node_dict"]     = ndict
            out["hetero_edges"]  = edges

        # 3. Plots
        if not skip_plots and not skip_stats:
            for name, png in viz.plot_all(cfg.stats_dir, cfg.plots_dir).items():
                out[f"plot:{name}"] = png

        # 4. Manifest — single small JSON describing everything we wrote
        manifest = cfg.out_dir / "export_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(
            {k: str(v) for k, v in out.items()}, indent=2, sort_keys=True
        ))
        out["manifest"] = manifest
        log.info("Export complete — %d artefacts under %s", len(out) - 1, cfg.out_dir)
        return out

    # ── ad-hoc query helper (used by notebooks) ──────────────────────
    def query(self, sparql: str) -> pd.DataFrame:
        """Convenience: forward to the client for ad-hoc SELECTs."""
        return self.client.select_df(sparql)
