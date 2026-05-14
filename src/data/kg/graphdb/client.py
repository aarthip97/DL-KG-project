"""Thin REST wrapper around GraphDB's Workbench + SPARQL endpoints.

Designed to be a *plain library* — no global state, no print-and-swallow.
Errors raise :class:`GraphDBError`; SELECT results come back as pandas
DataFrames so callers can keep their code declarative.

The Workbench REST API is documented at:
    https://graphdb.ontotext.com/documentation/standard/workbench-rest-api.html
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from .config import GraphDBConfig

log = logging.getLogger(__name__)


class GraphDBError(RuntimeError):
    """Raised when the GraphDB server returns a non-success status."""


# Mapping from common file suffix → MIME type GraphDB understands.
# Used by :meth:`GraphDBClient.upload_rdf` when no explicit type is given.
_CONTENT_TYPES = {
    ".ttl":   "text/turtle",
    ".nt":    "application/n-triples",
    ".n3":    "text/n3",
    ".rdf":   "application/rdf+xml",
    ".owl":   "application/rdf+xml",
    ".jsonld": "application/ld+json",
    ".trig":  "application/trig",
    ".nq":    "application/n-quads",
}


class GraphDBClient:
    """Stateless-ish REST client for a single GraphDB server.

    Re-usable across repositories — switch by passing a different
    :class:`GraphDBConfig` (or by building a new client).
    """

    def __init__(self, cfg: GraphDBConfig):
        self.cfg = cfg
        self._session = requests.Session()
        if cfg.has_credentials and cfg.username and cfg.password:
            self._session.auth = (cfg.username, cfg.password)
        self._session.verify = cfg.verify_ssl

    # ── infrastructure ──────────────────────────────────────────────────
    def ping(self) -> bool:
        """Return True iff the server responds to /rest/info/version."""
        try:
            r = self._session.get(
                f"{self.cfg.url.rstrip('/')}/rest/info/version",
                timeout=10,
            )
            return r.ok
        except requests.RequestException:
            return False

    def ensure_user(self) -> None:
        """Idempotent admin-user creation.  No-op if no credentials configured."""
        cfg = self.cfg
        if not cfg.has_credentials:
            log.info("No credentials configured; skipping user creation.")
            return
        url = f"{cfg.url.rstrip('/')}/rest/security/users/{cfg.username}"
        payload = {
            "password": cfg.password,
            "grantedAuthorities": ["ROLE_ADMIN", "ROLE_USER"],
        }
        r = self._session.post(url, json=payload,
                               headers={"Content-Type": "application/json"},
                               timeout=cfg.timeout)
        if r.status_code == 201:
            log.info("Created admin user %s", cfg.username)
        elif r.status_code in (400, 409):
            log.debug("User %s already exists", cfg.username)
        else:
            raise GraphDBError(
                f"User creation failed [{r.status_code}]: {r.text[:200]}"
            )

    def ensure_repository(self) -> None:
        """Idempotent repository creation via RDF4J Turtle config (PUT).

        GraphDB 11 uses the ``tag:rdf4j.org,2023:config/`` namespace and
        only accepts ``multipart/form-data`` on ``POST /rest/repositories``
        (the UI path).  We write the config.ttl directly into the GraphDB
        data directory and then let the repository manager pick it up via
        ``POST /rest/repositories`` with a form-data ``config`` file part,
        which avoids the charset-appending issues of a bare PUT.
        """
        cfg = self.cfg
        if self.repository_exists():
            log.debug("Repository %s already exists", cfg.repo_id)
            return

        disable_same_as = str(cfg.disable_same_as).lower()
        ttl = f"""\
@prefix config: <tag:rdf4j.org,2023:config/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix graphdb: <http://www.ontotext.com/config/graphdb#> .

<#{cfg.repo_id}> a config:Repository ;
   rdfs:label "{cfg.repo_title}" ;
   config:rep.id "{cfg.repo_id}" ;
   config:rep.impl [
       config:rep.type "graphdb:SailRepository" ;
       config:sail.impl [
           config:sail.type "graphdb:Sail" ;
           graphdb:ruleset "{cfg.ruleset}" ;
           graphdb:disable-sameAs "{disable_same_as}" ;
           graphdb:enablePredicateList "true" ;
           graphdb:in-memory-literal-properties "true" ;
           graphdb:repository-type "file-repository" ;
           graphdb:base-URL "http://example.org/owlim#" ;
           graphdb:defaultNS "" ;
           graphdb:imports ""
       ]
   ] .
"""
        r = self._session.post(
            f"{cfg.url.rstrip('/')}/rest/repositories",
            files={"config": ("config.ttl", ttl.encode("utf-8"), "text/turtle")},
            timeout=cfg.timeout,
        )
        if r.status_code == 201:
            log.info("Created repository %s", cfg.repo_id)
        elif r.status_code == 409:
            log.debug("Repository %s already exists (race condition)", cfg.repo_id)
        else:
            raise GraphDBError(
                f"Repository creation failed [{r.status_code}]: {r.text[:300]}"
            )

    def repository_exists(self) -> bool:
        r = self._session.get(
            f"{self.cfg.url.rstrip('/')}/rest/repositories/{self.cfg.repo_id}",
            timeout=10,
        )
        return r.status_code == 200

    def clear_repository(self) -> None:
        """Delete every triple in the repo (keeps the repo itself)."""
        r = self._session.delete(self.cfg.statements_endpoint,
                                 timeout=self.cfg.timeout)
        if not r.ok:
            raise GraphDBError(f"Clear failed [{r.status_code}]: {r.text[:200]}")

    # ── data ingestion ──────────────────────────────────────────────────
    def upload_rdf(self,
                   path: Path,
                   content_type: Optional[str] = None,
                   replace: bool = False) -> None:
        """POST a single RDF file to the repository's /statements endpoint.

        Suitable for files up to a few hundred MB.  For larger files prefer
        the GraphDB native loader (``loadrdf`` CLI run server-side).
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if content_type is None:
            content_type = _CONTENT_TYPES.get(path.suffix.lower())
            if content_type is None:
                raise ValueError(
                    f"Cannot infer content-type for {path.suffix}; "
                    f"pass content_type= explicitly."
                )
        url = self.cfg.statements_endpoint
        method = self._session.put if replace else self._session.post

        log.info("Uploading %s (%s, %.1f MiB) → %s",
                 path.name, content_type,
                 path.stat().st_size / (1 << 20), self.cfg.repo_id)
        with path.open("rb") as fh:
            r = method(url,
                       headers={"Content-Type": content_type},
                       data=fh,
                       timeout=self.cfg.timeout)
        if r.status_code != 204:
            raise GraphDBError(
                f"Upload of {path.name} failed [{r.status_code}]: "
                f"{r.text[:200]}"
            )

    def upload_many(self, paths: Iterable[Path], replace_first: bool = False) -> None:
        """Upload several files in order; optionally clear-replace with the first."""
        paths = list(paths)
        for i, p in enumerate(paths):
            self.upload_rdf(p, replace=replace_first and i == 0)

    # ── querying ────────────────────────────────────────────────────────
    def select_df(self, sparql: str) -> pd.DataFrame:
        """Run a SPARQL SELECT and return the result as a DataFrame."""
        r = self._session.post(
            self.cfg.repo_endpoint,
            headers={
                "Accept":       "text/csv",
                "Content-Type": "application/sparql-query",
            },
            data=sparql.encode("utf-8"),
            timeout=self.cfg.timeout,
        )
        if not r.ok:
            raise GraphDBError(
                f"SELECT failed [{r.status_code}]: {r.text[:300]}"
            )
        return pd.read_csv(io.StringIO(r.text))

    def select_tsv(self, sparql: str, out_path: Path) -> Path:
        """Run a SPARQL SELECT and stream the result straight to a TSV file.

        Avoids materialising the whole result set in Python memory — useful
        for the PyKEEN triples export which can have tens of millions of rows.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session.post(
            self.cfg.repo_endpoint,
            headers={
                "Accept":       "text/tab-separated-values",
                "Content-Type": "application/sparql-query",
            },
            data=sparql.encode("utf-8"),
            timeout=self.cfg.timeout,
            stream=True,
        ) as r:
            if not r.ok:
                raise GraphDBError(
                    f"SELECT-stream failed [{r.status_code}]: {r.text[:300]}"
                )
            with out_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        return out_path

    def construct_nt(self, sparql: str, out_path: Path) -> Path:
        """Run a CONSTRUCT/DESCRIBE query and stream N-Triples to ``out_path``."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session.post(
            self.cfg.repo_endpoint,
            headers={
                "Accept":       "application/n-triples",
                "Content-Type": "application/sparql-query",
            },
            data=sparql.encode("utf-8"),
            timeout=self.cfg.timeout,
            stream=True,
        ) as r:
            if not r.ok:
                raise GraphDBError(
                    f"CONSTRUCT failed [{r.status_code}]: {r.text[:300]}"
                )
            with out_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        return out_path

    def update(self, sparql: str) -> None:
        """Execute a SPARQL UPDATE (INSERT/DELETE) on the repo."""
        r = self._session.post(
            self.cfg.statements_endpoint,
            headers={"Content-Type": "application/sparql-update"},
            data=sparql.encode("utf-8"),
            timeout=self.cfg.timeout,
        )
        if r.status_code not in (200, 204):
            raise GraphDBError(
                f"UPDATE failed [{r.status_code}]: {r.text[:300]}"
            )

    # ── housekeeping ────────────────────────────────────────────────────
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "GraphDBClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
