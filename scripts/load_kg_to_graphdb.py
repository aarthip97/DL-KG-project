#!/usr/bin/env python3
"""End-to-end loader: KG files → GraphDB → CSV / TSV / Parquet / PNG artefacts.

Designed to be run *locally* (or on the cluster) on the host that has the
GraphDB server reachable.  All artefacts land in the directories defined
by :class:`graphdb.GraphDBConfig`, which by default sit under
``data/processed/graphdb`` and ``data/final/{kg_stats,kg_plots}``.
The Colab side of the workflow only needs to read those files.

Examples
--------
Bootstrap, upload the rich KG, run all exports::

    python scripts/load_kg_to_graphdb.py \\
        --rdf data/processed/kg_rich.ttl \\
        --rdf data/processed/listening.nt

Re-export only stats + plots (no upload)::

    python scripts/load_kg_to_graphdb.py --skip-upload

Force a re-upload (ignores the SHA-256 dedup state file)::

    python scripts/load_kg_to_graphdb.py --force --rdf data/processed/kg_rich.ttl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src/` importable when run as a plain script (no setuptools install)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.kg.graphdb import GraphDBClient, GraphDBConfig, KGRepo  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rdf", action="append", type=Path, default=[],
                   help="RDF file to upload (can be repeated). Skipped if "
                        "absent and --skip-upload is on.")
    p.add_argument("--out-dir",   type=Path, help="Override processed/graphdb dir")
    p.add_argument("--stats-dir", type=Path, help="Override final/kg_stats dir")
    p.add_argument("--plots-dir", type=Path, help="Override final/kg_plots dir")
    p.add_argument("--force",        action="store_true",
                   help="Ignore upload state file; re-upload everything.")
    p.add_argument("--replace-first", action="store_true",
                   help="PUT (clear-and-replace) the very first RDF file "
                        "instead of POST-appending.")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--skip-stats",  action="store_true")
    p.add_argument("--skip-pykeen", action="store_true")
    p.add_argument("--skip-plots",  action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {
        k: v for k, v in {
            "out_dir":   args.out_dir,
            "stats_dir": args.stats_dir,
            "plots_dir": args.plots_dir,
        }.items() if v is not None
    }
    cfg = GraphDBConfig.from_env(**overrides)

    with GraphDBClient(cfg) as cli:
        repo = KGRepo(cli, cfg)
        repo.bootstrap()

        if not args.skip_upload:
            if not args.rdf:
                print("[!] --skip-upload not given but no --rdf files passed; "
                      "nothing to upload.", file=sys.stderr)
            else:
                sent = repo.upload_kg(args.rdf,
                                      force=args.force,
                                      replace_first=args.replace_first)
                print(f"[✓] Uploaded {len(sent)} / {len(args.rdf)} files "
                      f"({len(args.rdf) - len(sent)} skipped via hash dedup)")

        out = repo.export_all(skip_pykeen=args.skip_pykeen,
                              skip_stats=args.skip_stats,
                              skip_plots=args.skip_plots)

    print(f"\n[✓] Wrote {len(out)} artefacts:")
    for name, path in sorted(out.items()):
        print(f"    {name:<28} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
