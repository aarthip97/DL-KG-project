# GraphDB workflow

This project keeps the *heavy* knowledge-graph work — bulk loading, SPARQL
analytics, PyKEEN triple export — in a real **GraphDB** server, and ships
only the small derived artefacts (CSVs / TSVs / Parquet / PNGs) up to
Google Drive so the downstream notebooks (Colab) can run on commodity
hardware without re-loading 150 M+ triples in-memory.

```
 ┌──────────────────┐    upload     ┌──────────────────┐    export     ┌────────────────────────┐
 │ kg_rich.ttl      │ ───────────►  │   GraphDB repo   │ ───────────►  │ data/processed/graphdb │
 │ listening.nt     │               │  music_recsys    │               │ data/final/kg_stats    │
 └──────────────────┘               └──────────────────┘               │ data/final/kg_plots    │
                                                                       └──────────┬─────────────┘
                                                                                  │  rsync / Drive
                                                                                  ▼
                                                                       ┌────────────────────────┐
                                                                       │  Colab notebook reads  │
                                                                       │  the precomputed CSVs  │
                                                                       └────────────────────────┘
```

## 1. Install GraphDB

### Local workstation (with sudo / systemd)

```bash
bash scripts/setup_graphdb.sh
sudo systemctl enable --now graphdb
```

GraphDB listens on `http://localhost:7200`.

### Cluster / SLURM (no root)

Submit a long-running job that runs the foreground launcher:

```bash
#!/bin/bash
#SBATCH -J graphdb
#SBATCH -t 12:00:00
#SBATCH --mem=16G
module load java/21
bash scripts/setup_graphdb_cluster.sh
```

Then point your client at `http://<compute-node>:7200` (the script prints
the exact URL on startup).

## 2. Configure credentials

Add the following to `.env` (already templated in this repo):

```env
GRAPHDB_URL=http://localhost:7200
GRAPHDB_REPO=music_recsys
GRAPHDB_USERNAME=admin
GRAPHDB_PASSWORD=changeme
GRAPHDB_RULESET=rdfsplus-optimized

# Optional output overrides — defaults shown
# GRAPHDB_OUT_DIR=data/processed/graphdb
# GRAPHDB_STATS_DIR=data/final/kg_stats
# GRAPHDB_PLOTS_DIR=data/final/kg_plots
```

## 3. Run the loader

```bash
python scripts/load_kg_to_graphdb.py \
    --rdf data/processed/kg_rich.ttl \
    --rdf data/processed/listening.nt
```

The script is **idempotent**: the SHA-256 of every uploaded file is
recorded in `data/processed/graphdb/.graphdb_uploaded.json`, so re-runs
that change nothing skip the upload step entirely.

Useful flags:

| Flag                | Purpose                                           |
|---------------------|---------------------------------------------------|
| `--skip-upload`     | Re-export CSVs without touching the server        |
| `--skip-stats`      | Skip the per-query CSVs                            |
| `--skip-pykeen`     | Skip `kg_triples.tsv` + `node_dict.json` + parquet |
| `--skip-plots`      | Skip the PNG generation                           |
| `--force`           | Ignore the upload-state cache, re-upload all      |
| `--replace-first`   | First file uses `PUT /statements` (clear-replace) |

## 4. Outputs

After a successful run you'll find:

```
data/processed/graphdb/
    kg_triples.tsv            # head\trel\ttail   ← pykeen
    node_dict.json            # {ntype: {uri: int}} ← pyg HeteroData
    hetero_edges.parquet      # rel/head/head_type/tail/tail_type
    export_manifest.json      # registry of everything written
    .graphdb_uploaded.json    # internal SHA-256 cache

data/final/kg_stats/          # one CSV per named query in queries.STATS_QUERIES
    triple_count.csv
    node_type_histogram.csv
    relation_histogram.csv
    genres_simple.csv
    genres_rich.csv
    confident_keys_simple.csv
    confident_keys_rich.csv
    genre_hierarchy.csv
    instrument_hierarchy.csv
    decade_chain.csv
    top5_popular_songs.csv

data/final/kg_plots/          # matplotlib PNGs
    genres_rich.png
    confident_keys_rich.png
    node_type_histogram.png
    relation_histogram.png
    ...
```

## 5. Push to Drive

The Colab side of the pipeline reads everything below `data/final/` and
the parquet/tsv under `data/processed/graphdb/`.  Either let the symlink
in cell 0.2h handle it (recommended, see notebook), or push manually:

```bash
rsync -av --delete \
    data/processed/graphdb/ data/final/kg_stats/ data/final/kg_plots/ \
    "$GDRIVE_PROJ_PATH/"
```

## 6. Use from a notebook (Colab)

```python
from data.kg.graphdb import GraphDBConfig, GraphDBClient, KGRepo

ON_COLAB    = "google.colab" in sys.modules
USE_GRAPHDB = (not ON_COLAB) and bool(os.environ.get("GRAPHDB_URL"))

if USE_GRAPHDB:
    cfg  = GraphDBConfig.from_env()
    cli  = GraphDBClient(cfg)
    repo = KGRepo(cli, cfg)
    df   = repo.query(queries.QUERY_GENRES_RICH)
else:
    df   = pd.read_csv(KG_STATS_DIR / "genres_rich.csv")
```

That's the only branching needed — the cells that produce the artefacts
are wrapped behind `if USE_GRAPHDB`, and every consumer cell reads from
the CSV/TSV path so Colab can run unmodified.
