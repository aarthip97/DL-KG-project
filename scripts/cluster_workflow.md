# Headless / Slurm workflow

The `04_DL_pipeline.ipynb` notebook is the canonical experimentation surface,
but the project also ships a CLI runner that mirrors its DL stages so the
exact same code path can run on a faculty GPU cluster (or any SSH-only host).

## Files

| File | Purpose |
|------|---------|
| `scripts/run_pipeline.py`           | Single-entry CLI wrapping the autoencoder, KGE and HGT trainers. |
| `scripts/slurm/run_pipeline.sbatch` | Reference Slurm submission script. |

---

## 1 — SSH into the cluster

```bash
# Replace with your actual username and cluster hostname
ssh <your_username>@<cluster.hostname.edu>
```

If the faculty cluster uses an SSH certificate / key file:

```bash
ssh -i ~/.ssh/<cert_or_key_file> <your_username>@<cluster.hostname.edu>
```

---

## 2 — One-time setup on the cluster

Run these commands **once** after your first SSH login.

```bash
# ── 2a. Clone the repo and check out the working branch ────────────────────
git clone https://github.com/aarthip97/DL-KG-project.git $HOME/DL-KG-project
cd $HOME/DL-KG-project
git checkout graph_contruction          # the active development branch

# ── 2b. Create and activate a Python virtual environment ───────────────────
# Check which Python version is available first:
#   python3 --version   OR   module avail python
# On many HPC clusters you may need to load a module before python3 exists:
# module load python/3.11 cuda/12.1

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ── 2c. Install all dependencies ───────────────────────────────────────────
# requirements.txt contains PyTorch, PyKEEN, torch-geometric, wandb, rdflib…
pip install -r requirements.txt

# If the cluster has its own CUDA-enabled PyTorch wheel (recommended):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# ── 2d. Store your W&B API key ─────────────────────────────────────────────
# Get the key from https://wandb.ai/authorize
echo 'export WANDB_API_KEY=<paste_key_here>' >> ~/.bashrc
source ~/.bashrc

# ── 2e. Create directories the pipeline writes to ──────────────────────────
mkdir -p $HOME/DL-KG-project/{models/{autoencoder,kge,hgt},logs,data/{interim,final}}
```

---

## 3 — Transfer data FROM your PC TO the cluster

The GPU stages need only a handful of pre-built artefacts — **not** the raw
MIDI files.  Transfer them once from your local machine before submitting jobs.

```bash
# Run these commands ON YOUR LOCAL MACHINE (not the cluster).
# Replace <user>@<cluster.hostname.edu> with your actual credentials.

CLUSTER="<user>@<cluster.hostname.edu>"
DEST="$CLUSTER:~/DL-KG-project"

# The populated knowledge-graph (produced by notebook 02)
rsync -avP data/final/MusicRecSyst_populated.ttl   $DEST/data/final/

# The per-track jSymbolic feature CSV (produced by notebook 01)
rsync -avP data/interim/interim.csv                $DEST/data/interim/

# (Optional) if you already trained the AE / KGE locally and want to skip those stages:
rsync -avP models/autoencoder/ae_model.pt          $DEST/models/autoencoder/
rsync -avP models/kge/kge_checkpoint.pt            $DEST/models/kge/
rsync -avP data/interim/ae_embeddings.parquet      $DEST/data/interim/
rsync -avP data/interim/kg_triples.tsv             $DEST/data/interim/
rsync -avP data/interim/edge_dict.json             $DEST/data/interim/
```

> **Tip** — `rsync -avP` shows per-file progress and skips files that are
> already up-to-date, so it is safe to re-run.  For large files through a
> slow connection, add `--compress` (`-z`).

---

## 4 — Activate the environment and run (interactive test)

Every SSH session and every sbatch script must re-activate the venv:

```bash
cd $HOME/DL-KG-project
source .venv/bin/activate

# Quick smoke-test on CPU (no GPU allocation needed)
python scripts/run_pipeline.py autoencoder \
    --data-root data --epochs-ae 2 --device cpu
```

---

## 5 — Submit Slurm jobs

```bash
# Run the full pipeline (extract → AE → KGE → HGT)
sbatch scripts/slurm/run_pipeline.sbatch all

# Re-train only HGT (AE + KGE artefacts already on disk)
sbatch scripts/slurm/run_pipeline.sbatch hgt

# Only KGE
sbatch scripts/slurm/run_pipeline.sbatch kge

# Check your job queue
squeue -u $USER

# Cancel a job
scancel <JOBID>

# Watch the live log (stdout) while the job is running
tail -f logs/dl-kg-pipeline-<JOBID>.out
```

Adjust `--partition`, `--gres`, `--time` and the `module load` lines in
`scripts/slurm/run_pipeline.sbatch` to match your cluster's conventions.

---

## 6 — Transfer results BACK to your PC for analysis

Run these on **your local machine** after the job finishes.

```bash
CLUSTER="<user>@<cluster.hostname.edu>"
SRC="$CLUSTER:~/DL-KG-project"

# Slurm logs (stdout / stderr)
rsync -avP $SRC/logs/                    results/cluster_logs/

# Trained model weights
rsync -avP $SRC/models/                  models/

# Intermediate embeddings + artefacts
rsync -avP $SRC/data/interim/            data/interim/
```

---

## 7 — Analyse training results locally

Once `models/hgt/hgt_results.pkl` is on your machine, open it in a notebook
or a script:

```python
import pickle, pandas as pd, matplotlib.pyplot as plt

with open("models/hgt/hgt_results.pkl", "rb") as f:
    r = pickle.load(f)

# r["history"] → dict of lists: train_loss, val_loss, …
# r["metrics"] → dict of final evaluation numbers

history = pd.DataFrame(r["history"])
print(history.tail())
print("\nFinal metrics:", r["metrics"])

fig, ax = plt.subplots()
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"],   label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150)
```

For richer dashboards, every metric is also synced to **W&B** in real time
when `--wandb-project` is passed.  Open
`https://wandb.ai/<your-entity>/music-recommender-system`
while the job is still running — no need to wait for it to finish.

---

## 8 — Keeping the repo in sync

After any local edits, push to GitHub and pull on the cluster:

```bash
# On your local machine
git add -A && git commit -m "..." && git push

# On the cluster (before re-submitting)
cd $HOME/DL-KG-project && git pull
```

---

## Outputs reference

```
data/interim/kg_triples.tsv          PyKEEN triple file
data/interim/edge_dict.json          HeteroData edge index dict
data/interim/ae_embeddings.parquet   Per-track 128-D audio embeddings
models/autoencoder/ae_model.pt       Autoencoder weights
models/kge/kge_checkpoint.pt         RotatE / ComplEx entity + relation embeddings
models/hgt/hgt_model.pt              HGT recommender weights
models/hgt/hgt_results.pkl           Loss history + evaluation metrics dict
logs/dl-kg-pipeline-<JOBID>.out      Full Slurm stdout (all stage logs)
logs/dl-kg-pipeline-<JOBID>.err      Slurm stderr (CUDA warnings, etc.)
```
