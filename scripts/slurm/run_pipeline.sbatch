#!/bin/bash
#SBATCH --job-name=dl-kg-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# ── Usage ────────────────────────────────────────────────────────────────────
# Submit the full pipeline:
#   sbatch scripts/slurm/run_pipeline.sbatch
#
# Override the stage (default: all):
#   STAGE=hgt sbatch scripts/slurm/run_pipeline.sbatch
#   STAGE=kge sbatch scripts/slurm/run_pipeline.sbatch
#
# ── Environment expectations ─────────────────────────────────────────────────
# Before submitting for the first time, do the ONE-TIME SETUP (see
# scripts/cluster_workflow.md, Section 2):
#
#   git clone https://github.com/aarthip97/DL-KG-project.git $HOME/DL-KG-project
#   cd $HOME/DL-KG-project && git checkout graph_contruction
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip && pip install -r requirements.txt
#   echo 'export WANDB_API_KEY=<your_key>' >> ~/.bashrc && source ~/.bashrc
#   mkdir -p models/{autoencoder,kge,hgt} logs data/{interim,final}
#
# Then transfer the data artefacts from your PC (see Section 3 of the same doc).

set -euo pipefail

REPO="${REPO:-$HOME/DL-KG-project}"
# STAGE can be: all | extract | autoencoder | kge | hgt
STAGE="${STAGE:-all}"

cd "$REPO"
mkdir -p logs

# ── Load cluster modules (uncomment / edit to match your HPC environment) ────
# module load python/3.11 cuda/12.1

# ── Activate the virtualenv ──────────────────────────────────────────────────
source "$REPO/.venv/bin/activate"

echo "[$(date)] host=$(hostname)  stage=$STAGE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ──────────────────────────────────────────────────────────────────────
srun python scripts/run_pipeline.py "$STAGE" \
    --data-root        "$REPO/data" \
    --kg-ttl           "$REPO/data/final/MusicRecSyst_populated.ttl" \
    --device           cuda \
    --wandb-project    music-recommender-system \
    --epochs-ae        30 \
    --epochs-kge       500 \
    --epochs-hgt       100 \
    --user-batch-hgt   1024 \
    --lambda-reg       0.2 \
    --temperature      0.1

echo "[$(date)] finished"
