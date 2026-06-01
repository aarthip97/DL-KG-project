#!/bin/bash
# ==============================================================================
# GraphDB Setup for HPC / SLURM clusters (no systemd, foreground process)
# ==============================================================================
#
# This is a counterpart to ``setup_graphdb.sh`` that does **not** require root
# (no systemd unit, no sudo).  Use it from a SLURM job script::
#
#     #SBATCH -J graphdb
#     #SBATCH -t 04:00:00
#     #SBATCH --mem=16G
#     module load java/21
#     bash scripts/setup_graphdb_cluster.sh
#
# The server runs in the foreground until the job is killed.  Point client
# code at ``http://<compute-node-host>:7200`` (Set ``GRAPHDB_URL`` in .env).
#
# To use a non-default port, install dir or heap size, override via env::
#
#     GRAPHDB_INSTALL_DIR=$SCRATCH/graphdb-server \
#     GRAPHDB_PORT=17200 GRAPHDB_HEAP=8g \
#     bash scripts/setup_graphdb_cluster.sh
# ==============================================================================

set -euo pipefail

GRAPHDB_VERSION="${GRAPHDB_VERSION:-11.3.3}"
GRAPHDB_INSTALL_DIR="${GRAPHDB_INSTALL_DIR:-$HOME/graphdb-server}"
GRAPHDB_PORT="${GRAPHDB_PORT:-7200}"
GRAPHDB_HEAP="${GRAPHDB_HEAP:-8g}"
ZIP_FILE="${TMPDIR:-/tmp}/graphdb-${GRAPHDB_VERSION}-dist.zip"
DOWNLOAD_URL="https://download.ontotext.com/owlim/0521929f-94ab-4ac0-adce-84fc0426b69e/graphdb-11.3.3-dist.zip?_gl=1*15600kb*_ga*MTE2MzA3NTI2OS4xNzc4MTMyOTEw*_ga_HGSKWBWCRK*czE3Nzg2OTk1ODMkbzMkZzEkdDE3Nzg3MDAxMTYkajU5JGwwJGgw"

echo "=== GraphDB cluster setup ==="
echo "    install_dir : $GRAPHDB_INSTALL_DIR"
echo "    port        : $GRAPHDB_PORT"
echo "    heap        : $GRAPHDB_HEAP"

# 1. Download + extract (one-time, idempotent)
if [ ! -d "$GRAPHDB_INSTALL_DIR" ]; then
    echo "[*] Downloading GraphDB ${GRAPHDB_VERSION}..."
    wget -q -O "$ZIP_FILE" "$DOWNLOAD_URL"
    echo "[*] Extracting to $(dirname "$GRAPHDB_INSTALL_DIR")..."
    unzip -q "$ZIP_FILE" -d "$(dirname "$GRAPHDB_INSTALL_DIR")"
    mv "$(dirname "$GRAPHDB_INSTALL_DIR")/graphdb-${GRAPHDB_VERSION}" "$GRAPHDB_INSTALL_DIR"
    rm -f "$ZIP_FILE"
else
    echo "[*] GraphDB already installed — skipping download."
fi

# 2. License (optional)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIC="${SCRIPT_DIR}/../graphdb.license"
if [ -f "$LIC" ] && [ ! -f "${GRAPHDB_INSTALL_DIR}/conf/graphdb.license" ]; then
    echo "[*] Injecting graphdb.license"
    cp "$LIC" "${GRAPHDB_INSTALL_DIR}/conf/"
fi

# 3. Foreground launch — log to stdout for SLURM to capture
export GRAPHDB_HEAP_SIZE="$GRAPHDB_HEAP"
echo "[*] Starting GraphDB in foreground (Ctrl-C / scancel to stop)..."
echo "    URL: http://$(hostname):${GRAPHDB_PORT}"
exec "${GRAPHDB_INSTALL_DIR}/bin/graphdb" \
    -Dgraphdb.connector.port="$GRAPHDB_PORT"
