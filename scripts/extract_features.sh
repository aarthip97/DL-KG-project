#!/usr/bin/env bash
# =============================================================================
# scripts/extract_features.sh
# -----------------------------------------------------------------------------
# Runs the three DIDONEproject/music_symbolic_features extractors:
#   1. musif     → data/interim/didone_features/<dataset>/musif-mid.csv
#   2. music21   → data/interim/didone_features/<dataset>/music21-mid.csv
#   3. jsymbolic → data/interim/didone_features/<dataset>/jsymbolic-mid.csv
#
# Prerequisites
# -------------
#   bash scripts/setup_didone.sh   (run once beforehand)
#
# Usage
# -----
#   bash scripts/extract_features.sh [--only musif|music21|jsymbolic]
#
# Environment variables (all optional)
# --------------------------------------
#   DATASET_NAME      Name of the dataset dir in vendor/…/datasets/.
#                     Default: lmd_matched
#   ONLY              Comma-separated extractors to run (default: all three).
#   N_TRIALS          Number of extraction trials (default: 1).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/music_symbolic_features"
VENV="$VENDOR_DIR/.venv"
PYTHON="$VENV/bin/python"
LOG_DIR="$REPO_ROOT/logs"

DATASET_NAME="${DATASET_NAME:-lmd_matched}"
N_TRIALS="${N_TRIALS:-1}"
ONLY="${ONLY:-musif,music21,jsymbolic}"

mkdir -p "$LOG_DIR"

# ── parse CLI flags ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) echo "[WARN] Unknown flag: $1"; shift ;;
    esac
done

# ── sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENDOR_DIR/.git" ]]; then
    echo "[ERROR] Vendor repo not found: $VENDOR_DIR"
    echo "        Run:  bash scripts/setup_didone.sh"
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] venv Python not found: $PYTHON"
    echo "        Run:  bash scripts/setup_didone.sh"
    exit 1
fi

DATASETS_DIR="$VENDOR_DIR/datasets"
if [[ ! -d "$DATASETS_DIR/$DATASET_NAME" ]]; then
    echo "[ERROR] Dataset not found: $DATASETS_DIR/$DATASET_NAME"
    echo "        Check that scripts/setup_didone.sh ran successfully and that"
    echo "        your MIDI directory is available."
    exit 1
fi

# ── helper: run one extractor ─────────────────────────────────────────────────
run_extractor() {
    local extractor="$1"
    local log_file="$LOG_DIR/${extractor}-mid.log"

    echo ""
    echo "──────────────────────────────────────────"
    echo " Running extractor: $extractor"
    echo " Log: $log_file"
    echo "──────────────────────────────────────────"

    cd "$VENDOR_DIR"

    # clear musif's own cache to avoid stale data
    if [[ "$extractor" == "musif" && -d "musif_cache" ]]; then
        echo "[${extractor}] Clearing musif_cache …"
        rm -rf musif_cache
    fi

    # Build CLI args:
    #   python -m symbolic_features.features extract <extractor> \
    #       --extension .mid --n_trials_extraction <N>
    "$PYTHON" -m symbolic_features.features extract "$extractor" \
        --extension .mid \
        --n_trials_extraction "$N_TRIALS" \
        2>&1 | tee "$log_file"

    echo "[${extractor}] Done.  Log written to $log_file"
    cd "$REPO_ROOT"
}

# ── main ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " DIDONEproject feature extraction"
echo " Extractors: $ONLY"
echo " Dataset   : $DATASET_NAME"
echo "=========================================="

IFS=',' read -ra EXTRACTORS <<< "$ONLY"
for ext in "${EXTRACTORS[@]}"; do
    ext="$(echo "$ext" | tr -d '[:space:]')"   # trim whitespace
    case "$ext" in
        musif|music21|jsymbolic|musif-harm)
            run_extractor "$ext"
            ;;
        *)
            echo "[WARN] Unknown extractor: $ext  (valid: musif, music21, jsymbolic)"
            ;;
    esac
done

echo ""
echo "=========================================="
echo " All done!"
echo " Features saved under:"
echo "   $REPO_ROOT/data/interim/didone_features/"
echo "=========================================="
