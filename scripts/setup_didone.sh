#!/usr/bin/env bash
# =============================================================================
# scripts/setup_didone.sh
# -----------------------------------------------------------------------------
# One-time setup for the DIDONEproject/music_symbolic_features pipeline.
#
# What it does
# ------------
#   1. Clones the repo into vendor/music_symbolic_features (if not present).
#   2. Creates a Python 3.10 venv inside it and installs all dependencies
#      via pip (we avoid pdm to keep things simple; all deps are declared in
#      pyproject.toml and can be installed with pip ≥ 21).
#   3. Symlinks our MIDI directory into vendor/music_symbolic_features/datasets/
#   4. Patches settings.py so OUTPUT points at data/interim/didone_features/
#   5. Downloads jSymbolic 2.2 jar (optional – only needed for jsymbolic extractor).
#
# Usage
# -----
#   bash scripts/setup_didone.sh [--midi-dir <path>] [--skip-jsymbolic]
#
# Environment variables (all optional)
# -------------------------------------
#   MIDI_DIR          Path to the MIDI directory to register as a dataset.
#                     Default: data/raw/lmd_matched
#   DATASET_NAME      Name of the symlink inside datasets/.
#                     Default: lmd_matched
#   SKIP_JSYMBOLIC    Set to 1 to skip downloading jSymbolic.
#   PYTHON310         Path to python3.10 binary.
#                     Default: auto-detected via 'python3.10' or 'python3'
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/music_symbolic_features"
DIDONE_REPO="https://github.com/DIDONEproject/music_symbolic_features.git"

# ── defaults ──────────────────────────────────────────────────────────────────
MIDI_DIR="${MIDI_DIR:-$REPO_ROOT/data/raw/lmd_matched}"
DATASET_NAME="${DATASET_NAME:-lmd_matched}"
SKIP_JSYMBOLIC="${SKIP_JSYMBOLIC:-0}"
OUTPUT_DIR="$REPO_ROOT/data/interim/didone_features"
LOG_DIR="$REPO_ROOT/logs"

# ── parse CLI flags ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --midi-dir)   MIDI_DIR="$2"; shift 2 ;;
        --skip-jsymbolic) SKIP_JSYMBOLIC=1; shift ;;
        *) echo "[WARN] Unknown flag: $1"; shift ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " DIDONEproject / music_symbolic_features"
echo " Setup script"
echo "=========================================="
echo "  Vendor dir : $VENDOR_DIR"
echo "  MIDI dir   : $MIDI_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo ""

# ── Step 1: Clone ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENDOR_DIR/.git" ]]; then
    echo "[1/5] Cloning music_symbolic_features …"
    git clone --depth 1 "$DIDONE_REPO" "$VENDOR_DIR"
else
    echo "[1/5] Repo already cloned — skipping."
fi

# ── Step 2: Python 3.10 venv ──────────────────────────────────────────────────
VENV="$VENDOR_DIR/.venv"

# auto-detect python3.10
if [[ -z "${PYTHON310:-}" ]]; then
    if command -v python3.10 &>/dev/null; then
        PYTHON310="$(command -v python3.10)"
    elif python3 --version 2>&1 | grep -q "3\.10"; then
        PYTHON310="$(command -v python3)"
    else
        echo "[ERROR] python3.10 not found. Install it or set PYTHON310= env var."
        exit 1
    fi
fi

echo "[2/5] Using Python: $PYTHON310  ($(${PYTHON310} --version))"

if [[ ! -d "$VENV" ]]; then
    echo "[2/5] Creating venv in $VENV …"
    "$PYTHON310" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

echo "[2/5] Installing dependencies (this may take several minutes) …"
"$PIP" install --quiet --upgrade pip

# Install the package in editable mode – this picks up all [project.dependencies]
cd "$VENDOR_DIR"
"$PIP" install --quiet -e ".[musif,music21,jsymbolic]" 2>/dev/null \
    || "$PIP" install --quiet -e "." \
    || true   # best-effort; user may need to fix extras manually

# Ensure the main extras are present regardless of extras syntax
"$PIP" install --quiet musif music21 2>/dev/null || true

echo "[2/5] Venv ready."
cd "$REPO_ROOT"

# ── Step 3: Datasets symlink ───────────────────────────────────────────────────
DATASETS_DIR="$VENDOR_DIR/datasets"
mkdir -p "$DATASETS_DIR"

LINK="$DATASETS_DIR/$DATASET_NAME"
if [[ -L "$LINK" ]]; then
    echo "[3/5] Symlink already exists: $LINK → $(readlink "$LINK")"
elif [[ -d "$MIDI_DIR" ]]; then
    ln -s "$(realpath "$MIDI_DIR")" "$LINK"
    echo "[3/5] Created symlink: $LINK → $MIDI_DIR"
else
    echo "[3/5] WARN: MIDI directory does not exist yet: $MIDI_DIR"
    echo "           Run scripts/download_lmd.py first, then re-run this script."
    # create a placeholder so settings.py does not fail on empty datasets dir
fi

# ── Step 4: Patch settings.py ─────────────────────────────────────────────────
SETTINGS="$VENDOR_DIR/symbolic_features/settings.py"
OUTPUT_DIR_ESCAPED="${OUTPUT_DIR//\//\\/}"   # escape slashes for sed

mkdir -p "$OUTPUT_DIR"

echo "[4/5] Patching $SETTINGS …"
# Replace OUTPUT line only if it still points to the default "features/"
if grep -q 'OUTPUT = "features/"' "$SETTINGS"; then
    sed -i "s|OUTPUT = \"features/\"|OUTPUT = \"$OUTPUT_DIR/\"|g" "$SETTINGS"
    echo "[4/5] OUTPUT patched to $OUTPUT_DIR/"
else
    echo "[4/5] OUTPUT already patched or custom — leaving as-is."
fi

# ── Step 5: Download jSymbolic 2.2 jar ────────────────────────────────────────
JSYMBOLIC_DIR="$VENDOR_DIR/tools/jSymbolic_2_2_user"
JSYMBOLIC_JAR="$JSYMBOLIC_DIR/jSymbolic2.jar"

if [[ "$SKIP_JSYMBOLIC" == "1" ]]; then
    echo "[5/5] Skipping jSymbolic download (--skip-jsymbolic)."
elif [[ -f "$JSYMBOLIC_JAR" ]]; then
    echo "[5/5] jSymbolic jar already present: $JSYMBOLIC_JAR"
else
    echo "[5/5] Downloading jSymbolic 2.2 …"
    mkdir -p "$JSYMBOLIC_DIR"
    JSYMBOLIC_URL="https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/jSymbolic_2_2_user.zip/download"
    TMP_ZIP="$(mktemp --suffix=.zip)"
    if curl -L --silent --show-error -o "$TMP_ZIP" "$JSYMBOLIC_URL"; then
        unzip -q -o "$TMP_ZIP" -d "$VENDOR_DIR/tools/"
        rm -f "$TMP_ZIP"
        echo "[5/5] jSymbolic extracted to $JSYMBOLIC_DIR"
    else
        echo "[5/5] WARN: Failed to download jSymbolic. Download manually from:"
        echo "       https://sourceforge.net/projects/jmir/files/jSymbolic/"
        echo "       and place jSymbolic2.jar at: $JSYMBOLIC_JAR"
        rm -f "$TMP_ZIP"
    fi
fi

# ── also patch JSYMBOLIC_JAR path in settings.py ──────────────────────────────
if [[ -f "$JSYMBOLIC_JAR" ]] && grep -q 'JSYMBOLIC_JAR' "$SETTINGS"; then
    JSYM_ESCAPED="${JSYMBOLIC_JAR//\//\\/}"
    sed -i "s|JSYMBOLIC_JAR = .*|JSYMBOLIC_JAR = \"$JSYMBOLIC_JAR\"|g" "$SETTINGS"
    echo "[5/5] Patched JSYMBOLIC_JAR path in settings.py"
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Next step:"
echo "   bash scripts/extract_features.sh"
echo "=========================================="
