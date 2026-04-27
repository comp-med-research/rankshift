#!/usr/bin/env bash
# Extract DiT+DINO features on every Real5 partition.
# Resume-safe: skips partitions whose .npy already exists.
set -euo pipefail

ROOT=/home/halimatmac/projects/rankshift
ROOT_DATA="$ROOT/data/real5/real5_omnidocbench"
PY="$ROOT/.venv/bin/python"
BATCH="${BATCH:-8}"

partitions=(
    "Scanning"
    "Warping"
    "Illumination"
    "Skew"
    "Screen-Photography"
)

cd "$ROOT"

for p in "${partitions[@]}"; do
    images_dir="$ROOT_DATA/Real5-OmniDocBench-$p"
    safe=$(echo "$p" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    out="$ROOT/features/real5_${safe}_features.npy"

    if [[ ! -d "$images_dir" ]]; then
        echo "[skip] $p — directory missing: $images_dir"
        continue
    fi
    if [[ -f "$out" ]]; then
        echo "[skip] $p — features already exist: $out"
        continue
    fi

    echo "===================================================================="
    echo "[$(date +%H:%M:%S)] Extracting features for partition: $p"
    echo "  images: $images_dir"
    echo "  output: $out"
    echo "===================================================================="
    "$PY" scripts/extract_features.py "$images_dir" "$out" --backbone both --batch-size "$BATCH"
done

echo "All partitions processed."
