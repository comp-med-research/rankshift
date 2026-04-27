#!/usr/bin/env bash
# Extract DiT + DINOv2 CLS features on OmniDocBench + Real5.
# Runs from the rankshift venv. extract_features.py loads one backbone at a
# time (DiT then DINOv2, freeing VRAM between) → ~3 GB peak, safe to run
# concurrently with run_all_inference.sh on the same GPU.
#
# Usage:
#   tmux new -s features 'bash ~/projects/rankshift/scripts/run_all_features.sh'
#   tmux attach -t features    # Ctrl-b d to detach

set -u
set -o pipefail

REPO=~/projects/rankshift
OMNI_DIR=$REPO/data/omnidocbench/omnidocbench/images
REAL5_DIR=$REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR" "$REPO/features"

cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

START=$(date +%s)

for spec in "omnidocbench:$OMNI_DIR:$REPO/features/omnidocbench_features.npy" \
            "real5:$REAL5_DIR:$REPO/features/real5_features.npy"; do
    IFS=: read -r name img_dir out_npy <<< "$spec"
    log=$LOG_DIR/features_${name}.log
    echo "==============================================================="
    echo "[$(date '+%F %T')] features on $name"
    echo "  img_dir: $img_dir"
    echo "  out_npy: $out_npy"
    echo "  log:     $log"
    echo "==============================================================="
    python "$REPO/scripts/extract_features.py" "$img_dir" "$out_npy" 2>&1 | tee -a "$log"
done

END=$(date +%s)
echo
echo "=== FEATURE EXTRACTION COMPLETE ==="
printf 'Total wall time: %dh %dm %ds\n' \
    $(( (END-START)/3600 )) $(( ((END-START)%3600)/60 )) $(( (END-START)%60 ))
echo "Features in: $REPO/features/{omnidocbench,real5}_features.npy"
