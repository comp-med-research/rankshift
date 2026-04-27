#!/usr/bin/env bash
# Run glm_ocr on real5 only.
# Use this AFTER the original `ocr` tmux finishes glm_ocr/omnidocbench
# (so you don't have two glm processes hammering the GPU).
#
# Usage:
#   tmux new -s glm_real5 'bash ~/projects/rankshift/scripts/run_glm_real5.sh'
#   tmux attach -t glm_real5
#
# Resume safe.

set -eo pipefail

REPO=$HOME/projects/rankshift
LOG=$REPO/logs/glm_ocr_real5.log
mkdir -p "$REPO/logs"

cd "$REPO"
source "$REPO/.venv/bin/activate"

python "$REPO/scripts/run_inference.py" \
    --model glm_ocr \
    --images-dir "$REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning" \
    --out-dir   "$REPO/predictions/real5/glm_ocr" \
    2>&1 | tee -a "$LOG"

echo "=== glm_ocr / real5 DONE ==="
