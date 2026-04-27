#!/usr/bin/env bash
# Run glm_ocr on omnidocbench AND real5 in parallel in tmux session `ocr`,
# each in its own window.  This is the relaunch after we proved vLLM
# produces inferior OCR for this model (drops detail + loops); we use
# the transformers path but parallelise across datasets to halve wall time.
#
# Memory: each glm_ocr instance peaks at ~6 GB on L40S.
# Coexists with the `ocr2` session running paddle/deepseek/dolphin
# (~26 GB).  Total ~38 GB out of 46 GB — fits with margin.
#
# Usage:
#   bash ~/projects/rankshift/scripts/run_glm_parallel.sh
#   tmux attach -t ocr     # then Ctrl-b 0/1 to switch, Ctrl-b d to detach
#
# Resume safe.

set -u

SESSION=ocr
REPO=$HOME/projects/rankshift
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    echo "Or kill it first:  tmux kill-session -t $SESSION"
    exit 1
fi

OMNI_CMD=$(cat <<EOF
set -eo pipefail
cd $REPO
source $REPO/.venv/bin/activate
echo "=== glm_ocr / omnidocbench ==="
python $REPO/scripts/run_inference.py \
    --model glm_ocr \
    --images-dir $REPO/data/omnidocbench/omnidocbench/images \
    --out-dir   $REPO/predictions/omnidocbench/glm_ocr \
    2>&1 | tee -a $LOG_DIR/glm_ocr_omnidocbench.log
echo "=== glm_ocr / omnidocbench DONE ==="
exec bash
EOF
)

REAL5_CMD=$(cat <<EOF
set -eo pipefail
cd $REPO
source $REPO/.venv/bin/activate
echo "=== glm_ocr / real5 ==="
python $REPO/scripts/run_inference.py \
    --model glm_ocr \
    --images-dir $REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning \
    --out-dir   $REPO/predictions/real5/glm_ocr \
    2>&1 | tee -a $LOG_DIR/glm_ocr_real5.log
echo "=== glm_ocr / real5 DONE ==="
exec bash
EOF
)

tmux new-session -d -s "$SESSION" -n omni  "bash -c $(printf %q "$OMNI_CMD")"
tmux new-window  -t "$SESSION":1 -n real5  "bash -c $(printf %q "$REAL5_CMD")"

echo "Started tmux session '$SESSION' with 2 windows: omni, real5"
echo "Attach: tmux attach -t $SESSION"
echo "Logs:"
echo "  tail -f $LOG_DIR/glm_ocr_omnidocbench.log"
echo "  tail -f $LOG_DIR/glm_ocr_real5.log"
