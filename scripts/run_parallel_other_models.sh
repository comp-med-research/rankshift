#!/usr/bin/env bash
# Launch paddle/deepseek/dolphin in parallel in tmux session `ocr2`,
# each in its own window. Skips images already present in --out-dir
# (so it's safe to re-run / safe even if the existing `ocr` session
# eventually races on the same out-dir).
#
# Designed to run alongside the existing `ocr` tmux that's running
# glm_ocr.  L40S has ~40 GB free with glm loaded; budget here is
# ~25 GB across the three new models.
#
# Usage:
#   bash ~/projects/rankshift/scripts/run_parallel_other_models.sh
#   tmux attach -t ocr2          # then Ctrl-b n / p to switch windows
#                                # Ctrl-b d to detach
#
# After glm finishes on omnidocbench and you kill the `ocr` session,
# kick off glm on real5 with run_glm_real5.sh (separate script).

set -u

SESSION=ocr2
REPO=$HOME/projects/rankshift
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

# Same image dirs as scripts/h200_launch_inference.sh (some trees use flat paths).
if [ -d "$REPO/data/omnidocbench/images" ]; then
  IMG_OMNI="$REPO/data/omnidocbench/images"
elif [ -d "$REPO/data/omnidocbench/omnidocbench/images" ]; then
  IMG_OMNI="$REPO/data/omnidocbench/omnidocbench/images"
else
  echo "ERROR: OmniDocBench images not found under $REPO/data/omnidocbench"
  exit 1
fi
if [ -d "$REPO/data/real5/Real5-OmniDocBench-Scanning" ]; then
  IMG_REAL5="$REPO/data/real5/Real5-OmniDocBench-Scanning"
elif [ -d "$REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning" ]; then
  IMG_REAL5="$REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning"
else
  echo "ERROR: Real5 scanning images not found under $REPO/data/real5"
  exit 1
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    echo "Or kill it first:  tmux kill-session -t $SESSION"
    exit 1
fi

# Build per-model commands. Each runs omnidoc then real5 inside the
# correct venv/cwd, tee'd to a log file. set -e + pipefail so we don't
# silently swallow errors mid-run.
PADDLE_CMD=$(cat <<EOF
set -eo pipefail
cd $REPO
source $REPO/.venv/bin/activate
echo "=== paddleocr_vl_1_5 / omnidocbench ==="
python $REPO/scripts/run_inference.py \
    --model paddleocr_vl_1_5 \
    --images-dir $IMG_OMNI \
    --out-dir   $REPO/predictions/omnidocbench/paddleocr_vl_1_5 \
    2>&1 | tee -a $LOG_DIR/paddleocr_vl_1_5_omnidocbench.log
echo "=== paddleocr_vl_1_5 / real5 ==="
python $REPO/scripts/run_inference.py \
    --model paddleocr_vl_1_5 \
    --images-dir $IMG_REAL5 \
    --out-dir   $REPO/predictions/real5/paddleocr_vl_1_5 \
    2>&1 | tee -a $LOG_DIR/paddleocr_vl_1_5_real5.log
echo "=== paddle DONE ===  (Ctrl-b d to detach, Ctrl-b & to close window)"
exec bash
EOF
)

DEEPSEEK_CMD=$(cat <<EOF
set -eo pipefail
cd $REPO
source $REPO/.venv-deepseek/bin/activate
echo "=== deepseek_ocr_2 / omnidocbench ==="
python $REPO/scripts/run_inference.py \
    --model deepseek_ocr_2 \
    --images-dir $IMG_OMNI \
    --out-dir   $REPO/predictions/omnidocbench/deepseek_ocr_2 \
    2>&1 | tee -a $LOG_DIR/deepseek_ocr_2_omnidocbench.log
echo "=== deepseek_ocr_2 / real5 ==="
python $REPO/scripts/run_inference.py \
    --model deepseek_ocr_2 \
    --images-dir $IMG_REAL5 \
    --out-dir   $REPO/predictions/real5/deepseek_ocr_2 \
    2>&1 | tee -a $LOG_DIR/deepseek_ocr_2_real5.log
echo "=== deepseek DONE ===  (Ctrl-b d to detach, Ctrl-b & to close window)"
exec bash
EOF
)

DOLPHIN_CMD=$(cat <<EOF
set -eo pipefail
cd $HOME/projects/Dolphin
source $HOME/projects/Dolphin/.venv/bin/activate
echo "=== dolphin_1_5 / omnidocbench ==="
python $REPO/scripts/run_dolphin.py \
    --images-dir $IMG_OMNI \
    --out-dir   $REPO/predictions/omnidocbench/dolphin_1_5 \
    2>&1 | tee -a $LOG_DIR/dolphin_1_5_omnidocbench.log
echo "=== dolphin_1_5 / real5 ==="
python $REPO/scripts/run_dolphin.py \
    --images-dir $IMG_REAL5 \
    --out-dir   $REPO/predictions/real5/dolphin_1_5 \
    2>&1 | tee -a $LOG_DIR/dolphin_1_5_real5.log
echo "=== dolphin DONE ===  (Ctrl-b d to detach, Ctrl-b & to close window)"
exec bash
EOF
)

# Create session detached. Window 0 = paddle, then add deepseek and dolphin.
tmux new-session -d -s "$SESSION" -n paddle   "bash -c $(printf %q "$PADDLE_CMD")"
tmux new-window  -t "$SESSION":1 -n deepseek  "bash -c $(printf %q "$DEEPSEEK_CMD")"
tmux new-window  -t "$SESSION":2 -n dolphin   "bash -c $(printf %q "$DOLPHIN_CMD")"

echo "Started tmux session '$SESSION' with 3 windows: paddle, deepseek, dolphin"
echo
echo "Attach:        tmux attach -t $SESSION"
echo "Switch window: Ctrl-b n  (next)  /  Ctrl-b p  (prev)  /  Ctrl-b 0/1/2"
echo "Detach:        Ctrl-b d"
echo
echo "Tail any single log without attaching:"
echo "  tail -f $LOG_DIR/paddleocr_vl_1_5_omnidocbench.log"
echo "  tail -f $LOG_DIR/deepseek_ocr_2_omnidocbench.log"
echo "  tail -f $LOG_DIR/dolphin_1_5_omnidocbench.log"
