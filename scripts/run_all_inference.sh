#!/usr/bin/env bash
# Run all 4 OCR models on OmniDocBench + Real5, sequentially.
# Each model uses the venv it was set up in:
#   glm_ocr, paddleocr_vl_1_5  → ~/projects/rankshift/.venv
#   deepseek_ocr_2             → ~/projects/rankshift/.venv-deepseek
#   dolphin_1_5                → ~/projects/Dolphin/.venv  (cwd must be ~/projects/Dolphin)
#
# Usage:
#   tmux new -s ocr 'bash ~/projects/rankshift/scripts/run_all_inference.sh'
#   tmux attach -t ocr     # Ctrl-b d to detach
#
# Resume safe: re-running skips images whose .md already exists in --out-dir.

set -u
set -o pipefail

REPO=~/projects/rankshift
OMNI_DIR=$REPO/data/omnidocbench/omnidocbench/images
REAL5_DIR=$REPO/data/real5/real5_omnidocbench/Real5-OmniDocBench-Scanning
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

run_model() {
    local model=$1 venv_activate=$2 cwd=$3 script=$4
    for dataset in omnidocbench real5; do
        local img_dir
        if [[ $dataset == omnidocbench ]]; then img_dir=$OMNI_DIR; else img_dir=$REAL5_DIR; fi
        local out_dir=$REPO/predictions/$dataset/$model
        local log=$LOG_DIR/${model}_${dataset}.log

        echo "==============================================================="
        echo "[$(date '+%F %T')] $model on $dataset"
        echo "  out_dir: $out_dir"
        echo "  log:     $log"
        echo "==============================================================="

        (
            set -e
            cd "$cwd"
            # shellcheck disable=SC1090
            source "$venv_activate"
            python "$script" \
                --model "$model" \
                --images-dir "$img_dir" \
                --out-dir "$out_dir"
        ) 2>&1 | tee -a "$log"
    done
}

run_dolphin() {
    # Dolphin uses run_dolphin.py (no --model arg; one model only).
    for dataset in omnidocbench real5; do
        local img_dir
        if [[ $dataset == omnidocbench ]]; then img_dir=$OMNI_DIR; else img_dir=$REAL5_DIR; fi
        local out_dir=$REPO/predictions/$dataset/dolphin_1_5
        local log=$LOG_DIR/dolphin_1_5_${dataset}.log

        echo "==============================================================="
        echo "[$(date '+%F %T')] dolphin_1_5 on $dataset"
        echo "  out_dir: $out_dir"
        echo "  log:     $log"
        echo "==============================================================="

        (
            set -e
            cd ~/projects/Dolphin
            # shellcheck disable=SC1091
            source ~/projects/Dolphin/.venv/bin/activate
            python "$REPO/scripts/run_dolphin.py" \
                --images-dir "$img_dir" \
                --out-dir "$out_dir"
        ) 2>&1 | tee -a "$log"
    done
}

START=$(date +%s)

# Light models first (fast feedback loop), heaviest last
run_model glm_ocr          "$REPO/.venv/bin/activate"          "$REPO" "$REPO/scripts/run_inference.py"
run_model paddleocr_vl_1_5 "$REPO/.venv/bin/activate"          "$REPO" "$REPO/scripts/run_inference.py"
run_dolphin
run_model deepseek_ocr_2   "$REPO/.venv-deepseek/bin/activate" "$REPO" "$REPO/scripts/run_inference.py"

END=$(date +%s)
echo
echo "=== ALL OCR RUNS COMPLETE ==="
printf 'Total wall time: %dh %dm %ds\n' \
    $(( (END-START)/3600 )) $(( ((END-START)%3600)/60 )) $(( (END-START)%60 ))
echo "Predictions in: $REPO/predictions/{omnidocbench,real5}/<model>/"
