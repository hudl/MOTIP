#!/bin/bash
# Watch for RF-DETR stage-2 resume checkpoints 5-8 and auto-eval each one
set -euo pipefail

S3_PREFIX="s3://hudl-experiments-v1/finlay/motip_rfdetr_stage2_hockey_v2/checkpoints/motip-hockey-rfdetr-stage2-hockey-resume-2026-09-13-11-26-49"
WORK_DIR="/workspaces/sip-tracking-clean"
MOTIP_DIR="$WORK_DIR/third_party/MOTIP"
PYTHON="$WORK_DIR/.venv/bin/python"

run_trackeval() {
    local N=$1
    local EVAL_DIR="$WORK_DIR/rfdetr_stage2_eval_ckpt${N}"
    echo "=== TrackEval for checkpoint_${N} ==="
    $PYTHON "$MOTIP_DIR/TrackEval/scripts/run_mot_challenge.py" \
      --SPLIT_TO_EVAL val \
      --METRICS HOTA CLEAR Identity \
      --GT_FOLDER "$WORK_DIR/data/motip_hockey_data/Hockey/val" \
      --TRACKERS_FOLDER "$EVAL_DIR/evaluate/default/Hockey/val" \
      --TRACKER_SUB_FOLDER tracker \
      --SEQMAP_FILE "$WORK_DIR/data/motip_hockey_data/Hockey/val_seqmap.txt" \
      --SKIP_SPLIT_FOL True \
      --GT_LOC_FORMAT "{gt_folder}/{seq}/gt/gt.txt" \
      --PRINT_RESULTS True 2>&1 | grep -E "COMBINED|HOTA |AssA |DetA |IDF1|MOTA"
}

eval_checkpoint() {
    local N=$1
    local CKPT_PATH="/tmp/rfdetr_resume_checkpoint_${N}.pth"
    local EVAL_DIR="/workspaces/sip-tracking-clean/rfdetr_stage2_eval_ckpt${N}"

    echo "=== Downloading checkpoint_${N} ==="
    aws s3 cp "$S3_PREFIX/checkpoint_${N}.pth" "$CKPT_PATH" --region us-east-1

    echo "=== Running inference for checkpoint_${N} ==="
    cd "$MOTIP_DIR"
    /workspaces/sip-tracking-clean/.venv/bin/accelerate launch --num_processes=1 submit_and_evaluate.py \
      --config-path ./configs/eval_stage2_rfdetr_hockey.yaml \
      --inference-model "$CKPT_PATH" \
      --data-root "$WORK_DIR/data/motip_hockey_data" \
      --outputs-dir "$EVAL_DIR" 2>&1

    # submit_and_evaluate.py crashes at TrackEval step — run directly
    run_trackeval "$N"
    echo "=== checkpoint_${N} DONE ==="
}

for N in 5 6 7 8; do
    echo "Waiting for checkpoint_${N}..."
    while true; do
        if aws s3 ls "$S3_PREFIX/checkpoint_${N}.pth" --region us-east-1 &>/dev/null; then
            echo "checkpoint_${N} found!"
            eval_checkpoint "$N"
            break
        fi
        sleep 300  # poll every 5 min
    done
done

echo "All checkpoints evaluated."
