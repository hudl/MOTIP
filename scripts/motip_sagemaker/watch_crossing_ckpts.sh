#!/bin/bash
# Watch for RF-DETR crossing finetune checkpoints 1-13 and auto-eval each one
# Evaluates on the standard 9-clip hockey val set (same as hockey model evals)
set -euo pipefail

S3_PREFIX="s3://hudl-experiments-v1/finlay/motip_rfdetr_crossing_finetune_v1/checkpoints/motip-hockey-rfdetr-crossing-finetune-2026-09-14-13-03-48"
HOST_WORK="/home/ubuntu/sip-tracking-clean"
HOST_CKPT_DIR="$HOST_WORK/tmp_ckpts"
C_WORK="/workspaces/sip-tracking-clean"
C_MOTIP="$C_WORK/third_party/MOTIP"
C_DATA_ROOT="$C_WORK/data/motip_hockey_data"
C_CKPT_DIR="$C_WORK/tmp_ckpts"
CONTAINER="amazing_spence"

mkdir -p "$HOST_CKPT_DIR"

run_trackeval() {
    local N=$1
    local C_EVAL_DIR="$C_WORK/rfdetr_crossing_eval_ckpt${N}"
    echo "=== TrackEval for checkpoint_${N} ==="
    docker exec "$CONTAINER" bash -c "
        $C_WORK/.venv/bin/python $C_MOTIP/TrackEval/scripts/run_mot_challenge.py \
          --SPLIT_TO_EVAL val \
          --METRICS HOTA CLEAR Identity \
          --GT_FOLDER $C_DATA_ROOT/Hockey/val \
          --TRACKERS_FOLDER $C_EVAL_DIR/submit/default/Hockey/val \
          --TRACKER_SUB_FOLDER tracker \
          --SEQMAP_FILE $C_DATA_ROOT/Hockey/val_seqmap.txt \
          --SKIP_SPLIT_FOL True \
          --GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt.txt' \
          --PRINT_RESULTS True 2>&1 | grep -E 'COMBINED|HOTA:|AssA|DetA|IDF1|MOTA'
    "
}

eval_checkpoint() {
    local N=$1
    local HOST_CKPT="$HOST_CKPT_DIR/checkpoint_${N}.pth"
    local C_CKPT="$C_CKPT_DIR/checkpoint_${N}.pth"
    local C_EVAL_DIR="$C_WORK/rfdetr_crossing_eval_ckpt${N}"

    echo "=== Downloading checkpoint_${N} ==="
    aws s3 cp "$S3_PREFIX/checkpoint_${N}.pth" "$HOST_CKPT" --region us-east-1

    echo "=== Running inference for checkpoint_${N} ==="
    docker exec "$CONTAINER" bash -c "
        cd $C_MOTIP && \
        $C_WORK/.venv/bin/accelerate launch --num_processes=1 submit_and_evaluate.py \
          --config-path ./configs/finetune_crossing_rfdetr_v1.yaml \
          --inference-model $C_CKPT \
          --data-root $C_DATA_ROOT \
          --outputs-dir $C_EVAL_DIR
    " 2>&1 | tail -30

    run_trackeval "$N"
    echo "=== checkpoint_${N} DONE ==="
}

for N in $(seq 1 13); do
    echo "Waiting for checkpoint_${N}..."
    while true; do
        if aws s3 ls "$S3_PREFIX/checkpoint_${N}.pth" --region us-east-1 &>/dev/null; then
            echo "checkpoint_${N} found!"
            eval_checkpoint "$N"
            break
        fi
        sleep 300
    done
done

echo "All checkpoints evaluated."
