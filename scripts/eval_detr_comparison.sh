#!/bin/bash
# Compare D-DETR vs RF-DETR detection quality on the hockey val set.
# Uses MOTIP's existing submit_and_evaluate in inference_only_detr mode.
#
# Usage:
#   bash scripts/eval_detr_comparison.sh \
#       --ddetr  /path/to/ddetr_stage1_checkpoint.pth \
#       --rfdetr /path/to/rfdetr_stage1_checkpoint.pth \
#       [--data-root /path/to/data] \
#       [--out-dir /tmp/detr_eval]
#
# Run from MOTIP root.

set -euo pipefail

DDETR_CKPT=""
RFDETR_CKPT=""
DATA_ROOT="${SM_CHANNEL_TRAIN:-/workspaces/sip-tracking-clean/data/motip_hockey_data}"
OUT_DIR="/tmp/detr_eval_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ddetr)   DDETR_CKPT="$2";  shift 2 ;;
        --rfdetr)  RFDETR_CKPT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$DDETR_CKPT" || -z "$RFDETR_CKPT" ]] && { echo "Usage: $0 --ddetr <ckpt> --rfdetr <ckpt>"; exit 1; }

run_eval() {
    local label=$1
    local config=$2
    local ckpt=$3
    local outdir="$OUT_DIR/$label"
    mkdir -p "$outdir"

    echo ""
    echo "=== $label ==="
    accelerate launch --num_processes=1 submit_and_evaluate.py \
        --config-path "$config" \
        --inference-model "$ckpt" \
        --data-root "$DATA_ROOT" \
        --outputs-dir "$outdir" \
        --inference-only-detr True \
        --det-thresh 0.5

    # Extract DetA / HOTA from the summary printed to stdout
    grep -E "DetA|HOTA|IDF1|Recall|Precision" "$outdir"/evaluate/*/*/pedestrian_summary.txt 2>/dev/null || \
    grep -E "DetA|HOTA|IDF1" "$outdir"/**/*.txt 2>/dev/null || \
    echo "(metrics in $outdir)"
}

run_eval "ddetr"  ./configs/eval_stage1_hockey.yaml       "$DDETR_CKPT"
run_eval "rfdetr" ./configs/eval_stage1_rfdetr_hockey.yaml "$RFDETR_CKPT"

echo ""
echo "=== SUMMARY ==="
echo "D-DETR:  $DDETR_CKPT"
echo "RF-DETR: $RFDETR_CKPT"
echo "Results in $OUT_DIR"
