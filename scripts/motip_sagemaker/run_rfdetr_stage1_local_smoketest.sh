#!/usr/bin/env bash
# Local smoke test for RF-DETR stage-1 (DETR detection pretrain) in MOTIP.
# Run from inside the devcontainer:
#   bash third_party/MOTIP/scripts/motip_sagemaker/run_rfdetr_stage1_local_smoketest.sh
set -euo pipefail

REPO_ROOT=/workspaces/sip-tracking-clean
MOTIP_DIR=$REPO_ROOT/third_party/MOTIP
DATA_DIR=/tmp/motip_rfdetr_stage1_data
OUT_DIR=/tmp/motip_rfdetr_stage1_out

S3_DATA=s3://hudl-experiments-v1/finlay/motip_hockey_smoketest/data/motip_hockey_data

echo "=== [1/4] Installing transformers<5 (required by DINOv2 backbone) ==="
uv pip install --quiet "transformers>=4.46.0,<5.0.0"

echo "=== [2/4] Syncing hockey data from S3 ==="
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/Hockey" ]; then
  aws s3 sync "$S3_DATA/Hockey" "$DATA_DIR/Hockey" --quiet
  aws s3 sync "$S3_DATA/SportsMOT" "$DATA_DIR/SportsMOT" --quiet
  echo "  data synced"
else
  echo "  already present, skipping"
fi

echo "=== [3/4] Setting LD_LIBRARY_PATH for torch ==="
TORCH_LIB=$($REPO_ROOT/.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}

echo "=== [4/4] Running RF-DETR stage-1 smoke test (1 GPU, 2 epochs) ==="
mkdir -p "$OUT_DIR"
cd "$MOTIP_DIR"
$REPO_ROOT/.venv/bin/accelerate launch --num_processes=1 train.py \
  --data-root "$DATA_DIR" \
  --exp-name rfdetr_hockey_stage1_smoketest \
  --config-path ./configs/pretrain_rfdetr_hockey_smoketest.yaml \
  --epochs 2 \
  --num-workers 2 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir "$OUT_DIR"

echo ""
echo "=== Stage-1 smoke test done ==="
