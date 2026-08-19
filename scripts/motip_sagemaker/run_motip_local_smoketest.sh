#!/usr/bin/env bash
# Local smoke test for MOTIP AMF stage-2 training.
# Run from inside the devcontainer:
#   bash third_party/MOTIP/scripts/motip_sagemaker/run_motip_local_smoketest.sh
set -euo pipefail

REPO_ROOT=/workspaces/sip-tracking-clean
MOTIP_DIR=$REPO_ROOT/third_party/MOTIP
DATA_DIR=/tmp/motip_smoketest_data
CKPT_DIR=/tmp/motip_smoketest_ckpt
OUT_DIR=/tmp/motip_smoketest_out

S3_DATA=s3://hudl-experiments-v1/finlay/amfb_motip_stage2/motip_hockey_data
S3_CKPT=s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/motip-hockey-stage1-amf-2026-07-23-10-44-08/checkpoint_14.pth

echo "=== [1/4] Downloading 3 AMF sequences (~240MB) ==="
mkdir -p "$DATA_DIR/AMFTracking/train"
for SEQ in \
  1074021_p1_bcd6fd71-2a56-4e6b-9e4f-cb85a9a08890 \
  1074021_p1_c8b11a4b-1636-4339-8805-6062a1f6223c \
  1074021_p2_759eef92-abc0-4af1-9440-280209557c74
do
  if [ ! -d "$DATA_DIR/AMFTracking/train/$SEQ" ]; then
    aws s3 cp "$S3_DATA/AMFTracking/train/$SEQ/" "$DATA_DIR/AMFTracking/train/$SEQ/" --recursive --quiet
    echo "  downloaded $SEQ"
  else
    echo "  already have $SEQ"
  fi
done

echo "=== [2/4] Downloading stage-1 checkpoint ==="
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/checkpoint_14.pth" ]; then
  aws s3 cp "$S3_CKPT" "$CKPT_DIR/checkpoint_14.pth"
else
  echo "  already have checkpoint_14.pth"
fi

echo "=== [3/4] Setting LD_LIBRARY_PATH for torch ==="
TORCH_LIB=$($REPO_ROOT/.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}

echo "=== [4/4] Running 1-epoch smoke train (1 GPU, large stride) ==="
mkdir -p "$OUT_DIR"
cd "$MOTIP_DIR"
$REPO_ROOT/.venv/bin/accelerate launch --num_processes=1 train.py \
  --data-root "$DATA_DIR" \
  --exp-name motip_amf_smoketest \
  --config-path ./configs/r50_deformable_detr_motip_amf_smoketest.yaml \
  --detr-pretrain "$CKPT_DIR/checkpoint_14.pth" \
  --num-workers 2 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir "$OUT_DIR"

echo ""
echo "=== Smoke test passed — checkpoint at $OUT_DIR/motip_amf_smoketest/ ==="
