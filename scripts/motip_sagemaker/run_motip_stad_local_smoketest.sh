#!/usr/bin/env bash
# Local smoke test for MOTIP AMF STAD stage-2 training.
# Run from inside the devcontainer:
#   bash third_party/MOTIP/scripts/motip_sagemaker/run_motip_stad_local_smoketest.sh
set -euo pipefail

REPO_ROOT=/workspaces/sip-tracking-clean
MOTIP_DIR=$REPO_ROOT/third_party/MOTIP
DATA_DIR=/tmp/motip_stad_smoketest_data
CKPT_DIR=/tmp/motip_stad_smoketest_ckpt
OUT_DIR=/tmp/motip_stad_smoketest_out

S3_DATA=s3://hudl-experiments/touchdown/datasets/tracking_stad_v2/train
S3_CKPT=s3://hudl-experiments-v1/finlay/motip_amf_stage1_pretrain_v1/checkpoints/motip-hockey-stage1-amf-2026-08-16-10-09-44/checkpoint_19.pth

echo "=== [1/4] Downloading 3 STAD sequences ==="
mkdir -p "$DATA_DIR/train"

# Pick the first 3 clips — capture full listing before slicing to avoid broken pipe
# (head -3 in a pipeline causes SIGPIPE with set -euo pipefail).
mapfile -t SEQS < <(aws s3 ls "$S3_DATA/" | awk '{print $2}' | tr -d '/')
SEQS=("${SEQS[@]:0:3}")
if [ ${#SEQS[@]} -eq 0 ]; then
  echo "ERROR: no sequences found at $S3_DATA" >&2
  exit 1
fi

for SEQ in "${SEQS[@]}"; do
  if [ ! -d "$DATA_DIR/train/$SEQ" ]; then
    aws s3 cp "$S3_DATA/$SEQ/" "$DATA_DIR/train/$SEQ/" --recursive --quiet
    echo "  downloaded $SEQ"
  else
    echo "  already have $SEQ"
  fi
done

echo "=== [2/4] Downloading stage-1 checkpoint (checkpoint_19) ==="
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/checkpoint_19.pth" ]; then
  aws s3 cp "$S3_CKPT" "$CKPT_DIR/checkpoint_19.pth"
else
  echo "  already have checkpoint_19.pth"
fi

echo "=== [3/4] Setting LD_LIBRARY_PATH for torch ==="
TORCH_LIB=$($REPO_ROOT/.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}

echo "=== [4/4] Running 1-epoch smoke train (1 GPU, stride 50) ==="
mkdir -p "$OUT_DIR"
cd "$MOTIP_DIR"
$REPO_ROOT/.venv/bin/accelerate launch --num_processes=1 train.py \
  --data-root "$DATA_DIR" \
  --exp-name motip_amf_stad_smoketest \
  --config-path ./configs/r50_deformable_detr_motip_amf_stad_smoketest.yaml \
  --detr-pretrain "$CKPT_DIR/checkpoint_19.pth" \
  --num-workers 2 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir "$OUT_DIR"

echo ""
echo "=== Smoke test passed — checkpoint at $OUT_DIR/motip_amf_stad_smoketest/ ==="
