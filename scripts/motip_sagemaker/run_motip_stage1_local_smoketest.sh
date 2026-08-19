#!/usr/bin/env bash
# Local smoke test for MOTIP AMF stage-1 (DETR detection pretrain).
# Run from inside the devcontainer:
#   bash third_party/MOTIP/scripts/motip_sagemaker/run_motip_stage1_local_smoketest.sh
set -euo pipefail

REPO_ROOT=/workspaces/sip-tracking-clean
MOTIP_DIR=$REPO_ROOT/third_party/MOTIP
DATA_DIR=/tmp/motip_stage1_smoketest_data
CKPT_DIR=/tmp/motip_smoketest_ckpt
OUT_DIR=/tmp/motip_stage1_smoketest_out

S3_DATA=s3://hudl-experiments-v1/finlay/amfb_detection
S3_INIT_CKPT=s3://hudl-experiments-v1/finlay/motip_hockey_smoketest/pretrain/r50_deformable_detr_coco_sportsmot.pth

echo "=== [1/4] Downloading 50 AMF detection frames (~20MB) ==="
mkdir -p "$DATA_DIR/AMFDetection/train/images"
mkdir -p "$DATA_DIR/AMFDetection/train/gts"

$REPO_ROOT/.venv/bin/python - <<'PYEOF'
import boto3, os, sys

data_dir = "/tmp/motip_stage1_smoketest_data"
bucket   = "hudl-experiments-v1"
img_pfx  = "finlay/amfb_detection/train/images/"
gt_pfx   = "finlay/amfb_detection/train/gts/"

s3   = boto3.client("s3")
resp = s3.list_objects_v2(Bucket=bucket, Prefix=img_pfx, MaxKeys=50)
for obj in resp.get("Contents", []):
    key   = obj["Key"]
    fname = os.path.basename(key)
    name  = fname[:-4]
    local_img = f"{data_dir}/AMFDetection/train/images/{fname}"
    local_gt  = f"{data_dir}/AMFDetection/train/gts/{name}.txt"
    if not os.path.exists(local_img):
        s3.download_file(bucket, key, local_img)
    if not os.path.exists(local_gt):
        try:
            s3.download_file(bucket, gt_pfx + name + ".txt", local_gt)
        except Exception:
            pass
n = len(os.listdir(f"{data_dir}/AMFDetection/train/images"))
print(f"  downloaded {n} frames")
PYEOF

echo "=== [2/4] Downloading COCO/SportsMOT init checkpoint (~488MB) ==="
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/r50_deformable_detr_coco_sportsmot.pth" ]; then
  aws s3 cp "$S3_INIT_CKPT" "$CKPT_DIR/r50_deformable_detr_coco_sportsmot.pth"
else
  echo "  already have r50_deformable_detr_coco_sportsmot.pth"
fi

echo "=== [3/4] Setting LD_LIBRARY_PATH for torch ==="
TORCH_LIB=$($REPO_ROOT/.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}

echo "=== [4/4] Running 1-epoch DETR pretrain smoke test (1 GPU) ==="
mkdir -p "$OUT_DIR"
cd "$MOTIP_DIR"
$REPO_ROOT/.venv/bin/accelerate launch --num_processes=1 train.py \
  --data-root "$DATA_DIR" \
  --exp-name motip_amf_stage1_smoketest \
  --config-path ./configs/pretrain_r50_deformable_detr_amf_smoketest.yaml \
  --detr-pretrain "$CKPT_DIR/r50_deformable_detr_coco_sportsmot.pth" \
  --num-workers 2 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir "$OUT_DIR"

echo ""
echo "=== Stage-1 smoke test passed — checkpoint at $OUT_DIR/motip_amf_stage1_smoketest/ ==="
