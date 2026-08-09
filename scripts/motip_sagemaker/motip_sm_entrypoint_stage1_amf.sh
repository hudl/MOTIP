#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-1 (DETR pretrain) — AMF detection dataset ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"

# g5.12xlarge EFA fork-safety (same as hockey stage-1 real).
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

# MOTIP expects data_root/AMFDetection/train/images but S3 channel is
# already at .../amfb_detection/train/images — create a symlink so no
# data copying is needed.
mkdir -p /tmp/amf_dataset
ln -s "$SM_CHANNEL_TRAIN" /tmp/amf_dataset/AMFDetection

echo "=== Launching training (4x GPU) ==="
accelerate launch --num_processes=4 train.py \
  --data-root /tmp/amf_dataset \
  --exp-name amf_detr_pretrain_v1 \
  --config-path ./configs/pretrain_r50_deformable_detr_amf.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/r50_deformable_detr_coco_sportsmot.pth" \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 5 \
  --outputs-dir /opt/ml/checkpoints
