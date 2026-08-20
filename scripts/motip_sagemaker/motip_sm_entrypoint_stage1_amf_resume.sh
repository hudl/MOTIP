#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-1 (DETR pretrain) — AMF, resuming from checkpoint_9 ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

mkdir -p /tmp/amf_dataset
ln -s "$SM_CHANNEL_TRAIN" /tmp/amf_dataset/AMFDetection

echo "=== Resume checkpoint contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Launching training (4x GPU, resuming from checkpoint_9.pth) ==="
accelerate launch --num_processes=4 train.py \
  --data-root /tmp/amf_dataset \
  --exp-name amf_detr_pretrain_v1 \
  --config-path ./configs/pretrain_r50_deformable_detr_amf_resume.yaml \
  --resume-model "$SM_CHANNEL_PRETRAIN/checkpoint_9.pth" \
  --resume-optimizer true \
  --resume-scheduler true \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 5 \
  --outputs-dir /opt/ml/checkpoints
