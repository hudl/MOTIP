#!/bin/bash
set -euo pipefail
echo "=== RF-DETR stage-1 (DETR pretrain) — Hockey dataset ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow "transformers>=4.46.0,<5.0.0" timm peft

echo "=== Compiling Deformable Attention extension (needed by MOTIP criterion) ==="
cd models/ops
python setup.py build install
cd ../..

# DINOv2 backbone weights load automatically from HuggingFace at model-build time.
# No pretrain channel needed — positional_encoding_size=37 matches DINOv2's 518px training.

mkdir -p /tmp/motip_hockey_data
ln -s "$SM_CHANNEL_TRAIN" /tmp/motip_hockey_data/Hockey

echo "=== Launching RF-DETR stage-1 training (1 GPU) ==="
accelerate launch --num_processes=1 train.py \
  --data-root /tmp/motip_hockey_data \
  --exp-name rfdetr_hockey_stage1 \
  --config-path ./configs/pretrain_rfdetr_hockey_smoketest.yaml \
  --num-workers 8 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 5 \
  --outputs-dir /opt/ml/checkpoints
