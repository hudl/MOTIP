#!/bin/bash
set -euo pipefail
echo "=== RF-DETR stage-1 (DETR pretrain) — full Hockey dataset, 4x GPU ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')":${LD_LIBRARY_PATH:-}
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow "transformers>=4.46.0,<5.0.0" timm peft

echo "=== Compiling Deformable Attention extension (needed by MOTIP criterion) ==="
cd models/ops
python setup.py build install
cd ../..

# DINOv2 backbone weights load from HuggingFace — no pretrain channel needed.
# SM_CHANNEL_TRAIN = motip_hockey_data/ (contains Hockey/ as subdirectory).

echo "=== Launching RF-DETR stage-1 training (4x GPU) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name rfdetr_hockey_stage1_real \
  --config-path ./configs/pretrain_rfdetr_hockey_real.yaml \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
