#!/bin/bash
set -euo pipefail
echo "=== MOTIP crossing fine-tune stride-1 — from checkpoint 12, REL_PE_LENGTH=60 ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"

export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== Data channel contents ==="
ls "$SM_CHANNEL_TRAIN" | head -20
echo "... (total sequences:)"
ls "$SM_CHANNEL_TRAIN/Hockey/train" | wc -l

echo "=== Pretrain channel contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Verifying checkpoint ==="
python -c "
import torch
ckpt = torch.load('$SM_CHANNEL_PRETRAIN/stage2_g7e_checkpoint_12.pth', map_location='cpu', weights_only=False)
keys = list(ckpt['model'].keys())
print(f'Checkpoint has {len(keys)} keys')
has_detr_prefix = any(k.startswith('detr.') for k in keys)
print(f'Keys have detr. prefix: {has_detr_prefix}')
"

echo "=== Launching training (2x GPU, stride-1, 21 epochs from ckpt12) ==="
accelerate launch --num_processes=2 train.py   --data-root "$SM_CHANNEL_TRAIN"   --exp-name motip_crossing_finetune_s1   --config-path ./configs/finetune_crossing_s1.yaml   --resume-model "$SM_CHANNEL_PRETRAIN/stage2_g7e_checkpoint_12.pth"   --resume-optimizer false   --resume-scheduler false   --num-workers 12   --prefetch-factor 6   --save-checkpoint-per-epoch 1   --outputs-dir /opt/ml/checkpoints
