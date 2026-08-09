#!/bin/bash
set -euo pipefail
echo "=== MOTIP crossing fine-tune v2 — no trajectory augmentation, verifying checkpoint load ==="

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
ls "$SM_CHANNEL_TRAIN"

echo "=== Pretrain channel contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Verifying checkpoint load ==="
python -c "
import torch
ckpt = torch.load('$SM_CHANNEL_PRETRAIN/stage2_g7e_checkpoint_12.pth', map_location='cpu', weights_only=False)
keys = list(ckpt['model'].keys())
print(f'Checkpoint has {len(keys)} keys')
print(f'First 5: {keys[:5]}')
has_detr_prefix = any(k.startswith('detr.') for k in keys)
has_bare_bbox = 'bbox_embed.0.layers.0.weight' in ckpt['model']
print(f'Keys have detr. prefix: {has_detr_prefix}')
print(f'Has bare bbox_embed (would trigger wrong load path): {has_bare_bbox}')
if has_bare_bbox:
    print('ERROR: checkpoint will be loaded via load_detr_pretrain path - ID head will be lost!')
    exit(1)
print('Checkpoint structure OK - full MOTIP model with detr. prefix')
"

echo "=== Launching training (2x GPU, epochs 13-21, crossing fine-tune v2 — no traj aug) ==="
accelerate launch --num_processes=2 train.py   --data-root "$SM_CHANNEL_TRAIN"   --exp-name motip_crossing_finetune_v2   --config-path ./configs/finetune_crossing_v1.yaml   --resume-model "$SM_CHANNEL_PRETRAIN/stage2_g7e_checkpoint_12.pth"   --resume-optimizer false   --resume-scheduler false   --num-workers 12   --prefetch-factor 6   --save-checkpoint-per-epoch 1   --outputs-dir /opt/ml/checkpoints
