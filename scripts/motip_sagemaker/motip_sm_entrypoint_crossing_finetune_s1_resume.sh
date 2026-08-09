#!/bin/bash
set -euo pipefail
echo "=== MOTIP crossing fine-tune stride-1 RESUME — from checkpoint 13, epochs 14-16 ==="

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

echo "=== Pretrain channel (checkpoint_13) ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Verifying checkpoint ==="
python -c "
import torch
ckpt = torch.load('$SM_CHANNEL_PRETRAIN/checkpoint_13.pth', map_location='cpu', weights_only=False)
print(f'Keys: {list(ckpt.keys())}')
print(f'Epoch in ckpt: {ckpt.get(\"epoch\", \"N/A\")}')
"

echo "=== Launching training (2x GPU, stride-1, epochs 14-16 from ckpt13) ==="
accelerate launch --num_processes=2 train.py   --data-root "$SM_CHANNEL_TRAIN"   --exp-name motip_crossing_finetune_s1_resume   --config-path ./configs/finetune_crossing_s1_resume.yaml   --resume-model "$SM_CHANNEL_PRETRAIN/checkpoint_13.pth"   --resume-optimizer true   --resume-scheduler true   --num-workers 12   --prefetch-factor 6   --save-checkpoint-per-epoch 1   --outputs-dir /opt/ml/checkpoints
