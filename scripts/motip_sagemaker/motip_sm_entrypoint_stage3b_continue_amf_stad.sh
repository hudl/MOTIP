#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-3b continue — STAD, epochs 3-5 from stage3b checkpoint_2 ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== Resume checkpoint contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Launching training (4x GPU, epochs 3-5, from stage3b checkpoint_2) ==="
accelerate launch --num_processes=4 train.py   --data-root "$SM_CHANNEL_TRAIN"   --exp-name motip_amf_stad_stage3b_v1   --config-path ./configs/r50_deformable_detr_motip_amf_stad_stage3b_continue.yaml   --resume-model "$SM_CHANNEL_PRETRAIN/checkpoint_2.pth"   --num-workers 12   --prefetch-factor 6   --save-checkpoint-per-epoch 1   --outputs-dir /opt/ml/checkpoints
