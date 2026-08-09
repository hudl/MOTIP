#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-2 (full MOTIP, detector + ID head) — AMF, resuming from stage-2 checkpoint ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

# g5.12xlarge EFA fork-safety.
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

echo "=== Resume checkpoint contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Launching training (4x GPU, resuming from checkpoint_1.pth) ==="
accelerate launch --num_processes=4 train.py   --data-root "$SM_CHANNEL_TRAIN"   --exp-name motip_amf_stage2_v1   --config-path ./configs/r50_deformable_detr_motip_amf.yaml   --resume-model "$SM_CHANNEL_PRETRAIN/checkpoint_1.pth"   --resume-optimizer true   --resume-scheduler true   --epochs 6   --num-workers 12   --prefetch-factor 6   --save-checkpoint-per-epoch 1   --outputs-dir /opt/ml/checkpoints
