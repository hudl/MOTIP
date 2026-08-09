#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-2 (full MOTIP, detector + ID head) — REAL run, from stage-1 hockey checkpoint ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

# g5.12xlarge EFA fork-safety fix — same root cause as stage-1, DataLoader
# num_workers forks without this and libfabric SIGABRTs.
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== Launching training (4x GPU, full stage-2 recipe: 13 epochs, sample-length 60, from stage-1 hockey checkpoint_1) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name motip_hockey_stage2_real_v1 \
  --config-path ./configs/r50_deformable_detr_motip_hockey_smoketest.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/checkpoint_1.pth" \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
