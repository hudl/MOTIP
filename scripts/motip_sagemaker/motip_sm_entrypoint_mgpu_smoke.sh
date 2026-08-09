#!/bin/bash
set -euo pipefail
echo "=== MOTIP multi-GPU smoke test — small dataset, isolate GPU/EFA issues fast ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

# g5.12xlarge has an EFA (Elastic Fabric Adapter) network interface for fast
# multi-GPU communication. Libfabric's EFA provider aborts (SIGABRT) any
# process that calls fork() without fork-safety enabled — our DataLoader's
# num_workers does exactly that. Confirmed root cause from a real crash on
# the first real-run attempt.
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== Launching training (4x GPU, 20-sequence smoke data, 2 epochs) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name motip_hockey_mgpu_smoketest \
  --config-path ./configs/pretrain_r50_deformable_detr_hockey_smoketest.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/r50_deformable_detr_coco_sportsmot.pth" \
  --epochs 2 \
  --num-workers 4 \
  --outputs-dir /opt/ml/checkpoints
