#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-1 ablation — DETR pretrain on STAD v2 data (detection-only) ==="
echo "Ablation: same training strategy as stage1, same data as stage2. Isolates data vs strategy."\n
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

# SM_CHANNEL_TRAIN is the STAD v2 dataset root (same as stage2).
# STADTracking expects: data_root/train/<clip_id>/{img1/, gt/gt.txt, seqinfo.ini}
echo "=== Clip count ==="
ls "$SM_CHANNEL_TRAIN/train" | wc -l

echo "=== Launching training (4x GPU) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name stad_detr_pretrain_ablation \
  --config-path ./configs/pretrain_r50_deformable_detr_stad.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/r50_deformable_detr_coco_sportsmot.pth" \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
