#!/bin/bash
set -euo pipefail
echo "=== MOTIP stage-2 — STAD tracking v2, from AMF stage-1 checkpoint (epoch 19) ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

# SM_CHANNEL_TRAIN is the STAD dataset root.
# Layout: train/<clip_id>/{img1/, gt/gt.txt, seqinfo.ini} — no AMFTracking/ wrapper.
echo "=== Data channel contents ==="
ls "$SM_CHANNEL_TRAIN"
echo "=== Clip count ==="
ls "$SM_CHANNEL_TRAIN/train" | wc -l

echo "=== Pretrain channel contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Launching training (4x GPU, 8 epochs, from AMF stage-1 checkpoint_19) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name motip_amf_stad_stage2_v1 \
  --config-path ./configs/r50_deformable_detr_motip_amf_stad.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/checkpoint_19.pth" \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
