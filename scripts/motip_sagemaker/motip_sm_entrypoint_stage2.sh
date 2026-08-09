#!/bin/bash
set -euo pipefail
echo "=== MOTIP SageMaker stage-2 (full MOTIP) smoke test entrypoint ==="

# The compiled MultiScaleDeformableAttention extension links against torch's
# bundled libc10.so/libtorch.so, but torch's own lib/ dir isn't on the
# dynamic linker's default search path in this container — without this,
# `import MultiScaleDeformableAttention` fails with
# "ImportError: libc10.so: cannot open shared object file".
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== Launching training (stage 2: full MOTIP, detector + ID head) ==="
# Same lighter memory settings that fixed the OOM on the devbox T4 —
# full MOTIP's joint multi-frame ID decoder is much heavier per-step than
# stage 1's single-frame detector-only training.
accelerate launch --num_processes=1 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name motip_hockey_stage2_sm_smoketest \
  --config-path ./configs/r50_deformable_detr_motip_hockey_smoketest.yaml \
  --detr-pretrain "$SM_CHANNEL_PRETRAIN/r50_deformable_detr_coco_sportsmot.pth" \
  --epochs 1 \
  --num-workers 4 \
  --sample-lengths 10 \
  --aug-max-size 640 \
  --detr-num-checkpoint-frames 2 \
  --batch-size 1 \
  --outputs-dir /opt/ml/checkpoints
