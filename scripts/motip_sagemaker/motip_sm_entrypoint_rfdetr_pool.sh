#!/bin/bash
set -euo pipefail
# Data-recipe pool arm (frozen RF-DETR from the production v1 checkpoint_12, detector outputs from the motip_pool_v1 DETR cache, ID decoder trains).
# POOL_ARM = c1 | w1 | m50 | a1; the arms differ only by configs/finetune_pool_<arm>.yaml (its SAMPLE_MANIFEST and augmentation).
echo "=== data-recipe pool arm ${POOL_ARM} ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"
export FI_EFA_FORK_SAFE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow "transformers==5.16.1" timm "peft==0.20.0"
echo "=== Installing scipy fallback for torch_linear_assignment ==="
python - << PYEOF
import site, pathlib
stub = 'import torch\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment as _lsa\n\ndef batch_linear_assignment(cost_matrix):\n    cost_np = cost_matrix.detach().cpu().to(torch.float32).numpy()\n    B, R, C = cost_np.shape\n    out = torch.full((B, R), -1, dtype=torch.int64)\n    for i in range(B):\n        rows, cols = _lsa(cost_np[i])\n        out[i, rows] = torch.from_numpy(cols.astype("int64"))\n    return out\n'
sp = pathlib.Path(site.getsitepackages()[0]) / "torch_linear_assignment"
sp.mkdir(exist_ok=True)
(sp / "__init__.py").write_text(stub)
print("torch_linear_assignment scipy stub ->", sp)
PYEOF

echo "=== Compiling Deformable Attention extension ==="
cd models/ops
python setup.py build install
cd ../..
python -c "import MultiScaleDeformableAttention; print('deformable attn OK')"

echo "=== channels ==="
ls "$SM_CHANNEL_PRETRAIN"
ls "$SM_CHANNEL_CACHE" | head
ls "$SM_CHANNEL_CACHE/Hockey/train" | wc -l
ls "$SM_CHANNEL_TRAIN/Hockey/train" | wc -l
ls "$SM_CHANNEL_TRAIN/manifests"

echo "=== Launching pool arm ${POOL_ARM} (4x A10G, frozen RF-DETR from v1 checkpoint_12, DETR cache) ==="
NPROC=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
echo "processes: $NPROC"
accelerate launch --num_processes=$NPROC train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name "rfdetr_pool_${POOL_ARM}" \
  --config-path "./configs/finetune_pool_${POOL_ARM}.yaml" \
  --resume-model "$SM_CHANNEL_PRETRAIN/${RESUME_CKPT_NAME:-checkpoint_12.pth}" \
  --resume-optimizer false \
  --resume-scheduler false \
  --num-workers 4 \
  --prefetch-factor 4 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
