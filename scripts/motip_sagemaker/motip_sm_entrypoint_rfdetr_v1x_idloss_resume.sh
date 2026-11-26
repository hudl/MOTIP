#!/bin/bash
set -euo pipefail
echo "=== v1x idloss RESUMED from checkpoint_2 (epochs 3-12) after the epoch-3 NCCL hang ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):"
export FI_EFA_FORK_SAFE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

# transformers 5.x to match the staged rfdetr 1.3.1 (transformers-5 backbone API); <5 crashed v1x at model build.
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

echo "=== Resume checkpoint contents ==="
ls "$SM_CHANNEL_PRETRAIN"

echo "=== Launching v1x idloss resume (4x GPU) ==="
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000   # flight recorder: a stack trace of the collective if anything hangs again
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name rfdetr_v1x_idloss_resume \
  --config-path ./configs/finetune_crossing_rfdetr_v1x_idloss_resume.yaml \
  --resume-model "$SM_CHANNEL_PRETRAIN/${RESUME_CKPT_NAME:-checkpoint_2.pth}" \
  --resume-optimizer true \
  --resume-scheduler true \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
