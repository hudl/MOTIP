#!/bin/bash
set -euo pipefail
echo "=== RF-DETR stage-1 (DETR pretrain) — full Hockey dataset, 4x GPU ==="

export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')":${LD_LIBRARY_PATH:-}
export FI_EFA_FORK_SAFE=1

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

pip install --no-cache-dir accelerate wandb einops pyyaml mlflow "transformers>=4.46.0,<5.0.0" timm peft
echo "=== Installing scipy fallback for torch_linear_assignment ==="
python - << PYEOF
import site, pathlib
stub = """import torch\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment as _lsa\n\ndef batch_linear_assignment(cost_matrix):\n    cost_np = cost_matrix.detach().cpu().to(torch.float32).numpy()\n    B, R, C = cost_np.shape\n    out = torch.full((B, R), -1, dtype=torch.int64)\n    for i in range(B):\n        rows, cols = _lsa(cost_np[i])\n        out[i, rows] = torch.from_numpy(cols.astype("int64"))\n    return out\n"""
sp = pathlib.Path(site.getsitepackages()[0]) / "torch_linear_assignment"
sp.mkdir(exist_ok=True)
(sp / "__init__.py").write_text(stub)
print("torch_linear_assignment scipy stub ->", sp)
PYEOF

echo "=== Compiling Deformable Attention extension (needed by MOTIP criterion) ==="
cd models/ops
python setup.py build install
cd ../..

# DINOv2 backbone weights load from HuggingFace — no pretrain channel needed.
# SM_CHANNEL_TRAIN = motip_hockey_data/ (contains Hockey/ as subdirectory).

echo "=== Launching RF-DETR stage-1 training (4x GPU) ==="
accelerate launch --num_processes=4 train.py \
  --data-root "$SM_CHANNEL_TRAIN" \
  --exp-name rfdetr_hockey_stage1_real \
  --config-path ./configs/pretrain_rfdetr_hockey_real.yaml \
  --num-workers 12 \
  --prefetch-factor 6 \
  --save-checkpoint-per-epoch 1 \
  --outputs-dir /opt/ml/checkpoints
