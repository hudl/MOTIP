#!/bin/bash
# SageMaker entrypoint: install deps, build TRT engine, bench detector, bench tracker.
# Outputs JSON result lines to /opt/ml/model/ (SageMaker copies to S3 automatically).
set -euo pipefail

echo "=== TRT bench entrypoint ==="
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device_count', torch.cuda.device_count())"

# --- deps ---
pip install --no-cache-dir --quiet \
    "transformers>=4.46.0,<5.0.0" \
    "rfdetr==1.5.2" \
    timm pyyaml einops

# TRT: try the NVIDIA PyPI index (TRT 10.x, compatible with CUDA 12).
# If the SageMaker image already bundles TRT this is a no-op.
pip install --no-cache-dir --quiet \
    tensorrt \
    --extra-index-url https://pypi.nvidia.com || true
python -c "import tensorrt as trt; print('TRT', getattr(trt, '__version__', 'unknown'))"

# scipy stub for torch_linear_assignment (same as other MOTIP entrypoints)
python - << 'PYEOF'
import site, pathlib
stub = (
    "import torch\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment as _lsa\n\n"
    "def batch_linear_assignment(cost_matrix):\n"
    "    cost_np = cost_matrix.detach().cpu().to(torch.float32).numpy()\n"
    "    B, R, C = cost_np.shape\n"
    "    out = torch.full((B, R), -1, dtype=torch.int64)\n"
    "    for i in range(B):\n"
    "        rows, cols = _lsa(cost_np[i])\n"
    "        out[i, rows] = torch.from_numpy(cols.astype('int64'))\n"
    "    return out\n"
)
sp = pathlib.Path(site.getsitepackages()[0]) / "torch_linear_assignment"
sp.mkdir(exist_ok=True)
(sp / "__init__.py").write_text(stub)
print("torch_linear_assignment stub ->", sp)
PYEOF

# --- config from env (set by submit.py via SageMaker environment dict) ---
VARIANT="${TRT_VARIANT:-large}"
RES="${TRT_RES:-1088}"
BATCHES="${TRT_BATCHES:-1,2,4,8}"
SKIP_TRACKER="${TRT_SKIP_TRACKER:-0}"
MOTIP_CONFIG="${TRT_MOTIP_CONFIG:-configs/eval_stage2_hockey.yaml}"

OUT_DIR="/opt/ml/model"
ENGINES_DIR="$OUT_DIR/engines"
RESULTS_JSON="$OUT_DIR/results.jsonl"
mkdir -p "$ENGINES_DIR"

export MOTIP_ROOT="$(pwd)"

echo ""
echo "=== Building TRT engine: rfdetr-$VARIANT @ $RES (detector mode) ==="
python trt/build_engine.py "$VARIANT" "$RES" \
    --out "$ENGINES_DIR" \
    --results-json "$RESULTS_JSON"

# Locate the engine we just built
ENGINE=$(ls "$ENGINES_DIR"/rfdetr_"$VARIANT"_"$RES"_det_*.engine 2>/dev/null | head -1)
if [[ -z "$ENGINE" ]]; then
    echo "ERROR: no engine found after build" >&2
    exit 1
fi
echo "Engine: $ENGINE"

echo ""
echo "=== Building MOTIP engine (with query_embeds output) ==="
python trt/build_engine.py "$VARIANT" "$RES" --motip \
    --out "$ENGINES_DIR" \
    --results-json "$RESULTS_JSON"

echo ""
echo "=== Bench detector ==="
python trt/bench_detector.py "$ENGINE" \
    --batches "$BATCHES" \
    --out "$OUT_DIR/bench_detector.json" \
    --results-json "$RESULTS_JSON"

if [[ "$SKIP_TRACKER" != "1" ]]; then
    echo ""
    echo "=== Bench tracker ==="
    python trt/bench_tracker.py \
        --config "$MOTIP_CONFIG" \
        --arms eager,cuda_graphs \
        --out "$OUT_DIR/bench_tracker.json" \
        --results-json "$RESULTS_JSON"
fi

echo ""
echo "=== Done. Results at $RESULTS_JSON ==="
cat "$RESULTS_JSON"
