#!/bin/bash
# k8s entrypoint for MOTIP TRT eval on A10G (SM86).
# Image: nvcr.io/nvidia/pytorch:23.09-py3  (TRT 8.6.1, CUDA 12.2, PyTorch 2.1)
#
# Required env vars:
#   S3_CHECKPOINT  s3://... MOTIP checkpoint .pth
#   S3_DATA        s3://... prefix with img1/ and gt/gt.txt
#   S3_OUT         s3://... where to push results
# Optional:
#   S3_SRC         s3://... path to motip_src.tar.gz  (default: k8s/motip_src.tar.gz next to S3_OUT)
#   TRT_VARIANT    small|large    (default: small)
#   TRT_RES        resolution     (default: 576)
#   N_FRAMES       frames to eval (default: 300)
set -euo pipefail

echo "=== MOTIP TRT k8s eval (A10G / SM86) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
python -c "import tensorrt as trt; print('TRT', trt.__version__)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

S3_CHECKPOINT="${S3_CHECKPOINT:?S3_CHECKPOINT not set}"
S3_DATA="${S3_DATA:?S3_DATA not set}"
S3_OUT="${S3_OUT:?S3_OUT not set}"
S3_BASE=$(dirname "$(dirname "$S3_OUT")")
S3_SRC="${S3_SRC:-${S3_BASE}/k8s/motip_src.tar.gz}"
VARIANT="${TRT_VARIANT:-small}"
RES="${TRT_RES:-576}"
N_FRAMES="${N_FRAMES:-300}"

WORKDIR="/workspace/motip"
DATA_DIR="/workspace/data"
OUT_DIR="/workspace/out"
ENGINES_DIR="$OUT_DIR/engines"
CKPT="/workspace/checkpoint.pth"

mkdir -p "$DATA_DIR/img1" "$DATA_DIR/gt" "$OUT_DIR" "$ENGINES_DIR"

# ── deps ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing Python deps ==="
pip install --no-cache-dir --quiet \
    "transformers>=4.46.0,<5.0.0" \
    "rfdetr==1.5.2" \
    timm pyyaml einops scipy opencv-python-headless

# torch_linear_assignment: stub via scipy (same pattern as SageMaker entrypoint)
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
print("stub ->", sp)
PYEOF

# ── fetch + unpack MOTIP source ───────────────────────────────────────────────
echo ""
echo "=== Fetching MOTIP source from S3 ==="
mkdir -p "$WORKDIR"
aws s3 cp "$S3_SRC" /tmp/motip_src.tar.gz
tar -xzf /tmp/motip_src.tar.gz -C "$WORKDIR"
cd "$WORKDIR"

echo "=== Building MultiScaleDeformableAttention CUDA op ==="
cd models/ops && python setup.py build_ext --inplace 2>&1 | tail -3 && cd ../..

export MOTIP_ROOT="$(pwd)"
export PYTHONPATH="$MOTIP_ROOT/models/ops:$MOTIP_ROOT"

# ── fetch data ────────────────────────────────────────────────────────────────
echo ""
echo "=== Fetching data from S3 ==="
aws s3 cp "$S3_CHECKPOINT" "$CKPT"
echo "  checkpoint: $(du -sh $CKPT | cut -f1)"

aws s3 cp "$S3_DATA/" "$DATA_DIR/" --recursive --quiet
echo "  frames: $(ls $DATA_DIR/img1 | wc -l)  GT rows: $(wc -l < $DATA_DIR/gt/gt.txt)"

# ── build TRT engines (SM86) ──────────────────────────────────────────────────
echo ""
echo "=== Building fp16 TRT engine (SM86) ==="
python trt/build_engine.py "$VARIANT" "$RES" --motip \
    --checkpoint "$CKPT" \
    --out "$ENGINES_DIR"

echo ""
echo "=== Building fp32 TRT engine (SM86) ==="
python trt/build_engine.py "$VARIANT" "$RES" --motip --no-fp16 \
    --checkpoint "$CKPT" \
    --out "$ENGINES_DIR"

FP16_ENGINE=$(ls "$ENGINES_DIR"/rfdetr_*_motip_ckpt_*_fp16.engine 2>/dev/null | head -1)
FP32_ENGINE=$(ls "$ENGINES_DIR"/rfdetr_*_motip_ckpt_*_fp32.engine 2>/dev/null | head -1)
echo "fp16 engine: $FP16_ENGINE"
echo "fp32 engine: $FP32_ENGINE"

if [[ -z "$FP16_ENGINE" || -z "$FP32_ENGINE" ]]; then
    echo "ERROR: expected both fp16 and fp32 engines" >&2; exit 1
fi

# ── detector throughput ───────────────────────────────────────────────────────
echo ""
echo "=== Bench detector ==="
python trt/bench_detector.py "$FP16_ENGINE" \
    --batches "1,2,4,8" \
    --out "$OUT_DIR/bench_detector_fp16.json"
python trt/bench_detector.py "$FP32_ENGINE" \
    --batches "1,2,4,8" \
    --out "$OUT_DIR/bench_detector_fp32.json"

# ── 3-way tracking eval (PT fp32 / TRT fp32 / TRT fp16) ──────────────────────
echo ""
echo "=== 3-way eval: tracking + TrackEval (HOTA / MOTA / IDF1) ==="
python trt/eval_trt.py \
    "$DATA_DIR/img1" \
    "$DATA_DIR/gt/gt.txt" \
    "$FP16_ENGINE" \
    "$FP32_ENGINE" \
    --checkpoint "$CKPT" \
    --n_frames "$N_FRAMES" \
    --res "$RES"

cp /tmp/eval_trt_results.json "$OUT_DIR/eval_results.json"

# ── render 3-way side-by-side video ──────────────────────────────────────────
echo ""
echo "=== Rendering 3-way comparison video ==="
python trt/render_3way.py \
    "$DATA_DIR/img1" \
    "$OUT_DIR/eval_results.json" \
    --out "$OUT_DIR/3way_compare.mp4" \
    --n_frames "$N_FRAMES" \
    --fps 15 \
    --res "$RES"

echo "  video: $(du -sh $OUT_DIR/3way_compare.mp4 | cut -f1)"

# ── upload results ────────────────────────────────────────────────────────────
echo ""
echo "=== Uploading results -> $S3_OUT ==="
aws s3 cp "$OUT_DIR/" "$S3_OUT/" --recursive
echo ""
echo "=== DONE ==="
aws s3 ls "$S3_OUT/" --recursive --human-readable
