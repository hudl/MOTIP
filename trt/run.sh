#!/usr/bin/env bash
# Local launcher for the TRT scripts on the devbox.
#
# TRT 8.6's Python bindings need cuDNN 8 while torch 2.8 ships cuDNN 9.
# Both are installed in separate directories; putting both on LD_LIBRARY_PATH
# lets each find the soname it wants.  This matches the approach in
# experiments/rfdetr-bench/trt.sh.
#
# Usage (from MOTIP repo root):
#   ./trt/run.sh build_engine large 1088 --motip
#   ./trt/run.sh build_engine large 1088 --batch 4
#   ./trt/run.sh bench_detector trt/engines/rfdetr_large_1088_det_sm75_fp16.engine
#   ./trt/run.sh bench_tracker --arms eager,cuda_graphs
#   ./trt/run.sh bench_detector --compare trt/engines/t4.json trt/engines/a10g.json
#
# On SageMaker: the entrypoint calls the Python scripts directly without this
# wrapper -- TRT is installed from PyPI there and cuDNN versions are consistent.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SCRIPT="$1"; shift

# Isolated venv where rfdetr + TRT 8.6 are installed together
BENCH_VENV="${RFDETR_BENCH_VENV:-/home/ubuntu/experiments/rfdetr-bench/.venv}"
PYTHON="$BENCH_VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $BENCH_VENV" >&2
    echo "       set RFDETR_BENCH_VENV to point at your rfdetr+TRT venv" >&2
    exit 1
fi

SP="$BENCH_VENV/lib/python3.10/site-packages"
CUDNN8_LIB="${CUDNN8_LIB:-/home/ubuntu/experiments/cudnn8/nvidia/cudnn/lib}"

export LD_LIBRARY_PATH="$SP/tensorrt_libs:$CUDNN8_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SP/tensorrt_bindings:${PYTHONPATH:-}"
export MOTIP_ROOT="$(pwd)"

exec "$PYTHON" "trt/$SCRIPT.py" "$@"
