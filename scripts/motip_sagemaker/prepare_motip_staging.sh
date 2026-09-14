#!/bin/bash
# Stage sip-tracking-experiments MOTIP source for SageMaker upload.
# Copies MOTIP source + entrypoints into /tmp/motip_sm_staging (the
# SOURCE_DIR expected by submit_motip_sagemaker.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTIP_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
STAGE="/tmp/motip_sm_staging"
REPO_ROOT="$(dirname "$(dirname "$MOTIP_ROOT")")"

echo "Staging from: ${MOTIP_ROOT}"
echo "Staging to:   ${STAGE}"

rm -rf "${STAGE}"
cp -r "${MOTIP_ROOT}/" "${STAGE}"

# Copy entrypoints to staging root (SageMaker runs them from there)
cp "${SCRIPT_DIR}"/motip_sm_entrypoint*.sh "${STAGE}/"

# Bundle rfdetr Python package so models/motip/__init__.py can find it at
# the _RFDETR_BUNDLED path (code/third_party/rfdetr/).
RF_DETR_PKG="${REPO_ROOT}/third_party/aml-ice-hockey/ihc-od/third_party/rf-detr/rfdetr"
if [ -d "${RF_DETR_PKG}" ]; then
    mkdir -p "${STAGE}/third_party"
    cp -r "${RF_DETR_PKG}" "${STAGE}/third_party/"
    echo "Bundled rfdetr -> ${STAGE}/third_party/rfdetr"
else
    echo "WARNING: rfdetr package not found at ${RF_DETR_PKG}" >&2
fi

# Prune noise
find "${STAGE}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" -type d -name build -path '*/ops/build' -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" -type d -name datasets -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" -type d -name outputs -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}" -type f -name '*.pyc' -delete
find "${STAGE}" -type f -name '*.pth' -delete
find "${STAGE}" -type f \( -name '*.mp4' -o -name '*.avi' \) -delete

echo "Done"
du -sh "${STAGE}" 2>/dev/null | sed 's/^/  /'
