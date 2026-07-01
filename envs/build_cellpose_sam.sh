#!/usr/bin/env bash
# Build the `cp-sam` sidecar conda env for the Cellpose-SAM (CP4) bake-off
# detector (scripts/cv_bakeoff/detectors/cp_sam.py -> workers/cp_sam_worker.py).
#
# Why a script and not just `conda env create`: torch + cellpose>=4 are
# pip-installed so the torch wheel can be matched to this workstation's CUDA
# (RTX 5080 / Blackwell wants a recent cu12x build). Adjust the index URL below
# if the default wheel lacks kernels for your GPU.
#
# Usage:  bash envs/build_cellpose_sam.sh
set -euo pipefail

ENV_NAME="cp-sam"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Creating conda env '${ENV_NAME}' from ${HERE}/cellpose-sam.yml"
conda env create -f "${HERE}/cellpose-sam.yml"

echo ">> Installing torch (CUDA) + cellpose>=4 via pip"
conda run -n "${ENV_NAME}" pip install torch --index-url https://download.pytorch.org/whl/cu124
conda run -n "${ENV_NAME}" pip install "cellpose>=4.0.0"

echo ">> Verifying cellpose>=4 import + model construction"
conda run -n "${ENV_NAME}" python -c \
    "import cellpose; from cellpose import models; \
print('OK: cellpose', getattr(cellpose, 'version', '?'), 'ready in env ${ENV_NAME}')"
