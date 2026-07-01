#!/usr/bin/env bash
# Build the `stardist` sidecar conda env for the StarDist 2D bake-off detector
# (scripts/cv_bakeoff/detectors/stardist.py -> workers/stardist_worker.py).
#
# Why a script: stardist + tensorflow are pip-installed so the tensorflow wheel
# can be matched to this workstation's CUDA. The pretrained 2D_versatile_fluo
# weights download on first model construction.
#
# Usage:  bash envs/build_stardist.sh
set -euo pipefail

ENV_NAME="stardist"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Creating conda env '${ENV_NAME}' from ${HERE}/stardist.yml"
conda env create -f "${HERE}/stardist.yml"

echo ">> Installing tensorflow + stardist + csbdeep via pip"
conda run -n "${ENV_NAME}" pip install "tensorflow[and-cuda]" stardist csbdeep

echo ">> Verifying stardist import + pretrained model fetch"
conda run -n "${ENV_NAME}" python -c \
    "import stardist; from stardist.models import StarDist2D; \
StarDist2D.from_pretrained('2D_versatile_fluo'); \
print('OK: stardist', stardist.__version__, 'ready in env ${ENV_NAME}')"
