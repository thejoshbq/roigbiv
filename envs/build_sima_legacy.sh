#!/usr/bin/env bash
# Build the `sima-legacy` sidecar conda env for the legacy SIMA motion-correction
# backend (roigbiv/pipeline/legacy_mc.py → scripts/sima_mc_worker.py).
#
# Why a script and not just `conda env create`:
#   SIMA 1.3.2's PyPI sdist ships only a stale pre-generated `_motion.c` that
#   references CPython exception fields removed in 3.7+, so it won't compile.
#   We install from the GitHub source (tag 1.3.2), which carries the `.pyx`, and
#   force `--no-build-isolation` so SIMA's setup.py re-cythonizes with this env's
#   Cython 0.29 into 3.8-compatible C. That flag cannot be expressed in the
#   conda yml `pip:` section, hence this wrapper.
#
# Usage:  bash envs/build_sima_legacy.sh
set -euo pipefail

ENV_NAME="sima-legacy"
SIMA_REF="1.3.2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Creating conda env '${ENV_NAME}' from ${HERE}/sima-legacy.yml"
conda env create -f "${HERE}/sima-legacy.yml"

echo ">> Installing SIMA ${SIMA_REF} from git (re-cythonized, no build isolation)"
conda run -n "${ENV_NAME}" pip install --no-build-isolation \
    "git+https://github.com/losonczylab/sima.git@${SIMA_REF}"

echo ">> Verifying SIMA import + HiddenMarkov2D construction"
conda run -n "${ENV_NAME}" python -c \
    "import sima, sima.motion; from sima.motion import HiddenMarkov2D; \
HiddenMarkov2D(granularity='row', max_displacement=[50,50], verbose=False); \
print('OK: SIMA', sima.__version__, 'ready in env ${ENV_NAME}')"
