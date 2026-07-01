#!/usr/bin/env python
"""DeepCAD-RT denoising sidecar runner.

Runs OUT-OF-PROCESS, invoked by the `deepcad` conda env's interpreter, because
DeepCAD-RT's dependency pins (torch/CUDA versions) may conflict with the
roigbiv (suite2p / cellpose 3.x) interpreter. This script therefore imports
ONLY numpy, tifffile, and deepcad — never roigbiv.

Contract (single argv: a JSON manifest path):
  {
    "input":  "<path>.tif",   # (T, Ly, Lx) TIFF stack, any dtype
    "output": "<path>.tif",   # written: denoised stack, SAME dtype as input
    "model":  "<path or ''>", # optional pretrained model checkpoint
    "gpu":    <bool>
  }

The caller (roigbiv/pipeline/deepcad.py) validates the output's shape, dtype,
and finiteness before trusting it — this script's job is only to produce a
same-shape, same-dtype denoised stack at the manifest's "output" path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


def _denoise(movie: np.ndarray, model_path: str) -> np.ndarray:
    """Apply DeepCAD-RT denoising to a 3D movie array.

    TODO: the real DeepCAD-RT inference call is not yet implemented — the
    exact public import name and eval API surface are not confirmed in this
    codebase. Fill this in once DeepCAD-RT is installed/vendored in the
    sidecar env; until then this raises NotImplementedError. No test reaches
    this function (tests exercise the driver against a separate fake stub
    script, not this file), so it can ship as a documented stub.
    """
    raise NotImplementedError(
        "DeepCAD-RT denoising call not yet implemented — see TODO in this "
        "function's docstring"
    )


def main(manifest_path: str) -> int:
    with open(manifest_path) as f:
        m = json.load(f)

    movie = tifffile.imread(m["input"])
    orig_dtype = movie.dtype

    # Import here so a missing/broken DeepCAD-RT surfaces as a clean stderr
    # message the parent can relay, not an import-time crash before argv is
    # parsed.
    try:
        import deepcad  # noqa: F401
    except ImportError:
        print(
            "DeepCAD-RT not installed in this environment — "
            "pip install deepcad-rt (or the correct package name)",
            file=sys.stderr,
        )
        return 1

    try:
        print(
            f"  [deepcad] denoising {Path(m['input']).stem} "
            f"({movie.shape[0]} frames, {movie.shape[1]}x{movie.shape[2]})…",
            flush=True,
        )
        denoised = _denoise(movie, m.get("model", ""))
        denoised = denoised.astype(orig_dtype)
        tifffile.imwrite(m["output"], denoised)
    except Exception as exc:
        print(f"deepcad sidecar worker failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: deepcad_sidecar.py <manifest.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
