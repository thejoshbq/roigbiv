"""Cellpose-SAM worker: one image in, one label image out.

Runs under an env with Cellpose 4 (not the ``roigbiv`` env); imports nothing
from this repo. ``--gpu`` refuses to start when CUDA is unavailable instead of
letting Cellpose fall back to the CPU without saying so.

    python cpsam_worker.py IMAGE.npy LABELS.npy INFO.json --model cpsam_v2 --cellprob 0 --flow 0.4 --gpu
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("labels")
    ap.add_argument("info")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cellprob", type=float, required=True)
    ap.add_argument("--flow", type=float, required=True)
    dev = ap.add_mutually_exclusive_group(required=True)
    dev.add_argument("--gpu", action="store_true")
    dev.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    if a.gpu:
        import torch

        if not torch.cuda.is_available():
            print("--gpu given but CUDA is not available", file=sys.stderr)
            return 3
    from cellpose import models

    img = np.load(a.image)
    model = models.CellposeModel(gpu=a.gpu, pretrained_model=a.model)
    labels = model.eval(img, cellprob_threshold=a.cellprob, flow_threshold=a.flow)[0]
    np.save(a.labels, np.asarray(labels, dtype=np.int32))
    try:
        cp_version = version("cellpose")
    except PackageNotFoundError:
        cp_version = "unknown"
    with open(a.info, "w") as f:
        json.dump({"cellpose_version": cp_version, "device": "cuda" if a.gpu else "cpu"}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
