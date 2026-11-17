"""Cellpose-SAM on one named summary image, as a centroid proposer.

Cellpose-SAM (``cpsam_v2``, Cellpose 4) has no diameter setting, so it does not
depend on the 12 px soma default that the deployed model and the stage 1 gates
assume. The ``roigbiv`` env pins Cellpose 3, so the model runs in a separate
Python (an env with Cellpose 4) through ``cpsam_worker.py``; only the image
and the label image cross the process boundary, as ``.npy`` files.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import center_of_mass

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult

WORKER = Path(__file__).resolve().parents[1] / "cpsam_worker.py"


def run_cpsam_subprocess(
    image: np.ndarray, *, python: str, model: str, cellprob_threshold: float,
    flow_threshold: float, gpu: bool,
) -> tuple[np.ndarray, dict]:
    """Run ``cpsam_worker.py`` under ``python`` on ``image``.

    Returns the ``(H, W)`` int32 label image and the worker's info dict
    (``cellpose_version``, ``device``). A non-zero worker exit raises
    ``RuntimeError`` carrying the last lines of its stderr.
    """
    with tempfile.TemporaryDirectory(prefix="cpsam_") as tmp:
        src, dst, info = (Path(tmp) / n for n in ("image.npy", "labels.npy", "info.json"))
        np.save(src, np.asarray(image, dtype=np.float32))
        cmd = [python, str(WORKER), str(src), str(dst), str(info), "--model", model,
               "--cellprob", repr(float(cellprob_threshold)), "--flow", repr(float(flow_threshold)),
               "--gpu" if gpu else "--cpu"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            raise RuntimeError(f"cpsam worker exited {proc.returncode}: {tail}")
        return np.load(dst).astype(np.int32), json.loads(info.read_text())


class CellposeSamImageDetector:
    """Run Cellpose-SAM on ``inputs.summary[image_key]`` and return label centroids.

    ``python`` is the interpreter of an env with Cellpose 4 and the model
    weights. ``run_fn`` replaces ``run_cpsam_subprocess`` (tests). Defaults
    are the values ROI Suite's discovery recipe uses: cellprob 0, flow 0.4.
    """

    def __init__(
        self,
        image_key: str,
        python: str,
        name: Optional[str] = None,
        model: str = "cpsam_v2",
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        gpu: bool = True,
        run_fn: Optional[Callable] = None,
    ):
        self.image_key = image_key
        self.python = python
        self.name = name or f"cpsam_{image_key}"
        self.model = model
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.gpu = gpu
        self.run_fn = run_fn or run_cpsam_subprocess

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        summary = inputs.summary
        if self.image_key not in summary:
            raise KeyError(f"image_key {self.image_key!r} not in summary; have {sorted(summary)}")
        t0 = time.time()
        labels, info = self.run_fn(
            summary[self.image_key], python=self.python, model=self.model,
            cellprob_threshold=self.cellprob_threshold, flow_threshold=self.flow_threshold, gpu=self.gpu,
        )
        elapsed = time.time() - t0
        ids = [int(i) for i in np.unique(labels) if i != 0]
        centroids = np.asarray(center_of_mass(labels > 0, labels, ids), dtype=np.float32).reshape(-1, 2)
        return CentroidDetectorResult(
            centroids=centroids,
            scores=None,
            meta={
                "method": self.name,
                "image_key": self.image_key,
                "model": self.model,
                "n": len(ids),
                "runtime_s": round(elapsed, 2),
                "cellprob_threshold": self.cellprob_threshold,
                "flow_threshold": self.flow_threshold,
                "cellpose_version": info.get("cellpose_version"),
                "device": info.get("device"),
            },
        )
