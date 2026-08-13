"""Cellpose centroid detector — wraps the production detection call directly.

Calls ``roigbiv.pipeline.stage1.run_cellpose_detection`` rather than
reimplementing Cellpose inference, so this arm inherits model loading
(``models/deployed/current_model``), GPU/CPU dispatch, diameter
auto-calibration, and — critically — the *actual* production defaults, which
live in ``roigbiv.pipeline.types.PipelineConfig`` and differ from
``configs/pipeline.yaml``'s documentary values: ``flow_threshold=0.4`` (not
0.6) and ``stage1_ch2_source="vcorr_max_fused"`` (not ``"vcorr_S"``). A
reimplementation would silently drift from whatever stage1.py actually ships.

``cellprob_threshold`` (default ``None`` -> ``cfg``'s own value, production
``-2.0``) is exposed as a constructor override for the sweep's structural
grid: it gates which pixels form a mask at all inside ``model.eval()``, so
masks below it never exist as candidates — unlike ``probs_list`` (the mean
cellprob per already-formed mask, returned in ``scores``), which can only
trim precision within whatever got formed at the current threshold, not
recover missed recall. ``flow_threshold`` is intentionally not overridable
here; the sweep holds it fixed at the production default.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import center_of_mass

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult


class CellposeCentroidDetector:
    name = "cellpose"

    def __init__(self, cfg=None, cellprob_threshold: float = None):
        self.cfg = cfg
        self.cellprob_threshold = cellprob_threshold

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        import dataclasses

        from roigbiv.pipeline.stage1 import run_cellpose_detection
        from roigbiv.pipeline.types import PipelineConfig

        cfg = self.cfg or inputs.cfg or PipelineConfig(fs=inputs.fs)
        if self.cellprob_threshold is not None:
            cfg = dataclasses.replace(cfg, cellprob_threshold=self.cellprob_threshold)

        mean_M = inputs.summary.get("mean_M")
        vcorr_S = inputs.summary.get("vcorr_S")
        max_S = inputs.summary.get("max_S")
        if mean_M is None or vcorr_S is None:
            raise KeyError(
                f"cellpose detector needs mean_M and vcorr_S; have {sorted(inputs.summary)}"
            )

        t0 = time.time()
        masks_list, probs_list, label_image, _cellprob_map = run_cellpose_detection(
            mean_M, vcorr_S, cfg, max_S=max_S,
        )
        elapsed = time.time() - t0

        if not masks_list:
            return CentroidDetectorResult(
                centroids=np.zeros((0, 2), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                meta={"method": self.name, "model": cfg.cellpose_model,
                      "stage1_ch2_source": cfg.stage1_ch2_source,
                      "flow_threshold": cfg.flow_threshold, "n": 0,
                      "runtime_s": round(elapsed, 2)},
            )

        # Per-mask center_of_mass -- same convention as roi_stamp.py::canonicalize
        # (ADR-0003's canonical centroid definition), applied per boolean mask
        # rather than run_cellpose_detection's already-split label list.
        centroids = np.asarray(
            [center_of_mass(m) for m in masks_list], dtype=np.float32,
        ).reshape(-1, 2)
        scores = np.asarray(probs_list, dtype=np.float32)

        return CentroidDetectorResult(
            centroids=centroids,
            scores=scores,
            meta={
                "method": self.name, "model": cfg.cellpose_model,
                "stage1_ch2_source": cfg.stage1_ch2_source,
                "flow_threshold": cfg.flow_threshold,
                "cellprob_threshold": cfg.cellprob_threshold,
                "n": int(len(masks_list)), "runtime_s": round(elapsed, 2),
            },
        )
