"""Cellpose CP3 detector — in-process, the existing pipeline baseline.

Delegates to ``roigbiv.pipeline.stage1.run_cellpose_detection`` so the bake-off
runs Cellpose exactly as the live Stage 1 does (dual-channel [morph, vcorr_S],
denoise, tile-norm) — no reimplementation. This is the control the other
methods must beat.
"""
from __future__ import annotations

import time

import numpy as np

from cv_bakeoff.detector import DetectorInputs, DetectorResult


class CP3Detector:
    name = "cp3"

    def __init__(
        self,
        model: str = "models/deployed/current_model",
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = -2.0,
        morph_channel: str = "mean_M",
        vcorr_channel: str = "vcorr_S",
        force_cpu: bool = False,
    ):
        self.model = model
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.morph_channel = morph_channel
        self.vcorr_channel = vcorr_channel
        self.force_cpu = force_cpu

    def detect(self, inputs: DetectorInputs) -> DetectorResult:
        from roigbiv.pipeline.types import PipelineConfig
        from roigbiv.pipeline.stage1 import (
            run_cellpose_detection, _resolve_model_path,
        )

        morph = inputs.summary[self.morph_channel]
        vcorr = inputs.summary.get(self.vcorr_channel)
        if vcorr is None:
            # Single-channel fallback when no correlation map is available.
            vcorr = np.zeros_like(morph)

        cfg = PipelineConfig(fs=inputs.fs)
        cfg.cellpose_model = self.model
        if self.diameter is not None:
            cfg.diameter = float(self.diameter)
        cfg.diameter_auto = False
        cfg.flow_threshold = self.flow_threshold
        cfg.cellprob_threshold = self.cellprob_threshold
        cfg.force_cpu = self.force_cpu

        resolved = _resolve_model_path(cfg.cellpose_model)
        t0 = time.time()
        _candidates, _probs, label_image, _cellprob = run_cellpose_detection(
            morph.astype(np.float32), vcorr.astype(np.float32), cfg,
        )
        elapsed = time.time() - t0

        label_mask = np.asarray(label_image, dtype=np.uint16)
        return DetectorResult(
            label_mask=label_mask,
            meta={
                "method": self.name,
                "model": resolved,
                "diameter": cfg.diameter,
                "flow_threshold": cfg.flow_threshold,
                "cellprob_threshold": cfg.cellprob_threshold,
                "morph_channel": self.morph_channel,
                "n_rois": int(label_mask.max()),
                "runtime_s": round(elapsed, 2),
            },
        )
