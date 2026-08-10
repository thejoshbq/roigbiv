"""Suite2p centroid detector — wraps run_suite2p_fov, reads stat['ypix']/['xpix'].

Centroid convention: unweighted mean of each ROI's mask pixel coordinates
(``ypix.mean()``, ``xpix.mean()``) — algebraically identical to
``scipy.ndimage.center_of_mass`` on the boolean mask, which is what GT,
Cellpose, and OpenCV all use (see ``roigbiv.pipeline.roi_stamp``, ADR-0003).
Suite2p's own ``stat['med']`` field is the nearest-*actual*-pixel to the
centroid, not the centroid itself — using it here would bake a small,
method-specific localization bias into every comparison, so it's deliberately
not used.

Returns every raw candidate ROI with its ``iscell`` probability in ``scores``,
unfiltered — the main pipeline's ``iscell_threshold`` cutoff moves to a
post-hoc filter (``sweep.filter_by_score``) applied by the caller, so a single
detection call can support both the production-faithful single-point report
and a free rescore sweep across the whole ``iscell`` probability range.

Lean by default: passes ``spikedetect: False`` (the only override; every other
detection knob is left unset so ``_build_ops()``'s own hardcoded defaults
apply -- the same defaults ``foundation.py``'s real call path actually
exercises in production, since it only forwards *registration* keys, never
detection ones). Spike deconvolution runs strictly after ``stat.npy`` is
finalized, so this shouldn't change detection -- ``--suite2p-full`` disables
the override for a paranoid parity check (see the implementation plan's
byte-diff verification step).

``threshold_scaling`` (default ``None`` -> Suite2p's own default of 1.0) is
exposed as a constructor arg for the sweep's structural grid: unlike
``iscell``, it changes which ROIs Suite2p finds in the first place, so it
requires a real rerun rather than a post-hoc filter — see
``roigbiv/suite2p.py::_build_ops``. When set, ``detect()`` runs in a
``work_dir`` subdirectory keyed by the value, so distinct sweep points don't
collide with each other (or with a prior default run) under
``run_suite2p_fov``'s ``resume=True`` short-circuit, which only checks whether
``stat.npy`` already exists at that output path.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult


class Suite2pCentroidDetector:
    name = "suite2p"

    def __init__(
        self,
        work_dir: Path,
        iscell_threshold: float = 0.3,
        lean: bool = True,
        threshold_scaling: float = None,
    ):
        self.work_dir = Path(work_dir)
        self.iscell_threshold = iscell_threshold
        self.lean = lean
        self.threshold_scaling = threshold_scaling

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        from roigbiv.suite2p import run_suite2p_fov

        if inputs.raw_tif_path is None:
            raise ValueError(
                "Suite2pCentroidDetector needs inputs.raw_tif_path (operates on the movie, not summaries)"
            )
        tif_path = Path(inputs.raw_tif_path).resolve()

        s2p_cfg = {"suite2p": {}}
        if self.lean:
            s2p_cfg["suite2p"]["spikedetect"] = False

        work_dir = self.work_dir
        if self.threshold_scaling is not None:
            work_dir = self.work_dir / f"ts_{self.threshold_scaling:g}"

        t0 = time.time()
        run_suite2p_fov(
            tif_path, work_dir, fs=inputs.fs,
            do_registration=False, cfg=s2p_cfg, resume=True,
            threshold_scaling=self.threshold_scaling,
        )
        elapsed = time.time() - t0

        # run_suite2p_fov strips "_mc" from the tif stem for its output dir
        # naming (roigbiv/suite2p.py::run_suite2p_fov) -- mirror that exactly
        # rather than trusting inputs.fov_stem to already be mc-stripped.
        stem = tif_path.stem.replace("_mc", "")
        plane_dir = work_dir / stem / "suite2p" / "plane0"
        stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
        iscell = np.load(plane_dir / "iscell.npy")

        centroids: list[list[float]] = []
        scores: list[float] = []
        for i, s in enumerate(stat):
            prob = float(iscell[i, 1]) if i < len(iscell) else 0.0
            centroids.append([float(np.mean(s["ypix"])), float(np.mean(s["xpix"]))])
            scores.append(prob)

        result = CentroidDetectorResult(
            centroids=np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
            scores=np.asarray(scores, dtype=np.float32),
            meta={
                "method": self.name, "iscell_threshold": self.iscell_threshold,
                "lean": self.lean, "threshold_scaling": self.threshold_scaling,
                "n_raw": int(len(stat)), "n": len(centroids),
                "runtime_s": round(elapsed, 2),
            },
        )

        from centroid_bakeoff.sweep import filter_by_score
        return filter_by_score(result, self.iscell_threshold)
