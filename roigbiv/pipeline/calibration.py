"""
ROI G. Biv pipeline — per-FOV centroid-discovery calibration.

Centroid discovery (:mod:`roigbiv.pipeline.centroids`) detects on the
anatomical mean image with Cellpose. This module persists the per-FOV overrides
a user sets in the UI — a measured soma diameter, a detection threshold, and
optionally a different Cellpose checkpoint — as ``calibration.json`` in the
FOV's output directory. No calibration on disk means unchanged (config
default) behavior; the file is written by the UI and read by both the UI and
the CLI (``roigbiv-pipeline --centroids``).

Every field maps onto a real Cellpose control (``diameter``,
``cellprob_threshold``, ``pretrained_model``). This replaces an earlier
diameter-to-Suite2p-``spatial_scale`` mapping that did not do what it claimed:
Suite2p's ``spatial_scale`` never constrained detected cell size — its only
effect was the accept threshold ``Th2 = threshold_scaling * 5 * max(1, scale)``
— so a measured diameter silently became a threshold change, and neighbouring
diameters (40 px and 60 px both map to scale 4) were the same run. Calibration
files from that era still load: ``diameter_px`` carries over unchanged and the
dead fields are ignored.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Mirrors PipelineConfig.cellprob_threshold — lower is more permissive.
DEFAULT_CELLPROB_THRESHOLD = -2.0


@dataclass
class Calibration:
    """A saved centroid-discovery calibration for one FOV."""

    diameter_px: float
    cellprob_threshold: float
    cellpose_model: Optional[str]
    generated_at: float


def write_calibration(output_dir: Path, diameter_px: float,
                      cellprob_threshold: float = DEFAULT_CELLPROB_THRESHOLD,
                      cellpose_model: Optional[str] = None) -> Calibration:
    """Persist a measured soma diameter (px) and its detection settings.

    ``cellpose_model`` is a spec understood by
    :func:`roigbiv.pipeline.stage1._resolve_model_path` — a built-in name
    (``"cyto3"``), or a path to a checkpoint. ``None`` leaves
    ``cfg.cellpose_model`` alone. It is exposed per-FOV because the deployed
    checkpoint is not the best model for every preparation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = Calibration(
        diameter_px=float(diameter_px),
        cellprob_threshold=float(cellprob_threshold),
        cellpose_model=cellpose_model or None,
        generated_at=time.time(),
    )
    (output_dir / "calibration.json").write_text(json.dumps(asdict(calib), indent=2))
    return calib


def load_calibration(output_dir: Path) -> Optional[Calibration]:
    """Read a prior calibration, or ``None`` if missing/corrupt."""
    path = Path(output_dir) / "calibration.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Calibration(
            diameter_px=float(data["diameter_px"]),
            cellprob_threshold=float(
                data.get("cellprob_threshold", DEFAULT_CELLPROB_THRESHOLD)),
            cellpose_model=data.get("cellpose_model") or None,
            generated_at=float(data["generated_at"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
