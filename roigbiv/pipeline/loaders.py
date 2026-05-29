"""
ROI G. Biv pipeline — disk → FOVData loader.

Symmetric counterpart to `outputs.py::save_pipeline_outputs`. Reconstitutes a
FOVData populated with the fields that `napari_viewer.display_pipeline_results`
reads (summary images, corr_contrast_maps, rois with masks + activity + gate
outcome + source_stage). Heavy artifacts — data.bin memmap, residuals, SVD
factors — are not reloaded because the viewer does not need them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.types import FOVData, ROI


def _maybe_read_tif(path: Path):
    if path.exists():
        return tifffile.imread(str(path)).astype(np.float32)
    return None


def load_fov_from_output_dir(
    output_dir: Path,
) -> tuple[FOVData, list]:
    """Reconstitute (FOVData, review_queue) from a pipeline output directory.

    Expects the layout written by `save_pipeline_outputs` + the Foundation /
    Stage 4 side-effects:

        output_dir/
            summary/{mean_M, mean_S, mean_L, max_S, std_S, vcorr_S, dog_map}.tif
            stage4/corr_contrast_{fast,medium,slow}.tif        (optional)
            merged_masks.tif
            roi_metadata.json
            pipeline_log.json
            review_queue.json                                  (optional)
    """
    output_dir = Path(output_dir)
    log_path = output_dir / "pipeline_log.json"
    meta_path = output_dir / "roi_metadata.json"
    masks_path = output_dir / "merged_masks.tif"
    if not log_path.exists():
        raise FileNotFoundError(f"pipeline_log.json not found in {output_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"roi_metadata.json not found in {output_dir}")
    if not masks_path.exists():
        raise FileNotFoundError(f"merged_masks.tif not found in {output_dir}")

    log = json.loads(log_path.read_text())
    meta = json.loads(meta_path.read_text())
    merged_masks = tifffile.imread(str(masks_path))

    # ── Summary images ────────────────────────────────────────────────────
    summary = output_dir / "summary"
    mean_M  = _maybe_read_tif(summary / "mean_M.tif")
    mean_S  = _maybe_read_tif(summary / "mean_S.tif")
    mean_L  = _maybe_read_tif(summary / "mean_L.tif")
    max_S   = _maybe_read_tif(summary / "max_S.tif")
    std_S   = _maybe_read_tif(summary / "std_S.tif")
    vcorr_S = _maybe_read_tif(summary / "vcorr_S.tif")
    dog_map = _maybe_read_tif(summary / "dog_map.tif")

    # ── Stage 4 correlation contrast maps ─────────────────────────────────
    stage4_dir = output_dir / "stage4"
    corr_contrast_maps = {}
    for window in ("fast", "medium", "slow"):
        img = _maybe_read_tif(stage4_dir / f"corr_contrast_{window}.tif")
        if img is not None:
            corr_contrast_maps[window] = img

    # ── Rebuild ROI list ──────────────────────────────────────────────────
    rois: list[ROI] = []
    for entry in meta:
        label_id = int(entry["label_id"])
        mask = (merged_masks == label_id)
        if not mask.any():
            continue
        roi = ROI(
            mask=mask,
            label_id=label_id,
            source_stage=int(entry["source_stage"]),
            confidence=str(entry["confidence"]),
            gate_outcome=str(entry["gate_outcome"]),
            area=int(entry.get("area", 0)),
            solidity=float(entry.get("solidity", 0.0)),
            eccentricity=float(entry.get("eccentricity", 0.0)),
            nuclear_shadow_score=float(entry.get("nuclear_shadow_score", 0.0)),
            soma_surround_contrast=float(entry.get("soma_surround_contrast", 0.0)),
            cellpose_prob=entry.get("cellpose_prob"),
            iscell_prob=entry.get("iscell_prob"),
            event_count=entry.get("event_count"),
            corr_contrast=entry.get("corr_contrast"),
            activity_type=entry.get("activity_type"),
            gate_reasons=list(entry.get("gate_reasons", [])),
            features=dict(entry.get("features", {})),
        )
        rois.append(roi)

    # ── Build FOVData shell (paths that napari viewer never reads stay None) ─
    shape = tuple(log.get("shape", (0, *merged_masks.shape)))
    fov = FOVData(
        raw_path=Path(log.get("input", output_dir / "unknown.tif")),
        output_dir=output_dir,
        data_bin_path=output_dir / "suite2p" / "plane0" / "data.bin",
        shape=shape,
        residual_S_path=output_dir / "residual_S.dat",
        mean_M=mean_M,
        mean_S=mean_S,
        mean_L=mean_L,
        max_S=max_S,
        std_S=std_S,
        vcorr_S=vcorr_S,
        dog_map=dog_map,
        k_background=int(log.get("k_background", 30)),
        rois=rois,
        stage_counts=dict(log.get("stage_counts", {})),
        corr_contrast_maps=corr_contrast_maps,
    )

    # ── Review queue (optional) ───────────────────────────────────────────
    review_queue: list = []
    rq_path = output_dir / "review_queue.json"
    if rq_path.exists():
        review_queue = json.loads(rq_path.read_text())

    return fov, review_queue
