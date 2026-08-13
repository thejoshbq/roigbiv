"""
ROI G. Biv pipeline — standalone centroid discovery.

Independent of the stage cascade (Stage 1-4, cascade_legacy): this module turns
a motion-corrected FOV into a list of soma centroids for the annotation
workflow, and never warps pixels itself.

Detector
--------
Cellpose on the **anatomical** mean image (``summary/mean_M.tif``), via
:func:`roigbiv.pipeline.stage1.run_cellpose_detection` so the model selector,
diameter estimation and denoise behavior match Stage 1 exactly.

Suite2p's activity-based sparse detection was the original detector here and is
now retained only as a cross-check: each Cellpose centroid records whether a
Suite2p candidate corroborates it (``activity_support``). It was demoted because
its substrate inverts on this preparation — ``sparsery`` normalizes every pixel
by its own temporal SD (``mov / sdmov``), which lifts empty low-variance
background to parity with real tissue. On a prism FOV, where much of the frame
images nothing, the peak search then prefers the background: measured on the
reference FOV, 3519 of 4833 candidates landed in the *darkest* intensity
quintile and 14 in the brightest, and only 1 of the 62 inside the tissue sat on
a local maximum of the mean image. Cellpose on the same mean image put every one
of its masks on a real soma.

When Foundation never ran, the mean projection detection falls back to is also
written to ``summary/mean_M.tif`` — the cross-session registry needs an
anatomical image to align sessions, and this makes a centroids-only workspace
trackable without paying for Foundation.

Recompute is keyed on the resolved detection parameters and a schema version,
both recorded in ``centroids.json``: a changed ``calibration.json`` recomputes,
an unchanged one resumes.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

# Bumped whenever the centroids.json payload changes shape. Part of the
# recompute key: an artifact written by an older schema is recomputed rather
# than reused, so a stale pre-fix result can't survive an upgrade untouched.
_SCHEMA = 4

# A Suite2p candidate this close to a Cellpose centroid counts as corroborating
# it. One soma radius at the diameters this workflow sees (40-80 px).
_ACTIVITY_SUPPORT_RADIUS_PX = 25.0


@dataclass
class CentroidResult:
    """Outcome of one :func:`run_centroid_discovery` call."""

    output_path: Path
    count: int


class _Substrate(NamedTuple):
    morph: np.ndarray      # mean_M — the anatomical channel
    ch2: np.ndarray        # vcorr_S — Stage 1's second channel
    max_S: Optional[np.ndarray]
    source: str            # "mean_M" (Foundation's) or "mean_projection" (ours)


def _with_overrides(cfg, **overrides):
    """Copy of *cfg* with fields replaced — never mutate the caller's config."""
    if is_dataclass(cfg):
        return replace(cfg, **overrides)
    clone = copy.copy(cfg)
    for key, value in overrides.items():
        setattr(clone, key, value)
    return clone


def _resolved_params(cfg, calib) -> dict:
    """The detection parameters this run will actually use.

    Doubles as the recompute key written into ``centroids.json``.
    """
    return {
        "detector": "cellpose",
        # None = no measurement for this FOV; leave cfg.diameter/diameter_auto
        # alone rather than pinning inference to the config's generic default.
        "diameter_px": float(calib.diameter_px) if calib else None,
        "cellprob_threshold": (float(calib.cellprob_threshold) if calib
                               else float(getattr(cfg, "cellprob_threshold", -2.0))),
        "cellpose_model": ((calib.cellpose_model if calib and calib.cellpose_model
                            else getattr(cfg, "cellpose_model", None))),
        "tissue_mask": bool(getattr(cfg, "centroid_tissue_mask", False)),
    }


# Frames averaged when deriving a mean projection straight from a stack. The
# anatomical image converges long before this; the cap keeps a multi-hour
# session from being read end to end for what is only a morphology reference.
_MEAN_PROJECTION_FRAMES = 2000


def _mean_projection(mc_tif_path: Path) -> np.ndarray:
    """Mean projection of an already motion-corrected stack.

    Only used when Foundation never ran for this FOV (centroids-only mode on a
    pre-corrected input), which is the one case with no ``summary/`` on disk.
    """
    import tifffile

    with tifffile.TiffFile(str(mc_tif_path)) as tif:
        n = len(tif.pages)
        idx = np.unique(np.linspace(0, n - 1, min(n, _MEAN_PROJECTION_FRAMES))
                        .astype(int))
        total = np.zeros(tif.pages[0].shape, dtype=np.float64)
        for i in idx:
            total += np.asarray(tif.pages[int(i)].asarray(), dtype=np.float64)
    return (total / len(idx)).astype(np.float32)


def _load_substrate(output_dir: Path, mc_tif_path: Path) -> _Substrate:
    """The anatomical images centroid discovery detects on.

    ``mean_M`` is the morphological channel, not ``mean_S``: under truncated-SVD
    L+S the residual mean is ~0 everywhere (measured std 0.07 on the reference
    FOV), so it carries no morphology. This matches what Stage 1 passes at
    ``roigbiv/pipeline/run.py:610``.

    ``ch2``/``max_S`` are inert under this module's pinned single-channel
    config and are read only so the Stage 1 entry point keeps its signature.
    """
    import tifffile

    summary = Path(output_dir) / "summary"
    mean_path = summary / "mean_M.tif"
    if not mean_path.exists():
        if not mc_tif_path.exists() or mc_tif_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"no anatomical image for this FOV: neither {mean_path} nor a "
                f"readable stack at {mc_tif_path} — run motion correction first")
        morph = _mean_projection(mc_tif_path)
        return _Substrate(morph, morph, None, "mean_projection")

    morph = np.asarray(tifffile.imread(mean_path), dtype=np.float32)
    ch2_path = summary / "vcorr_S.tif"
    ch2 = (np.asarray(tifffile.imread(ch2_path), dtype=np.float32)
           if ch2_path.exists() else morph)
    max_path = summary / "max_S.tif"
    max_S = (np.asarray(tifffile.imread(max_path), dtype=np.float32)
             if max_path.exists() else None)
    return _Substrate(morph, ch2, max_S, "mean_M")


def _persist_substrate(output_dir: Path, morph: np.ndarray) -> None:
    """Write the detection substrate to ``summary/mean_M.tif`` if nothing is there.

    The cross-session registry needs an anatomical image to align sessions
    (``roicat_adapter.load_session_input``), and a centroids-only workspace has
    never run Foundation. This is the same mean projection detection just ran
    on, so persisting it costs nothing and makes the FOV trackable.

    Never overwrites: Foundation's own ``mean_M`` is the better image and wins
    whenever it exists.
    """
    import tifffile

    summary = Path(output_dir) / "summary"
    mean_path = summary / "mean_M.tif"
    if mean_path.exists():
        return
    summary.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(mean_path), np.asarray(morph, dtype=np.float32))


def _tissue_mask(morph: np.ndarray, sigma: float) -> tuple[Optional[np.ndarray], dict]:
    """Binary "this pixel is inside the imaged tissue" mask from the mean image.

    Off by default: Cellpose detects on the anatomical image directly and does
    not have the background-inversion problem that made this necessary for
    Suite2p. Kept as an opt-in guard for FOVs with bright non-tissue artifacts.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    smoothed = ndi.gaussian_filter(morph, sigma)
    threshold = float(threshold_otsu(smoothed))
    mask = smoothed > threshold
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, np.ones((15, 15))))

    labels, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    return mask, {
        "applied": True,
        "sigma": float(sigma),
        "otsu_threshold": round(threshold, 2),
        "coverage": round(float(mask.mean()), 4),
    }


def _suite2p_candidates(output_dir: Path, stem: str) -> Optional[np.ndarray]:
    """Centroids of Foundation's Suite2p detection, as an ``(N, 2)`` y/x array.

    Read-only — whatever Foundation already wrote. Absent output simply means
    no cross-check is available (``None``), never a re-run.
    """
    stat_path = Path(output_dir) / stem / "suite2p" / "plane0" / "stat.npy"
    if not stat_path.exists():
        return None
    try:
        stat = np.load(str(stat_path), allow_pickle=True)
    except (OSError, ValueError):
        return None

    points = [
        (float(np.asarray(s["ypix"]).mean()), float(np.asarray(s["xpix"]).mean()))
        for s in stat
        if np.asarray(s["ypix"]).size and np.asarray(s["xpix"]).size
    ]
    return np.asarray(points, dtype=np.float64) if points else None


def _centroids_from_masks(masks, probs, activity: Optional[np.ndarray]) -> list[dict]:
    """Unweighted mask centroid per ROI — the convention in ADR-0003."""
    from scipy.ndimage import center_of_mass

    centroids = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        npix = int(mask.sum())
        if npix == 0:
            continue
        y, x = center_of_mass(mask)
        entry = {
            "label_id": i,
            "y": float(y),
            "x": float(x),
            "npix": npix,
            "equiv_diameter_px": round(float(2 * np.sqrt(npix / np.pi)), 2),
            "cellpose_prob": float(probs[i]) if i < len(probs) else 0.0,
        }
        if activity is not None:
            d = np.hypot(activity[:, 0] - y, activity[:, 1] - x).min()
            entry["activity_support"] = bool(d <= _ACTIVITY_SUPPORT_RADIUS_PX)
            entry["activity_distance_px"] = round(float(d), 2)
        centroids.append(entry)
    return centroids


def run_centroid_discovery(
    mc_tif_path: Path,
    output_dir: Path,
    cfg,
    gpu_lock=None,
) -> CentroidResult:
    """Detect soma centroids on a motion-corrected FOV's anatomical mean image.

    ``output_dir`` is the FOV's pipeline output directory. Detection runs on
    Foundation's ``summary/mean_M.tif`` when present, otherwise on a mean
    projection taken from ``mc_tif_path`` itself. Writes
    ``output_dir/centroids.json`` and returns a :class:`CentroidResult`; a prior
    run with identical resolved parameters and schema is reused as-is.
    """
    from roigbiv.pipeline.calibration import load_calibration
    from roigbiv.pipeline.stage1 import run_cellpose_detection

    mc_tif_path = Path(mc_tif_path)
    output_dir = Path(output_dir)
    stem = mc_tif_path.stem.replace("_mc", "")
    output_path = output_dir / "centroids.json"

    calib = load_calibration(output_dir)
    params = _resolved_params(cfg, calib)

    if output_path.exists():
        try:
            prior = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            prior = None
        if (prior and prior.get("params") == params
                and prior.get("schema") == _SCHEMA):
            return CentroidResult(output_path=output_path,
                                  count=len(prior.get("centroids", [])))

    substrate = _load_substrate(output_dir, mc_tif_path)
    if substrate.source == "mean_projection":
        _persist_substrate(output_dir, substrate.morph)

    overrides = {
        "cellprob_threshold": params["cellprob_threshold"],
        # Single-channel (grayscale) detection. Cellpose's channel-2 slot means
        # "nuclear stain"; Stage 1 fills it with vcorr_S/max_S, which works on
        # bright cranial-window FOVs but collapses here — on the reference prism
        # FOV vcorr_S is essentially noise (std 0.03), and feeding it as ch2 took
        # detection from 8 somata to 0-1. Stage 1's own 2-channel convention is
        # deliberately left untouched; this is centroid discovery's substrate.
        "channels": (0, 0),
        # Global rather than tiled normalization: a prism FOV's dark, cell-free
        # tiles get stretched to parity with tissue under per-tile norm.
        "tile_norm_blocksize": 0,
        # denoise_cyto3 is trained on conventional 2P and erases real structure
        # on shot-noise-dominated data (reference FOV: per-pixel temporal SD 350
        # at mean 155). Measured 8 somata without it, 5 with.
        "use_denoise": False,
    }
    if params["diameter_px"] is not None:
        # An explicit measurement beats the per-image estimator.
        overrides["diameter"] = int(round(params["diameter_px"]))
        overrides["diameter_auto"] = False
    if params["cellpose_model"]:
        overrides["cellpose_model"] = params["cellpose_model"]
    det_cfg = _with_overrides(cfg, **overrides)

    masks, probs, _labels, _cellprob = run_cellpose_detection(
        substrate.morph, substrate.ch2, det_cfg, max_S=substrate.max_S)

    activity = _suite2p_candidates(output_dir, stem)
    centroids = _centroids_from_masks(masks, probs, activity)
    n_detected = len(centroids)

    mask_meta: dict = {"applied": False}
    if centroids and params["tissue_mask"]:
        sigma = float(getattr(cfg, "centroid_tissue_mask_sigma", 8.0))
        mask, mask_meta = _tissue_mask(substrate.morph, sigma)
        height, width = mask.shape
        centroids = [
            c for c in centroids
            if 0 <= int(round(c["y"])) < height
            and 0 <= int(round(c["x"])) < width
            and mask[int(round(c["y"])), int(round(c["x"]))]
        ]

    output_path.write_text(json.dumps({
        "stem": stem,
        "schema": _SCHEMA,
        "source": "cellpose",
        "substrate_source": substrate.source,
        "generated_at": time.time(),
        "params": params,
        "n_detected": n_detected,
        "n_outside_tissue": n_detected - len(centroids),
        "tissue_mask": mask_meta,
        "activity_cross_check": {
            "available": activity is not None,
            "n_suite2p_candidates": 0 if activity is None else int(len(activity)),
            "radius_px": _ACTIVITY_SUPPORT_RADIUS_PX,
            "n_supported": sum(1 for c in centroids if c.get("activity_support")),
        },
        "centroids": centroids,
    }, indent=2))

    return CentroidResult(output_path=output_path, count=len(centroids))


def clear_centroid_output(output_dir: Path, stem: str) -> None:
    """Delete a FOV's ``centroids.json``.

    Recompute is normally keyed on the recorded parameters and schema, so this
    is only needed to force a re-detect under *unchanged* settings (e.g. after
    the underlying summary images themselves were replaced). Foundation's
    Suite2p output is left alone — centroid discovery only reads it, for the
    activity cross-check.
    """
    output_dir = Path(output_dir)
    centroids_path = output_dir / "centroids.json"
    if centroids_path.exists():
        centroids_path.unlink()
