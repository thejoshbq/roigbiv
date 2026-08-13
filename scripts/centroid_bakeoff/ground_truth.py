"""Ground-truth centroid loaders — real (ImageJ RoiSet) and synthetic (injection).

Real-FOV discovery/parsing is adapted from
``scripts/process_external_data.py``'s ``discover_pairs``/``roi_zip_to_mask``
(which already solves this exact problem for a similar external-data ingestion
job), extended to prefer a ``*_RoiSet_FINAL.zip`` over ``*_RoiSet.zip`` when
both exist.

Centroid convention: rasterize each ROI polygon (``skimage.draw.polygon``,
identical to ``roi_zip_to_mask``) into a labeled mask, then
``scipy.ndimage.center_of_mass`` per label — the same convention
``roigbiv.pipeline.roi_stamp`` uses for every detector-native mask in the main
pipeline (see docs/adr/0003-centroid-canonical-roi-stamps.md), so real-GT
centroids and detector-predicted centroids are computed the same way.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import roifile
from scipy.ndimage import center_of_mass
from skimage.draw import polygon as _draw_polygon

log = logging.getLogger("centroid_bakeoff.ground_truth")


# ---------------------------------------------------------------------------
# Real ground truth — ImageJ RoiSet.zip
# ---------------------------------------------------------------------------

def discover_real_pairs(
    search_roots: list[Path],
    exclude_patterns: Optional[list[str]] = None,
) -> list[tuple[Path, Path, str]]:
    """Find ``*_mc.tif`` + matching RoiSet pairs under *search_roots*.

    Prefers ``<base>_RoiSet_FINAL.zip`` over ``<base>_RoiSet.zip`` when both
    exist for the same ``*_mc.tif``. Returns ``(mc_tif_path, roi_zip_path, mc_stem)``.
    """
    exclude = exclude_patterns or []
    seen_stems: set[str] = set()
    pairs: list[tuple[Path, Path, str]] = []
    unpaired: list[Path] = []

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            log.warning("search root does not exist: %s", root)
            continue

        for mc_tif in sorted(root.rglob("*_mc.tif")):
            if any(ex in str(mc_tif) for ex in exclude):
                continue

            mc_stem = mc_tif.stem
            if mc_stem in seen_stems:
                continue
            seen_stems.add(mc_stem)

            base = mc_stem.removesuffix("_mc")
            roi_final = mc_tif.parent / f"{base}_RoiSet_FINAL.zip"
            roi_plain = mc_tif.parent / f"{base}_RoiSet.zip"
            roi_zip = roi_final if roi_final.exists() else roi_plain

            if not roi_zip.exists():
                unpaired.append(mc_tif)
                continue

            pairs.append((mc_tif, roi_zip, mc_stem))

    if unpaired:
        log.warning("%d _mc.tif file(s) without a matching RoiSet: %s",
                     len(unpaired), [p.name for p in unpaired])

    return pairs


def _polygon_roi_centroid(coords_xy: np.ndarray, shape: tuple[int, int]) -> Optional[tuple[float, float]]:
    """Rasterize one ROI's polygon and return its (y, x) center_of_mass.

    Mirrors ``process_external_data.py::roi_zip_to_mask``'s rasterization
    exactly (``skimage.draw.polygon`` on the same coordinate order), so real-GT
    centroids use the same pixel-level definition the rest of the pipeline
    already relies on. Returns ``None`` for degenerate input (<3 vertices).
    """
    if coords_xy is None or len(coords_xy) < 3:
        return None
    H, W = shape
    rr, cc = _draw_polygon(coords_xy[:, 1], coords_xy[:, 0], shape=(H, W))
    if rr.size == 0:
        return None
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    cy, cx = center_of_mass(mask)
    return float(cy), float(cx)


def imagej_roiset_to_centroids(
    roi_zip_path: Path, shape: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Parse an ImageJ RoiSet .zip into (N, 2) (y, x) centroids + ROI names.

    POINT-type ROIs (including ImageJ multi-point selections, which store
    several coordinates under one ROI object) emit one centroid per coordinate
    directly — no rasterization needed, the point *is* the centroid.
    POLYGON/FREEHAND/OVAL/TRACED ROIs are rasterized then center-of-mass'd
    (see ``_polygon_roi_centroid``). Verified against the real TDT4_ENSURESA
    data: all 5 FOVs' RoiSets are 100% POLYGON-type freehand somas, but the
    POINT branch is kept for robustness against other RoiSets.
    """
    rois = roifile.roiread(str(roi_zip_path))
    if not isinstance(rois, list):
        rois = [rois]

    centroids: list[tuple[float, float]] = []
    names: list[str] = []
    for idx, roi in enumerate(rois):
        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            continue
        if roi.roitype == roifile.ROI_TYPE.POINT:
            for x, y in coords:
                centroids.append((float(y), float(x)))
                names.append(roi.name or f"pt{idx}")
        else:
            c = _polygon_roi_centroid(coords, shape)
            if c is not None:
                centroids.append(c)
                names.append(roi.name or f"roi{idx}")

    arr = np.asarray(centroids, dtype=np.float32).reshape(-1, 2)
    return arr, names


# ---------------------------------------------------------------------------
# Synthetic ground truth — soma injection
# ---------------------------------------------------------------------------

def _layout_specs(H: int, W: int, rng: np.random.Generator, margin: int = 40):
    """A grid of somas spanning every soma_type, plus one crowded overlapping
    pair, laid out with even spacing so the injected FOV isn't pathologically
    dense. Mirrors the soma_type vocabulary in
    ``roigbiv.benchmark.synthetic._DEFAULT_SNR_BANDS``.
    """
    from roigbiv.benchmark.synthetic import SomaSpec, overlapping_pair

    soma_types = ["dim", "sparse_transient", "slow_modulation", "elevated_baseline"]
    specs: list = []
    label_id = 1

    n_cols = 4
    n_rows = 3
    ys = np.linspace(margin, H - margin, n_rows)
    xs = np.linspace(margin, W - margin, n_cols)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            soma_type = soma_types[(i * n_cols + j) % len(soma_types)]
            jitter = rng.uniform(-3, 3, size=2)
            cy = int(round(float(y) + jitter[0]))
            cx = int(round(float(x) + jitter[1]))
            specs.append(SomaSpec(soma_type=soma_type, center=(cy, cx),
                                   radius=6.0, label_id=label_id))
            label_id += 1

    # One crowded overlapping pair, offset from the grid, to stress-test
    # matching/detection near the soma_scale-derived min-separation.
    # label_id left at its 0 default -- inject_somas auto-assigns from
    # max(existing label_id) + 1, so these land at label_id 13/14.
    specs.extend(overlapping_pair((H // 2, W // 2), offset=(5, 5), radius=6.0))

    return specs


def build_synthetic_fov(
    shape: tuple[int, int, int] = (300, 512, 512),
    fs: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Gaussian-noise background + injected somas across soma_types (incl. a
    crowded pair). Returns ``(movie, gt_centroids (N,2) y/x, specs)``.

    The Gaussian-noise background matches
    ``roigbiv/benchmark/tests/test_synthetic.py``'s own background convention
    exactly, deliberately avoiding a real ``_mc.tif`` as substrate — reusing
    real footage would contaminate precision/recall with real, un-annotated
    neurons sitting underneath the injected somas.
    """
    from roigbiv.benchmark.synthetic import inject_somas

    T, H, W = shape
    rng = np.random.default_rng(seed)
    movie = rng.normal(scale=0.5, size=(T, H, W)).astype(np.float32)

    specs = _layout_specs(H, W, rng)
    result = inject_somas(movie, specs, fs=fs, seed=seed)
    gt = np.asarray([s.center for s in result.specs], dtype=np.float32)
    return result.movie, gt, result.specs
