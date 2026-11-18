"""A small synthetic benchmark in the v0 manifest format, for tests only.

Four FOVs on a 64 x 64 image. Every ROI is an 8-pixel square polygon with
integer corners, which ``skimage.draw.polygon`` rasterizes to 81 pixels with
its centre of mass at ``(y0 + 4, x0 + 4)``: hand-checkable centroids and
areas. No lab data.

- ``syn-a1``: animal A1, dev, core, GRIN, 4 ROIs, 12-frame TIFF movie
- ``syn-a2``: animal A2, dev, core, GRIN, two raters (r1: 4 ROIs; r2: 3 shared + 1 other)
- ``syn-a3``: animal A3, test, core, prism, 3 ROIs, 10-frame folder movie
- ``syn-a4``: animal A4, dev, flagged, GRIN, 1 ROI, no movie
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import roifile
import tifffile

SHAPE = (64, 64)
SIDE = 8
ROI_AREA = 81  # (SIDE + 1) ** 2 pixels for a square with integer corners

# Top-left corners (y0, x0) per FOV and rater.
CORNERS = {
    "syn-a1": [(8, 8), (8, 40), (40, 8), (40, 40)],
    "syn-a2/r1": [(8, 8), (8, 40), (40, 8), (40, 40)],
    "syn-a2/r2": [(8, 8), (8, 40), (40, 8), (20, 24)],
    "syn-a3": [(10, 10), (10, 30), (30, 20)],
    "syn-a4": [(16, 16)],
}


def centers(key: str) -> np.ndarray:
    """``(N, 2)`` float32 ``(y, x)`` centres for a ``CORNERS`` key."""
    return np.asarray([(y0 + SIDE / 2, x0 + SIDE / 2) for y0, x0 in CORNERS[key]], dtype=np.float32)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def square_roi(y0: int, x0: int, name: str) -> roifile.ImagejRoi:
    xs = [x0, x0 + SIDE, x0 + SIDE, x0]
    ys = [y0, y0, y0 + SIDE, y0 + SIDE]
    roi = roifile.ImagejRoi.frompoints(np.stack([xs, ys], axis=1).astype(float))
    roi.roitype = roifile.ROI_TYPE.POLYGON
    roi.name = name
    return roi


def write_roiset(path: Path, corners: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roifile.roiwrite(str(path), [square_roi(y0, x0, f"r{i}") for i, (y0, x0) in enumerate(corners)])


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(100, 200, size=SHAPE, dtype=np.uint16)


def write_synthetic_benchmark(root: Path) -> Path:
    """Write the benchmark under ``root`` and return ``root``."""
    root = Path(root)
    (root / "fovs").mkdir(parents=True, exist_ok=True)

    def fov_entry(fov_id, animal, split, tier, lens, annotations, movie):
        d = root / "fovs" / fov_id
        d.mkdir(parents=True, exist_ok=True)
        std = d / "STD.tif"
        tifffile.imwrite(str(std), _image(len(fov_id) + ord(fov_id[-1])))
        anns = []
        for i, (rater, key) in enumerate(annotations):
            zp = d / f"annotation_{'ab'[i]}.zip"
            write_roiset(zp, CORNERS[key])
            anns.append({
                "path": str(zp.relative_to(root)), "source": f"synthetic/{key}",
                "sha256": sha256_file(zp), "rater": rater, "n_rois": len(CORNERS[key]),
                "n_centroids": len(CORNERS[key]), "n_rasterized": len(CORNERS[key]), "contrast": 1.0,
            })
        return {
            "id": fov_id, "animal": animal, "split_group": animal, "tier": tier, "lens": lens,
            "region": "synthetic", "cell_type": "neuron", "fs_hz": None, "averaging": None,
            "notes": [], "images": {"std": {"path": str(std.relative_to(root)), "source": "synthetic",
                                            "sha256": sha256_file(std), "derived": False}},
            "annotations": anns, "movie": movie, "source_root": "synthetic",
            "image_shape": list(SHAPE), "split": split,
        }

    # syn-a1 movie: one multi-page TIFF.
    m1 = root / "fovs" / "syn-a1" / "movie.tif"
    m1.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(m1), np.stack([_image(100 + t) for t in range(12)]))
    movie1 = {"kind": "tif", "frames_read": 12, "header_frames": None, "bad_frames": 0,
              "frame_shape": list(SHAPE), "path": str(m1.relative_to(root)),
              "source": "synthetic", "sha256": sha256_file(m1)}

    # syn-a3 movie: a folder of one-frame TIFFs.
    m3 = root / "fovs" / "syn-a3" / "movie_frames"
    m3.mkdir(parents=True, exist_ok=True)
    for t in range(10):
        tifffile.imwrite(str(m3 / f"frame_{t + 1:06d}.tif"), _image(200 + t))
    movie3 = {"kind": "frame_dir", "files": 10, "frames_read": 10, "bad_frames": 0,
              "frame_shape": list(SHAPE), "path": str(m3.relative_to(root)), "source": "synthetic"}

    fovs = [
        fov_entry("syn-a1", "A1", "dev", "core", "GRIN", [(None, "syn-a1")], movie1),
        fov_entry("syn-a2", "A2", "dev", "core", "GRIN", [("r1", "syn-a2/r1"), ("r2", "syn-a2/r2")], None),
        fov_entry("syn-a3", "A3", "test", "core", "prism", [(None, "syn-a3")], movie3),
        fov_entry("syn-a4", "A4", "dev", "flagged", "GRIN", [(None, "syn-a4")], None),
    ]
    manifest = {
        "benchmark": "roigbiv-bench-synthetic", "version": "syn0", "frozen": "2026-11-12",
        "regime": {"lens": "mixed"}, "source": "synthetic", "summary": {}, "g0": {}, "split": {},
        "excluded": [], "fovs": fovs,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return root
