"""Boundary bake-off: GT rasterization and the seeded-subset metric.

The arms themselves are the production code paths and are covered by
``roigbiv/pipeline/tests/``; what is unique to this script is turning RoiSets
into label images and deciding which GT cells a seeded arm could have helped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.boundary_bakeoff.ground_truth import (  # noqa: E402
    imagej_roiset_to_labels,
    label_centroids,
)
from scripts.boundary_bakeoff.score import (  # noqa: E402
    score_arm,
    seeded_gt_labels,
)

H = W = 64


class _Arm:
    def __init__(self, labels):
        self.name = "test"
        self.labels = labels
        self.n_disk_fallback = 0
        self.n_orphan_basin_px = 0


def _square(labels, label, y, x, r):
    labels[y - r:y + r + 1, x - r:x + r + 1] = label
    return labels


def _roiset(tmp_path: Path, polygons, name="RoiSet.zip") -> Path:
    """Write an ImageJ RoiSet of freehand polygons."""
    roifile = pytest.importorskip("roifile")
    path = tmp_path / name
    rois = []
    for i, (y0, y1, x0, x1) in enumerate(polygons):
        # roifile takes (x, y) points, matching ImageJ's own order — the same
        # order ground_truth.py unpacks when it rasterizes.
        coords = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        rois.append(roifile.ImagejRoi.frompoints(coords, name=f"cell{i}"))
    roifile.roiwrite(str(path), rois, mode="w")
    return path


# ── ground-truth rasterization ────────────────────────────────────────────


def test_roiset_becomes_one_label_per_roi(tmp_path):
    path = _roiset(tmp_path, [(10, 20, 10, 20), (40, 50, 40, 50)])

    labels, names = imagej_roiset_to_labels(path, (H, W))

    assert labels.dtype == np.uint16
    assert len(names) == 2
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[15, 15] == 1
    assert labels[45, 45] == 2


def test_label_centroids_land_inside_their_own_roi(tmp_path):
    path = _roiset(tmp_path, [(10, 20, 10, 20)])
    labels, _ = imagej_roiset_to_labels(path, (H, W))

    centroids = label_centroids(labels)

    cy, cx = centroids[1]
    assert 10 <= cy <= 20 and 10 <= cx <= 20
    assert labels[int(round(cy)), int(round(cx))] == 1


# ── the seeded subset ─────────────────────────────────────────────────────


def test_seeded_subset_is_containment_not_proximity():
    """A centroid just outside a GT cell is not evidence anyone marked it."""
    gt = _square(np.zeros((H, W), np.uint16), 1, 20, 20, 5)

    inside = seeded_gt_labels(gt, {1: (20.0, 20.0)})
    outside = seeded_gt_labels(gt, {1: (20.0, 30.0)})

    assert inside == {1}
    assert outside == set()


def test_out_of_bounds_seed_is_ignored():
    gt = _square(np.zeros((H, W), np.uint16), 1, 20, 20, 5)
    assert seeded_gt_labels(gt, {1: (-5.0, 500.0)}) == set()


def test_seeded_iou_ignores_cells_no_one_confirmed():
    """The headline mean IoU understates a change seeding could not reach."""
    gt = np.zeros((H, W), np.uint16)
    _square(gt, 1, 15, 15, 5)      # confirmed, predicted exactly
    _square(gt, 2, 45, 45, 5)      # never confirmed, predicted poorly

    pred = np.zeros((H, W), np.uint16)
    _square(pred, 1, 15, 15, 5)
    _square(pred, 2, 45, 45, 3)

    score = score_arm(_Arm(pred), gt, "fovA", {1: (15.0, 15.0)}, min_iou=0.3)

    assert score.n_gt == 2
    assert score.n_gt_seeded == 1
    assert score.mean_iou_seeded == pytest.approx(1.0)
    assert score.mean_iou < 1.0
    assert score.notes, "an unreachable-cell caveat must be recorded"


def test_score_reports_precision_recall_and_f1():
    gt = np.zeros((H, W), np.uint16)
    _square(gt, 1, 15, 15, 5)
    _square(gt, 2, 45, 45, 5)

    pred = np.zeros((H, W), np.uint16)
    _square(pred, 1, 15, 15, 5)     # tp
    _square(pred, 5, 45, 15, 5)     # fp; GT 2 is an fn

    score = score_arm(_Arm(pred), gt, "fovA", {}, min_iou=0.3)

    assert (score.n_tp, score.n_fp, score.n_fn) == (1, 1, 1)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(0.5)
    assert score.mean_iou_seeded is None, "no seeds means no seeded subset"


def test_empty_prediction_scores_zero_rather_than_raising():
    gt = _square(np.zeros((H, W), np.uint16), 1, 15, 15, 5)

    score = score_arm(_Arm(np.zeros((H, W), np.uint16)), gt, "fovA", {})

    assert score.n_tp == 0 and score.n_fn == 1
    assert score.mean_iou == 0.0
    assert score.precision is None
    assert score.f1 is None
