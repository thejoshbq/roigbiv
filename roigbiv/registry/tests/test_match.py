from __future__ import annotations

import numpy as np

from roigbiv.registry.fingerprint import CellFeature
from roigbiv.registry.match import (
    AUTO_ACCEPT_THRESHOLD,
    match_cells,
    match_fov,
    phase_correlate,
)


def _plant_cells(img: np.ndarray, centroids: list[tuple[int, int]], radius: int = 4) -> None:
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    for (cy, cx) in centroids:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        img[mask] += 1.0


def _make_fov(centroids, H=128, W=128, noise_scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(scale=noise_scale, size=(H, W)).astype(np.float32)
    _plant_cells(img, centroids)
    cells = [
        CellFeature(
            local_label_id=i + 1,
            centroid_y=float(cy),
            centroid_x=float(cx),
            area=50,
            solidity=0.9,
            eccentricity=0.2,
        )
        for i, (cy, cx) in enumerate(centroids)
    ]
    return img, cells


def test_phase_correlate_recovers_translation():
    centroids = [(30, 40), (60, 70), (90, 50), (40, 100)]
    img_a, _ = _make_fov(centroids)
    shifted_centroids = [(cy + 3, cx - 5) for cy, cx in centroids]
    img_b, _ = _make_fov(shifted_centroids, seed=0)
    (dy, dx), peak = phase_correlate(img_a, img_b)
    assert abs(dy - 3) < 1.0
    assert abs(dx - (-5)) < 1.0
    assert peak > 0.5


def test_match_cells_identifies_all_pairs_when_aligned():
    cells_a = [
        CellFeature(local_label_id=i + 1, centroid_y=float(cy), centroid_x=float(cx),
                    area=50, solidity=0.9, eccentricity=0.2)
        for i, (cy, cx) in enumerate([(30, 40), (60, 70), (90, 50)])
    ]
    cells_b = [
        CellFeature(local_label_id=100 + i, centroid_y=float(cy), centroid_x=float(cx),
                    area=50, solidity=0.9, eccentricity=0.2)
        for i, (cy, cx) in enumerate([(30, 40), (60, 70), (90, 50)])
    ]
    result = match_cells(cells_a, cells_b)
    assert len(result.matches) == 3
    assert not result.new_query_labels
    assert not result.missing_candidate_labels


def test_match_cells_detects_new_and_missing():
    cells_a = [
        CellFeature(local_label_id=1, centroid_y=30.0, centroid_x=40.0,
                    area=50, solidity=0.9, eccentricity=0.2),
        CellFeature(local_label_id=2, centroid_y=60.0, centroid_x=70.0,
                    area=50, solidity=0.9, eccentricity=0.2),
        CellFeature(local_label_id=3, centroid_y=90.0, centroid_x=50.0,
                    area=50, solidity=0.9, eccentricity=0.2),
    ]
    cells_b = [
        CellFeature(local_label_id=11, centroid_y=30.2, centroid_x=40.1,
                    area=50, solidity=0.9, eccentricity=0.2),
        CellFeature(local_label_id=12, centroid_y=60.1, centroid_x=70.0,
                    area=50, solidity=0.9, eccentricity=0.2),
        CellFeature(local_label_id=13, centroid_y=10.0, centroid_x=10.0,
                    area=50, solidity=0.9, eccentricity=0.2),
    ]
    result = match_cells(cells_a, cells_b)
    assert len(result.matches) == 2
    assert result.new_query_labels == [3]
    assert result.missing_candidate_labels == [13]


def test_match_fov_auto_accepts_identical_scene():
    centroids = [(30, 40), (60, 70), (90, 50), (40, 100)]
    img, cells = _make_fov(centroids)
    result = match_fov(img, cells, img, cells)
    assert result.fov_sim >= AUTO_ACCEPT_THRESHOLD
    assert result.decision == "auto_match"


def test_match_fov_rejects_unrelated_scenes():
    img_a, cells_a = _make_fov([(20, 20), (40, 80), (100, 30)])
    img_b, cells_b = _make_fov([(10, 110), (110, 10), (70, 90)], seed=7)
    result = match_fov(img_a, cells_a, img_b, cells_b)
    assert result.decision in ("reject", "review")
