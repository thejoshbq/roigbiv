"""Unit tests for :mod:`roigbiv.registry.roicat_adapter`.

Pure-numpy helpers (``footprints_from_merged_masks``, ``centroids_from_merged_masks``,
``load_session_input``, single-session ``cluster_sessions`` degenerate path) are
tested unconditionally — they don't touch ROICaT.

The full ROICaT pipeline test (multi-session ``cluster_sessions``) is gated
behind a ``skipif`` marker that checks whether the ~75 MB ROInet weights are
already cached locally; if not, the test is skipped so CI stays fast. Run
locally with the weights present to exercise the end-to-end path.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from roigbiv.registry.roicat_adapter import (
    AdapterConfig,
    ClusterResult,
    SessionInput,
    centroids_from_merged_masks,
    cluster_sessions,
    footprints_from_merged_masks,
    load_session_input,
)


# ── Synthetic fixture helpers ──────────────────────────────────────────────


def _disk_mask(H: int, W: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _make_label_image(
    shape: tuple[int, int], centers: list[tuple[int, int]], radius: int = 3
) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint16)
    for label_id, (cy, cx) in enumerate(centers, start=1):
        out[_disk_mask(H, W, cy, cx, radius)] = label_id
    return out


def _make_mean_m(
    shape: tuple[int, int], centers: list[tuple[int, int]], sigma: float = 2.0
) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    img = np.zeros((H, W), dtype=np.float32)
    for cy, cx in centers:
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    return img


# ── footprints_from_merged_masks ───────────────────────────────────────────


def test_footprints_from_merged_masks_basic_shape_and_order():
    mask = _make_label_image((16, 16), [(4, 4), (10, 10), (4, 10)], radius=2)
    footprints, label_ids = footprints_from_merged_masks(mask)

    assert isinstance(footprints, sparse.csr_matrix)
    assert footprints.shape == (3, 16 * 16)
    # label_ids sorted ascending, matching row order
    assert label_ids.tolist() == [1, 2, 3]
    # Row k is the binary mask of label k+1 flattened (C order)
    for row_idx, lbl in enumerate(label_ids.tolist()):
        expected = (mask == lbl).astype(np.float32).ravel()
        got = footprints.getrow(row_idx).toarray().ravel()
        np.testing.assert_array_equal(got, expected)


def test_footprints_from_merged_masks_empty():
    mask = np.zeros((8, 8), dtype=np.uint16)
    footprints, label_ids = footprints_from_merged_masks(mask)
    assert footprints.shape == (0, 64)
    assert label_ids.shape == (0,)


def test_footprints_from_merged_masks_rejects_3d():
    with pytest.raises(ValueError, match="2-D"):
        footprints_from_merged_masks(np.zeros((2, 4, 4), dtype=np.uint16))


def test_footprints_label_ids_preserved_with_gaps():
    # Labels aren't required to be contiguous — parser must preserve the original ids.
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[_disk_mask(8, 8, 2, 2, 1)] = 5
    mask[_disk_mask(8, 8, 5, 5, 1)] = 42
    _, label_ids = footprints_from_merged_masks(mask)
    assert label_ids.tolist() == [5, 42]


# ── centroids_from_merged_masks ────────────────────────────────────────────


def test_centroids_from_merged_masks_matches_disk_centers():
    centers = [(4, 4), (10, 10)]
    mask = _make_label_image((16, 16), centers, radius=2)
    centroids = centroids_from_merged_masks(mask)
    assert centroids.shape == (2, 2)
    for computed, (cy, cx) in zip(centroids.tolist(), centers):
        assert abs(computed[0] - cy) <= 1
        assert abs(computed[1] - cx) <= 1


def test_centroids_empty_mask():
    centroids = centroids_from_merged_masks(np.zeros((4, 4), dtype=np.uint16))
    assert centroids.shape == (0, 2)


# ── load_session_input ─────────────────────────────────────────────────────


def test_load_session_input_roundtrip(tmp_path: Path):
    import tifffile

    session_dir = tmp_path / "T1_230101_PrL-NAc_FOV1_PRE-000"
    (session_dir / "summary").mkdir(parents=True)
    mask = _make_label_image((12, 12), [(4, 4), (8, 8)], radius=1)
    mean_m = _make_mean_m((12, 12), [(4, 4), (8, 8)])
    tifffile.imwrite(str(session_dir / "merged_masks.tif"), mask)
    tifffile.imwrite(str(session_dir / "summary" / "mean_M.tif"), mean_m)

    session = load_session_input(session_dir)
    assert session.session_key == session_dir.name
    assert session.mean_m.shape == (12, 12)
    assert session.merged_masks.shape == (12, 12)
    assert session.merged_masks.dtype == np.uint16
    np.testing.assert_array_equal(session.merged_masks, mask)


def test_load_session_input_missing_files_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="merged_masks"):
        load_session_input(tmp_path)


# ── cluster_sessions: degenerate single-session path ───────────────────────


def test_cluster_sessions_single_session_degenerate():
    mask = _make_label_image((16, 16), [(4, 4), (10, 10), (6, 12)], radius=2)
    mean_m = _make_mean_m((16, 16), [(4, 4), (10, 10), (6, 12)])
    session = SessionInput(session_key="s0", mean_m=mean_m, merged_masks=mask)

    result = cluster_sessions([session])

    assert isinstance(result, ClusterResult)
    assert result.per_session_roi_count == [3]
    assert result.labels.shape == (3,)
    assert result.labels.tolist() == [0, 1, 2]
    assert result.session_bool.shape == (3, 1)
    assert result.session_bool.all()
    assert result.alignment_method == "none"
    assert result.alignment_inlier_rate == 1.0
    assert result.fov_height == 16
    assert result.fov_width == 16
    assert result.per_session_label_ids[0].tolist() == [1, 2, 3]


def test_cluster_sessions_single_session_empty_mask():
    mask = np.zeros((8, 8), dtype=np.uint16)
    mean_m = np.zeros((8, 8), dtype=np.float32)
    session = SessionInput(session_key="empty", mean_m=mean_m, merged_masks=mask)
    result = cluster_sessions([session])
    assert result.labels.shape == (0,)
    assert result.session_bool.shape == (0, 1)
    assert result.per_session_roi_count == [0]


def test_cluster_sessions_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        cluster_sessions([])


def test_cluster_sessions_pads_mismatched_sizes_in_single_session_path():
    # Two sessions of different sizes — but total_rois == 0 forces the degenerate
    # path, which verifies the padding bookkeeping without invoking ROICaT.
    s0 = SessionInput(
        session_key="s0",
        mean_m=np.zeros((8, 8), dtype=np.float32),
        merged_masks=np.zeros((8, 8), dtype=np.uint16),
    )
    s1 = SessionInput(
        session_key="s1",
        mean_m=np.zeros((16, 20), dtype=np.float32),
        merged_masks=np.zeros((16, 20), dtype=np.uint16),
    )
    result = cluster_sessions([s0, s1])
    assert result.fov_height == 16
    assert result.fov_width == 20
    assert result.per_session_roi_count == [0, 0]


# ── cluster_sessions: full ROICaT path (gated on cached weights) ───────────


def _roinet_weights_cached() -> bool:
    override = os.environ.get("ROIGBIV_ROINET_CACHE")
    cache = Path(override) if override else Path.home() / ".cache" / "roigbiv" / "roinet"
    if not cache.exists():
        return False
    # Presence of any .pth file inside the cache dir tree indicates prior download.
    for _ in cache.rglob("*.pth"):
        return True
    return False


@pytest.mark.skipif(
    not _roinet_weights_cached(),
    reason="ROInet weights not cached locally — end-to-end ROICaT path skipped in CI",
)
def test_cluster_sessions_two_sessions_shared_cells():
    """Two synthetic sessions with 3 shared cell positions + 1 extra each.

    After ROICaT clustering, at least 2 of the 3 shared positions should land
    in clusters that span both sessions (label appears in both session rows of
    session_bool within the same cluster).
    """
    shape = (64, 64)
    shared = [(12, 12), (32, 32), (50, 20)]
    extra_s0 = [(22, 50)]
    extra_s1 = [(45, 50)]

    mask_s0 = _make_label_image(shape, shared + extra_s0, radius=3)
    mask_s1 = _make_label_image(shape, shared + extra_s1, radius=3)
    mean_s0 = _make_mean_m(shape, shared + extra_s0)
    mean_s1 = _make_mean_m(shape, shared + extra_s1)

    sessions = [
        SessionInput(session_key="s0", mean_m=mean_s0, merged_masks=mask_s0),
        SessionInput(session_key="s1", mean_m=mean_s1, merged_masks=mask_s1),
    ]
    # Lenient d_cutoff: synthetic 8-ROI input is too small for ROICaT's
    # same/diff distribution crossover inference; bypass it here. Real data
    # with hundreds of ROIs (Phase 4) uses the default d_cutoff=None.
    cfg = AdapterConfig(
        alignment_method="PhaseCorrelation", d_cutoff=1.0, verbose=False
    )
    result = cluster_sessions(sessions, cfg)

    assert result.per_session_roi_count == [4, 4]
    assert result.session_bool.shape == (8, 2)
    assert result.labels.shape == (8,)

    # Find clusters containing ROIs from both sessions.
    shared_clusters = 0
    for cluster_label in np.unique(result.labels):
        if cluster_label == -1:
            continue
        members = result.labels == cluster_label
        s0_present = (members & result.session_bool[:, 0]).any()
        s1_present = (members & result.session_bool[:, 1]).any()
        if s0_present and s1_present:
            shared_clusters += 1

    assert shared_clusters >= 2, (
        f"Expected ≥2 cross-session clusters on synthetic overlapping cells, "
        f"got {shared_clusters}. Labels: {result.labels.tolist()}"
    )
