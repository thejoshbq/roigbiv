"""Tests for the v3 footprint-derived fingerprint."""
from __future__ import annotations

import numpy as np
import pytest

from roigbiv.registry.fingerprint import (
    FINGERPRINT_VERSION,
    compute_fingerprint,
    deserialize_centroids,
    deserialize_mean_m,
    deserialize_merged_masks,
)


def _mask_image(H: int, W: int, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint16)
    yy, xx = np.ogrid[:H, :W]
    for lbl, (cy, cx) in enumerate(centers, start=1):
        out[((yy - cy) ** 2 + (xx - cx) ** 2) <= 4] = lbl
    return out


def _mean_m(shape):
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


def test_fingerprint_version_is_3():
    assert FINGERPRINT_VERSION == 3


def test_fingerprint_deterministic_across_calls():
    mask = _mask_image(16, 16, [(4, 4), (10, 10)])
    mean_m = _mean_m((16, 16))
    a = compute_fingerprint(mask, mean_m)
    b = compute_fingerprint(mask, mean_m)
    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.fingerprint_version == 3


def test_fingerprint_hash_ignores_mean_projection():
    mask = _mask_image(16, 16, [(4, 4), (10, 10)])
    a = compute_fingerprint(mask, _mean_m((16, 16)))
    b = compute_fingerprint(mask, _mean_m((16, 16)) + 100.0)
    assert a.fingerprint_hash == b.fingerprint_hash


def test_fingerprint_hash_changes_with_centroid_shift():
    mask_a = _mask_image(16, 16, [(4, 4), (10, 10)])
    mask_b = _mask_image(16, 16, [(5, 4), (10, 10)])  # shifted one pixel
    mean_m = _mean_m((16, 16))
    a = compute_fingerprint(mask_a, mean_m)
    b = compute_fingerprint(mask_b, mean_m)
    assert a.fingerprint_hash != b.fingerprint_hash


def test_fingerprint_blob_roundtrip():
    mask = _mask_image(12, 12, [(3, 3), (8, 9)])
    mean_m = _mean_m((12, 12))
    fp = compute_fingerprint(mask, mean_m)

    np.testing.assert_array_equal(deserialize_merged_masks(fp.merged_masks_blob), mask)
    np.testing.assert_allclose(deserialize_mean_m(fp.mean_m_blob), mean_m)
    cent = deserialize_centroids(fp.centroids_blob)
    assert cent.shape == (2, 2)
    np.testing.assert_array_equal(cent, fp.centroids)


def test_fingerprint_empty_mask():
    mask = np.zeros((8, 8), dtype=np.uint16)
    mean_m = np.zeros((8, 8), dtype=np.float32)
    fp = compute_fingerprint(mask, mean_m)
    assert fp.label_ids.shape == (0,)
    assert fp.centroids.shape == (0, 2)
    assert fp.areas.shape == (0,)
    assert isinstance(fp.fingerprint_hash, str)
    assert len(fp.fingerprint_hash) == 64


def test_fingerprint_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        compute_fingerprint(
            np.zeros((8, 8), dtype=np.uint16),
            np.zeros((10, 10), dtype=np.float32),
        )
