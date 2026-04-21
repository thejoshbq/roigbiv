from __future__ import annotations

import numpy as np

from roigbiv.registry.fingerprint import (
    CellFeature,
    compute_fingerprint,
    deserialize_cells,
    deserialize_mean_m,
    downsample,
)


def _synthetic_cells(n: int = 5) -> list[CellFeature]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        out.append(CellFeature(
            local_label_id=i + 1,
            centroid_y=float(rng.uniform(10, 200)),
            centroid_x=float(rng.uniform(10, 200)),
            area=int(rng.integers(40, 200)),
            solidity=float(rng.uniform(0.7, 1.0)),
            eccentricity=float(rng.uniform(0.0, 0.9)),
            nuclear_shadow_score=float(rng.uniform(0.0, 0.5)),
        ))
    return out


def test_fingerprint_is_deterministic():
    rng = np.random.default_rng(42)
    img = rng.normal(size=(256, 256)).astype(np.float32)
    cells = _synthetic_cells()
    a = compute_fingerprint(img, cells)
    b = compute_fingerprint(img, cells)
    assert a.fingerprint_hash == b.fingerprint_hash


def test_fingerprint_changes_on_image_edit():
    rng = np.random.default_rng(1)
    img = rng.normal(size=(256, 256)).astype(np.float32)
    cells = _synthetic_cells()
    a = compute_fingerprint(img, cells)
    img2 = img.copy()
    img2[0, 0] += 10_000.0
    b = compute_fingerprint(img2, cells)
    assert a.fingerprint_hash != b.fingerprint_hash


def test_fingerprint_changes_on_cell_drop():
    rng = np.random.default_rng(2)
    img = rng.normal(size=(128, 128)).astype(np.float32)
    cells = _synthetic_cells()
    a = compute_fingerprint(img, cells)
    b = compute_fingerprint(img, cells[:-1])
    assert a.fingerprint_hash != b.fingerprint_hash


def test_downsample_shape():
    img = np.random.rand(500, 500).astype(np.float32)
    ds = downsample(img)
    assert ds.shape == (64, 64)


def test_roundtrip_serialization():
    rng = np.random.default_rng(3)
    img = rng.normal(size=(96, 96)).astype(np.float32)
    cells = _synthetic_cells(7)
    fp = compute_fingerprint(img, cells)
    restored_img = deserialize_mean_m(fp.mean_m_blob)
    np.testing.assert_allclose(restored_img, img, rtol=1e-5)
    restored_cells = deserialize_cells(fp.centroid_blob)
    assert len(restored_cells) == len(cells)
    for a, b in zip(cells, restored_cells):
        assert a.local_label_id == b.local_label_id
        assert abs(a.centroid_y - b.centroid_y) < 1e-3
        assert abs(a.centroid_x - b.centroid_x) < 1e-3
