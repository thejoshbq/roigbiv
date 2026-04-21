from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from roigbiv.registry.blob.local import LocalBlobStore
from roigbiv.registry.fingerprint import CellFeature
from roigbiv.registry.match import AUTO_ACCEPT_THRESHOLD
from roigbiv.registry.orchestrator import register_or_match
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _plant_cells(img: np.ndarray, centroids, radius: int = 4) -> None:
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    for (cy, cx) in centroids:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        img[mask] += 1.0


def _make_fov(centroids, H=128, W=128, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(scale=0.02, size=(H, W)).astype(np.float32)
    _plant_cells(img, centroids)
    cells = [
        CellFeature(
            local_label_id=i + 1,
            centroid_y=float(cy),
            centroid_x=float(cx),
            area=50, solidity=0.9, eccentricity=0.2,
        )
        for i, (cy, cx) in enumerate(centroids)
    ]
    return img, cells


def _build_backends(tmp_path: Path):
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    blob = LocalBlobStore(root=tmp_path / "blobs")
    return store, blob


def test_first_run_mints_new_fov(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    img, cells = _make_fov([(30, 40), (60, 70), (90, 50)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
        mean_m=img,
        cells=cells,
        output_dir=out_dir,
        store=store,
        blob_store=blob,
    )
    assert report["decision"] == "new_fov"
    assert len(store.list_fovs()) == 1
    fov = store.list_fovs()[0]
    assert fov.animal_id == "PrL-NAc-G6-5M_HI-D1"
    assert fov.region == "PrL-NAc"
    assert fov.latest_session_date == date(2022, 12, 9)
    assert len(store.list_cells(fov.fov_id)) == 3


def test_second_run_same_input_hash_matches(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    img, cells = _make_fov([(30, 40), (60, 70), (90, 50)])

    out1 = tmp_path / "out1"; out1.mkdir()
    r1 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        mean_m=img, cells=cells, output_dir=out1,
        store=store, blob_store=blob,
    )
    out2 = tmp_path / "out2"; out2.mkdir()
    r2 = register_or_match(
        fov_stem="T1_221215_PrL-NAc-G6-5M_HI-D1_FOV1",
        mean_m=img, cells=cells, output_dir=out2,
        store=store, blob_store=blob,
    )
    assert r2["decision"] == "hash_match"
    assert r2["fov_id"] == r1["fov_id"]
    assert len(store.list_fovs()) == 1
    sessions = store.list_sessions(r1["fov_id"])
    assert len(sessions) == 2


def test_second_run_with_dropped_cell_auto_matches(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    img, cells = _make_fov([(30, 40), (60, 70), (90, 50), (40, 100)])

    out1 = tmp_path / "out1"; out1.mkdir()
    r1 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        mean_m=img, cells=cells, output_dir=out1,
        store=store, blob_store=blob,
    )

    cells_day2 = cells[:-1]
    out2 = tmp_path / "out2"; out2.mkdir()
    r2 = register_or_match(
        fov_stem="T1_221215_PrL-NAc-G6-5M_HI-D1_FOV1",
        mean_m=img, cells=cells_day2, output_dir=out2,
        store=store, blob_store=blob,
    )
    assert r2["decision"] == "auto_match"
    assert r2["fov_id"] == r1["fov_id"]
    assert r2["n_matched"] == 3
    assert r2["n_missing"] == 1


def test_unrelated_fov_mints_new(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    img_a, cells_a = _make_fov([(20, 20), (40, 80), (100, 30)])
    img_b, cells_b = _make_fov([(10, 110), (110, 10), (70, 90)], seed=7)

    out1 = tmp_path / "out1"; out1.mkdir()
    register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        mean_m=img_a, cells=cells_a, output_dir=out1,
        store=store, blob_store=blob,
    )
    out2 = tmp_path / "out2"; out2.mkdir()
    r2 = register_or_match(
        fov_stem="T1_221210_PrL-NAc-G6-5M_HI-D1_FOV2",
        mean_m=img_b, cells=cells_b, output_dir=out2,
        store=store, blob_store=blob,
    )
    assert r2["decision"] in ("new_fov", "review")
