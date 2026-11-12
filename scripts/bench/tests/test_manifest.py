"""Accept tests for U1 (``bench.manifest``). Written by the contract; workers do not edit them."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import roifile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.manifest import (  # noqa: E402
    ENV_ROOT, ManifestError, abs_path, load_annotation, load_benchmark, resolve_root, select,
)
from bench.tests.synthetic import ROI_AREA, centers, write_synthetic_benchmark  # noqa: E402
from centroid_bakeoff.ground_truth import imagej_roiset_to_centroids  # noqa: E402


@pytest.fixture
def root(tmp_path):
    return write_synthetic_benchmark(tmp_path / "bench")


def _edit_manifest(root, fn):
    p = root / "manifest.json"
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


def test_load_fields(root):
    b = load_benchmark(root)
    assert b.name == "roigbiv-bench-synthetic" and b.version == "syn0"
    assert Path(b.root) == root
    assert [f.id for f in b.fovs] == ["syn-a1", "syn-a2", "syn-a3", "syn-a4"]
    a1, a2, a3, a4 = b.fovs
    assert a1.image_shape == (64, 64) and isinstance(a1.image_shape, tuple)
    assert a1.animal == "A1" and a1.split_group == "A1" and a1.split == "dev" and a1.tier == "core"
    assert a1.lens == "GRIN" and a3.lens == "prism" and a4.tier == "flagged"
    assert a1.images["std"].path == "fovs/syn-a1/STD.tif" and a1.images["std"].derived is False
    assert a1.movie.kind == "tif" and a1.movie.n_frames == 12 and len(a1.movie.sha256) == 64
    assert a3.movie.kind == "frame_dir" and a3.movie.n_frames == 10 and a3.movie.sha256 is None
    assert a4.movie is None
    assert [a.rater for a in a2.annotations] == ["r1", "r2"]
    assert [a.n_rois for a in a2.annotations] == [4, 4]
    assert isinstance(a1.annotations, tuple) and isinstance(a1.notes, tuple)
    assert a1.fs_hz is None and a1.averaging is None


def test_root_from_env(root, monkeypatch):
    monkeypatch.setenv(ENV_ROOT, str(root))
    assert Path(resolve_root()) == root
    assert load_benchmark().version == "syn0"


def test_explicit_root_wins_over_env(root, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_ROOT, str(tmp_path / "nowhere"))
    assert load_benchmark(root).version == "syn0"


def test_no_root(monkeypatch):
    monkeypatch.delenv(ENV_ROOT, raising=False)
    with pytest.raises(ManifestError):
        load_benchmark()


def test_root_without_manifest(tmp_path):
    with pytest.raises(ManifestError):
        load_benchmark(tmp_path)


def test_bad_split(root):
    _edit_manifest(root, lambda m: m["fovs"][0].update(split="train"))
    with pytest.raises(ManifestError, match=r"fovs\[0\]\.split"):
        load_benchmark(root)


def test_bad_tier(root):
    _edit_manifest(root, lambda m: m["fovs"][2].update(tier="gold"))
    with pytest.raises(ManifestError, match=r"fovs\[2\]\.tier"):
        load_benchmark(root)


def test_missing_fov_key(root):
    _edit_manifest(root, lambda m: m["fovs"][1].pop("lens"))
    with pytest.raises(ManifestError, match=r"fovs\[1\]\.lens"):
        load_benchmark(root)


def test_missing_top_key(root):
    _edit_manifest(root, lambda m: m.pop("version"))
    with pytest.raises(ManifestError, match="version"):
        load_benchmark(root)


def test_duplicate_id(root):
    _edit_manifest(root, lambda m: m["fovs"][1].update(id="syn-a1"))
    with pytest.raises(ManifestError):
        load_benchmark(root)


def test_bad_movie_kind(root):
    _edit_manifest(root, lambda m: m["fovs"][0]["movie"].update(kind="avi"))
    with pytest.raises(ManifestError):
        load_benchmark(root)


def test_empty_annotations(root):
    _edit_manifest(root, lambda m: m["fovs"][0].update(annotations=[]))
    with pytest.raises(ManifestError):
        load_benchmark(root)


@pytest.mark.parametrize("bad", ["../outside.tif", "/etc/passwd", "fovs/../../x.tif"])
def test_path_must_stay_inside_root(root, bad):
    _edit_manifest(root, lambda m: m["fovs"][0]["images"]["std"].update(path=bad))
    with pytest.raises(ManifestError):
        load_benchmark(root)


def test_select(root):
    b = load_benchmark(root)
    assert [f.id for f in select(b, split="dev")] == ["syn-a1", "syn-a2", "syn-a4"]
    assert [f.id for f in select(b, split="test")] == ["syn-a3"]
    assert [f.id for f in select(b, tier="core", lens="GRIN")] == ["syn-a1", "syn-a2"]
    assert [f.id for f in select(b)] == ["syn-a1", "syn-a2", "syn-a3", "syn-a4"]


def test_abs_path(root):
    b = load_benchmark(root)
    assert Path(abs_path(b, "fovs/syn-a1/STD.tif")) == root / "fovs/syn-a1/STD.tif"


def _manifest_sha(root):
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_acquisition_fills_nulls(root, tmp_path):
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps({
        "manifest_sha256": _manifest_sha(root),
        "movies": {"syn-a1": {"fs_hz": 7.5, "averaging": 4},
                   "syn-a3": {"fs_hz": None, "averaging": None}},
    }))
    b = load_benchmark(root, acquisition=acq)
    by_id = {f.id: f for f in b.fovs}
    assert by_id["syn-a1"].fs_hz == 7.5 and by_id["syn-a1"].averaging == 4
    assert by_id["syn-a3"].fs_hz is None and by_id["syn-a3"].averaging is None


def test_acquisition_without_sha_key(root, tmp_path):
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps({"movies": {"syn-a1": {"fs_hz": 7.5, "averaging": 4}}}))
    assert load_benchmark(root, acquisition=acq).fovs[0].fs_hz == 7.5


def test_acquisition_wrong_manifest_sha(root, tmp_path):
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps({"manifest_sha256": "0" * 64, "movies": {}}))
    with pytest.raises(ManifestError):
        load_benchmark(root, acquisition=acq)


def test_acquisition_unknown_fov(root, tmp_path):
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps({"movies": {"syn-zz": {"fs_hz": 7.5, "averaging": 4}}}))
    with pytest.raises(ManifestError):
        load_benchmark(root, acquisition=acq)


def test_acquisition_conflict_and_agreement(root, tmp_path):
    _edit_manifest(root, lambda m: m["fovs"][0].update(fs_hz=30.0))
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps({"movies": {"syn-a1": {"fs_hz": 7.5, "averaging": None}}}))
    with pytest.raises(ManifestError):
        load_benchmark(root, acquisition=acq)
    acq.write_text(json.dumps({"movies": {"syn-a1": {"fs_hz": 30.0, "averaging": 4}}}))
    f = load_benchmark(root, acquisition=acq).fovs[0]
    assert f.fs_hz == 30.0 and f.averaging == 4


def test_verify_hashes(root):
    load_benchmark(root, verify=True)
    std = root / "fovs/syn-a2/STD.tif"
    data = bytearray(std.read_bytes())
    data[-1] ^= 0xFF
    std.write_bytes(bytes(data))
    load_benchmark(root)  # no hashing without verify
    with pytest.raises(ManifestError):
        load_benchmark(root, verify=True)


def test_verify_missing_file(root):
    (root / "fovs/syn-a1/annotation_a.zip").unlink()
    with pytest.raises(ManifestError):
        load_benchmark(root, verify=True)


def test_load_annotation_polygons(root):
    b = load_benchmark(root)
    a1, a2 = b.fovs[0], b.fovs[1]
    pts = load_annotation(b, a1)
    assert pts.centroids.dtype == np.float32 and pts.centroids.shape == (4, 2)
    np.testing.assert_array_equal(pts.centroids, centers("syn-a1"))
    np.testing.assert_array_equal(pts.areas, np.full(4, float(ROI_AREA)))
    assert pts.rater is None and pts.path == "fovs/syn-a1/annotation_a.zip"

    r2 = load_annotation(b, a2, index=1)
    assert r2.rater == "r2"
    np.testing.assert_array_equal(r2.centroids, centers("syn-a2/r2"))
    ref, _ = imagej_roiset_to_centroids(root / a2.annotations[0].path, a2.image_shape)
    np.testing.assert_array_equal(load_annotation(b, a2, 0).centroids, ref)


def test_load_annotation_points_and_degenerate(root):
    b = load_benchmark(root)
    a1 = b.fovs[0]
    zp = root / a1.annotations[0].path
    point = roifile.ImagejRoi.frompoints(np.array([[5.0, 6.0], [7.0, 9.0]]))
    point.roitype = roifile.ROI_TYPE.POINT
    line = roifile.ImagejRoi.frompoints(np.array([[1.0, 1.0], [3.0, 3.0]]))
    line.roitype = roifile.ROI_TYPE.POLYGON
    square = roifile.ImagejRoi.frompoints(np.array([[20.0, 30.0], [28.0, 30.0], [28.0, 38.0], [20.0, 38.0]]))
    square.roitype = roifile.ROI_TYPE.POLYGON
    zp.unlink()
    roifile.roiwrite(str(zp), [point, line, square])
    pts = load_annotation(b, a1)
    np.testing.assert_array_equal(pts.centroids, np.array([[6, 5], [9, 7], [34, 24]], dtype=np.float32))
    assert math.isnan(pts.areas[0]) and math.isnan(pts.areas[1]) and pts.areas[2] == ROI_AREA
