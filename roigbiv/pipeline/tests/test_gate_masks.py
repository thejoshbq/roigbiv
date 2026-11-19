"""Gate-time mask sidecar: every candidate's mask as its gate saw it, written before canonicalize."""

import pathlib

import numpy as np

from roigbiv.pipeline import gate_masks as gm
from roigbiv.pipeline.roi_stamp import canonicalize
from roigbiv.pipeline.types import ROI

H, W = 30, 40


def _roi(label, outcome, ys, xs, stage=1):
    m = np.zeros((H, W), bool)
    m[ys, xs] = True
    return ROI(mask=m, label_id=label, source_stage=stage, confidence="moderate", gate_outcome=outcome)


def _rois():
    return [
        _roi(7, "accept", slice(2, 9), slice(3, 6)),  # a 7 x 3 bar, not a disk
        _roi(3, "reject", [10, 10, 11], [20, 21, 21]),
        _roi(9, "flag", slice(20, 24), slice(30, 38)),
    ]


def test_saved_masks_round_trip_for_every_candidate_rejects_included(tmp_path):
    rois = _rois()
    path = gm.save_gate_masks(tmp_path, 1, rois)
    assert path == tmp_path / "stage1" / "stage1_gate_masks.npz"
    shape, masks = gm.load_gate_masks(path)
    assert shape == (H, W)
    assert set(masks) == {7, 3, 9}
    for r in rois:
        np.testing.assert_array_equal(masks[r.label_id], np.flatnonzero(r.mask))


def test_masks_saved_before_canonicalize_keep_the_detected_shape(tmp_path):
    rois = _rois()
    bar = np.flatnonzero(rois[0].mask)
    gm.save_gate_masks(tmp_path, 2, rois)
    canonicalize(rois[0], 3, (H, W))
    assert not np.array_equal(np.flatnonzero(rois[0].mask), bar)
    _, masks = gm.load_gate_masks(gm.gate_masks_path(tmp_path, 2))
    np.testing.assert_array_equal(masks[7], bar)


def test_saving_leaves_the_rois_untouched(tmp_path):
    rois = _rois()
    before = [(r.label_id, r.gate_outcome, r.mask.copy()) for r in rois]
    gm.save_gate_masks(tmp_path, 3, rois)
    for r, (lab, out, m) in zip(rois, before):
        assert r.label_id == lab and r.gate_outcome == out
        np.testing.assert_array_equal(r.mask, m)


def test_an_empty_stage_saves_an_empty_sidecar(tmp_path):
    path = gm.save_gate_masks(tmp_path, 4, [], shape=(H, W))
    shape, masks = gm.load_gate_masks(path)
    assert shape == (H, W) and masks == {}


def test_a_failed_write_warns_and_leaves_no_partial_file(tmp_path, monkeypatch, capsys):
    def boom(file, **k):
        pathlib.Path(file).write_bytes(b"partial")  # a write that dies part way, as on a full disk
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(gm.np, "savez_compressed", boom)
    assert gm.save_gate_masks(tmp_path, 1, _rois()) is None
    assert "gate masks" in capsys.readouterr().out.lower()
    assert not list((tmp_path / "stage1").glob("*gate_masks*"))


def test_a_rewrite_replaces_the_previous_sidecar(tmp_path):
    gm.save_gate_masks(tmp_path, 1, _rois())
    gm.save_gate_masks(tmp_path, 1, _rois()[:1])
    _, masks = gm.load_gate_masks(gm.gate_masks_path(tmp_path, 1))
    assert set(masks) == {7}
    assert sorted(p.name for p in (tmp_path / "stage1").iterdir()) == ["stage1_gate_masks.npz"]


def test_a_failed_write_leaves_no_sidecar_from_an_earlier_run(tmp_path, monkeypatch):
    gm.save_gate_masks(tmp_path, 3, _rois())

    def boom(file, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(gm.np, "savez_compressed", boom)
    assert gm.save_gate_masks(tmp_path, 3, _rois()[:1]) is None
    assert not gm.gate_masks_path(tmp_path, 3).exists()


def test_any_error_is_a_warning_never_an_exception_into_the_run(tmp_path, capsys):
    bad = _rois()
    bad[1].mask = np.zeros((H + 1, W), bool)  # a mask of another shape would give meaningless pixel indices
    assert gm.save_gate_masks(tmp_path, 1, bad) is None
    assert gm.save_gate_masks(tmp_path, 2, []) is None  # no ROIs and no shape
    assert capsys.readouterr().out.lower().count("gate masks") == 2
    assert not list(tmp_path.rglob("*gate_masks*"))
