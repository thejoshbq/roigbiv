"""Tests for :mod:`roigbiv.pipeline.corrections`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roigbiv.pipeline.corrections import (
    USER_STAGE_SENTINEL,
    CorrectionOp,
    append_correction,
    apply_corrections,
    corrected_masks_path,
    corrected_metadata_path,
    load_corrections,
    materialize,
    write_corrections,
)
from roigbiv.pipeline.types import ROI


SHAPE = (32, 32)


def _rect_mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _roi(label_id: int, mask: np.ndarray, stage: int = 1,
         activity: str = "phasic") -> ROI:
    return ROI(
        mask=mask,
        label_id=label_id,
        source_stage=stage,
        confidence="high",
        gate_outcome="accept",
        area=int(mask.sum()),
        activity_type=activity,
    )


def test_op_round_trips_through_dict() -> None:
    op = CorrectionOp.add([[0, 0], [0, 5], [5, 5], [5, 0]],
                          activity_type="tonic", notes="hello")
    restored = CorrectionOp.from_dict(op.to_jsonable())
    assert restored.op == "add"
    assert restored.polygon == op.polygon
    assert restored.activity_type == "tonic"
    assert restored.notes == "hello"


def test_apply_add_adds_user_roi() -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4))]
    op = CorrectionOp.add([[10, 10], [10, 20], [20, 20], [20, 10]])
    corrected = apply_corrections(base, [op], SHAPE)
    assert len(corrected) == 2
    user = [r for r in corrected if r.source_stage == USER_STAGE_SENTINEL]
    assert len(user) == 1
    assert user[0].area > 0
    assert user[0].label_id == 2   # one past the max existing


def test_apply_delete_removes_roi() -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4)), _roi(2, _rect_mask(5, 5, 9, 9))]
    corrected = apply_corrections(base, [CorrectionOp.delete(1)], SHAPE)
    assert {r.label_id for r in corrected} == {2}


def test_apply_relabel_is_in_place_semantically() -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4), activity="phasic")]
    corrected = apply_corrections(
        base, [CorrectionOp.relabel(1, "tonic")], SHAPE,
    )
    assert corrected[0].activity_type == "tonic"
    # pure: input ROI unchanged
    assert base[0].activity_type == "phasic"


def test_apply_merge_combines_masks_when_no_polygon_given() -> None:
    base = [
        _roi(1, _rect_mask(0, 0, 4, 4)),
        _roi(2, _rect_mask(5, 5, 9, 9)),
    ]
    corrected = apply_corrections(
        base, [CorrectionOp.merge([1, 2])], SHAPE,
    )
    assert len(corrected) == 1
    # union should cover both rectangles → at least 4*4 + 4*4 pixels
    assert corrected[0].area >= 16 + 16


def test_apply_split_creates_separate_rois() -> None:
    big = _rect_mask(0, 0, 10, 10)
    base = [_roi(1, big)]
    corrected = apply_corrections(
        base,
        [CorrectionOp.split(1, [
            [[0, 0], [0, 5], [4, 5], [4, 0]],
            [[5, 5], [5, 10], [10, 10], [10, 5]],
        ])],
        SHAPE,
    )
    assert len(corrected) == 2
    # new ROIs get fresh label ids; old label_id 1 is gone.
    assert all(r.source_stage == USER_STAGE_SENTINEL for r in corrected)
    assert all(r.label_id != 1 for r in corrected)


def test_apply_corrections_replay_is_deterministic() -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4))]
    ops = [
        CorrectionOp.add([[10, 10], [10, 20], [20, 20], [20, 10]]),
        CorrectionOp.relabel(1, "silent"),
        CorrectionOp.delete(1),
    ]
    a = apply_corrections(base, ops, SHAPE)
    b = apply_corrections(base, ops, SHAPE)
    assert [r.label_id for r in a] == [r.label_id for r in b]
    assert [r.activity_type for r in a] == [r.activity_type for r in b]


def test_apply_bad_op_is_skipped_not_raised() -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4))]
    # polygon is malformed — should be quietly dropped.
    ops = [CorrectionOp.add(polygon=[]),
           CorrectionOp.delete(label_id=9999)]   # nonexistent
    corrected = apply_corrections(base, ops, SHAPE)
    assert len(corrected) == 1   # base survives, garbage ops no-op


def test_materialize_writes_uint16_tif_and_metadata(tmp_path: Path) -> None:
    base = [_roi(1, _rect_mask(0, 0, 4, 4))]
    corrected = apply_corrections(
        base,
        [CorrectionOp.add([[10, 10], [10, 20], [20, 20], [20, 10]])],
        SHAPE,
    )
    m_path, meta_path = materialize(corrected, tmp_path, SHAPE)
    assert m_path == corrected_masks_path(tmp_path)
    assert meta_path == corrected_metadata_path(tmp_path)

    import tifffile

    img = tifffile.imread(str(m_path))
    assert img.dtype == np.uint16
    assert img.shape == SHAPE
    # at least one nonzero pixel per ROI
    assert (img == 1).sum() > 0
    user_label_id = max(r.label_id for r in corrected)
    assert (img == user_label_id).sum() > 0

    meta = json.loads(meta_path.read_text())
    assert isinstance(meta, list)
    assert {m["label_id"] for m in meta} == {r.label_id for r in corrected}


def test_append_then_load_preserves_order(tmp_path: Path) -> None:
    ops = [
        CorrectionOp.delete(1),
        CorrectionOp.relabel(2, "tonic"),
        CorrectionOp.add([[0, 0], [0, 5], [5, 5], [5, 0]]),
    ]
    for op in ops:
        append_correction(tmp_path, op)
    loaded = load_corrections(tmp_path)
    assert [op.op for op in loaded] == ["delete", "relabel", "add"]


def test_write_corrections_empty_removes_log(tmp_path: Path) -> None:
    append_correction(tmp_path, CorrectionOp.delete(1))
    assert (tmp_path / "corrections" / "corrections.jsonl").exists()
    write_corrections(tmp_path, [])
    assert not (tmp_path / "corrections" / "corrections.jsonl").exists()
