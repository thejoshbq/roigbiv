"""Tests for :mod:`roigbiv.pipeline.reingest`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.corrections import load_corrections
from roigbiv.pipeline.reingest import reingest_mask


SHAPE = (32, 32)


def _disk(tmp: Path, masks: dict[int, np.ndarray]) -> Path:
    """Lay out a minimal pipeline output directory at ``tmp/output``."""
    out = tmp / "output"
    out.mkdir()
    label_img = np.zeros(SHAPE, dtype=np.uint16)
    meta = []
    for lid, mask in masks.items():
        label_img[mask] = lid
        meta.append({
            "label_id": int(lid),
            "source_stage": 1,
            "confidence": "high",
            "gate_outcome": "accept",
            "area": int(mask.sum()),
        })
    tifffile.imwrite(str(out / "merged_masks.tif"), label_img)
    (out / "roi_metadata.json").write_text(json.dumps(meta))
    (out / "pipeline_log.json").write_text(json.dumps({
        "shape": [1, SHAPE[0], SHAPE[1]],
        "input": str(out / "fake.tif"),
        "k_background": 30,
        "stage_counts": {},
    }))
    return out


def _square(y0: int, x0: int, side: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def _write_label(path: Path, masks: dict[int, np.ndarray]) -> None:
    img = np.zeros(SHAPE, dtype=np.uint16)
    for lid, mask in masks.items():
        img[mask] = lid
    tifffile.imwrite(str(path), img)


def test_reingest_emits_add_for_new_label(tmp_path: Path) -> None:
    output_dir = _disk(tmp_path, {1: _square(2, 2, 6)})
    edited = tmp_path / "edited.tif"
    _write_label(edited, {
        1: _square(2, 2, 6),         # unchanged
        7: _square(15, 15, 6),       # new
    })

    result = reingest_mask(output_dir, edited, dry_run=True, notes="t")

    assert result.n_added == 1
    assert result.n_deleted == 0
    assert result.n_unchanged == 1
    assert any(op.op == "add" for op in result.ops)


def test_reingest_emits_delete_for_dropped_label(tmp_path: Path) -> None:
    output_dir = _disk(tmp_path, {
        1: _square(2, 2, 6),
        2: _square(15, 15, 6),
    })
    edited = tmp_path / "edited.tif"
    _write_label(edited, {1: _square(2, 2, 6)})

    result = reingest_mask(output_dir, edited, dry_run=True)

    assert result.n_deleted == 1
    assert result.n_added == 0
    delete_ops = [op for op in result.ops if op.op == "delete"]
    assert len(delete_ops) == 1
    assert delete_ops[0].label_id == 2


def test_reingest_emits_edit_when_iou_in_band(tmp_path: Path) -> None:
    """An ROI whose mask shifts by a small amount triggers an `edit` op."""
    original = _square(5, 5, 8)         # 64 px
    shifted = _square(6, 6, 8)          # mostly overlaps but not >=0.95 IoU
    output_dir = _disk(tmp_path, {1: original})
    edited = tmp_path / "edited.tif"
    _write_label(edited, {1: shifted})

    result = reingest_mask(output_dir, edited, dry_run=True)

    assert result.n_edited == 1
    assert result.n_added == 0
    assert result.n_deleted == 0
    edit_ops = [op for op in result.ops if op.op == "edit"]
    assert len(edit_ops) == 1
    assert edit_ops[0].label_id == 1


def test_reingest_writes_jsonl_and_corrected_tif(tmp_path: Path) -> None:
    output_dir = _disk(tmp_path, {1: _square(2, 2, 6)})
    edited = tmp_path / "edited.tif"
    _write_label(edited, {
        1: _square(2, 2, 6),
        9: _square(20, 20, 5),
    })

    result = reingest_mask(output_dir, edited, notes="cli test")

    assert result.n_added == 1

    log = load_corrections(output_dir)
    assert len(log) == len(result.ops)
    assert all(op.notes == "cli test" for op in log)

    corrected_tif = output_dir / "corrections" / "corrected_masks.tif"
    corrected_meta = output_dir / "corrections" / "corrected_metadata.json"
    assert corrected_tif.exists()
    assert corrected_meta.exists()
    img = tifffile.imread(str(corrected_tif))
    # Original ROI + the newly added one → at least 2 distinct non-zero labels.
    nonzero_labels = np.unique(img)
    nonzero_labels = nonzero_labels[nonzero_labels > 0]
    assert nonzero_labels.size >= 2


def test_reingest_unchanged_returns_empty_op_list(tmp_path: Path) -> None:
    output_dir = _disk(tmp_path, {1: _square(2, 2, 6), 2: _square(20, 20, 5)})
    edited = tmp_path / "edited.tif"
    _write_label(edited, {1: _square(2, 2, 6), 2: _square(20, 20, 5)})

    result = reingest_mask(output_dir, edited, dry_run=True)

    assert result.ops == []
    assert result.n_unchanged == 2


def test_reingest_rejects_shape_mismatch(tmp_path: Path) -> None:
    output_dir = _disk(tmp_path, {1: _square(2, 2, 6)})
    edited = tmp_path / "edited.tif"
    tifffile.imwrite(str(edited), np.zeros((16, 16), dtype=np.uint16))

    with pytest.raises(ValueError, match="shape mismatch"):
        reingest_mask(output_dir, edited, dry_run=True)
