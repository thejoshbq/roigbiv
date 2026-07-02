"""Tests for CandidateUnion container.

Tests cover:
- add() combines candidates from multiple detectors/branches
- add() rejects duplicate candidate_id and mismatched mask shapes
- iter_by_detector / iter_by_branch filter correctly
- to_label_image shape/dtype/pixel assignment, shape inference, empty-union
  error, uint16 candidate-count guard
- save/load round trip preserves provenance and mask geometry, including for
  overlapping candidates (the container's core use case pre-deconfliction)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roigbiv.pipeline.types import CandidateROI, CandidateUnion


def _square_mask(shape, y0, x0, size):
    """A (H, W) bool mask with a size x size True square at (y0, x0)."""
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y0 + size, x0:x0 + size] = True
    return mask


def _make_candidate(candidate_id, source_detector="cellpose", source_stage=1,
                     branch="raw", mask=None, **kwargs):
    if mask is None:
        mask = _square_mask((20, 20), 2, 2, 3)
    indices = np.flatnonzero(mask)
    sparse_pixels = kwargs.pop("sparse_pixels", (indices, np.ones(len(indices))))
    return CandidateROI(
        candidate_id=candidate_id,
        source_detector=source_detector,
        source_stage=source_stage,
        branch=branch,
        mask=mask,
        sparse_pixels=sparse_pixels,
        centroid=kwargs.pop("centroid", (3.5, 3.5)),
        bbox=kwargs.pop("bbox", (2, 2, 5, 5)),
        area=kwargs.pop("area", int(mask.sum())),
        **kwargs,
    )


def test_add_combines_candidates_from_multiple_detectors():
    """CandidateUnion.add() can combine candidates from multiple detectors."""
    union = CandidateUnion()
    union.add(_make_candidate("c1", source_detector="cellpose", branch="raw"))
    union.add(_make_candidate("c2", source_detector="suite2p", branch="raw"))
    union.add(_make_candidate("c3", source_detector="template_sweep", branch="denoised"))

    assert len(union.candidates) == 3
    assert set(union.candidates.keys()) == {"c1", "c2", "c3"}


def test_add_rejects_duplicate_candidate_id():
    """CandidateUnion.add() raises ValueError on a duplicate candidate_id."""
    union = CandidateUnion()
    union.add(_make_candidate("dup"))

    with pytest.raises(ValueError, match="dup"):
        union.add(_make_candidate("dup"))

    assert len(union.candidates) == 1


def test_add_rejects_mismatched_mask_shape():
    """CandidateUnion.add() raises ValueError when a candidate's mask shape
    disagrees with the rest of the union."""
    union = CandidateUnion()
    union.add(_make_candidate("c1", mask=_square_mask((20, 20), 2, 2, 3)))

    with pytest.raises(ValueError, match="shape"):
        union.add(_make_candidate("c2", mask=_square_mask((30, 30), 2, 2, 3)))

    assert len(union.candidates) == 1


def test_iter_by_detector_filters_correctly():
    """iter_by_detector() yields only candidates from the named detector."""
    union = CandidateUnion()
    union.add(_make_candidate("c1", source_detector="cellpose"))
    union.add(_make_candidate("c2", source_detector="suite2p"))
    union.add(_make_candidate("c3", source_detector="cellpose"))

    cellpose_ids = {c.candidate_id for c in union.iter_by_detector("cellpose")}
    assert cellpose_ids == {"c1", "c3"}

    suite2p_ids = {c.candidate_id for c in union.iter_by_detector("suite2p")}
    assert suite2p_ids == {"c2"}

    assert list(union.iter_by_detector("nonexistent")) == []


def test_iter_by_branch_filters_correctly():
    """iter_by_branch() yields only candidates from the named branch."""
    union = CandidateUnion()
    union.add(_make_candidate("c1", branch="raw"))
    union.add(_make_candidate("c2", branch="denoised"))
    union.add(_make_candidate("c3", branch="raw"))

    raw_ids = {c.candidate_id for c in union.iter_by_branch("raw")}
    assert raw_ids == {"c1", "c3"}

    denoised_ids = {c.candidate_id for c in union.iter_by_branch("denoised")}
    assert denoised_ids == {"c2"}


def test_to_label_image_shape_and_dtype():
    """to_label_image() produces a (H, W) uint16 array inferred from masks."""
    union = CandidateUnion()
    union.add(_make_candidate("c1", mask=_square_mask((20, 20), 2, 2, 3)))
    union.add(_make_candidate("c2", mask=_square_mask((20, 20), 10, 10, 3)))

    label_image = union.to_label_image()

    assert label_image.shape == (20, 20)
    assert label_image.dtype == np.uint16


def test_to_label_image_assigns_ids_in_insertion_order():
    """Each candidate's mask is painted with its 1-based insertion-order id."""
    union = CandidateUnion()
    mask1 = _square_mask((20, 20), 2, 2, 3)
    mask2 = _square_mask((20, 20), 10, 10, 3)
    union.add(_make_candidate("c1", mask=mask1))
    union.add(_make_candidate("c2", mask=mask2))

    label_image = union.to_label_image()

    assert np.all(label_image[mask1] == 1)
    assert np.all(label_image[mask2] == 2)
    assert np.all(label_image[~mask1 & ~mask2] == 0)


def test_to_label_image_empty_union_without_shape_raises():
    """An empty CandidateUnion with no explicit shape cannot infer one."""
    union = CandidateUnion()

    with pytest.raises(ValueError, match="empty"):
        union.to_label_image()


def test_to_label_image_empty_union_with_explicit_shape():
    """An empty CandidateUnion with an explicit shape returns an all-zero image."""
    union = CandidateUnion()

    label_image = union.to_label_image(shape=(10, 10))

    assert label_image.shape == (10, 10)
    assert np.all(label_image == 0)


def test_to_label_image_rejects_more_than_65535_candidates():
    """to_label_image() raises rather than silently wrapping uint16 label ids."""
    union = CandidateUnion()
    tiny_mask = np.zeros((1, 1), dtype=bool)
    for i in range(65536):
        union.add(CandidateROI(
            candidate_id=f"c{i}", source_detector="cellpose", source_stage=1,
            branch="raw", mask=tiny_mask,
            sparse_pixels=(np.array([], dtype=np.int64), np.array([], dtype=np.float32)),
            centroid=(0.0, 0.0), bbox=(0, 0, 1, 1), area=0,
        ))

    with pytest.raises(ValueError, match="65535"):
        union.to_label_image()


def test_save_load_round_trip_preserves_metadata(tmp_path: Path):
    """save()/load() round trip preserves every to_serializable() field."""
    union = CandidateUnion()
    union.add(_make_candidate(
        "c1", source_detector="cellpose", source_stage=1, branch="raw",
        mask=_square_mask((20, 20), 2, 2, 3),
        detector_score=0.87, validation_status="accepted",
        provenance={"model": "cyto3"}, seed_frames=[10, 20],
        summary_features={"skew": 0.5}, temporal_features={"n_transients": 3},
    ))
    union.add(_make_candidate(
        "c2", source_detector="suite2p", source_stage=2, branch="denoised",
        mask=_square_mask((20, 20), 10, 10, 4),
    ))

    out_dir = tmp_path / "candidates"
    union.save(out_dir)

    assert (out_dir / "candidate_masks.tif").exists()
    assert (out_dir / "candidate_metadata.json").exists()

    loaded = CandidateUnion.load(out_dir)

    assert set(loaded.candidates.keys()) == {"c1", "c2"}
    for candidate_id in ("c1", "c2"):
        original = union.candidates[candidate_id]
        restored = loaded.candidates[candidate_id]
        assert restored.to_serializable() == original.to_serializable()


def test_save_load_round_trip_preserves_mask_geometry(tmp_path: Path):
    """save()/load() preserves each candidate's boolean mask exactly."""
    union = CandidateUnion()
    mask1 = _square_mask((20, 20), 2, 2, 3)
    mask2 = _square_mask((20, 20), 10, 10, 4)
    union.add(_make_candidate("c1", mask=mask1))
    union.add(_make_candidate("c2", mask=mask2))

    out_dir = tmp_path / "candidates"
    union.save(out_dir)
    loaded = CandidateUnion.load(out_dir)

    assert np.array_equal(loaded.candidates["c1"].mask, mask1)
    assert np.array_equal(loaded.candidates["c2"].mask, mask2)


def test_save_load_round_trip_preserves_overlapping_masks(tmp_path: Path):
    """save()/load() preserves each candidate's exact mask even when candidates
    overlap — the container's core use case (multiple detectors proposing
    overlapping regions pre-deconfliction). A shared label-image persistence
    format would let the second candidate's pixels overwrite the first's."""
    union = CandidateUnion()
    mask1 = _square_mask((20, 20), 2, 2, 6)   # rows/cols 2-7
    mask2 = _square_mask((20, 20), 5, 5, 6)   # rows/cols 5-10, overlaps mask1 in 5-7
    assert (mask1 & mask2).any(), "test setup: masks must actually overlap"
    union.add(_make_candidate("c1", mask=mask1, area=int(mask1.sum())))
    union.add(_make_candidate("c2", mask=mask2, area=int(mask2.sum())))

    out_dir = tmp_path / "candidates"
    union.save(out_dir)
    loaded = CandidateUnion.load(out_dir)

    assert np.array_equal(loaded.candidates["c1"].mask, mask1)
    assert np.array_equal(loaded.candidates["c2"].mask, mask2)
    assert loaded.candidates["c1"].mask.sum() == mask1.sum()
    assert loaded.candidates["c2"].mask.sum() == mask2.sum()


def test_save_load_reconstructs_sparse_pixels_from_mask(tmp_path: Path):
    """sparse_pixels weights are not preserved (dropped by to_serializable());
    load() synthesizes placeholder indices/weights from the restored mask."""
    union = CandidateUnion()
    mask = _square_mask((20, 20), 2, 2, 3)
    union.add(_make_candidate("c1", mask=mask))

    out_dir = tmp_path / "candidates"
    union.save(out_dir)
    loaded = CandidateUnion.load(out_dir)

    restored_indices, restored_weights = loaded.candidates["c1"].sparse_pixels
    assert np.array_equal(restored_indices, np.flatnonzero(mask))
    assert np.all(restored_weights == 1.0)


def test_save_empty_union(tmp_path: Path):
    """save() on an empty union writes an empty label image and metadata list."""
    union = CandidateUnion()
    out_dir = tmp_path / "candidates"
    union.save(out_dir)

    loaded = CandidateUnion.load(out_dir)
    assert loaded.candidates == {}
