"""Canonical ROI stamps — disk geometry, centroid preservation, crowding guard.

See docs/adr/0003-centroid-canonical-roi-stamps.md for why the pipeline
replaces detector-native boundaries with fixed-radius disks post-gate.
"""
import numpy as np

from roigbiv.pipeline.roi_stamp import canonicalize, disk_mask, resolve_crowding
from roigbiv.pipeline.types import ROI

H = W = 128


def _make_roi(mask, *, label_id, source_stage=1, confidence="high", gate_outcome="accept"):
    return ROI(
        mask=mask,
        label_id=label_id,
        source_stage=source_stage,
        confidence=confidence,
        gate_outcome=gate_outcome,
        area=int(mask.sum()),
    )


def _ellipse(cy, cx, ry, rx):
    yy, xx = np.ogrid[:H, :W]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


# ── disk_mask ────────────────────────────────────────────────────────────────

def test_disk_mask_area_matches_pi_r_squared():
    r = 10
    m = disk_mask(64, 64, r, H, W)
    assert abs(int(m.sum()) - np.pi * r ** 2) / (np.pi * r ** 2) < 0.05


def test_disk_mask_clips_to_image_bounds():
    # Centroid near the top-left corner; disk must not index out of bounds
    # and must be truncated rather than wrapping/erroring.
    m = disk_mask(2, 2, 10, H, W)
    assert m.shape == (H, W)
    assert int(m.sum()) < int(np.pi * 10 ** 2)  # truncated, not a full circle


# ── canonicalize ─────────────────────────────────────────────────────────────

def test_canonicalize_replaces_irregular_mask_with_disk_at_same_centroid():
    raw = _ellipse(64, 70, 6, 18)  # elongated, off-center blob
    roi = _make_roi(raw, label_id=1)
    original_area, original_solidity = roi.area, roi.solidity

    canonicalize(roi, radius=8, shape=(H, W))

    expected = disk_mask(64, 70, 8, H, W)
    assert np.array_equal(roi.mask, expected)
    # Gate-time morphology fields are untouched by canonicalization.
    assert roi.area == original_area
    assert roi.solidity == original_solidity


def test_canonicalize_is_noop_on_empty_mask():
    roi = _make_roi(np.zeros((H, W), dtype=bool), label_id=1)
    canonicalize(roi, radius=8, shape=(H, W))
    assert not roi.mask.any()


# ── resolve_crowding ─────────────────────────────────────────────────────────

def test_resolve_crowding_demotes_weaker_of_close_pair():
    strong = _make_roi(disk_mask(64, 60, 8, H, W), label_id=1, confidence="high")
    weak = _make_roi(disk_mask(64, 65, 8, H, W), label_id=2, confidence="moderate")

    resolve_crowding([strong, weak], radius=8)

    assert strong.gate_outcome == "accept"
    assert weak.gate_outcome == "flag"
    assert weak.confidence == "moderate"
    assert any("crowded_neighbor" in r for r in weak.gate_reasons)


def test_resolve_crowding_leaves_distant_pair_untouched():
    a = _make_roi(disk_mask(20, 20, 8, H, W), label_id=1, confidence="high")
    b = _make_roi(disk_mask(100, 100, 8, H, W), label_id=2, confidence="high")

    resolve_crowding([a, b], radius=8)

    assert a.gate_outcome == "accept"
    assert b.gate_outcome == "accept"


def test_resolve_crowding_ignores_rejected_rois():
    accepted = _make_roi(disk_mask(64, 60, 8, H, W), label_id=1, confidence="high")
    rejected = _make_roi(
        disk_mask(64, 65, 8, H, W), label_id=2, confidence="requires_review",
        gate_outcome="reject",
    )

    resolve_crowding([accepted, rejected], radius=8)

    assert accepted.gate_outcome == "accept"
    assert rejected.gate_outcome == "reject"  # untouched, not re-demoted


def test_resolve_crowding_tiebreak_is_deterministic_on_equal_confidence():
    earlier = _make_roi(disk_mask(64, 60, 8, H, W), label_id=1, source_stage=1, confidence="high")
    later = _make_roi(disk_mask(64, 65, 8, H, W), label_id=5, source_stage=2, confidence="high")

    resolve_crowding([earlier, later], radius=8)

    assert earlier.gate_outcome == "accept"
    assert later.gate_outcome == "flag"
