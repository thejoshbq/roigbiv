"""Tests for :mod:`roigbiv.ui.services.loaders` helpers that don't hit disk."""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.types import ROI
from roigbiv.ui.services.loaders import (
    _gcid_by_label_from_registry,
    render_roi,
)


def _mask(shape, y0, x0, y1, x1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_render_roi_builds_contour_and_centroid() -> None:
    mask = _mask((32, 32), 10, 12, 20, 22)
    roi = ROI(
        mask=mask, label_id=7, source_stage=2,
        confidence="moderate", gate_outcome="flag",
        area=int(mask.sum()), activity_type="sparse",
    )
    rendered = render_roi(roi, gcid="abc-123")
    assert rendered.label_id == 7
    assert rendered.source_stage == 2
    assert rendered.gate_outcome == "flag"
    assert rendered.activity_type == "sparse"
    assert rendered.global_cell_id == "abc-123"
    assert len(rendered.contours) >= 1
    ys, xs = rendered.contours[0]
    assert len(ys) == len(xs) and len(ys) > 3
    # centroid should land inside the rectangle
    cy, cx = rendered.centroid_yx
    assert 10 <= cy <= 20
    assert 12 <= cx <= 22


def test_gcid_by_label_map_handles_missing_registry() -> None:
    assert _gcid_by_label_from_registry(None) == {}
    assert _gcid_by_label_from_registry({}) == {}


def test_gcid_by_label_map_extracts_entries() -> None:
    registry = {
        "cell_assignments": [
            {"local_label_id": 1, "global_cell_id": "g1",
             "match_kind": "matched"},
            {"local_label_id": 2, "global_cell_id": None,
             "match_kind": "new"},
            {"local_label_id": "bad", "global_cell_id": "g3"},
            {"local_label_id": 3, "global_cell_id": "g3"},
        ],
    }
    out = _gcid_by_label_from_registry(registry)
    assert out == {1: "g1", 3: "g3"}
