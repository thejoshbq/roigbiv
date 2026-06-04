"""Tests for roigbiv.pipeline.web_reingest."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.web_reingest import (
    _mask_to_polygon_yx,
    _polygon_to_mask,
    annotations_to_label_image,
    parse_svg_polygon,
)


# ── parse_svg_polygon ──────────────────────────────────────────────────────────


def test_parse_svg_polygon_basic():
    svg = '<svg><polygon points="10,20 30,40 50,60"></svg>'
    result = parse_svg_polygon(svg)
    assert result == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


def test_parse_svg_polygon_decimal():
    svg = '<svg><polygon points="10.5,20.3 30.1,40.9 50.0,60.0"></svg>'
    result = parse_svg_polygon(svg)
    assert len(result) == 3
    assert abs(result[0][0] - 10.5) < 1e-9
    assert abs(result[0][1] - 20.3) < 1e-9


def test_parse_svg_polygon_roundtrip_xy_yx():
    """Verify that (x,y) → (y,x) transposition is correct for CorrectionOps."""
    svg = '<svg><polygon points="5,10 15,25 30,5"></svg>'
    xy_pairs = parse_svg_polygon(svg)
    # Convert to (y, x) for rasterisation
    yx_pairs = [(y, x) for x, y in xy_pairs]
    assert yx_pairs == [(10.0, 5.0), (25.0, 15.0), (5.0, 30.0)]


def test_parse_svg_polygon_empty():
    assert parse_svg_polygon("") == []
    assert parse_svg_polygon("<svg></svg>") == []


def test_parse_svg_polygon_malformed():
    # Should not raise; just returns empty
    assert parse_svg_polygon("not xml at all <<<") == []


# ── _polygon_to_mask ───────────────────────────────────────────────────────────


def test_polygon_to_mask_fills_interior():
    shape = (50, 50)
    # Square centred in the image
    yx_pairs = [(10.0, 10.0), (10.0, 40.0), (40.0, 40.0), (40.0, 10.0)]
    mask = _polygon_to_mask(yx_pairs, shape)
    assert mask.shape == shape
    assert mask[25, 25]          # interior should be filled
    assert not mask[5, 5]        # corner should be empty


def test_polygon_to_mask_too_few_points():
    shape = (50, 50)
    mask = _polygon_to_mask([(0.0, 0.0), (10.0, 10.0)], shape)
    assert not mask.any()


# ── annotations_to_label_image ─────────────────────────────────────────────────


def _make_annotation(label_id: int, yx_box: tuple[int, int, int, int]) -> dict:
    """Build a W3C annotation dict for a rectangular polygon.

    ``yx_box`` = (y0, x0, y1, x1) in pixel coords; SVG needs (x, y).
    """
    y0, x0, y1, x1 = yx_box
    points_str = f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"
    svg_value = f'<svg><polygon points="{points_str}"></svg>'
    return {
        "type": "Annotation",
        "id": f"roi-{label_id}",
        "body": [],
        "target": {"selector": {"type": "SvgSelector", "value": svg_value}},
    }


def test_annotations_to_label_image_no_overlap():
    shape = (100, 100)
    ann1 = _make_annotation(1, (5, 5, 30, 30))
    ann2 = _make_annotation(2, (60, 60, 90, 90))
    label_img = annotations_to_label_image([ann1, ann2], shape)
    assert label_img.shape == shape
    assert label_img.dtype == np.uint16
    assert label_img[15, 15] == 1     # inside first box
    assert label_img[75, 75] == 2     # inside second box
    assert label_img[50, 50] == 0     # gap between boxes


def test_annotations_to_label_image_empty():
    shape = (50, 50)
    label_img = annotations_to_label_image([], shape)
    assert not label_img.any()


# ── reingest_from_annotations (integration) ────────────────────────────────────


def _make_pipeline_output_dir(tmp_path: Path, rois: list[np.ndarray]) -> Path:
    """Create a minimal pipeline output directory for testing."""
    out = tmp_path / "fov_test"
    out.mkdir()
    (out / "summary").mkdir()
    (out / "corrections").mkdir()

    H, W = 100, 100
    mean_img = np.zeros((H, W), dtype=np.float32)
    tifffile.imwrite(str(out / "summary" / "mean_M.tif"), mean_img)

    # Build merged_masks.tif from ROI list
    merged = np.zeros((H, W), dtype=np.uint16)
    meta = []
    for i, mask in enumerate(rois, start=1):
        merged[mask] = i
        yx = np.argwhere(mask)
        centroid_y = float(yx[:, 0].mean())
        centroid_x = float(yx[:, 1].mean())
        meta.append({
            "label_id": i,
            "source_stage": 1,
            "gate_outcome": "accept",
            "activity_type": "unknown",
            "confidence": 1.0,
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
        })
    tifffile.imwrite(str(out / "merged_masks.tif"), merged)

    (out / "roi_metadata.json").write_text(json.dumps(meta))
    (out / "pipeline_log.json").write_text(
        json.dumps({"fs": 7.5, "stages_run": [1]})
    )
    return out


def _rect_mask(y0: int, x0: int, y1: int, x1: int, shape=(100, 100)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _mask_to_annotation(roi_id: int, mask: np.ndarray) -> dict:
    """Build a W3C annotation from the actual sub-pixel contour of a mask.

    Uses ``_mask_to_polygon_yx`` so the annotation faithfully represents the
    mask boundary — rasterising it back via ``_polygon_to_mask`` should yield
    IoU ≥ 0.95 with the original.  Integer-corner rectangles do NOT work for
    round-trip tests because ``skimage.draw.polygon`` can include or exclude
    boundary rows/cols differently from a numpy boolean slice.
    """
    contour = _mask_to_polygon_yx(mask)  # [[y, x], ...]
    points_str = " ".join(f"{x},{y}" for y, x in contour)
    svg_value = f'<svg><polygon points="{points_str}"></svg>'
    return {
        "type": "Annotation",
        "id": f"roi-{roi_id}",
        "body": [],
        "target": {"selector": {"type": "SvgSelector", "value": svg_value}},
    }


def test_reingest_no_change(tmp_path):
    """Submitting identical ROIs should produce zero ops."""
    from roigbiv.pipeline.web_reingest import reingest_from_annotations

    roi1 = _rect_mask(10, 10, 30, 30)
    roi2 = _rect_mask(60, 60, 80, 80)
    roi3 = _rect_mask(10, 60, 30, 80)
    out = _make_pipeline_output_dir(tmp_path, [roi1, roi2, roi3])

    # Build annotations using actual sub-pixel contours for faithful round-trip
    anns = [_mask_to_annotation(i + 1, r) for i, r in enumerate([roi1, roi2, roi3])]
    result = reingest_from_annotations(out, anns, dry_run=True)
    assert result.n_unchanged == 3
    assert result.ops == []


def test_reingest_add_delete(tmp_path):
    """Submitting one extra + omitting one should emit add + delete ops."""
    from roigbiv.pipeline.corrections import load_corrections
    from roigbiv.pipeline.web_reingest import reingest_from_annotations

    roi1 = _rect_mask(10, 10, 30, 30)
    roi2 = _rect_mask(60, 60, 80, 80)
    out = _make_pipeline_output_dir(tmp_path, [roi1, roi2])

    # Keep roi1 (via actual contour), drop roi2, add a new roi
    anns = [
        _mask_to_annotation(1, roi1),          # unchanged
        _make_annotation(99, (40, 40, 55, 55)),  # new roi (integer box is fine here)
    ]
    result = reingest_from_annotations(out, anns)
    assert result.n_added == 1
    assert result.n_deleted == 1
    assert result.n_unchanged == 1

    ops = load_corrections(out)
    op_types = {op.op for op in ops}
    assert "add" in op_types
    assert "delete" in op_types
