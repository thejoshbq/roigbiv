"""Guards for the Stage-1 diameter calibration on the motion-correction preview.

The user sets the Cellpose ``diameter`` by dragging an editable circle on the MC
preview (or via "Suggest"). These tests pin the load-bearing pieces:

* the relayout→pixel-diameter conversion (axis-orientation-proof),
* the run-override contract (chosen diameter honoured, ``diameter_auto`` forced
  off so stage1 can't silently override it — stage1.py:259),
* the editable circle geometry and its presence on the figure,
* AppState calibration persistence + reset on a new scan.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import roigbiv.ui.pages.process as proc
from roigbiv.ui.services.app_state import AppState


# ── relayout → diameter (px) ─────────────────────────────────────────────────


def test_diameter_from_relayout_x_extent():
    # Incremental shape keys → diameter is the x-extent |x1 - x0|.
    relayout = {"shapes[0].x0": 100.0, "shapes[0].x1": 118.0,
                "shapes[0].y0": 50.0, "shapes[0].y1": 68.0}
    assert proc._diameter_from_relayout(relayout) == pytest.approx(18.0)


def test_diameter_from_relayout_reversed_y_is_orientation_proof():
    # The image figure's y-axis is reversed (range=[H-1, 0]), so a drag yields
    # y0 > y1. With only y keys present the diameter must still be positive and
    # correct (abs of the extent), never negative.
    relayout = {"shapes[0].y0": 68.0, "shapes[0].y1": 50.0}
    assert proc._diameter_from_relayout(relayout) == pytest.approx(18.0)


def test_diameter_from_relayout_shapes_list_form():
    # Plotly may emit the whole shapes list instead of incremental keys.
    relayout = {"shapes": [{"x0": 10.0, "x1": 30.0, "y0": 80.0, "y1": 60.0}]}
    assert proc._diameter_from_relayout(relayout) == pytest.approx(20.0)


def test_diameter_from_relayout_ignores_pan_zoom():
    # Pan/zoom relayouts carry axis-range keys, not shape keys → None, so the
    # callback no-ops and replacing the figure can't feed back into a loop.
    assert proc._diameter_from_relayout(
        {"xaxis.range[0]": 0, "xaxis.range[1]": 511}) is None
    assert proc._diameter_from_relayout({"autosize": True}) is None
    assert proc._diameter_from_relayout(None) is None
    # Degenerate zero-width circle is rejected.
    assert proc._diameter_from_relayout(
        {"shapes[0].x0": 5.0, "shapes[0].x1": 5.0}) is None


# ── coercion + run-override contract ─────────────────────────────────────────


def test_coerce_diameter_rejects_garbage_and_tiny():
    assert proc._coerce_diameter(12) == 12.0
    assert proc._coerce_diameter("15") == 15.0
    assert proc._coerce_diameter(None) is None
    assert proc._coerce_diameter("abc") is None
    assert proc._coerce_diameter(2) is None          # below the 3 px floor


def test_on_run_diameter_falls_back_to_default():
    assert proc._on_run_diameter(17.4) == 17
    assert proc._on_run_diameter(None) == 12
    assert proc._on_run_diameter("garbage") == 12
    assert proc._on_run_diameter(1) == 12            # too small → default


def test_diameter_overrides_calibration_wins_and_auto_off():
    # An AppState calibration (drag/type/Suggest) beats the raw field, and
    # diameter_auto is ALWAYS forced off — the core correctness guarantee.
    assert proc._diameter_overrides(18, 12) == {"diameter": 18, "diameter_auto": False}
    # No calibration → fall back to the field value.
    assert proc._diameter_overrides(None, 20) == {"diameter": 20, "diameter_auto": False}
    # No calibration + blank field → cfg default 12 (matches pre-feature run).
    assert proc._diameter_overrides(None, None) == {"diameter": 12, "diameter_auto": False}
    assert proc._diameter_overrides(None, "x") == {"diameter": 12, "diameter_auto": False}


# ── circle geometry + figure ─────────────────────────────────────────────────


def test_diameter_circle_shape_centered_with_radius():
    sh = proc._diameter_circle_shape(W=200, H=100, diameter_px=20)
    assert sh["type"] == "circle"
    assert sh["editable"] is True
    # Centered on the image; bounding box width == diameter.
    assert (sh["x0"] + sh["x1"]) / 2 == pytest.approx(100.0)
    assert (sh["y0"] + sh["y1"]) / 2 == pytest.approx(50.0)
    assert (sh["x1"] - sh["x0"]) == pytest.approx(20.0)


def test_mc_preview_figure_draws_circle_when_diameter_given(monkeypatch):
    monkeypatch.setattr(proc, "_maybe_read_tif",
                        lambda _p: np.zeros((64, 64), dtype=np.float32))
    fig = proc._mc_preview_figure("summary:/ws/output/s01", diameter_px=18)
    shapes = fig.layout.shapes
    assert len(shapes) == 1
    assert shapes[0].type == "circle"
    assert (shapes[0].x1 - shapes[0].x0) == pytest.approx(18.0)


def test_mc_preview_figure_no_circle_without_diameter(monkeypatch):
    monkeypatch.setattr(proc, "_maybe_read_tif",
                        lambda _p: np.zeros((64, 64), dtype=np.float32))
    fig = proc._mc_preview_figure("summary:/ws/output/s01", diameter_px=None)
    assert not fig.layout.shapes


def test_mc_preview_figure_no_circle_when_no_mean():
    # No FOV loaded (value None) → empty canvas, no circle even if a diameter
    # is supplied (nothing to scale against).
    fig = proc._mc_preview_figure(None, diameter_px=18)
    assert not fig.layout.shapes


# ── AppState persistence ─────────────────────────────────────────────────────


def _fake_ws():
    return SimpleNamespace(
        tifs=(Path("/ws/a_mc.tif"),),
        db_dsn="sqlite:///x.db", blob_root=Path("/b"),
        calibration_path=Path("/c.json"), db_path=Path("/x.db"),
    )


def test_calibration_round_trip_and_rounding():
    st = AppState()
    assert st.calibrated_diameter() is None
    st.set_calibration(17.6, fov_stem="a")
    assert st.calibration == {"diameter_px": 17.6, "fov_stem": "a"}
    assert st.calibrated_diameter() == 18          # rounded
    st.clear_calibration()
    assert st.calibration is None
    assert st.calibrated_diameter() is None


def test_set_workspace_clears_calibration():
    # A diameter measured on one workspace's FOVs must not leak into the next.
    st = AppState()
    st.set_calibration(22.0)
    st.set_workspace(_fake_ws())
    assert st.calibration is None
    assert st.calibrated_diameter() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
