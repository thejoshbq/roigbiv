"""Guards for the Centroids page — per-FOV calibration and the detection run.

Calibration is per-FOV because what it sets are per-FOV facts. These cases pin
the two things that make it usable: the fields reload for the FOV you switch
to, and the readout says plainly whether this FOV is calibrated and whether
saving invalidates prior output.

The page must also stay free of motion-correction controls: a centroids-only
run routes through ``workspace._run_centroids_only``, which never reaches the
registration path, so an MC tunable here would describe work that never runs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roigbiv.pipeline.calibration import write_calibration
from roigbiv.ui.pages import centroids
from roigbiv.ui.tests._tree import find_by_id, ids


class _FakeState:
    def __init__(self, output_root=Path("/ws/output")):
        self.workspace = (SimpleNamespace(output_root=output_root,
                                          input_root=Path("/ws"), tifs=())
                          if output_root is not None else None)
        self.registry_config = None


# ── the readout ────────────────────────────────────────────────────────────


def test_readout_says_uncalibrated(tmp_path):
    assert "Not calibrated" in centroids._readout_text(None, tmp_path)


def test_readout_names_the_saved_settings_and_warns_on_existing_output(tmp_path):
    calib = write_calibration(tmp_path, 45.0, cellprob_threshold=-1.0,
                              cellpose_model="cyto3")
    text = centroids._readout_text(calib, tmp_path)
    assert "45.0px diameter" in text
    assert "cellprob_threshold=-1" in text
    assert "model=cyto3" in text
    assert "already has centroid output" not in text

    (tmp_path / "centroids.json").write_text("{}")
    assert "already has centroid output" in centroids._readout_text(calib, tmp_path)


def test_readout_names_the_deployed_model_when_unset(tmp_path):
    """An unset model override reads as "deployed", not as an empty string."""
    calib = write_calibration(tmp_path, 40.0)
    assert "model=deployed" in centroids._readout_text(calib, tmp_path)


def test_readout_is_blank_without_a_fov():
    assert centroids._readout_text(None, None) == ""


# ── the reference circle ───────────────────────────────────────────────────


def test_preview_draws_the_calibration_circle(monkeypatch):
    # A diameter adds exactly one shape (the dashed reference circle), centered
    # on the image, so a real neuron can be lined up against it.
    monkeypatch.setattr(
        centroids.fov_select, "mean_and_title",
        lambda _v: (np.zeros((32, 32), dtype=np.float32), "sess01", None))

    assert not centroids._preview_figure("summary:/o/s", False, None).layout.shapes

    fig = centroids._preview_figure("summary:/o/s", False, 12.0)
    assert len(fig.layout.shapes) == 1
    shape = fig.layout.shapes[0]
    assert shape.type == "circle"
    assert shape.x1 - shape.x0 == pytest.approx(12.0)
    assert shape.y1 - shape.y0 == pytest.approx(12.0)
    assert (shape.x0 + shape.x1) / 2 == pytest.approx(16.0)
    assert (shape.y0 + shape.y1) / 2 == pytest.approx(16.0)


def test_no_circle_without_a_mean(monkeypatch):
    """An 'input:' FOV with no computable mean must not error on shape-adding."""
    monkeypatch.setattr(centroids.fov_select, "mean_and_title",
                        lambda _v: (None, None, None))
    fig = centroids._preview_figure("input:/data/sess01_mc.tif", False, 12.0)
    assert not fig.layout.shapes


def test_centroids_are_loaded_only_when_the_overlay_is_on(monkeypatch):
    from roigbiv.ui.services import loaders as loaders_mod
    from roigbiv.ui.services.loaders import ROIRender

    monkeypatch.setattr(
        centroids.fov_select, "mean_and_title",
        lambda _v: (np.zeros((8, 8), dtype=np.float32), "s", Path("/o/s")))
    calls = {"n": 0}

    def _fake_load_centroids(output_dir, shape, *a, **k):
        calls["n"] += 1
        return [ROIRender(label_id=0, source_stage=2, gate_outcome="accept",
                          activity_type=None, area=10,
                          centroid_yx=(4.0, 4.0), contours=[([1.0], [1.0])])]

    monkeypatch.setattr(loaders_mod, "load_centroids", _fake_load_centroids)

    fig_off = centroids._preview_figure("summary:/o/s", False, None)
    assert calls["n"] == 0, "centroids must not be loaded when the toggle is off"
    assert len(fig_off.data) == 1, "heatmap only — no scatter overlay traces"

    fig_on = centroids._preview_figure("summary:/o/s", True, None)
    assert calls["n"] == 1
    assert len(fig_on.data) == 2, "heatmap + one scatter trace for the centroid"


# ── the run ────────────────────────────────────────────────────────────────


def test_the_run_is_centroids_only():
    overrides = centroids.centroid_overrides(force_cpu=False, persist_flows=True)
    assert overrides["run_centroids"] is True
    assert overrides["foundation_only"] is False


def test_flow_persistence_is_on_by_default_and_reaches_the_config():
    """It is what decides whether seeded boundaries are possible for a FOV."""
    assert centroids.centroid_overrides(False, True)["centroid_persist_flows"] is True
    assert centroids.centroid_overrides(False, False)["centroid_persist_flows"] is False


def test_the_run_carries_no_motion_correction_keys():
    overrides = centroids.centroid_overrides(False, True)
    mc_keys = [k for k in overrides if k.startswith("mc_")
               or k == "motion_correction_backend"]
    assert not mc_keys, f"a centroids-only run never registers: {mc_keys}"


# ── isolation ──────────────────────────────────────────────────────────────


def test_the_page_carries_no_motion_correction_controls(monkeypatch):
    monkeypatch.setattr(centroids, "get_app_state", lambda: _FakeState())
    present = ids(centroids.layout())
    mc_ids = [i for i in present
              if isinstance(i, str) and i.startswith("roigbiv-param-mc")]
    assert not mc_ids, f"MC tunables do not belong on this page: {mc_ids}"


def test_the_page_carries_the_calibration_controls(monkeypatch):
    monkeypatch.setattr(centroids, "get_app_state", lambda: _FakeState())
    present = ids(centroids.layout())
    for cid in (centroids.DIAMETER_ID, centroids.THRESHOLD_ID,
                centroids.MODEL_ID, centroids.SAVE_ID, centroids.SAVE_CLEAR_ID,
                centroids.PERSIST_FLOWS_ID, centroids.RUN_ID):
        assert cid in present


def test_the_model_choices_include_stock_cyto3(monkeypatch):
    """The deployed checkpoint does not transfer to every preparation.

    On the reference prism FOV it found 2 somata where stock cyto3 found 9.
    """
    monkeypatch.setattr(centroids, "get_app_state", lambda: _FakeState())
    select = find_by_id(centroids.layout(), centroids.MODEL_ID)
    values = [o["value"] for o in select.options]
    assert "" in values, "the deployed checkpoint must remain the default"
    assert "cyto3" in values


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
