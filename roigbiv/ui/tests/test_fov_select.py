"""The FOV-value convention, shared by every page that picks one.

A dropdown value is self-describing (``summary:`` / ``input:``) because the
render callback receives only the value, not the option it came from. Three
pages resolve it and they must not fork.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from roigbiv.ui.components import fov_select


def test_options_keeps_the_current_selection(monkeypatch):
    monkeypatch.setattr(
        fov_select, "list_motion_corrected_fovs",
        lambda _ws: [("a", "summary:/o/a"), ("b (input)", "input:/i/b_mc.tif")])

    opts, value = fov_select.options_and_value(object(),
                                               current="input:/i/b_mc.tif")
    assert [o["value"] for o in opts] == ["summary:/o/a", "input:/i/b_mc.tif"]
    assert value == "input:/i/b_mc.tif"            # preserved

    _, fallback = fov_select.options_and_value(object(), current="gone")
    assert fallback == "summary:/o/a"              # falls back to first

    monkeypatch.setattr(fov_select, "list_motion_corrected_fovs", lambda _ws: [])
    assert fov_select.options_and_value(object()) == ([], None)


def test_processed_only_hides_unrun_inputs(monkeypatch):
    """A seeded boundary needs a cached flow field, which only exists once a run
    has written one — offering an unrun input would be offering a dead end."""
    monkeypatch.setattr(
        fov_select, "list_motion_corrected_fovs",
        lambda _ws: [("a", "summary:/o/a"), ("b (input)", "input:/i/b_mc.tif")])

    opts, value = fov_select.processed_options_and_value(object())
    assert [o["value"] for o in opts] == ["summary:/o/a"]
    assert value == "summary:/o/a"


def test_processed_only_drops_a_selection_that_no_longer_qualifies(monkeypatch):
    monkeypatch.setattr(
        fov_select, "list_motion_corrected_fovs",
        lambda _ws: [("a", "summary:/o/a"), ("b (input)", "input:/i/b_mc.tif")])
    _, value = fov_select.processed_options_and_value(
        object(), current="input:/i/b_mc.tif")
    assert value == "summary:/o/a"


def test_summary_value_reads_the_precomputed_mean(monkeypatch):
    import numpy as np

    seen = {}

    def _fake_read(path):
        seen["path"] = path
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(fov_select, "_maybe_read_tif", _fake_read)
    mean, title, out_dir = fov_select.mean_and_title("summary:/ws/output/sess01")

    assert mean is not None and title == "sess01"
    assert out_dir == Path("/ws/output/sess01")
    assert str(seen["path"]).endswith("sess01/summary/mean_M.tif")


def test_input_value_computes_a_mean_on_demand(monkeypatch):
    """An unrun stack has no summary tif — and no output dir either, so it has
    no centroids, flows or boundaries."""
    import numpy as np

    called = {}

    def _fake_mean(path):
        called["path"] = path
        return np.zeros((8, 8), dtype=np.float32)

    def _fail_read(_path):
        raise AssertionError("input branch must not read a summary tif")

    monkeypatch.setattr(fov_select, "mc_input_mean", _fake_mean)
    monkeypatch.setattr(fov_select, "_maybe_read_tif", _fail_read)

    mean, title, out_dir = fov_select.mean_and_title("input:/data/sess01_mc.tif")
    assert mean is not None and title == "sess01 (input)"
    assert out_dir is None
    assert str(called["path"]).endswith("sess01_mc.tif")


def test_an_unparseable_value_resolves_to_nothing():
    assert fov_select.mean_and_title(None) == (None, None, None)
    assert fov_select.mean_and_title("garbage") == (None, None, None)


def test_output_dir_resolves_for_a_not_yet_processed_input(monkeypatch):
    """Calibration is meant to work ahead of the first run, same as
    centroids-only mode itself."""
    monkeypatch.setattr(
        fov_select, "get_app_state",
        lambda: SimpleNamespace(workspace=SimpleNamespace(
            output_root=Path("/ws/output"))))

    assert fov_select.resolve_output_dir(
        "summary:/ws/output/sess01") == Path("/ws/output/sess01")
    assert fov_select.resolve_output_dir(
        "input:/ws/sess01_mc.tif") == Path("/ws/output/sess01")


def test_output_dir_is_none_without_a_workspace(monkeypatch):
    monkeypatch.setattr(fov_select, "get_app_state",
                        lambda: SimpleNamespace(workspace=None))
    assert fov_select.resolve_output_dir("summary:/ws/output/sess01") is None
    assert fov_select.resolve_output_dir(None) is None


def test_the_dropdown_does_not_persist_without_a_workspace(monkeypatch):
    """A constant key would leak one workspace's selection onto the next."""
    monkeypatch.setattr(fov_select, "list_motion_corrected_fovs", lambda _ws: [])
    assert fov_select.select("x", None).persistence is False
    assert fov_select.select(
        "x", SimpleNamespace(input_root=Path("/ws"))).persistence == "/ws"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
