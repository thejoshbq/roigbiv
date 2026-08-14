"""Guards for the Boundaries page and its preview service.

Two things make the page worth having, and both are pinned here:

* the sliders are *live*, which is only true because the expensive half of the
  computation (Cellpose's pixel dynamics) is cached and does not depend on them;
* the failure modes are on screen. A high disk-fallback rate means the detector
  never fired, which ``capture_px`` cannot fix — a page that showed only
  outlines would make a detection problem look like a tuning problem.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from roigbiv.ui.pages import boundaries
from roigbiv.ui.services import boundary_preview
from roigbiv.ui.tests._tree import ids, text

H = W = 96
_PARAMS = {"detector": "cellpose", "diameter_px": 20.0,
           "cellprob_threshold": -2.0, "cellpose_model": "cyto3",
           "tissue_mask": False}


@pytest.fixture(autouse=True)
def _clean_cache():
    boundary_preview.clear_cache()
    yield
    boundary_preview.clear_cache()


def _fov(out: Path, centroids, *, blobs=None, flows=True) -> Path:
    """A FOV with centroids.json, a summary image, and a cached flow field.

    The flow field points every cell pixel at its nearest blob centre — the
    shape Cellpose produces when it does not merge two cells.
    """
    (out / "summary").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out / "summary" / "mean_M.tif",
                     np.zeros((H, W), np.float32))
    out.joinpath("centroids.json").write_text(json.dumps({
        "stem": out.name, "schema": 5, "source": "cellpose", "params": _PARAMS,
        "centroids": [{"label_id": i, "y": y, "x": x, "npix": 100,
                       "equiv_diameter_px": 11.3, "cellpose_prob": 0.9}
                      for i, (y, x) in enumerate(centroids)],
    }))
    if not flows:
        return out

    blobs = centroids if blobs is None else blobs
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cellprob = np.zeros((H, W), np.float32)
    for cy, cx in blobs:
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 7.0 ** 2))
        cellprob = np.maximum(cellprob, g.astype(np.float32))
    cellprob = cellprob * 4.0 - 2.0

    best = np.full((H, W), np.inf, np.float32)
    nearest = np.full((H, W), -1, np.int32)
    for i, (cy, cx) in enumerate(blobs):
        d = np.hypot(yy - cy, xx - cx)
        take = d < best
        best, nearest = np.where(take, d, best), np.where(take, i, nearest)
    dy, dx = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for i, (cy, cx) in enumerate(blobs):
        sel = nearest == i
        dy[sel], dx[sel] = (cy - yy)[sel], (cx - xx)[sel]

    flow_dir = out / "flows"
    flow_dir.mkdir(exist_ok=True)
    np.save(flow_dir / "dP.npy", np.stack([dy, dx]).astype(np.float32) * 5.0)
    np.save(flow_dir / "cellprob.npy", cellprob)
    (flow_dir / "meta.json").write_text(json.dumps({
        "schema": 5, "params": _PARAMS, "niter": 200,
        "dp_scale": 5.0, "diameter": 20.0, "shape": [H, W],
    }))
    return out


def _point_at(monkeypatch, out_dir: Path):
    """Make every dropdown value on the page resolve to ``out_dir``."""
    monkeypatch.setattr(
        boundaries.fov_select, "mean_and_title",
        lambda _v: (np.zeros((H, W), np.float32), out_dir.name, out_dir))


# ── the preview service ────────────────────────────────────────────────────


def test_moving_the_sliders_does_not_re_run_the_dynamics(tmp_path, monkeypatch):
    """The whole reason the page is usable.

    ``converge_pixels`` is ~0.5-2 s on CPU at 512² and depends only on the flow
    cache, so it must run once per FOV no matter how far the sliders travel.
    """
    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])

    from roigbiv.pipeline import seeded_masks

    real = seeded_masks.converge_pixels
    calls = {"n": 0}

    def _counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr("roigbiv.pipeline.seeded_masks.converge_pixels", _counted)

    for capture in (6.0, 12.0, 18.0, 24.0):
        boundary_preview.preview(out, boundaries._cfg(), capture_px=capture)

    assert calls["n"] == 1, f"dynamics re-ran {calls['n']}x across 4 slider moves"


def test_a_fresh_detection_run_invalidates_the_cache(tmp_path):
    """Keyed on the flow cache's mtime, so nobody has to remember to clear it."""
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)])
    first = boundary_preview.preview(out, boundaries._cfg(), capture_px=12.0)

    cellprob = np.load(out / "flows" / "cellprob.npy")
    np.save(out / "flows" / "cellprob.npy", cellprob * 0 - 5.0)   # all sub-threshold

    second = boundary_preview.preview(out, boundaries._cfg(), capture_px=12.0)
    assert first.n_disk_fallback == 0
    assert second.n_disk_fallback == 1, "the new field should reach the preview"


def test_a_fov_without_a_flow_cache_is_a_state_not_a_crash(tmp_path):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)], flows=False)
    with pytest.raises(boundary_preview.NoFlowCache):
        boundary_preview.preview(out, boundaries._cfg(), capture_px=12.0)


def test_a_fov_without_centroids_is_the_same_state(tmp_path):
    out = tmp_path / "sess01"
    out.mkdir()
    with pytest.raises(boundary_preview.NoFlowCache):
        boundary_preview.preview(out, boundaries._cfg(), capture_px=12.0)


# ── the statistics line ────────────────────────────────────────────────────


def test_the_statistics_report_seeds_fallbacks_and_orphans(tmp_path, monkeypatch):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])
    _point_at(monkeypatch, out)

    body = text(boundaries._stats_for("summary:x", 12.0, 0))
    assert "seeds" in body and "disk fallbacks" in body and "orphan px" in body


def test_the_disk_area_is_named_so_a_boundary_can_be_judged(tmp_path, monkeypatch):
    """If the flow-derived area is no different from the disk it replaces, this
    whole geometry track is buying nothing on this FOV."""
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)])
    _point_at(monkeypatch, out)

    body = text(boundaries._stats_for("summary:x", 12.0, 0))
    assert "median flow-derived area" in body and "would be" in body


def test_a_mass_fallback_says_capture_px_is_not_the_fix(tmp_path, monkeypatch):
    """Measured: sweeping capture_px 6→45 px moved the fallback count only
    419→393. Silence here would make a detector problem look tunable."""
    out = _fov(tmp_path / "sess01",
               [(10.0, 10.0), (10.0, 30.0), (10.0, 50.0)],
               blobs=[(80.0, 80.0)])
    _point_at(monkeypatch, out)

    body = text(boundaries._stats_for("summary:x", 3.0, 0))
    assert "the detector never fired there" in body


def test_a_missing_flow_cache_offers_the_way_out(tmp_path, monkeypatch):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)], flows=False)
    _point_at(monkeypatch, out)

    body = text(boundaries._stats_for("summary:x", 12.0, 0))
    assert "re-run centroid discovery" in body


def test_without_a_fov_the_page_names_the_earlier_steps(monkeypatch):
    monkeypatch.setattr(boundaries.fov_select, "mean_and_title",
                        lambda _v: (None, None, None))
    body = text(boundaries._stats_for(None, 12.0, 0))
    assert "centroid discovery" in body


# ── saving ─────────────────────────────────────────────────────────────────


def test_save_writes_the_image_and_pins_the_settings(tmp_path, monkeypatch):
    from roigbiv.pipeline.boundaries import resolve_capture_px

    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])
    _point_at(monkeypatch, out)
    monkeypatch.setattr(
        boundaries, "get_app_state",
        lambda: SimpleNamespace(workspace=SimpleNamespace(output_root=tmp_path)))

    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    boundaries.register_callbacks(_App())

    class _Ctx:
        triggered_id = boundaries.SAVE_ID

    monkeypatch.setattr(boundaries.dash, "ctx", _Ctx())
    report = captured["_on_save"](1, None, "summary:x", 11.0, 4)

    assert (out / "boundaries.tif").exists()
    assert "1 FOV(s) written" in text(report)
    assert resolve_capture_px(out, boundaries._cfg()) == pytest.approx(11.0)


def test_apply_to_all_names_what_it_skipped(tmp_path, monkeypatch):
    """A silent skip count reads as success on the workspace where this page is
    least useful and most needs to say so."""
    good = _fov(tmp_path / "sess01", [(30.0, 30.0)])
    _fov(tmp_path / "sess02", [(30.0, 30.0)], flows=False)
    _point_at(monkeypatch, good)
    monkeypatch.setattr(
        boundaries, "get_app_state",
        lambda: SimpleNamespace(workspace=SimpleNamespace(output_root=tmp_path)))

    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    boundaries.register_callbacks(_App())

    class _Ctx:
        triggered_id = boundaries.SAVE_ALL_ID

    monkeypatch.setattr(boundaries.dash, "ctx", _Ctx())
    report = captured["_on_save"](None, 1, "summary:x", 12.0, 0)

    body = text(report)
    assert "1 FOV(s) written" in body
    assert "1 skipped" in body and "sess02" in body


# ── layout ─────────────────────────────────────────────────────────────────


def test_the_page_carries_the_two_controls_and_the_preview(monkeypatch):
    monkeypatch.setattr(boundaries, "get_app_state",
                        lambda: SimpleNamespace(workspace=None))
    present = ids(boundaries.layout())
    for cid in (boundaries.FOV_SELECT_ID, boundaries.CAPTURE_ID,
                boundaries.MIN_AREA_ID, boundaries.PREVIEW_ID,
                boundaries.STATS_ID, boundaries.SAVE_ID,
                boundaries.SAVE_ALL_ID):
        assert cid in present


def test_the_page_ignores_config_level_boundary_overrides():
    """The slider is what the user is dragging; a value in pipeline.yaml
    outranking it would make the control inert."""
    cfg = boundaries._cfg()
    assert cfg.boundary_capture_px is None
    assert cfg.boundary_min_area is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
