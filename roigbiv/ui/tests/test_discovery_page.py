"""Guards for the Discovery page — calibration, the detection run, boundary
tuning, and the OpenSeadragon viewer's gesture endpoints.

Ports the assertions from the retired ``test_centroids_page.py`` and
``test_boundaries_page.py`` onto the merged page, plus new coverage for
``discovery_edit_ops`` (the primitive centroid-op writer the viewer's
add/delete/move gestures post through) and ``discovery_api`` (the Flask
surface the viewer fetches and posts to).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.calibration import write_calibration
from roigbiv.ui.pages import discovery
from roigbiv.ui.services import boundary_preview
from roigbiv.ui.services import discovery_edit_ops as edit_ops
from roigbiv.ui.tests._tree import find_by_id, ids, text


class _FakeState:
    def __init__(self, output_root=Path("/ws/output")):
        self.workspace = (SimpleNamespace(output_root=output_root,
                                          input_root=Path("/ws"), tifs=())
                          if output_root is not None else None)
        self.registry_config = None


@pytest.fixture(autouse=True)
def _clean_cache():
    boundary_preview.clear_cache()
    yield
    boundary_preview.clear_cache()


# ── the readout (ported from test_centroids_page) ──────────────────────────


def test_readout_says_uncalibrated(tmp_path):
    assert "Not calibrated" in discovery._readout_text(None, tmp_path)


def test_readout_names_the_saved_settings_and_warns_on_existing_output(tmp_path):
    calib = write_calibration(tmp_path, 45.0, cellprob_threshold=-1.0,
                              cellpose_model="cyto3")
    body = discovery._readout_text(calib, tmp_path)
    assert "45.0px diameter" in body
    assert "cellprob_threshold=-1" in body
    assert "model=cyto3" in body
    assert "already has centroid output" not in body

    (tmp_path / "centroids.json").write_text("{}")
    assert "already has centroid output" in discovery._readout_text(calib, tmp_path)


def test_readout_names_the_deployed_model_when_unset(tmp_path):
    calib = write_calibration(tmp_path, 40.0)
    assert "model=deployed" in discovery._readout_text(calib, tmp_path)


def test_readout_is_blank_without_a_fov():
    assert discovery._readout_text(None, None) == ""


# ── the run (ported from test_centroids_page) ───────────────────────────────


def test_the_run_is_centroids_only():
    overrides = discovery.centroid_overrides(force_cpu=False, persist_flows=True)
    assert overrides["run_centroids"] is True
    assert overrides["foundation_only"] is False


def test_flow_persistence_is_on_by_default_and_reaches_the_config():
    assert discovery.centroid_overrides(False, True)["centroid_persist_flows"] is True
    assert discovery.centroid_overrides(False, False)["centroid_persist_flows"] is False


def test_the_run_carries_no_motion_correction_keys():
    overrides = discovery.centroid_overrides(False, True)
    mc_keys = [k for k in overrides if k.startswith("mc_")
               or k == "motion_correction_backend"]
    assert not mc_keys, f"a centroids-only run never registers: {mc_keys}"


# ── layout ────────────────────────────────────────────────────────────────


def test_the_page_carries_no_motion_correction_controls(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    present = ids(discovery.layout())
    mc_ids = [i for i in present
              if isinstance(i, str) and i.startswith("roigbiv-param-mc")]
    assert not mc_ids, f"MC tunables do not belong on this page: {mc_ids}"


def test_the_page_carries_the_calibration_and_run_controls(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    present = ids(discovery.layout())
    for cid in (discovery.FOV_SELECT_ID, discovery.DIAMETER_ID,
                discovery.THRESHOLD_ID, discovery.MODEL_ID, discovery.SAVE_ID,
                discovery.SAVE_CLEAR_ID, discovery.PERSIST_FLOWS_ID,
                discovery.RUN_ID):
        assert cid in present


def test_the_page_carries_the_boundary_controls(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    present = ids(discovery.layout())
    for cid in (discovery.CAPTURE_ID, discovery.MIN_AREA_ID, discovery.STATS_ID,
                discovery.SAVE_BOUNDARY_ID, discovery.SAVE_ALL_BOUNDARY_ID):
        assert cid in present


def test_the_page_carries_the_viewer_mount(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    present = ids(discovery.layout())
    assert discovery.SHEET_ID in present
    assert discovery.EDIT_ID in present


def test_the_model_choices_include_stock_cyto3(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    select = find_by_id(discovery.layout(), discovery.MODEL_ID)
    values = [o["value"] for o in select.options]
    assert "" in values, "the deployed checkpoint must remain the default"
    assert "cyto3" in values


def test_the_boundary_section_is_hidden_without_centroids(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "get_app_state",
                        lambda: _FakeState(output_root=tmp_path))
    section = find_by_id(discovery.layout(), discovery.BOUNDARY_SECTION_ID)
    assert section.style == {"display": "none"}


# ── the extraction section ──────────────────────────────────────────────────


def test_the_extraction_section_is_hidden_without_merged_masks(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "get_app_state",
                        lambda: _FakeState(output_root=tmp_path))
    section = find_by_id(discovery.layout(), discovery.EXTRACT_SECTION_ID)
    assert section.style == {"display": "none"}


def test_has_merged_masks_is_false_without_the_file(tmp_path):
    out = tmp_path / "sess01"
    out.mkdir()
    assert discovery._has_merged_masks(out) is False
    assert discovery._extraction_style(out) == {"display": "none"}


def test_has_merged_masks_is_true_once_written(tmp_path):
    out = tmp_path / "sess01"
    out.mkdir()
    tifffile.imwrite(out / "merged_masks.tif", np.zeros((8, 8), np.uint16))
    assert discovery._has_merged_masks(out) is True
    assert discovery._extraction_style(out) == {}


def test_extraction_status_without_a_bundle_says_so(tmp_path):
    out = tmp_path / "sess01"
    out.mkdir()
    status = discovery._extraction_status(out)
    assert "No trace bundle" in text(status)


def test_extraction_status_without_a_fov_is_none():
    assert discovery._extraction_status(None) is None


def test_extraction_section_carries_the_stats_checklist_and_button(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    present = ids(discovery.layout())
    for cid in (discovery.EXTRACT_STATS_ID, discovery.EXTRACT_BTN_ID,
                discovery.EXTRACT_STATUS_ID):
        assert cid in present


# ── the boundary preview (ported from test_boundaries_page) ────────────────

H = W = 96
_PARAMS = {"detector": "cellpose", "diameter_px": 20.0,
           "cellprob_threshold": -2.0, "cellpose_model": "cyto3",
           "tissue_mask": False}


def _fov(out: Path, centroids, *, blobs=None, flows=True) -> Path:
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


def test_moving_the_sliders_does_not_re_run_the_dynamics(tmp_path, monkeypatch):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])

    from roigbiv.pipeline import seeded_masks

    real = seeded_masks.converge_pixels
    calls = {"n": 0}

    def _counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr("roigbiv.pipeline.seeded_masks.converge_pixels", _counted)

    for capture in (6.0, 12.0, 18.0, 24.0):
        boundary_preview.preview(out, discovery._cfg(), capture_px=capture)

    assert calls["n"] == 1, f"dynamics re-ran {calls['n']}x across 4 slider moves"


def test_the_statistics_report_seeds_fallbacks_and_orphans(tmp_path):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])
    stats, _contours = discovery._boundary_preview_payload(out, 12.0, 0)
    body = text(stats)
    assert "seeds" in body and "disk fallbacks" in body and "orphan px" in body


def test_the_disk_area_is_named_so_a_boundary_can_be_judged(tmp_path):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)])
    stats, _contours = discovery._boundary_preview_payload(out, 12.0, 0)
    body = text(stats)
    assert "median flow-derived area" in body and "would be" in body


def test_a_mass_fallback_says_capture_px_is_not_the_fix(tmp_path):
    out = _fov(tmp_path / "sess01",
               [(10.0, 10.0), (10.0, 30.0), (10.0, 50.0)],
               blobs=[(80.0, 80.0)])
    stats, _contours = discovery._boundary_preview_payload(out, 3.0, 0)
    body = text(stats)
    assert "the detector never fired there" in body


def test_a_missing_flow_cache_offers_the_way_out(tmp_path):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0)], flows=False)
    stats, contours = discovery._boundary_preview_payload(out, 12.0, 0)
    assert "re-run centroid discovery" in text(stats)
    assert contours == {"contours": {}}


def test_without_a_fov_the_boundary_section_produces_no_stats():
    stats, contours = discovery._boundary_preview_payload(None, 12.0, 0)
    assert stats is None
    assert contours == {"contours": {}}


def test_the_contours_payload_carries_one_entry_per_label(tmp_path):
    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])
    _stats, contours = discovery._boundary_preview_payload(out, 12.0, 0)
    assert set(contours["contours"]) == {"1", "2"}
    for entry in contours["contours"].values():
        assert entry["origin"] in ("flow", "disk_fallback")
        assert entry["rings"]
        for ring in entry["rings"]:
            assert all(len(point) == 2 for point in ring)   # [x, y] pairs


def test_the_page_ignores_config_level_boundary_overrides():
    cfg = discovery._cfg()
    assert cfg.boundary_capture_px is None
    assert cfg.boundary_min_area is None


# ── saving boundaries (ported from test_boundaries_page) ───────────────────


def test_save_writes_the_image_and_pins_the_settings(tmp_path, monkeypatch):
    from roigbiv.pipeline.boundaries import resolve_capture_px

    out = _fov(tmp_path / "sess01", [(30.0, 30.0), (30.0, 66.0)])
    monkeypatch.setattr(
        discovery.fov_select, "resolve_output_dir", lambda _v: out)
    monkeypatch.setattr(
        discovery, "get_app_state",
        lambda: SimpleNamespace(workspace=SimpleNamespace(output_root=tmp_path)))

    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

        def clientside_callback(self, *a, **k):
            pass

    discovery.register_callbacks(_App())

    class _Ctx:
        triggered_id = discovery.SAVE_BOUNDARY_ID

    monkeypatch.setattr(discovery.dash, "ctx", _Ctx())
    report = captured["_on_save_boundary"](1, None, "summary:x", 11.0, 4)

    assert (out / "boundaries.tif").exists()
    assert "1 FOV(s) written" in text(report)
    assert resolve_capture_px(out, discovery._cfg()) == pytest.approx(11.0)


def test_apply_to_all_names_what_it_skipped(tmp_path, monkeypatch):
    good = _fov(tmp_path / "sess01", [(30.0, 30.0)])
    _fov(tmp_path / "sess02", [(30.0, 30.0)], flows=False)
    monkeypatch.setattr(discovery.fov_select, "resolve_output_dir", lambda _v: good)
    monkeypatch.setattr(
        discovery, "get_app_state",
        lambda: SimpleNamespace(workspace=SimpleNamespace(output_root=tmp_path)))

    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

        def clientside_callback(self, *a, **k):
            pass

    discovery.register_callbacks(_App())

    class _Ctx:
        triggered_id = discovery.SAVE_ALL_BOUNDARY_ID

    monkeypatch.setattr(discovery.dash, "ctx", _Ctx())
    report = captured["_on_save_boundary"](None, 1, "summary:x", 12.0, 0)

    body = text(report)
    assert "1 FOV(s) written" in body
    assert "1 skipped" in body and "sess02" in body


# ── discovery_edit_ops: the primitive op writer ─────────────────────────────


def _centroids_only(out: Path, points) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    out.joinpath("centroids.json").write_text(json.dumps({
        "stem": out.name, "schema": 5,
        "centroids": [{"label_id": i, "y": y, "x": x, "npix": 10}
                      for i, (y, x) in enumerate(points, start=1)],
    }))
    return out


def test_a_gesture_against_a_fov_with_no_centroids_is_refused(tmp_path):
    out = tmp_path / "sess01"
    out.mkdir()
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="add", y=1.0, x=1.0))
    assert not result.ok
    assert result.status == 409


def test_add_appends_a_centroid_and_returns_the_fresh_set(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="add", y=5.0, x=5.0))
    assert result.ok
    assert len(result.centroids) == 2
    assert (out / "corrections" / "centroids.jsonl").exists()


def test_delete_drops_the_named_label(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0), (20.0, 20.0)])
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="delete", label=1))
    assert result.ok
    assert [c[0] for c in result.centroids] == [2]


def test_deleting_an_unknown_label_is_refused(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="delete", label=99))
    assert not result.ok
    assert result.status == 400


def test_move_relocates_without_snapping(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="move", label=1, y=12.0, x=8.0))
    assert result.ok
    moved = {label: (y, x) for label, y, x in result.centroids}
    assert moved[1] == (12.0, 8.0)


def test_undo_reverses_the_last_op(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    edit_ops.apply_gesture(out, edit_ops.Gesture(kind="add", y=5.0, x=5.0))
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="undo"))
    assert result.ok
    assert len(result.centroids) == 1


def test_undo_with_nothing_written_says_so(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="undo"))
    assert not result.ok
    assert result.message == "nothing to undo"


def test_pipeline_output_is_never_mutated_by_an_edit(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    frozen = (out / "centroids.json").read_text()
    edit_ops.apply_gesture(out, edit_ops.Gesture(kind="add", y=5.0, x=5.0))
    edit_ops.apply_gesture(out, edit_ops.Gesture(kind="delete", label=1))
    assert (out / "centroids.json").read_text() == frozen


@pytest.mark.parametrize("payload,message", [
    ({"kind": "levitate"}, "unknown gesture kind"),
    ({"kind": "move"}, "requires a label"),
    ({"kind": "add", "label": 1}, "requires y and x"),
    ({"kind": "draw_boundary", "ring": [[0, 0], [0, 5], [5, 5]]},
     "requires a label"),
    ({"kind": "draw_boundary", "label": 1}, "requires a ring"),
    ({"kind": "draw_boundary", "label": 1, "ring": [[0, 0], [0, 5]]},
     "requires a ring"),
    ({"kind": "delete_boundary"}, "requires a label"),
])
def test_a_malformed_gesture_payload_is_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        edit_ops.Gesture.from_payload(payload)


# ── discovery_edit_ops: boundary gestures ───────────────────────────────────

_RING = [(2.0, 2.0), (2.0, 12.0), (12.0, 12.0), (12.0, 2.0)]


def test_draw_boundary_appends_an_op_and_leaves_centroids_untouched(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="draw_boundary", label=1, ring=_RING))
    assert result.ok
    assert [c[0] for c in result.centroids] == [1]
    assert (out / "corrections" / "boundaries.jsonl").exists()


def test_draw_boundary_for_an_unknown_label_is_refused(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="draw_boundary", label=99, ring=_RING))
    assert not result.ok
    assert result.status == 400


def test_delete_boundary_with_no_active_manual_boundary_is_refused(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="delete_boundary", label=1))
    assert not result.ok
    assert result.status == 400


def test_delete_boundary_after_a_draw_succeeds(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="draw_boundary", label=1, ring=_RING))
    result = edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="delete_boundary", label=1))
    assert result.ok

    from roigbiv.pipeline.boundary_edits import active_manual_labels, load_boundary_ops
    assert active_manual_labels(load_boundary_ops(out)) == set()


def test_undo_boundary_reverses_the_last_boundary_op(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="draw_boundary", label=1, ring=_RING))
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="undo_boundary"))
    assert result.ok

    from roigbiv.pipeline.boundary_edits import active_manual_labels, load_boundary_ops
    assert active_manual_labels(load_boundary_ops(out)) == set()


def test_undo_boundary_with_nothing_written_says_so(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    result = edit_ops.apply_gesture(out, edit_ops.Gesture(kind="undo_boundary"))
    assert not result.ok
    assert result.message == "nothing to undo"


def test_boundary_gestures_never_mutate_centroids_json(tmp_path):
    out = _centroids_only(tmp_path / "sess01", [(10.0, 10.0)])
    frozen = (out / "centroids.json").read_text()
    edit_ops.apply_gesture(
        out, edit_ops.Gesture(kind="draw_boundary", label=1, ring=_RING))
    edit_ops.apply_gesture(out, edit_ops.Gesture(kind="delete_boundary", label=1))
    assert (out / "centroids.json").read_text() == frozen


# ── discovery_api: the Flask surface the viewer talks to ───────────────────


@pytest.fixture
def api_workspace(tmp_path):
    out = _fov(tmp_path / "sess01", [(20.0, 20.0), (40.0, 40.0)])
    return SimpleNamespace(
        tmp_path=tmp_path, out_dir=out,
        state=SimpleNamespace(workspace=SimpleNamespace(output_root=tmp_path),
                              registry_config=None),
    )


@pytest.fixture
def api_client(api_workspace, monkeypatch):
    from roigbiv.ui.app import build_app
    from roigbiv.ui.routes import discovery_api

    app = build_app()
    monkeypatch.setattr(discovery_api, "get_app_state",
                        lambda: api_workspace.state)
    return app.server.test_client()


def test_the_state_carries_the_image_url_and_centroids(api_client):
    resp = api_client.get("/api/discovery/sess01")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["image_url"] == "/api/discovery/sess01/image.png"
    assert len(body["centroids"]) == 2
    assert body["radius"] > 0


def test_an_unknown_stem_is_a_404(api_client):
    assert api_client.get("/api/discovery/nope").status_code == 404


@pytest.mark.parametrize("stem", ["..%2f..%2fetc", "..", "a%2fb"])
def test_no_crafted_stem_ever_resolves(api_client, stem):
    """*stem* is joined onto ``output_root`` and must resolve to a direct,
    existing child of it. A stem containing an encoded path separator never
    reaches this route at all and falls through to Dash's page router (200,
    the app shell); a bare ``..`` reaches the route and is refused there
    (404). Neither ever answers with real FOV state."""
    resp = api_client.get(f"/api/discovery/{stem}")
    if resp.mimetype == "application/json":
        assert "centroids" not in (resp.get_json() or {})
    else:
        assert resp.status_code == 200   # Dash's SPA shell, not our route


def test_the_image_is_served_as_a_png(api_client):
    resp = api_client.get("/api/discovery/sess01/image.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_image_is_cacheable_and_revalidates_on_its_etag(api_client):
    first = api_client.get("/api/discovery/sess01/image.png")
    etag = first.headers.get("ETag")
    assert etag
    second = api_client.get("/api/discovery/sess01/image.png",
                            headers={"If-None-Match": etag})
    assert second.status_code == 304


def test_adding_returns_fresh_state(api_client, api_workspace):
    resp = api_client.post("/api/discovery/sess01/gesture",
                           json={"kind": "add", "y": 55.0, "x": 12.0})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"]
    assert len(body["state"]["centroids"]) == 3
    assert (api_workspace.out_dir / "corrections" / "centroids.jsonl").exists()


def test_deleting_drops_the_centroid(api_client):
    label = api_client.get("/api/discovery/sess01").get_json()["centroids"][0]["label_id"]
    body = api_client.post("/api/discovery/sess01/gesture",
                           json={"kind": "delete", "label": label}).get_json()
    assert body["ok"]
    assert label not in [c["label_id"] for c in body["state"]["centroids"]]


def test_a_malformed_gesture_is_a_400(api_client):
    resp = api_client.post("/api/discovery/sess01/gesture",
                           json={"kind": "levitate"})
    assert resp.status_code == 400


# ── discovery_api: boundary gestures ────────────────────────────────────────


def test_drawing_a_boundary_returns_fresh_contours(api_client, api_workspace):
    label = api_client.get("/api/discovery/sess01").get_json()["centroids"][0]["label_id"]
    resp = api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "draw_boundary", "label": label, "ring": _RING,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"]
    assert "boundaries" in body
    entry = body["boundaries"]["contours"][str(label)]
    assert entry["origin"] == "manual"
    assert (api_workspace.out_dir / "corrections" / "boundaries.jsonl").exists()


def test_drawing_a_boundary_for_an_unknown_label_is_a_400(api_client):
    resp = api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "draw_boundary", "label": 999, "ring": _RING,
    })
    assert resp.status_code == 400


def test_deleting_a_drawn_boundary_reverts_its_origin(api_client):
    label = api_client.get("/api/discovery/sess01").get_json()["centroids"][0]["label_id"]
    api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "draw_boundary", "label": label, "ring": _RING,
    })
    resp = api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "delete_boundary", "label": label,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"]
    entry = body["boundaries"]["contours"].get(str(label))
    if entry is not None:   # still present, just no longer manual
        assert entry["origin"] != "manual"


def test_deleting_a_never_drawn_boundary_is_a_400(api_client):
    label = api_client.get("/api/discovery/sess01").get_json()["centroids"][0]["label_id"]
    resp = api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "delete_boundary", "label": label,
    })
    assert resp.status_code == 400


def test_undo_boundary_via_the_api(api_client):
    label = api_client.get("/api/discovery/sess01").get_json()["centroids"][0]["label_id"]
    api_client.post("/api/discovery/sess01/gesture", json={
        "kind": "draw_boundary", "label": label, "ring": _RING,
    })
    resp = api_client.post("/api/discovery/sess01/gesture",
                           json={"kind": "undo_boundary"})
    assert resp.status_code == 200
    assert resp.get_json()["ok"]


def test_a_centroid_gesture_does_not_recompute_boundary_contours(api_client):
    """Boundary contours are only recomputed for boundary-kind gestures — a
    plain centroid add/move/delete does not pay for a redraw it didn't ask for."""
    resp = api_client.post("/api/discovery/sess01/gesture",
                           json={"kind": "add", "y": 55.0, "x": 12.0})
    assert "boundaries" not in resp.get_json()


# ── discovery_api: traces.h5 download ───────────────────────────────────────


def test_traces_download_404s_without_a_bundle(api_client):
    resp = api_client.get("/api/discovery/sess01/traces.h5")
    assert resp.status_code == 404


def test_traces_download_404s_for_an_unknown_stem(api_client):
    resp = api_client.get("/api/discovery/nope/traces.h5")
    assert resp.status_code == 404


def test_traces_download_serves_the_freshest_bundle(api_client, api_workspace):
    import pandas as pd

    from roigbiv.pipeline.traces_io import write_traces_bundle
    from roigbiv.pipeline.types import PipelineConfig, ROI

    out = api_workspace.out_dir
    mask = np.zeros((96, 96), dtype=bool)
    mask[0:4, 0:4] = True
    roi = ROI(mask=mask, label_id=1, source_stage=1, confidence="high",
              gate_outcome="accept", area=int(mask.sum()))
    F = np.ones((1, 5), dtype=np.float32)
    write_traces_bundle(
        [roi], F, np.zeros_like(F), F, out, PipelineConfig(fs=7.5),
        source="discovery",
    )

    resp = api_client.get("/api/discovery/sess01/traces.h5")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "application/x-hdf5"
    assert "sess01_traces.h5" in resp.headers["Content-Disposition"]

    tmp = tmp_path_for_response(resp)
    with pd.HDFStore(str(tmp), "r") as store:
        assert "/f" in store
        meta = store["/meta"]
        assert list(meta.index) == ["lcl:1"]


def tmp_path_for_response(resp) -> Path:
    """Write a test-client response's bytes to a scratch file so pandas'
    ``HDFStore`` (which needs a real path, not a file-like object) can read
    it back."""
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".h5")
    with open(fd, "wb") as f:
        f.write(resp.get_data())
    return Path(name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
