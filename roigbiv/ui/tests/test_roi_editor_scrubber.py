"""Tests for timeline scrubbing routes in roi_editor.py."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from flask import Flask

from roigbiv.ui.routes import roi_editor as re_mod
from roigbiv.ui.routes.roi_editor import _get_fov_meta, register_flask_routes


# ── autouse fixture: reset module-level caches between tests ─────────────────


@pytest.fixture(autouse=True)
def _reset_caches():
    re_mod._encode_frame_png.cache_clear()
    re_mod._memmap_pool.clear()
    re_mod._fov_meta_cache.clear()
    yield
    re_mod._encode_frame_png.cache_clear()
    re_mod._memmap_pool.clear()


# ── helpers ───────────────────────────────────────────────────────────────────


def _b64(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path).encode()).decode()


def _make_output_dir(tmp_path: Path, *, with_data_bin: bool = False,
                     with_ops: bool = False, T: int = 10,
                     Ly: int = 8, Lx: int = 8, fps: float = 7.5) -> Path:
    out = tmp_path / "fov_out"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    if with_data_bin:
        data = np.zeros((T, Ly, Lx), dtype=np.int16)
        data.tofile(out / "suite2p" / "plane0" / "data.bin" if False else
                    _ensure_s2p_dir(out) / "data.bin")
    if with_ops:
        ops_path = _ensure_s2p_dir(out) / "ops.npy"
        np.save(str(ops_path), {"nframes": T, "Ly": Ly, "Lx": Lx, "fs": fps})
    return out


def _ensure_s2p_dir(out: Path) -> Path:
    d = out / out.name / "suite2p" / "plane0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_data_bin(out: Path, T: int = 10, Ly: int = 8, Lx: int = 8) -> Path:
    d = _ensure_s2p_dir(out)
    data = np.zeros((T, Ly, Lx), dtype=np.int16)
    p = d / "data.bin"
    data.tofile(str(p))
    return p


def _make_ops(out: Path, T: int = 10, Ly: int = 8, Lx: int = 8, fps: float = 7.5):
    ops_path = _ensure_s2p_dir(out) / "ops.npy"
    np.save(str(ops_path), {"nframes": T, "Ly": Ly, "Lx": Lx, "fs": fps})


def _make_mean_m(out: Path, Ly: int = 8, Lx: int = 8):
    summary = out / "summary"
    summary.mkdir(exist_ok=True)
    arr = np.ones((Ly, Lx), dtype=np.float32)
    tifffile.imwrite(str(summary / "mean_M.tif"), arr)


def _app_with_routes() -> Flask:
    app = Flask(__name__)
    register_flask_routes(app)
    return app


# ── _get_fov_meta ─────────────────────────────────────────────────────────────


def test_get_fov_meta_no_data_bin(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    # clear cache in case previous test polluted it
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    assert _get_fov_meta(out) is None


def test_get_fov_meta_with_ops_npy(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=50, Ly=8, Lx=8)
    _make_ops(out, T=50, Ly=8, Lx=8, fps=7.5)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    meta = _get_fov_meta(out)
    assert meta is not None
    assert meta["n_frames"] == 50
    assert meta["fps"] == pytest.approx(7.5)
    assert meta["height"] == 8
    assert meta["width"] == 8


def test_get_fov_meta_fallback_no_ops(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=20, Ly=8, Lx=8)
    _make_mean_m(out, Ly=8, Lx=8)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    meta = _get_fov_meta(out)
    assert meta is not None
    assert meta["n_frames"] == 20


def test_get_fov_meta_caches(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out)
    _make_ops(out)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    m1 = _get_fov_meta(out)
    m2 = _get_fov_meta(out)
    assert m1 is m2  # same dict object → cache hit


# ── /api/fov-meta route ───────────────────────────────────────────────────────


def test_fov_meta_no_data_bin(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/fov-meta/fov?dir={_b64(out)}")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["n_frames"] == 0


def test_fov_meta_with_ops_npy_route(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=50)
    _make_ops(out, T=50, fps=7.5)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/fov-meta/fov?dir={_b64(out)}")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["n_frames"] == 50
    assert data["fps"] == pytest.approx(7.5)


# ── /api/frame route ──────────────────────────────────────────────────────────


def test_fov_frame_valid(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=10, Ly=8, Lx=8)
    _make_ops(out, T=10, Ly=8, Lx=8)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/frame/fov?dir={_b64(out)}&n=3")
    assert r.status_code == 200
    assert r.content_type == "image/png"
    assert r.headers["X-Frame-Index"] == "3"


def test_fov_frame_clamp(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=10, Ly=8, Lx=8)
    _make_ops(out, T=10, Ly=8, Lx=8)
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/frame/fov?dir={_b64(out)}&n=9999")
    assert r.status_code == 200
    assert r.headers["X-Frame-Index"] == "9"


def test_fov_frame_missing_data_bin(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    re_mod._fov_meta_cache.pop(str(out.resolve()), None)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/frame/fov?dir={_b64(out)}&n=0")
    assert r.status_code == 404


# ── server-side cache + memmap pool ──────────────────────────────────────────


def test_frame_server_cache_hit(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=10, Ly=8, Lx=8)
    _make_ops(out, T=10, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=3")
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=3")
    assert re_mod._encode_frame_png.cache_info().hits >= 1


def test_frame_different_frames_isolated(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=10, Ly=8, Lx=8)
    _make_ops(out, T=10, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=0")
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=1")
    assert re_mod._encode_frame_png.cache_info().currsize == 2


def test_memmap_pool_reuse(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_data_bin(out, T=10, Ly=8, Lx=8)
    _make_ops(out, T=10, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=2")
        c.get(f"/api/frame/fov?dir={_b64(out)}&n=5")
    assert len(re_mod._memmap_pool) == 1


def test_projection_button_in_html(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_mean_m(out, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/roi-editor/fov?dir={_b64(out)}")
    assert r.status_code == 200
    assert b"btn-proj" in r.data
    assert b"_returnToProjection" in r.data


# ── Playback controls ─────────────────────────────────────────────────────────


def _get_editor_html(tmp_path: Path) -> bytes:
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_mean_m(out, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/roi-editor/fov?dir={_b64(out)}")
    assert r.status_code == 200
    return r.data


def test_playback_controls_in_html(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"btn-play" in data
    assert b"btn-step" in data
    assert b"speed-select" in data
    assert b"playback-controls" in data


def test_play_button_unicode_icons(tmp_path):
    data = _get_editor_html(tmp_path)
    assert "&#x25B6;".encode() in data   # ▶ play triangle in HTML
    assert "_togglePlay".encode() in data


def test_speed_select_options(tmp_path):
    data = _get_editor_html(tmp_path)
    for opt in (b"0.25", b"0.5", b"2", b"4"):
        assert opt in data
    assert b'value="1" selected' in data


def test_playback_js_globals_in_html(tmp_path):
    data = _get_editor_html(tmp_path)
    for sym in (b"_playing", b"_frameInFlight", b"_speedMult", b"_playTick"):
        assert sym in data


def test_step_frame_buttons(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"_stepFrame(-1)" in data
    assert b"_stepFrame(1)" in data


def test_space_key_handler_in_html(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"e.key === ' '" in data
    assert b"_togglePlay" in data


def test_frameinflight_guard_wired_both_directions(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"_frameInFlight = true" in data
    assert b"_frameInFlight = false" in data


# ── Embedded-editor: color/overlay seeding + postMessage bridges ────────────


def test_editor_page_color_overlay_params(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_mean_m(out, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/roi-editor/fov?dir={_b64(out)}&color=feature&overlay=0")
    assert r.status_code == 200
    assert b'window._colorMode = "feature"' in r.data
    assert b'window._overlayOn = ("0"' in r.data


def test_editor_page_color_defaults_to_stage(tmp_path):
    out = tmp_path / "fov"
    out.mkdir()
    (out / "pipeline_log.json").write_text("{}")
    _make_mean_m(out, Ly=8, Lx=8)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/roi-editor/fov?dir={_b64(out)}")  # no color/overlay
    assert r.status_code == 200
    assert b'window._colorMode = "stage"' in r.data
    assert b'window._overlayOn = ("1"' in r.data


def test_editor_html_has_style_bridge(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"addEventListener('message'" in data
    assert b"roigbiv-style" in data
    assert b"formatRoi" in data


def test_editor_emits_roi_selected(tmp_path):
    data = _get_editor_html(tmp_path)
    assert b"roigbiv-roi-selected" in data
    assert b"window.parent.postMessage" in data


# ── Embedded-editor: per-ROI attributes + colors in /api/annotations ────────


def _make_output_dir_with_rois(tmp_path: Path) -> Path:
    """Minimal pipeline output dir with one ROI so _rois_to_annotations runs."""
    out = tmp_path / "fov_rois"
    out.mkdir()
    (out / "pipeline_log.json").write_text(json.dumps({"shape": [1, 8, 8]}))
    # One 3x3 ROI labelled 1 in an 8x8 field.
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[2:5, 2:5] = 1
    tifffile.imwrite(str(out / "merged_masks.tif"), masks)
    (out / "roi_metadata.json").write_text(json.dumps([
        {"label_id": 1, "source_stage": 2, "confidence": "high",
         "gate_outcome": "accept", "area": 9, "activity_type": "phasic"},
    ]))
    _make_mean_m(out, Ly=8, Lx=8)
    return out


def test_annotations_embed_attrs(tmp_path):
    out = _make_output_dir_with_rois(tmp_path)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/annotations/fov_rois?dir={_b64(out)}")
    assert r.status_code == 200
    anns = r.get_json()
    assert isinstance(anns, list) and len(anns) == 1
    bodies = anns[0]["body"]
    attr_body = next(b for b in bodies if b.get("purpose") == "roigbiv-attrs")
    attrs = json.loads(attr_body["value"])
    assert attrs["label_id"] == 1
    assert attrs["source_stage"] == 2
    assert attrs["activity_type"] == "phasic"
    assert "global_cell_id" in attrs            # None when unregistered
    assert set(attrs["colors"]) == {"single", "stage", "feature", "gcid"}
    # Stage-2 color is the Suite2p orange from the shared palette.
    from roigbiv.ui.services.colors import color_for_stage
    assert attrs["colors"]["stage"] == color_for_stage(2)
    # The tagging body (used by the editor for the label) is still present.
    assert any(b.get("purpose") == "tagging" for b in bodies)


def test_annotations_id_is_label_keyed(tmp_path):
    out = _make_output_dir_with_rois(tmp_path)
    app = _app_with_routes()
    with app.test_client() as c:
        r = c.get(f"/api/annotations/fov_rois?dir={_b64(out)}")
    assert r.get_json()[0]["id"] == "roi-1"
