"""Guards for the /discovery movie player — the frame source, the crop/decimate
read, and the two Flask routes the browser-side ring buffer talks to.

The assertions that matter here are the ones the plan's acceptance criteria
rest on: a chunk contains *exactly* the frames and pixels it claims to, and a
request the server had to clamp comes back describing what it actually served
rather than what was asked for. Those two together are what let the client
buffer without ever guessing.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from roigbiv.ui.pages import discovery
from roigbiv.ui.services import movie_source
from roigbiv.ui.tests._tree import find_by_id


T, LY, LX = 40, 24, 32


@pytest.fixture(autouse=True)
def _clear_caches():
    movie_source.clear_cache()
    yield
    movie_source.clear_cache()


# ── fixtures on disk ────────────────────────────────────────────────────────


def _ramp(t: int = T, ly: int = LY, lx: int = LX) -> np.ndarray:
    """A movie where every voxel's value identifies its own coordinates.

    ``value = t*1000 + y*32 + x`` is unique per (t, y, x) within these bounds,
    so a wrongly-strided or wrongly-offset read cannot coincidentally look
    right.
    """
    t_i, y_i, x_i = np.indices((t, ly, lx))
    return (t_i * 1000 + y_i * 32 + x_i).astype(np.int16)


def _fov(root: Path, stem: str = "sess01", *, nested: bool = True,
         fs: float = 7.5) -> Path:
    """A FOV output dir with a Suite2p ``data.bin`` + ``ops.npy``."""
    out = root / stem
    plane = (out / stem if nested else out) / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    _ramp().tofile(str(plane / "data.bin"))
    np.save(str(plane / "ops.npy"),
            {"nframes": T, "Ly": LY, "Lx": LX, "fs": fs})
    (out / "pipeline_log.json").write_text("{}")
    return out


def _fov_mc_tif_only(root: Path, stem: str = "sess02") -> Path:
    out = root / stem
    out.mkdir(parents=True)
    tifffile.imwrite(str(out / f"{stem}_mc.tif"),
                     np.clip(_ramp(), 0, None).astype(np.uint16))
    return out


def _expected(block_lo_hi, start, count, x, y, w, h, ds) -> np.ndarray:
    lo, hi = block_lo_hi
    raw = _ramp()[start:start + count, y:y + h:ds, x:x + w:ds].astype(np.float32)
    scaled = np.clip((raw - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8)


# ── resolve_movie ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("nested", [True, False])
def test_resolve_finds_data_bin_in_either_layout(tmp_path, nested):
    """``run_suite2p_fov`` writes to ``<stem>/<stem>/suite2p/plane0``; some
    FOVs have since been flattened. Both are real, on-disk shapes."""
    src = movie_source.resolve_movie(_fov(tmp_path, nested=nested))
    assert src is not None
    assert src.kind == "data_bin"
    assert src.shape == (T, LY, LX)
    assert src.fps == pytest.approx(7.5)


def test_resolve_falls_back_to_the_mc_tif(tmp_path):
    src = movie_source.resolve_movie(_fov_mc_tif_only(tmp_path))
    assert src is not None
    assert src.kind == "mc_tif"
    assert src.shape == (T, LY, LX)


def test_a_frameless_data_bin_is_not_a_movie(tmp_path):
    """An interrupted motion-correction run leaves a zero-length ``data.bin``.
    Left resolvable, it reaches ``np.memmap`` and raises "cannot mmap an empty
    file" out of ``/movie/meta`` as a 500, instead of the player simply saying
    this FOV has no movie."""
    out = _fov(tmp_path)
    (out / out.name / "suite2p" / "plane0" / "data.bin").write_bytes(b"")
    assert movie_source.resolve_movie(out) is None


def test_resolve_returns_none_without_a_movie(tmp_path):
    empty = tmp_path / "sess03"
    empty.mkdir()
    assert movie_source.resolve_movie(empty) is None


# ── memmap pool ─────────────────────────────────────────────────────────────


def test_the_memmap_is_pooled(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    assert movie_source.open_movie(src) is movie_source.open_movie(src)


def test_a_rewritten_movie_is_not_served_from_the_stale_pool(tmp_path):
    """Re-running motion correction rewrites ``data.bin`` in place. A pool
    keyed on path alone would keep handing back a mapping of the old length —
    the bug ``roi_editor``'s pool has, and the reason this one keys on
    (path, size, mtime)."""
    out = _fov(tmp_path)
    src = movie_source.resolve_movie(out)
    first = movie_source.open_movie(src)

    plane = out / out.name / "suite2p" / "plane0"
    _ramp(t=T + 10).tofile(str(plane / "data.bin"))
    np.save(str(plane / "ops.npy"),
            {"nframes": T + 10, "Ly": LY, "Lx": LX, "fs": 7.5})

    again = movie_source.resolve_movie(out)
    assert again.shape[0] == T + 10
    second = movie_source.open_movie(again)
    assert second is not first
    assert second.shape[0] == T + 10


# ── display window ──────────────────────────────────────────────────────────


def test_the_display_window_is_computed_once_per_movie(tmp_path, monkeypatch):
    src = movie_source.resolve_movie(_fov(tmp_path))
    calls = []
    real = movie_source._compute_window
    monkeypatch.setattr(movie_source, "_compute_window",
                        lambda s: (calls.append(s), real(s))[1])
    first = movie_source.display_window(src)
    assert movie_source.display_window(src) == first
    assert len(calls) == 1


def test_the_display_window_spans_the_whole_movie(tmp_path):
    """Fixed, not per-frame: a per-frame stretch re-normalises against each
    frame's own extremes, so the background pulses against any transient."""
    src = movie_source.resolve_movie(_fov(tmp_path))
    lo, hi = movie_source.display_window(src)
    assert lo < hi
    # The ramp's last frame is far brighter than its first; a window derived
    # from one frame could not span both.
    assert hi > float(_ramp()[0].max())


# ── read_block ──────────────────────────────────────────────────────────────


def test_a_full_frame_read_is_byte_exact(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=3, count=4, x=0, y=0,
                                     w=LX, h=LY, ds=1)
    block = movie_source.read_block(src, req)
    assert block.shape == (4, LY, LX)
    np.testing.assert_array_equal(
        block, _expected(movie_source.display_window(src), 3, 4, 0, 0, LX, LY, 1))


def test_a_cropped_read_lands_on_the_right_pixels(tmp_path):
    """The crop is what makes zoomed-in playback cheap, so its origin has to be
    exact — an off-by-one here shifts the movie against the markers drawn over
    it."""
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=0, count=2, x=5, y=7,
                                     w=8, h=6, ds=1)
    block = movie_source.read_block(src, req)
    assert block.shape == (2, 6, 8)
    np.testing.assert_array_equal(
        block, _expected(movie_source.display_window(src), 0, 2, 5, 7, 8, 6, 1))


def test_a_decimated_read_strides_from_the_crop_origin(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=1, count=2, x=4, y=4,
                                     w=16, h=12, ds=4)
    block = movie_source.read_block(src, req)
    assert (req.rows, req.cols) == (3, 4)
    assert block.shape == (2, 3, 4)
    np.testing.assert_array_equal(
        block, _expected(movie_source.display_window(src), 1, 2, 4, 4, 16, 12, 4))


def test_the_block_is_frame_major_and_contiguous(tmp_path):
    """The client slices the response by ``i * cols * rows``; anything but
    C-contiguous frame-major would tear each frame across the next."""
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=0, count=3, x=0, y=0,
                                     w=LX, h=LY, ds=1)
    block = movie_source.read_block(src, req)
    assert block.flags["C_CONTIGUOUS"]
    flat = block.tobytes()
    per = req.rows * req.cols
    for i in range(3):
        np.testing.assert_array_equal(
            np.frombuffer(flat[i * per:(i + 1) * per], dtype=np.uint8),
            block[i].ravel())


# ── clamping ────────────────────────────────────────────────────────────────


def test_a_start_past_the_end_clamps_to_the_last_frame(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=9999, count=8, x=0, y=0,
                                     w=LX, h=LY, ds=1)
    assert req.start == T - 1
    assert req.count == 1


def test_a_rect_running_off_the_edge_is_trimmed(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=0, count=1, x=LX - 4, y=LY - 3,
                                     w=100, h=100, ds=1)
    assert (req.x, req.y, req.w, req.h) == (LX - 4, LY - 3, 4, 3)


def test_a_count_over_the_cap_is_trimmed(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=0, count=10_000, x=0, y=0,
                                     w=LX, h=LY, ds=1)
    assert req.count <= movie_source.MAX_COUNT


def test_a_zero_or_negative_decimation_becomes_one(tmp_path):
    src = movie_source.resolve_movie(_fov(tmp_path))
    assert movie_source.clamp_request(src, start=0, count=1, x=0, y=0,
                                      w=LX, h=LY, ds=0).ds == 1
    assert movie_source.clamp_request(src, start=0, count=1, x=0, y=0,
                                      w=LX, h=LY, ds=-4).ds == 1


def test_a_request_cannot_exceed_the_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(movie_source, "MAX_BYTES", LY * LX * 3)
    src = movie_source.resolve_movie(_fov(tmp_path))
    req = movie_source.clamp_request(src, start=0, count=64, x=0, y=0,
                                     w=LX, h=LY, ds=1)
    assert req.count == 3


# ── the Flask routes ────────────────────────────────────────────────────────


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    from roigbiv.ui.app import build_app
    from roigbiv.ui.routes import discovery_api

    _fov(tmp_path)
    (tmp_path / "sess03").mkdir()
    app = build_app()
    monkeypatch.setattr(
        discovery_api, "get_app_state",
        lambda: SimpleNamespace(
            workspace=SimpleNamespace(output_root=tmp_path),
            registry_config=None),
    )
    return app.server.test_client()


def test_meta_describes_the_movie(api_client):
    body = api_client.get("/api/discovery/sess01/movie/meta").get_json()
    assert body["available"] is True
    assert body["kind"] == "data_bin"
    assert body["n_frames"] == T
    assert (body["height"], body["width"]) == (LY, LX)
    assert body["fps"] == pytest.approx(7.5)
    assert body["window"][0] < body["window"][1]
    assert body["max_count"] == movie_source.MAX_COUNT


def test_meta_survives_a_frameless_movie(api_client, tmp_path):
    (tmp_path / "sess01" / "sess01" / "suite2p" / "plane0"
     / "data.bin").write_bytes(b"")
    movie_source.clear_cache()
    resp = api_client.get("/api/discovery/sess01/movie/meta")
    assert resp.status_code == 200
    assert resp.get_json()["available"] is False


def test_meta_says_why_there_is_no_movie(api_client):
    body = api_client.get("/api/discovery/sess03/movie/meta").get_json()
    assert body["available"] is False
    assert "Motion Correction" in body["reason"]


def test_a_chunk_is_raw_bytes_matching_its_headers(api_client):
    resp = api_client.get(
        "/api/discovery/sess01/movie/chunk"
        "?start=2&count=5&x=4&y=4&w=16&h=8&ds=2")
    assert resp.status_code == 200
    assert resp.mimetype == "application/octet-stream"
    cols = int(resp.headers["X-Movie-Cols"])
    rows = int(resp.headers["X-Movie-Rows"])
    count = int(resp.headers["X-Movie-Count"])
    assert (cols, rows, count) == (8, 4, 5)
    assert resp.headers["X-Movie-Rect"] == "4,4,16,8"
    assert resp.headers["X-Movie-Ds"] == "2"
    assert len(resp.data) == count * rows * cols


def test_a_chunk_carries_the_pixels_it_claims_to(api_client, tmp_path):
    resp = api_client.get(
        "/api/discovery/sess01/movie/chunk?start=6&count=3&x=1&y=2&w=9&h=7&ds=1")
    src = movie_source.resolve_movie(tmp_path / "sess01")
    served = np.frombuffer(resp.data, dtype=np.uint8).reshape(3, 7, 9)
    np.testing.assert_array_equal(
        served, _expected(movie_source.display_window(src), 6, 3, 1, 2, 9, 7, 1))


def test_a_clamped_chunk_reports_what_it_actually_served(api_client):
    """The client trusts these headers over its own request, so a trimmed ask
    has to land as valid data rather than as a hole to retry around."""
    resp = api_client.get(
        "/api/discovery/sess01/movie/chunk"
        f"?start={T - 2}&count=32&x=0&y=0&w=999&h=999&ds=1")
    assert resp.status_code == 200
    assert int(resp.headers["X-Movie-Start"]) == T - 2
    assert int(resp.headers["X-Movie-Count"]) == 2
    assert resp.headers["X-Movie-Rect"] == f"0,0,{LX},{LY}"
    assert len(resp.data) == 2 * LY * LX


def test_a_chunk_is_never_http_cached(api_client):
    resp = api_client.get("/api/discovery/sess01/movie/chunk?start=0&count=1")
    assert resp.headers["Cache-Control"] == "no-store"


def test_a_malformed_parameter_is_a_400(api_client):
    resp = api_client.get(
        "/api/discovery/sess01/movie/chunk?start=0&count=1&ds=banana")
    assert resp.status_code == 400
    assert "ds" in resp.get_json()["error"]


def test_a_chunk_for_a_movieless_fov_is_a_404(api_client):
    assert api_client.get(
        "/api/discovery/sess03/movie/chunk?start=0&count=1").status_code == 404


def test_no_route_takes_a_filesystem_path(api_client, tmp_path):
    """Unlike ``roi_editor``'s base64 ``?dir=``, these resolve the stem against
    the session's own workspace. A path in the query string must be inert."""
    resp = api_client.get(
        f"/api/discovery/sess03/movie/meta?dir={tmp_path / 'sess01'}")
    assert resp.get_json()["available"] is False


# ── the page control ────────────────────────────────────────────────────────


class _FakeState:
    def __init__(self, output_root=Path("/ws/output")):
        self.workspace = SimpleNamespace(output_root=output_root,
                                         input_root=Path("/ws"), tifs=())
        self.registry_config = None


def test_the_page_carries_the_live_movie_switch(monkeypatch):
    monkeypatch.setattr(discovery, "get_app_state", lambda: _FakeState())
    assert find_by_id(discovery.layout(), discovery.PLAY_ID) is not None


def test_the_player_asset_exposes_what_the_sheet_calls(monkeypatch):
    """``discovery_sheet.js`` drives the player through these four entry
    points; there is no JS test runner here, so this is the guard that the two
    files do not drift apart."""
    js = (Path(__file__).resolve().parents[1] / "assets"
          / "discovery_player.js").read_text()
    for name in ("mount:", "setStem:", "setEnabled:", "setRect:", "destroy:"):
        assert name in js
    assert "window.roigbivDiscoveryPlayer" in js


def test_the_player_holds_the_frame_exactness_contract():
    """Runs ``tests/js/player_check.js`` — the real render loop against a
    stubbed DOM and a controllable movie server.

    This is the only automated guard on the two rules the feature is for: that
    playback paints every frame in order, and that an underrun holds the
    playhead instead of banking elapsed time and skipping ahead once the data
    lands. Skipped rather than failed without node, since nothing else in this
    repo needs it.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = Path(__file__).resolve().parent / "js" / "player_check.js"
    proc = subprocess.run([node, str(script)], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_sheet_stacks_the_movie_under_the_markers():
    """The canvas must be added to the overlay before the SVG (DOM order is
    z-order) and must never take pointer events, or it would swallow the edit
    gestures the SVG layer exists to receive."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    sheet = (assets / "discovery_sheet.js").read_text()
    player = (assets / "discovery_player.js").read_text()
    body = sheet.split("function attachOverlay")[1]
    assert body.index("attachFrameCanvas(viewer)") < body.index("createElementNS")
    assert "pointer-events: none" in player
