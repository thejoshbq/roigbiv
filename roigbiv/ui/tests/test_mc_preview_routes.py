"""Guards for the live motion-correction preview routes and page card.

The routes take a FOV ``stem`` straight off the query string, so the path
resolution is security-relevant: it must stay inside the *requesting session's*
workspace output root and never accept a client-supplied filesystem path.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roigbiv.pipeline.mc_preview import MCPreviewWriter, preview_dir
from roigbiv.ui.services.mc_preview import (
    STALE_AFTER_S,
    fov_preview_dir,
    latest_state,
    list_states,
    read_state,
)


def _sidecar(out_root: Path, stem: str, *, n=3, phase="registering"):
    """Write a real sidecar for ``stem`` using the production writer."""
    rng = np.random.default_rng(abs(hash(stem)) % 2**31)
    fov_dir = out_root / stem
    fov_dir.mkdir(parents=True, exist_ok=True)
    w = MCPreviewWriter(fov_dir, stem=stem, backend="phasecorr",
                        min_interval_s=0.0, max_dim=32, metrics=False)
    w.set_total(100)
    w.set_phase(phase)
    base = rng.normal(500, 50, (64, 64)).astype(np.float32)
    for i in range(n):
        w.record_shifts(i * 10, np.full(10, 2.0), np.full(10, -1.0),
                        np.full(10, 0.9))
        w.emit(np.roll(base, i + 1, axis=0), base, frame_index=i * 10,
               n_done=(i + 1) * 10)
    return w


@pytest.fixture
def workspace(tmp_path):
    out_root = tmp_path / "output"
    out_root.mkdir()
    return SimpleNamespace(output_root=out_root, input_root=tmp_path)


@pytest.fixture
def client(workspace, monkeypatch):
    """Flask test client whose session AppState points at ``workspace``."""
    from roigbiv.ui.app import build_app
    import roigbiv.ui.services.app_state as app_state

    app = build_app()
    state = app_state.AppState()
    state.workspace = workspace
    monkeypatch.setattr(app_state, "_instances", {"test-session": state})
    # The routes import get_app_state lazily from the module, so patching the
    # module attribute is enough — no session cookie juggling required.
    monkeypatch.setattr(app_state, "get_app_state", lambda: state)
    return app.server.test_client()


# ── path resolution ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("stem", ["", ".", "..", "../etc", "a/b", "a\\b"])
def test_fov_preview_dir_rejects_traversal(workspace, stem):
    with pytest.raises(ValueError):
        fov_preview_dir(workspace, stem)


def test_fov_preview_dir_requires_a_workspace():
    with pytest.raises(ValueError):
        fov_preview_dir(None, "fovA")


def test_fov_preview_dir_resolves_under_the_output_root(workspace):
    pdir = fov_preview_dir(workspace, "fovA")
    assert pdir == workspace.output_root / "fovA" / "mc_preview"


# ── state reading ───────────────────────────────────────────────────────────

def test_read_state_returns_none_when_absent(tmp_path):
    assert read_state(tmp_path) is None


def test_read_state_survives_a_truncated_file(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    assert read_state(tmp_path) is None


def test_stale_flag_tracks_updated_at(workspace):
    _sidecar(workspace.output_root, "fovA")
    pdir = preview_dir(workspace.output_root / "fovA")
    assert read_state(pdir)["stale"] is False

    state = json.loads((pdir / "state.json").read_text())
    state["updated_at"] = time.time() - (STALE_AFTER_S + 60)
    (pdir / "state.json").write_text(json.dumps(state))
    assert read_state(pdir)["stale"] is True


def test_list_states_surfaces_every_fov_newest_first(workspace):
    _sidecar(workspace.output_root, "fovA")
    time.sleep(0.01)
    _sidecar(workspace.output_root, "fovB")
    # Batch mode runs two FOVs concurrently; both must show up, with no shared
    # "current FOV" pointer to race over.
    stems = [s["stem"] for s in list_states(workspace)]
    assert set(stems) == {"fovA", "fovB"}
    assert stems[0] == "fovB"
    assert latest_state(workspace)["stem"] == "fovB"


def test_list_states_without_workspace_is_empty():
    assert list_states(None) == []
    assert latest_state(None) is None


# ── routes ──────────────────────────────────────────────────────────────────

def test_list_route(client, workspace):
    _sidecar(workspace.output_root, "fovA")
    resp = client.get("/api/mc-preview/list")
    assert resp.status_code == 200
    (entry,) = resp.get_json()
    assert entry["stem"] == "fovA"
    assert entry["phase"] == "registering"
    assert entry["seq"] >= 0
    assert entry["n_total"] == 100
    # The fast poll must stay small — traces and metrics belong to /state.
    assert "shifts" not in entry and "live_metrics" not in entry


def test_state_route(client, workspace):
    _sidecar(workspace.output_root, "fovA")
    resp = client.get("/api/mc-preview/state?stem=fovA")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["stem"] == "fovA"
    assert len(body["shifts"]["y"]) == 30
    assert body["valid_crop_frac"] is not None
    assert resp.headers["Cache-Control"] == "no-store"


def test_state_route_404s_for_a_fov_without_a_sidecar(client, workspace):
    (workspace.output_root / "fovA").mkdir()
    assert client.get("/api/mc-preview/state?stem=fovA").status_code == 404


@pytest.mark.parametrize("stem", ["../../etc", "a/b", ""])
def test_state_route_rejects_traversal(client, stem):
    assert client.get(f"/api/mc-preview/state?stem={stem}").status_code == 400


@pytest.mark.parametrize("kind", ["raw", "corr", "corrected", "avg"])
def test_image_route_serves_png(client, workspace, kind):
    w = _sidecar(workspace.output_root, "fovA")
    resp = client.get(f"/api/mc-preview/image?stem=fovA&seq={w._seq}&kind={kind}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data[:4] == b"\x89PNG"
    # Seq-suffixed frames are write-once, so they are safe to cache hard.
    assert "immutable" in resp.headers["Cache-Control"]


def test_image_route_bad_inputs(client, workspace):
    _sidecar(workspace.output_root, "fovA")
    assert client.get(
        "/api/mc-preview/image?stem=fovA&seq=9999&kind=raw").status_code == 404
    assert client.get(
        "/api/mc-preview/image?stem=fovA&seq=0&kind=bogus").status_code == 400
    assert client.get(
        "/api/mc-preview/image?stem=fovA&seq=x&kind=raw").status_code == 400
    assert client.get(
        "/api/mc-preview/image?stem=fovA&kind=raw").status_code == 400


def test_routes_are_quiet_without_a_workspace(client, monkeypatch):
    import roigbiv.ui.services.app_state as app_state

    monkeypatch.setattr(app_state, "get_app_state",
                        lambda: SimpleNamespace(workspace=None))
    assert client.get("/api/mc-preview/list").get_json() == []
    assert client.get("/api/mc-preview/state?stem=fovA").status_code == 400


# ── page rendering ──────────────────────────────────────────────────────────

def _ids(component):
    found = []
    stack = [component]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.append(cid)
        children = getattr(node, "children", None)
        if children is not None:
            stack.append(children)
    return found


def test_live_card_renders_above_the_existing_preview():
    from roigbiv.ui.pages.process import _live_mc_section, _mc_preview_section

    ids = _ids(_live_mc_section())
    for expected in ("roigbiv-mc-live-tick", "roigbiv-mc-live-raw",
                     "roigbiv-mc-live-corr", "roigbiv-mc-live-avg",
                     "roigbiv-mc-live-blink", "roigbiv-mc-live-shifts",
                     "roigbiv-mc-live-status", "roigbiv-mc-live-metrics",
                     "roigbiv-mc-live-corr-crop", "roigbiv-mc-live-scrub"):
        assert expected in ids
    # The pre-existing post-hoc card must be untouched by this feature.
    assert "roigbiv-mc-preview" not in ids


def test_fast_interval_starts_disabled():
    from roigbiv.ui.pages.process import _live_mc_section

    stack = [_live_mc_section()]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        if getattr(node, "id", None) == "roigbiv-mc-live-tick":
            assert node.disabled is True
            return
        children = getattr(node, "children", None)
        if children is not None:
            stack.append(children)
    pytest.fail("fast interval not found")


@pytest.mark.parametrize("phase,active", [
    ("registering", True), ("building_reference", True), ("starting", True),
    ("done", False), ("skipped_precorrected", False), ("unsupported", False),
    ("degraded", False), ("aborted", False), ("skipped_resume", False),
])
def test_live_tick_stops_on_terminal_phases(phase, active):
    from roigbiv.ui.pages.process import _live_tick_active

    assert _live_tick_active({"phase": phase}) is active
    assert _live_tick_active(None) is False


def test_status_text_explains_a_skipped_run():
    from roigbiv.ui.pages.process import _live_status_text

    text = _live_status_text({"stem": "fovA", "backend": "phasecorr",
                              "phase": "skipped_precorrected"})
    assert "already motion-corrected" in text
    live = _live_status_text({"stem": "fovA", "backend": "phasecorr",
                              "phase": "registering", "n_done": 40,
                              "n_total": 100})
    assert "40 / 100" in live


def test_crop_overlay_style_hides_without_data():
    from roigbiv.ui.pages.process import _crop_overlay_style

    assert _crop_overlay_style(None)["display"] == "none"
    assert _crop_overlay_style({"valid_crop_frac": None})["display"] == "none"
    style = _crop_overlay_style({"valid_crop_frac": [0.0, 0.0, 1.0, 0.94]})
    assert style["display"] == "block"
    assert style["height"] == "94.000%"
