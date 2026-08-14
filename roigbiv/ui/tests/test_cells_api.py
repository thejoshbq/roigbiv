"""The HTTP surface the /cells contact sheet is built on.

The browser draws whatever these routes return and posts every edit back
through them, so the wire format is a contract: the overlay cannot join, infer
or repair anything it is not handed. What is guarded here is that contract, the
write path behind it, and the fact that neither accepts a filesystem path.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.centroid_masks import load_effective_centroids, write_merged_masks
from roigbiv.pipeline.types import PipelineConfig
from roigbiv.registry.config import RegistryConfig
from roigbiv.registry.store.base import FOVRecord, SessionRecord
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
from roigbiv.ui.routes import cells_api

STEMS = ["sess-a", "sess-b"]


# ── a real two-session workspace on disk ───────────────────────────────────


def _write_session(root: Path, stem: str, centroids: list[tuple[float, float]]):
    out_dir = root / stem
    (out_dir / "summary").mkdir(parents=True)
    tifffile.imwrite(str(out_dir / "summary" / "mean_M.tif"),
                     np.linspace(0, 1, 64 * 64,
                                 dtype=np.float32).reshape(64, 64))
    (out_dir / "centroids.json").write_text(json.dumps({
        "stem": stem, "schema": 4,
        "centroids": [
            {"label_id": i, "y": y, "x": x, "npix": 50, "cellpose_prob": 0.9}
            for i, (y, x) in enumerate(centroids)
        ],
    }))
    write_merged_masks(out_dir, PipelineConfig())
    return out_dir


@pytest.fixture
def workspace(tmp_path):
    """Two sessions of one FOV, registered and matched into shared cells."""
    cfg = RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'registry.db'}", blob_backend="local",
        blob_root=tmp_path / "blobs", endpoint=None, api_key=None,
    )
    store = SQLAlchemyStore(dsn=cfg.dsn)
    store.ensure_schema()

    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="c" * 64, animal_id="DS-Prism-3",
        region="DS-Prism", mean_m_uri="file:///m", centroid_table_uri="file:///c",
        created_at=datetime.now(timezone.utc)))

    session_ids = []
    for i, stem in enumerate(STEMS):
        out_dir = _write_session(tmp_path, stem, [(20.0, 20.0), (40.0, 40.0)])
        session_id = str(uuid.uuid4())
        session_ids.append(session_id)
        store.upsert_session(SessionRecord(
            session_id=session_id, fov_id=fov_id, session_date=date(2026, 1, i + 1),
            output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
            sequence_index=i))

    # Bootstrap the observations. No ROICaT here: with no prior rows every
    # present label becomes its own deterministic cell, which is exactly the
    # unmatched starting point a link gesture is for.
    from roigbiv.registry.cell_edits import apply_tracking_edits
    apply_tracking_edits(fov_id, tmp_path, store)

    return SimpleNamespace(
        tmp_path=tmp_path, cfg=cfg, store=store, fov_id=fov_id,
        session_ids=session_ids,
        state=SimpleNamespace(
            workspace=SimpleNamespace(input_root=tmp_path),
            registry_config=cfg,
        ),
    )


@pytest.fixture
def client(workspace, monkeypatch):
    """Flask test client whose AppState points at the built workspace.

    ``cells_api`` binds ``get_app_state`` at import, so the route module's own
    attribute is the one that has to be replaced.
    """
    from roigbiv.ui.app import build_app

    app = build_app()
    monkeypatch.setattr(cells_api, "get_app_state", lambda: workspace.state)
    # Every route runs the edit path; nothing in these tests wants the real
    # runner's global state deciding whether a write is allowed.
    monkeypatch.setattr(
        "roigbiv.ui.services.cell_edit_ops._tracking_is_active", lambda: False)
    return app.server.test_client()


def _state(client, workspace):
    resp = client.get(f"/api/cells/{workspace.fov_id}")
    assert resp.status_code == 200
    return resp.get_json()


def _gesture(client, workspace, payload):
    return client.post(f"/api/cells/{workspace.fov_id}/gesture", json=payload)


def _label_of(state, session_index, cell_index):
    """The local label a display number carries in one session."""
    for roi in state["sessions"][session_index]["rois"]:
        if roi["cell_index"] == cell_index and not roi["ghost"]:
            return roi["label_id"]
    return None


def _write_boundaries(out_dir: Path, shape: tuple[int, int],
                      labels: dict[int, tuple[float, float]]) -> None:
    """A seeded-boundary label image, deliberately much bigger than the radius-8
    disk stamps ``write_merged_masks`` draws — big enough that a test can tell
    which geometry a response actually came from just from a contour's extent.
    """
    masks = np.zeros(shape, dtype=np.uint16)
    for label_id, (y, x) in labels.items():
        y, x = int(round(y)), int(round(x))
        masks[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13] = label_id
    tifffile.imwrite(str(out_dir / "boundaries.tif"), masks)


def _ring_width(roi: dict) -> float:
    """The x-extent of an roi's contours — >24px for the fake boundary above,
    well under it for a radius-8 disk."""
    xs = [point[0] for ring in roi["contours"] for point in ring]
    return max(xs) - min(xs) if xs else 0.0


def _roi(state: dict, session_index: int, label_id: int) -> dict:
    return next(r for r in state["sessions"][session_index]["rois"]
               if r["label_id"] == label_id)


# ── the wire format ────────────────────────────────────────────────────────


def test_the_state_carries_every_session_in_timeline_order(client, workspace):
    state = _state(client, workspace)
    assert [s["stem"] for s in state["sessions"]] == STEMS
    assert [s["index"] for s in state["sessions"]] == [0, 1]


def test_each_session_carries_the_frame_size_the_panel_is_shaped_from(
        client, workspace):
    """The panel's aspect ratio comes from these, which is what stops the
    projection letterboxing inside a viewport-height box."""
    for session in _state(client, workspace)["sessions"]:
        assert (session["height"], session["width"]) == (64, 64)


def test_every_roi_carries_geometry_the_overlay_can_draw_without_joining(
        client, workspace):
    session = _state(client, workspace)["sessions"][0]
    assert session["rois"]
    for roi in session["rois"]:
        assert set(roi) >= {"label_id", "gcid", "cell_index", "match_status",
                            "ghost", "centroid", "contours"}
        assert len(roi["centroid"]) == 2
        # Contours are [[x, y], ...] — SVG order, not array order.
        assert all(len(point) == 2 for ring in roi["contours"] for point in ring)


def test_contours_close_into_rings_the_overlay_can_fill(client, workspace):
    """The fill is the click target; an open path would fill as a wedge."""
    roi = _state(client, workspace)["sessions"][0]["rois"][0]
    assert roi["contours"]
    for ring in roi["contours"]:
        assert len(ring) >= 3


def test_the_palette_travels_with_the_state(client, workspace):
    """Colour is decided server-side so the overlay and the page legend cannot
    drift apart."""
    palette = _state(client, workspace)["palette"]
    assert {"matched", "new", "lost"} <= set(palette)


def test_the_cell_roster_carries_the_presence_timeline(client, workspace):
    state = _state(client, workspace)
    assert state["cells"]
    for cell in state["cells"]:
        assert len(cell["present"]) == len(state["sessions"])
        assert set(cell) >= {"gcid", "index", "present", "anomalies"}


def test_the_caption_counts_what_the_panel_is_actually_drawing(client, workspace):
    session = _state(client, workspace)["sessions"][0]
    drawn = sum(1 for r in session["rois"] if r["match_status"] == "matched")
    assert session["counts"].startswith(f"{drawn} matched")


def test_an_unknown_fov_is_a_404_rather_than_a_stack_trace(client):
    assert client.get("/api/cells/not-a-fov").status_code == 404


# ── which geometry the state draws ─────────────────────────────────────────


def test_the_default_geometry_is_the_disk_stamps(client, workspace):
    """No ``show_boundaries`` means the canonical disks — even when a
    boundaries.tif exists on disk."""
    out_dir = Path(workspace.tmp_path) / STEMS[0]
    _write_boundaries(out_dir, (64, 64), {1: (20.0, 20.0), 2: (40.0, 40.0)})

    state = _state(client, workspace)
    assert _ring_width(_roi(state, 0, 1)) < 20, \
        "default view must be the disk stamps, not boundaries.tif"


def test_show_boundaries_query_param_switches_the_geometry(client, workspace):
    out_dir = Path(workspace.tmp_path) / STEMS[0]
    _write_boundaries(out_dir, (64, 64), {1: (20.0, 20.0), 2: (40.0, 40.0)})

    resp = client.get(f"/api/cells/{workspace.fov_id}?show_boundaries=1")
    assert resp.status_code == 200
    state = resp.get_json()
    assert _ring_width(_roi(state, 0, 1)) >= 20, \
        "the query param must select boundaries.tif"


def test_a_falsy_query_param_still_reads_as_disks(client, workspace):
    out_dir = Path(workspace.tmp_path) / STEMS[0]
    _write_boundaries(out_dir, (64, 64), {1: (20.0, 20.0), 2: (40.0, 40.0)})

    resp = client.get(f"/api/cells/{workspace.fov_id}?show_boundaries=0")
    state = resp.get_json()
    assert _ring_width(_roi(state, 0, 1)) < 20


def test_without_a_workspace_the_routes_say_so(client, workspace, monkeypatch):
    monkeypatch.setattr(cells_api, "get_app_state",
                        lambda: SimpleNamespace(workspace=None,
                                                registry_config=None))
    resp = client.get(f"/api/cells/{workspace.fov_id}")
    assert resp.status_code == 409


# ── the projection image ───────────────────────────────────────────────────


def test_a_session_projection_is_served_as_a_png(client, workspace):
    resp = client.get(f"/api/cells/{workspace.fov_id}/image/{STEMS[0]}.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_projection_is_cacheable_and_revalidates_on_its_etag(client, workspace):
    """A 1024² projection is the page's whole payload budget; it changes only
    when the pipeline re-runs, and the ETag is what catches that."""
    first = client.get(f"/api/cells/{workspace.fov_id}/image/{STEMS[0]}.png")
    etag = first.headers.get("ETag")
    assert etag
    assert "max-age" in first.headers.get("Cache-Control", "")

    second = client.get(f"/api/cells/{workspace.fov_id}/image/{STEMS[0]}.png",
                        headers={"If-None-Match": etag})
    assert second.status_code == 304


@pytest.mark.parametrize("stem", ["..%2f..%2fetc", "..", "a%2fb", "%2eetc"])
def test_no_crafted_stem_ever_yields_image_bytes(client, workspace, stem):
    """Sessions are addressed by fov_id + stem and resolved by matching against
    the registry's own session list — the stem is never joined onto a path, so
    there is nothing to traverse. Asserted on the payload rather than the
    status: an unrouted URL falls through to Dash's page router, which answers
    200 with the app shell."""
    resp = client.get(f"/api/cells/{workspace.fov_id}/image/{stem}.png")
    assert resp.mimetype != "image/png"
    assert not resp.get_data().startswith(b"\x89PNG")


def test_an_unregistered_stem_is_a_404(client, workspace):
    resp = client.get(f"/api/cells/{workspace.fov_id}/image/nope.png")
    assert resp.status_code == 404


# ── gestures ───────────────────────────────────────────────────────────────


def test_a_malformed_gesture_is_a_400(client, workspace):
    assert _gesture(client, workspace, {"kind": "levitate"}).status_code == 400
    assert _gesture(client, workspace,
                    {"kind": "move", "stem": STEMS[0]}).status_code == 400


def test_adding_returns_fresh_state_containing_the_new_centroid(client, workspace):
    before = _state(client, workspace)
    n_before = len(before["sessions"][0]["rois"])

    resp = _gesture(client, workspace,
                    {"kind": "add", "stem": STEMS[0], "y": 55.0, "x": 12.0})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"]
    assert "added" in body["message"]
    # The response is the whole sheet, so the browser repaints from it rather
    # than re-fetching — this is what keeps an edit off the viewer.
    assert len(body["state"]["sessions"][0]["rois"]) == n_before + 1
    assert (Path(workspace.tmp_path) / STEMS[0] / "corrections"
            / "centroids.jsonl").exists()


def test_a_short_move_lands_where_it_was_dropped(client, workspace):
    """The regression this page was rebuilt around: the Plotly path resolved a
    move's destination through a nearest-centroid snap, so any nudge inside the
    stamp radius silently wrote a no-op."""
    state = _state(client, workspace)
    roi = state["sessions"][0]["rois"][0]
    origin_y, origin_x = roi["centroid"]

    resp = _gesture(client, workspace, {
        "kind": "move", "stem": STEMS[0], "label": roi["label_id"],
        "y": origin_y + 2.0, "x": origin_x - 2.0,
    })
    assert resp.status_code == 200
    assert resp.get_json()["ok"]

    effective, _warnings = load_effective_centroids(
        Path(workspace.tmp_path) / STEMS[0])
    moved = effective[roi["label_id"]]
    assert (round(moved[0]), round(moved[1])) == (round(origin_y + 2.0),
                                                 round(origin_x - 2.0))


def test_deleting_drops_the_centroid_from_the_returned_state(client, workspace):
    state = _state(client, workspace)
    roi = state["sessions"][0]["rois"][0]

    body = _gesture(client, workspace, {
        "kind": "delete", "stem": STEMS[0], "label": roi["label_id"],
    }).get_json()

    assert body["ok"]
    remaining = [r["label_id"] for r in body["state"]["sessions"][0]["rois"]]
    assert roi["label_id"] not in remaining


def test_linking_two_cells_merges_them_into_one_identity(client, workspace):
    state = _state(client, workspace)
    # Two different cells, one member each, in different sessions.
    first = state["sessions"][0]["rois"][0]
    other = next(r for r in state["sessions"][1]["rois"]
                 if r["gcid"] != first["gcid"] and not r["ghost"])

    body = _gesture(client, workspace, {
        "kind": "link", "stem": STEMS[1], "label": other["label_id"],
        "selected_gcid": first["gcid"],
    }).get_json()

    assert body["ok"]
    assert body["message"] == "linked"
    after = body["state"]
    merged = [c for c in after["cells"] if all(c["present"])]
    assert merged, "the linked pair should now be present in both sessions"


def test_linking_without_a_selection_explains_itself(client, workspace):
    state = _state(client, workspace)
    roi = state["sessions"][0]["rois"][0]
    body = _gesture(client, workspace, {
        "kind": "link", "stem": STEMS[0], "label": roi["label_id"],
    }).get_json()
    assert not body["ok"]
    assert "select a cell first" in body["message"]
    # A refusal leaves the sheet exactly as the browser already has it.
    assert "state" not in body


def test_confirming_a_ghost_adopts_the_outline_underneath_it(client, workspace):
    """Ctrl-click on a ghost, with nothing selected first. Both sessions detect
    the same soma here but tracking did not match them, so the repair is to
    adopt the outline already there rather than stack a centroid on it."""
    state = _state(client, workspace)
    ghost = next(r for r in state["sessions"][1]["rois"] if r["ghost"])

    body = _gesture(client, workspace, {
        "kind": "confirm", "stem": STEMS[1],
        "y": ghost["centroid"][0], "x": ghost["centroid"][1],
        "selected_gcid": ghost["gcid"],
    }).get_json()

    assert body["ok"], body["message"]
    assert "an outline was already there" in body["message"]
    after = body["state"]
    assert [c for c in after["cells"] if all(c["present"])], \
        "the ghost's cell should now be present in both sessions"
    # Adopted, not duplicated.
    assert len(after["sessions"][1]["rois"]) <= len(state["sessions"][1]["rois"])


def test_confirming_a_cell_that_is_already_here_is_refused(client, workspace):
    state = _state(client, workspace)
    roi = next(r for r in state["sessions"][0]["rois"] if not r["ghost"])
    body = _gesture(client, workspace, {
        "kind": "confirm", "stem": STEMS[0],
        "y": roi["centroid"][0], "x": roi["centroid"][1],
        "selected_gcid": roi["gcid"],
    }).get_json()
    assert not body["ok"]
    assert "already has a centroid" in body["message"]
    assert "state" not in body


def test_a_confirm_without_a_coordinate_is_a_bad_request(client, workspace):
    resp = _gesture(client, workspace,
                    {"kind": "confirm", "stem": STEMS[1], "selected_gcid": "x"})
    assert resp.status_code == 400


def test_gesture_response_honours_show_boundaries(client, workspace):
    """The gesture route's own reload always fetches the disk-geometry cache
    entry (it has no notion of which track the caller is viewing) — the route
    has to re-fetch under the request's own flag so the response matches what
    the browser has on screen."""
    out_dir = Path(workspace.tmp_path) / STEMS[0]
    _write_boundaries(out_dir, (64, 64), {1: (20.0, 20.0), 2: (40.0, 40.0)})

    # The edit lands in the other session so STEMS[0]'s boundaries.tif — and
    # the label this asserts on — is untouched by the write itself.
    body = _gesture(client, workspace, {
        "kind": "add", "stem": STEMS[1], "y": 55.0, "x": 12.0,
        "show_boundaries": True,
    }).get_json()

    assert body["ok"]
    assert _ring_width(_roi(body["state"], 0, 1)) >= 20, \
        "the gesture response must honour the request's show_boundaries"


def test_gesture_response_defaults_to_disks_without_the_flag(client, workspace):
    out_dir = Path(workspace.tmp_path) / STEMS[0]
    _write_boundaries(out_dir, (64, 64), {1: (20.0, 20.0), 2: (40.0, 40.0)})

    body = _gesture(client, workspace,
                    {"kind": "add", "stem": STEMS[1], "y": 55.0, "x": 12.0}
                    ).get_json()

    assert body["ok"]
    assert _ring_width(_roi(body["state"], 0, 1)) < 20


def test_undo_reverses_the_last_edit(client, workspace):
    before = len(_state(client, workspace)["sessions"][0]["rois"])
    _gesture(client, workspace,
             {"kind": "add", "stem": STEMS[0], "y": 55.0, "x": 12.0})

    body = _gesture(client, workspace, {"kind": "undo"}).get_json()
    assert body["ok"]
    assert len(body["state"]["sessions"][0]["rois"]) == before


def test_undo_with_nothing_written_says_so(client, workspace):
    body = _gesture(client, workspace, {"kind": "undo"}).get_json()
    assert not body["ok"]
    assert body["message"] == "nothing to undo"


def test_every_gesture_is_refused_with_a_409_while_tracking_runs(
        client, workspace, monkeypatch):
    monkeypatch.setattr(
        "roigbiv.ui.services.cell_edit_ops._tracking_is_active", lambda: True)
    resp = _gesture(client, workspace,
                    {"kind": "add", "stem": STEMS[0], "y": 5.0, "x": 5.0})
    assert resp.status_code == 409
    assert "tracking is running" in resp.get_json()["message"]
    assert not (Path(workspace.tmp_path) / STEMS[0] / "corrections"
                / "centroids.jsonl").exists()


def test_pipeline_output_is_never_mutated_by_an_edit(client, workspace):
    """ADR-0004: corrections are additive. ``centroids.json`` is detector
    output and rewriting it would let a later discovery run consume an edit as
    if it were a detection."""
    frozen = Path(workspace.tmp_path) / STEMS[0] / "centroids.json"
    before = frozen.read_text()

    _gesture(client, workspace,
             {"kind": "add", "stem": STEMS[0], "y": 55.0, "x": 12.0})
    _gesture(client, workspace,
             {"kind": "delete", "stem": STEMS[0], "label": 1})

    assert frozen.read_text() == before
