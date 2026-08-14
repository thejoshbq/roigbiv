"""One FOV assembled across its sessions (:mod:`roigbiv.ui.services.tracked_cells`).

These go through a real on-disk store and real label images rather than mocks:
the join being tested *is* the one between ``merged_masks.tif`` label values and
``cell_observation.local_label_id``, and a mock of either side would assert only
that the test author agreed with themselves.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.registry.config import RegistryConfig
from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    SessionRecord,
)
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
from roigbiv.ui.services.tracked_cells import (
    _cache,
    invalidate_tracked_fov,
    load_tracked_fov,
    load_tracked_fov_cached,
)

STEMS = ["fov_pre-005", "fov_beh-006", "fov_post-007"]

# label_id -> (y, x) per session. Cell A is in every session, B drops out after
# the second, C only ever appears in the third.
SESSION_LABELS = [
    {1: (10, 10), 2: (10, 30)},          # A, B
    {1: (11, 11), 2: (10, 31)},          # A, B
    {1: (12, 12), 2: (30, 10)},          # A, C
]
# global cell -> [(session_index, label_id), ...]
OBSERVATIONS = {
    "A": [(0, 1), (1, 1), (2, 1)],
    "B": [(0, 2), (1, 2)],
    "C": [(2, 2)],
}


def _write_session(out_dir: Path, labels: dict[int, tuple[int, int]]) -> None:
    """A minimal tracked output dir: a label image and its mean projection."""
    masks = np.zeros((40, 40), dtype=np.uint16)
    for label_id, (y, x) in labels.items():
        masks[y - 3:y + 4, x - 3:x + 4] = label_id
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / "merged_masks.tif"), masks)
    tifffile.imwrite(str(out_dir / "summary" / "mean_M.tif"),
                     np.zeros((40, 40), dtype=np.float32))


@pytest.fixture
def tracked(tmp_path):
    """A three-session FOV on disk, with a store that ``build_store`` can reopen."""
    cfg = RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'registry.db'}", blob_backend="local",
        blob_root=tmp_path / "blobs", endpoint=None, api_key=None,
    )
    store = SQLAlchemyStore(dsn=cfg.dsn)
    store.ensure_schema()

    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="a" * 64, animal_id="DS-Prism-3",
        region="DS-Prism", mean_m_uri="file:///m.npy",
        centroid_table_uri="file:///c.npy",
        created_at=datetime.now(timezone.utc),
    ))

    session_ids = []
    for i, stem in enumerate(STEMS):
        out_dir = tmp_path / stem
        _write_session(out_dir, SESSION_LABELS[i])
        session_id = str(uuid.uuid4())
        store.upsert_session(SessionRecord(
            session_id=session_id, fov_id=fov_id, session_date=date(2026, 5, 21),
            output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
            sequence_index=i, n_matched=len(SESSION_LABELS[i]),
        ))
        session_ids.append(session_id)

    gcids = {}
    for name, sightings in OBSERVATIONS.items():
        gcid = str(uuid.uuid4())
        gcids[name] = gcid
        store.insert_cell(CellRecord(global_cell_id=gcid, fov_id=fov_id))
        store.insert_observations([
            ObservationRecord(
                observation_id=str(uuid.uuid4()), global_cell_id=gcid,
                session_id=session_ids[i], local_label_id=label_id)
            for i, label_id in sightings
        ])

    return cfg, fov_id, tmp_path, gcids, session_ids


def _status_by_label(session) -> dict[int, str]:
    return {r.label_id: r.match_status for r in session.rois}


# ── shape of the result ────────────────────────────────────────────────────


def test_sessions_come_back_in_timeline_order(tracked):
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert [s.stem for s in fov.sessions] == STEMS
    assert fov.animal_id == "DS-Prism-3"
    assert fov.ordering_is_confirmed is True


def test_every_cell_gets_a_presence_vector_one_slot_per_session(tracked):
    cfg, fov_id, *_ = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert len(fov.cells) == 3
    assert all(len(c.present) == 3 for c in fov.cells)


def test_cells_seen_throughout_are_counted_as_complete(tracked):
    cfg, fov_id, *_ = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert fov.n_complete == 1        # only A


# ── the status a viewer colors by ──────────────────────────────────────────


def test_a_cell_is_new_the_first_time_and_matched_after(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    a = fov.cell_by_gcid(gcids["A"])
    assert _status_by_label(fov.sessions[0])[a.label_in(0)] == "new"
    assert _status_by_label(fov.sessions[1])[a.label_in(1)] == "matched"
    assert _status_by_label(fov.sessions[2])[a.label_in(2)] == "matched"


def test_a_cell_first_seen_late_is_new_in_that_session_not_matched(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    c = fov.cell_by_gcid(gcids["C"])
    assert c.present == [False, False, True]
    assert _status_by_label(fov.sessions[2])[c.label_in(2)] == "new"
    assert "late_arrival" in c.anomalies


def test_a_dropout_is_drawn_where_it_last_was(tracked):
    """Without the carried-forward ghost a dropout is simply invisible."""
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    b = fov.cell_by_gcid(gcids["B"])
    assert b.present == [True, True, False]

    ghosts = [r for r in fov.sessions[2].rois if r.match_status == "lost"]
    assert len(ghosts) == 1
    ghost = ghosts[0]
    assert ghost.global_cell_id == gcids["B"]
    # Its last real sighting was session 2, at (10, 31).
    assert ghost.centroid_yx == pytest.approx((10.0, 31.0), abs=1.0)
    assert ghost.contours, "a ghost with no contour cannot be drawn"


def test_no_ghost_before_a_cell_has_ever_been_seen(tracked):
    """C is absent from sessions 1-2, but it has no last position yet."""
    cfg, fov_id, *_ = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    for session in fov.sessions[:2]:
        assert not [r for r in session.rois if r.match_status == "lost"]


def test_ghost_labels_cannot_be_mistaken_for_real_ones(tracked):
    cfg, fov_id, *_ = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    for session in fov.sessions:
        real = {r.label_id for r in session.rois if r.match_status != "lost"}
        ghosts = {r.label_id for r in session.rois if r.match_status == "lost"}
        assert all(g < 0 for g in ghosts)
        assert not (real & ghosts)


# ── display numbering ──────────────────────────────────────────────────────


def test_numbers_follow_first_appearance_then_label_order(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    numbers = {c.global_cell_id: c.index for c in fov.cells}
    assert numbers[gcids["A"]] == 1     # session 0, label 1
    assert numbers[gcids["B"]] == 2     # session 0, label 2
    assert numbers[gcids["C"]] == 3     # session 2


def test_numbers_are_stable_across_loads(tracked):
    cfg, fov_id, *_ = tracked
    first = {c.global_cell_id: c.index for c in load_tracked_fov(fov_id, cfg=cfg).cells}
    second = {c.global_cell_id: c.index for c in load_tracked_fov(fov_id, cfg=cfg).cells}
    assert first == second


# ── resolving a click back to a cell ───────────────────────────────────────


def test_a_click_on_any_session_resolves_to_the_same_cell(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    a = fov.cell_by_gcid(gcids["A"])
    for i in range(3):
        assert fov.gcid_for_label(i, a.label_in(i)) == gcids["A"]


def test_a_click_on_a_ghost_resolves_to_the_cell_that_went_missing(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    ghost = next(r for r in fov.sessions[2].rois if r.match_status == "lost")
    assert fov.gcid_for_label(2, ghost.label_id) == gcids["B"]


def test_per_session_centroids_are_recorded_where_the_cell_was_seen(tracked):
    cfg, fov_id, _tmp, gcids, _sids = tracked
    fov = load_tracked_fov(fov_id, cfg=cfg)
    b = fov.cell_by_gcid(gcids["B"])
    assert b.centroids[0] == pytest.approx((10.0, 30.0), abs=1.0)
    assert b.centroids[2] is None, "not seen in session 3 — no position of its own"


# ── degraded inputs ────────────────────────────────────────────────────────


def test_a_missing_output_directory_is_skipped_not_fatal(tracked):
    """A workspace can be moved or partly deleted; the rest must still load."""
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    for path in sorted((tmp_path / STEMS[1]).rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (tmp_path / STEMS[1]).rmdir()

    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert [s.stem for s in fov.sessions] == [STEMS[0], STEMS[2]]
    assert all(len(c.present) == 2 for c in fov.cells)


def test_a_label_with_no_observation_carries_no_status(tracked):
    """An ROI the registry never recorded is not evidence of a match."""
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    masks = tifffile.imread(str(tmp_path / STEMS[0] / "merged_masks.tif"))
    masks[35:38, 35:38] = 9        # a label no observation points at
    tifffile.imwrite(str(tmp_path / STEMS[0] / "merged_masks.tif"), masks)

    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert _status_by_label(fov.sessions[0])[9] is None


def test_no_sessions_at_all_loads_empty_rather_than_raising(tmp_path):
    cfg = RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'empty.db'}", blob_backend="local",
        blob_root=tmp_path / "blobs", endpoint=None, api_key=None,
    )
    SQLAlchemyStore(dsn=cfg.dsn).ensure_schema()
    fov = load_tracked_fov(str(uuid.uuid4()), cfg=cfg)
    assert fov.sessions == [] and fov.cells == []


# ── registry / disk disagreement ───────────────────────────────────────────


def test_a_session_is_clean_when_its_match_json_agrees(tracked):
    cfg, fov_id, tmp_path, _gcids, session_ids = tracked
    (tmp_path / STEMS[0] / "registry_match.json").write_text(
        json.dumps({"session_id": session_ids[0], "decision": "auto_match"}))
    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert fov.sessions[0].stale is False


def test_a_match_json_naming_another_session_is_flagged_stale(tracked):
    """The registry says one thing, the FOV's own match record says another."""
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    (tmp_path / STEMS[0] / "registry_match.json").write_text(
        json.dumps({"session_id": str(uuid.uuid4()), "decision": "auto_match"}))
    fov = load_tracked_fov(fov_id, cfg=cfg)
    assert fov.sessions[0].stale is True
    assert fov.sessions[1].stale is False


def test_an_unreadable_match_json_is_not_called_stale(tracked):
    cfg, fov_id, tmp_path, *_ = tracked
    (tmp_path / STEMS[0] / "registry_match.json").write_text("{ not json")
    assert load_tracked_fov(fov_id, cfg=cfg).sessions[0].stale is False


# ── cache ──────────────────────────────────────────────────────────────────


def _cache_keys(fov_id: str, dsn) -> list:
    return [k for k in _cache if k[0] == fov_id and k[1] == dsn]


def test_a_second_call_with_the_same_fingerprint_reuses_the_cached_object(
        tracked):
    cfg, fov_id, _tmp_path, *_ = tracked
    first = load_tracked_fov_cached(fov_id, cfg=cfg)
    second = load_tracked_fov_cached(fov_id, cfg=cfg)
    assert first is second


def test_a_changed_fingerprint_evicts_the_stale_entry_instead_of_leaking(
        tracked):
    """Every edit bumps a mask's mtime and adds a new cache key; the old
    fingerprint's entry must not survive alongside it, or a 30-edit session
    leaks 30 full TrackedFOVs."""
    cfg, fov_id, tmp_path, *_ = tracked
    load_tracked_fov_cached(fov_id, cfg=cfg)
    assert len(_cache_keys(fov_id, cfg.dsn)) == 1

    # Bump one session's mask mtime — the fingerprint tracked_cells uses.
    masks = tmp_path / STEMS[0] / "merged_masks.tif"
    masks.touch()
    import os
    os.utime(masks, None)

    load_tracked_fov_cached(fov_id, cfg=cfg)
    assert len(_cache_keys(fov_id, cfg.dsn)) == 1


def test_invalidate_tracked_fov_drops_the_cached_entry(tracked):
    cfg, fov_id, _tmp_path, *_ = tracked
    load_tracked_fov_cached(fov_id, cfg=cfg)
    assert _cache_keys(fov_id, cfg.dsn)

    invalidate_tracked_fov(fov_id, cfg=cfg)
    assert _cache_keys(fov_id, cfg.dsn) == []


def test_invalidate_tracked_fov_does_not_touch_other_fovs(tracked):
    cfg, fov_id, _tmp_path, *_ = tracked
    load_tracked_fov_cached(fov_id, cfg=cfg)
    other_key = ("some-other-fov", cfg.dsn, ("marker",))
    _cache[other_key] = object()

    invalidate_tracked_fov(fov_id, cfg=cfg)

    assert other_key in _cache
    del _cache[other_key]


# ── which geometry the page draws ──────────────────────────────────────────


def _write_boundaries(out_dir: Path, labels: dict[int, tuple[int, int]]) -> None:
    """Seeded boundaries for the same labels, at a visibly different size.

    Bigger than ``_write_session``'s 7x7 stamps so a test can tell which of the
    two images was drawn without depending on their exact shapes.
    """
    masks = np.zeros((40, 40), dtype=np.uint16)
    for label_id, (y, x) in labels.items():
        masks[y - 5:y + 6, x - 5:x + 6] = label_id
    tifffile.imwrite(str(out_dir / "boundaries.tif"), masks)


def test_boundaries_are_off_by_default(tracked):
    """The canonical disks load first, even when boundaries exist on disk.

    ADR-0003's stamps are what the registry matched on; a reviewer should see
    that geometry unless they ask for the other track.
    """
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    _write_boundaries(tmp_path / STEMS[0], SESSION_LABELS[0])

    fov = load_tracked_fov(fov_id, cfg=cfg)

    areas = {r.label_id: r.area for r in fov.sessions[0].rois if r.area}
    assert areas[1] == 7 * 7, "default view is the disk stamps, not boundaries.tif"


def test_boundaries_are_drawn_when_requested(tracked):
    """Seeded boundaries replace the canonical disks, opt-in.

    ``merged_masks.tif`` stays what the registry matched on (ADR-0003); it is
    only what the *viewer* renders that changes.
    """
    cfg, fov_id, tmp_path, _gcids, _sids = tracked
    _write_boundaries(tmp_path / STEMS[0], SESSION_LABELS[0])

    fov = load_tracked_fov(fov_id, cfg=cfg, show_boundaries=True)

    areas = {r.label_id: r.area for r in fov.sessions[0].rois if r.area}
    assert areas[1] == 11 * 11, "boundaries.tif should be the geometry source"
    # A session without boundaries keeps the stamps.
    other = {r.label_id: r.area for r in fov.sessions[1].rois if r.area}
    assert other[1] == 7 * 7


def test_label_to_cell_association_survives_the_geometry_swap(tracked):
    """Both images carry the same label ids, so identity cannot shift."""
    cfg, fov_id, tmp_path, gcids, _sids = tracked
    before = load_tracked_fov(fov_id, cfg=cfg)
    before_map = {r.label_id: r.global_cell_id for r in before.sessions[0].rois}

    _write_boundaries(tmp_path / STEMS[0], SESSION_LABELS[0])
    after = load_tracked_fov(fov_id, cfg=cfg, show_boundaries=True)

    after_map = {r.label_id: r.global_cell_id for r in after.sessions[0].rois}
    assert after_map == before_map
    assert after_map[1] == gcids["A"]


def test_boundaries_of_the_wrong_shape_are_ignored(tracked):
    """A leftover from a differently-sized run would misplace every contour."""
    cfg, fov_id, tmp_path, *_ = tracked
    tifffile.imwrite(str(tmp_path / STEMS[0] / "boundaries.tif"),
                     np.zeros((20, 20), dtype=np.uint16))

    fov = load_tracked_fov(fov_id, cfg=cfg, show_boundaries=True)

    areas = {r.label_id: r.area for r in fov.sessions[0].rois if r.area}
    assert areas[1] == 7 * 7, "must fall back to merged_masks.tif"


def test_redrawn_boundaries_invalidate_the_cache(tracked):
    """A centroid edit redraws boundaries; the page must not serve the old ones."""
    cfg, fov_id, tmp_path, *_ = tracked
    load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=True)
    assert len(_cache_keys(fov_id, cfg.dsn)) == 1

    _write_boundaries(tmp_path / STEMS[0], SESSION_LABELS[0])
    fov = load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=True)

    areas = {r.label_id: r.area for r in fov.sessions[0].rois if r.area}
    assert areas[1] == 11 * 11


def test_each_geometry_caches_separately_and_does_not_cross_serve(tracked):
    """A tab with boundaries on must never be handed the other tab's disks.

    ``show_boundaries`` is part of the cache key precisely so this can't
    happen — see ``load_tracked_fov_cached``'s docstring.
    """
    cfg, fov_id, tmp_path, *_ = tracked
    _write_boundaries(tmp_path / STEMS[0], SESSION_LABELS[0])

    disks = load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=False)
    boundaries = load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=True)

    disk_areas = {r.label_id: r.area for r in disks.sessions[0].rois if r.area}
    boundary_areas = {r.label_id: r.area
                      for r in boundaries.sessions[0].rois if r.area}
    assert disk_areas[1] == 7 * 7
    assert boundary_areas[1] == 11 * 11
    assert len(_cache_keys(fov_id, cfg.dsn)) == 2

    # Re-fetching each is a cache hit, not a re-derivation of the other.
    assert load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=False) is disks
    assert (load_tracked_fov_cached(fov_id, cfg=cfg, show_boundaries=True)
            is boundaries)
