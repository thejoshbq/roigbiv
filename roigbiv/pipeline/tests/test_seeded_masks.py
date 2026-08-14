"""Seeded boundary formation — extent from flows, identity from confirmed seeds.

The partition logic is exercised with hand-built ``converged`` arrays rather
than real Cellpose output: ``seeded_labels`` takes them as parameters precisely
so these tests need neither a GPU nor an inference call. The one thing that
cannot be faked — that Cellpose's flow field really does merge two touching
somata into a single attractor — is what motivated the watershed step and is
recorded in the module docstring of ``seeded_masks.py``.
"""
import numpy as np
import pytest

from roigbiv.pipeline.seeded_masks import (
    ORIGIN_DISK_FALLBACK,
    ORIGIN_FLOW,
    seeded_labels,
)

H = W = 128


def _gauss(cy, cx, sigma):
    yy, xx = np.ogrid[:H, :W]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def _basin(mask, attractor):
    """``(inds, converged)`` for every pixel of *mask*, all pulled to *attractor*.

    Mimics what ``follow_flows`` produces for one Cellpose basin: every member
    pixel lands on the same point.
    """
    ys, xs = np.nonzero(mask)
    inds = np.vstack([ys, xs]).astype(np.int32)
    converged = np.vstack([
        np.full(ys.size, attractor[0], dtype=np.float32),
        np.full(xs.size, attractor[1], dtype=np.float32),
    ])
    return inds, converged


def _merge(*basins):
    inds = np.hstack([b[0] for b in basins])
    conv = np.hstack([b[1] for b in basins])
    return inds, conv


def _call(seeds, inds, converged, cellprob, **over):
    kwargs = dict(capture_px=20.0, fallback_radius=6, min_area=0, max_area=None)
    kwargs.update(over)
    return seeded_labels(seeds, (H, W), inds=inds, converged=converged,
                         cellprob=cellprob, **kwargs)


# --------------------------------------------------------------------------
# The merge split — the reason this module exists
# --------------------------------------------------------------------------

def test_single_attractor_basin_splits_between_two_seeds():
    """Two somata merged into one flow basin are separated by their seeds.

    This is the measured failure mode: Cellpose emits one label whose flow
    field has a single attractor equidistant from both true centroids. A
    nearest-seed rule in converged space assigns nothing; the watershed on
    -cellprob assigns everything, to the right cell.
    """
    left, right = (64.0, 46.0), (64.0, 82.0)
    cellprob = (_gauss(*left, 12.0) + _gauss(*right, 12.0)).astype(np.float32)
    blob = cellprob > 0.25
    # One attractor, midway between the two seeds — 18 px from each.
    inds, converged = _basin(blob, (64.0, 64.0))

    out = _call({1: left, 2: right}, inds, converged, cellprob)

    assert out.origins == {1: ORIGIN_FLOW, 2: ORIGIN_FLOW}
    assert out.areas[1] > 0 and out.areas[2] > 0
    # The split partitions the basin: no pixel is lost, none double-counted.
    assert out.areas[1] + out.areas[2] == int(blob.sum())
    # Each label lands on its own side of the divide.
    ys, xs = np.nonzero(out.labels == 1)
    assert xs.mean() < 64
    ys, xs = np.nonzero(out.labels == 2)
    assert xs.mean() > 64


# --------------------------------------------------------------------------
# Recall: a confirmed cell can never disappear
# --------------------------------------------------------------------------

def test_seed_in_empty_background_gets_disk_fallback():
    seed = (30.0, 100.0)                       # nowhere near the basin
    cellprob = _gauss(64.0, 40.0, 10.0).astype(np.float32)
    inds, converged = _basin(cellprob > 0.25, (64.0, 40.0))

    out = _call({7: seed}, inds, converged, cellprob, fallback_radius=6)

    assert out.origins[7] == ORIGIN_DISK_FALLBACK
    assert out.n_disk_fallback == 1
    assert out.areas[7] > 0
    cy, cx = np.argwhere(out.labels == 7).mean(axis=0)
    assert abs(cy - seed[0]) < 1.0 and abs(cx - seed[1]) < 1.0


def test_every_seed_gets_exactly_one_label():
    seeds = {3: (40.0, 40.0), 9: (40.0, 90.0), 12: (95.0, 64.0)}
    cellprob = _gauss(40.0, 40.0, 9.0).astype(np.float32)
    inds, converged = _basin(cellprob > 0.25, (40.0, 40.0))

    out = _call(seeds, inds, converged, cellprob)

    assert set(out.present_labels) == set(seeds)
    assert set(out.origins) == set(seeds)
    assert out.n_seeds == 3


# --------------------------------------------------------------------------
# Precision: basins nobody confirmed are not cells
# --------------------------------------------------------------------------

def test_basin_with_no_seed_is_dropped_and_counted():
    real, spurious = (40.0, 40.0), (100.0, 100.0)
    cellprob = (_gauss(*real, 10.0) + _gauss(*spurious, 10.0)).astype(np.float32)
    good = _basin(_gauss(*real, 10.0) > 0.25, real)
    bad = _basin(_gauss(*spurious, 10.0) > 0.25, spurious)
    inds, converged = _merge(good, bad)

    out = _call({1: real}, inds, converged, cellprob)

    assert out.origins[1] == ORIGIN_FLOW
    assert out.n_orphan_basin_px == bad[0].shape[1]
    # Nothing was drawn where the unconfirmed basin was.
    assert out.labels[95:110, 95:110].sum() == 0


# --------------------------------------------------------------------------
# Label-id invariants — CellObservation.local_label_id references these
# --------------------------------------------------------------------------

def test_sparse_label_ids_survive_unrenumbered():
    """Labels are the caller's, never positional 1..N.

    f27453c made centroid labels explicit exactly so a delete could not
    renumber later ones and silently repoint the registry.
    """
    seeds = {5: (40.0, 40.0), 41: (40.0, 88.0)}
    cellprob = (_gauss(40.0, 40.0, 10.0) + _gauss(40.0, 88.0, 10.0)).astype(np.float32)
    inds, converged = _merge(
        _basin(_gauss(40.0, 40.0, 10.0) > 0.25, (40.0, 40.0)),
        _basin(_gauss(40.0, 88.0, 10.0) > 0.25, (40.0, 88.0)),
    )

    out = _call(seeds, inds, converged, cellprob)

    assert out.present_labels == (5, 41)
    assert all(isinstance(v, int) for v in out.present_labels)


def test_repeat_calls_are_deterministic():
    seeds = {1: (40.0, 40.0), 2: (40.0, 80.0)}
    cellprob = (_gauss(40.0, 40.0, 11.0) + _gauss(40.0, 80.0, 11.0)).astype(np.float32)
    inds, converged = _basin(cellprob > 0.25, (40.0, 60.0))

    a = _call(seeds, inds, converged, cellprob)
    b = _call(seeds, inds, converged, cellprob)

    assert np.array_equal(a.labels, b.labels)


# --------------------------------------------------------------------------
# Cleanup rules
# --------------------------------------------------------------------------

def test_area_bounds_demote_to_disk_fallback():
    seed = (64.0, 64.0)
    cellprob = _gauss(*seed, 14.0).astype(np.float32)
    inds, converged = _basin(cellprob > 0.2, seed)

    out = _call({1: seed}, inds, converged, cellprob, max_area=50)

    assert out.origins[1] == ORIGIN_DISK_FALLBACK
    assert any("outside area bounds" in w for w in out.warnings)


def test_disconnected_fragment_away_from_seed_is_dropped():
    seed = (40.0, 40.0)
    main = _gauss(*seed, 9.0) > 0.25
    stray = np.zeros((H, W), bool)
    stray[100:106, 100:106] = True
    cellprob = np.where(main | stray, 1.0, 0.0).astype(np.float32)
    inds, converged = _basin(main | stray, seed)

    out = _call({1: seed}, inds, converged, cellprob)

    assert out.origins[1] == ORIGIN_FLOW
    assert not out.labels[100:106, 100:106].any()
    assert out.areas[1] == int(main.sum())


def test_no_seeds_returns_empty_with_warning():
    cellprob = np.zeros((H, W), np.float32)
    empty = np.zeros((2, 0), np.float32)

    out = seeded_labels({}, (H, W), inds=empty, converged=empty,
                        cellprob=cellprob, capture_px=20.0, fallback_radius=6)

    assert out.n_seeds == 0
    assert out.present_labels == ()
    assert out.warnings


def test_no_cell_pixels_falls_back_for_every_seed():
    seeds = {1: (40.0, 40.0), 2: (80.0, 80.0)}
    cellprob = np.zeros((H, W), np.float32)
    empty = np.zeros((2, 0), np.float32)

    out = _call(seeds, empty, empty, cellprob)

    assert out.n_disk_fallback == 2
    assert set(out.present_labels) == {1, 2}
