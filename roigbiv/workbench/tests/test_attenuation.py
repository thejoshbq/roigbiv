"""Event-window attenuation: how much of a cell's event survives background removal and source subtraction."""

import numpy as np
import pytest

from roigbiv.pipeline.residual import SourceLayer
from roigbiv.workbench import attenuation as at

SHAPE = (30, 40)
N = SHAPE[0] * SHAPE[1]
FS = 7.5


def disk(cy, cx, r=3):
    ys, xs = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return np.flatnonzero(((ys - cy) ** 2 + (xs - cx) ** 2) <= r**2)


def transient(n, peaks, amp=10.0, tau_frames=6.0):
    x = np.zeros(n)
    t = np.arange(n)
    for p in peaks:
        after = t >= p
        x[after] += amp * np.exp(-(t[after] - p) / tau_frames)
    return x


def test_averaging_matrix_rows_are_footprint_means():
    fps = [disk(10, 10), disk(20, 30, 2)]
    a = at.averaging_matrix(fps, N)
    img = np.random.default_rng(0).normal(size=N)
    np.testing.assert_allclose(a @ img, [img[fps[0]].mean(), img[fps[1]].mean()])


def test_averaging_matrix_refuses_an_empty_footprint():
    with pytest.raises(ValueError):
        at.averaging_matrix([disk(10, 10), np.array([], dtype=np.int64)], N)


def test_movie_region_traces_match_dense_means_across_uneven_chunks():
    rng = np.random.default_rng(1)
    movie = rng.normal(size=(103, N)).astype(np.float32)
    fps = [disk(10, 10), disk(20, 30)]
    got = at.movie_region_traces(lambda t0, t1: movie[t0:t1], 103, at.averaging_matrix(fps, N), chunk=25)
    want = np.stack([movie[:, fp].mean(axis=1) for fp in fps])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_background_region_traces_are_the_low_rank_movie_averaged_over_each_footprint():
    rng = np.random.default_rng(2)
    us, v = rng.normal(size=(N, 3)).astype(np.float32), rng.normal(size=(50, 3)).astype(np.float32)
    fps = [disk(10, 10), disk(20, 30)]
    got = at.background_region_traces(at.averaging_matrix(fps, N), us, v)
    low_rank = v @ us.T
    np.testing.assert_allclose(got, np.stack([low_rank[:, fp].mean(axis=1) for fp in fps]), rtol=1e-4, atol=1e-5)


def test_source_region_traces_are_the_subtracted_sources_averaged_over_each_footprint():
    rng = np.random.default_rng(3)
    idx = np.union1d(disk(10, 10), disk(12, 14))
    layer = SourceLayer(
        flat_idx=idx,
        W_design=rng.random((2, idx.size)).astype(np.float32),
        traces=rng.normal(size=(2, 40)).astype(np.float32),
    )
    fps = [disk(10, 12), disk(25, 35)]  # the second footprint touches no source pixel
    got = at.source_region_traces(at.averaging_matrix(fps, N), layer)
    dense = np.zeros((40, N), dtype=np.float32)
    dense[:, idx] = layer.traces.T @ layer.W_design
    np.testing.assert_allclose(got, np.stack([dense[:, fp].mean(axis=1) for fp in fps]), rtol=1e-4, atol=1e-5)
    assert np.all(got[1] == 0)


def test_annulus_leaves_a_gap_and_skips_occupied_pixels():
    fp = disk(15, 15, 3)
    occupied = np.zeros(SHAPE, bool)
    occupied.flat[fp] = True
    occupied[15, 22:25] = True  # a neighbour inside the ring
    ring = at.annulus(fp, SHAPE, occupied, gap=2, ring=4)
    ys, xs = np.unravel_index(ring, SHAPE)
    d = np.hypot(ys - 15, xs - 15)
    assert ring.size > 0
    assert d.min() > 3 + 1  # the gap keeps the ring off the cell edge
    assert not occupied.flat[ring].any()
    assert (15 * SHAPE[1] + 23) not in set(ring)


def test_detect_events_finds_transients_over_drift_and_noise():
    rng = np.random.default_rng(4)
    n = 1500
    peaks = [200, 520, 900, 1300]
    trace = 100 + np.linspace(0, 30, n) + transient(n, peaks) + rng.normal(0, 1.0, n)
    ev = at.detect_events(trace, FS, z=4.0)
    assert len(ev.peaks) == len(peaks)
    assert np.all(np.abs(np.asarray(ev.peaks) - peaks) <= 2)
    assert 0.8 < ev.robust_sd < 1.3
    assert min(ev.z) > 4.0


def test_detect_events_ignores_dips_and_counts_them_on_the_flipped_trace():
    rng = np.random.default_rng(5)
    n = 1500
    trace = 50 - transient(n, [300, 800]) + rng.normal(0, 1.0, n)
    assert len(at.detect_events(trace, FS, z=4.0).peaks) == 0
    assert len(at.detect_events(-trace, FS, z=4.0).peaks) == 2


def test_detect_events_keeps_the_higher_of_two_peaks_closer_than_the_separation():
    n = 1000
    trace = transient(n, [400], amp=10) + transient(n, [405], amp=20) + np.random.default_rng(6).normal(0, 1, n)
    ev = at.detect_events(trace, FS, z=4.0, min_separation_s=2.0)
    assert len(ev.peaks) == 1
    assert abs(ev.peaks[0] - 405) <= 1


def test_detect_events_on_pure_noise_finds_none_at_a_high_threshold():
    trace = np.random.default_rng(7).normal(0, 1, 3000)
    assert len(at.detect_events(trace, FS, z=6.0).peaks) == 0


def test_detect_events_refuses_a_trace_with_nan_rather_than_reporting_no_events():
    trace = transient(500, [200]) + np.random.default_rng(11).normal(0, 1, 500)
    trace[10] = np.nan
    with pytest.raises(ValueError):
        at.detect_events(trace, FS)


def test_event_amplitudes_are_nan_when_the_pre_window_would_overlap_the_peak_window():
    trace = np.zeros(100)
    trace[50:] = 4.0
    # at 1 Hz the pre-window [p-1, p) lies inside the peak window: no clean baseline, so no amplitude (not 0)
    assert np.isnan(at.event_amplitudes(trace, [51], fs=1.0)[0])
    assert at.event_amplitudes(trace, [51], fs=2.0)[0] == pytest.approx(4.0)


def test_event_amplitudes_measure_the_step_from_the_pre_event_window():
    n = 200
    trace = np.zeros(n)
    trace[100:] = 5.0
    trace[150:] = 12.0
    amps = at.event_amplitudes(trace, [101, 151], FS)
    np.testing.assert_allclose(amps, [5.0, 7.0])


def test_event_amplitudes_are_nan_when_the_pre_window_falls_off_the_start():
    amps = at.event_amplitudes(np.arange(50, dtype=float), [2, 30], FS)
    assert np.isnan(amps[0]) and np.isfinite(amps[1])


def test_cell_attenuation_splits_each_event_between_background_and_residual():
    rng = np.random.default_rng(8)
    n = 1500
    peaks = [200, 600, 1000]
    event = transient(n, peaks, amp=20)
    noise = rng.normal(0, 0.5, n)
    m = 100 + event + noise
    bg = 100 + 0.25 * event  # the low-rank background absorbs a quarter of each event
    sub = 0.5 * event  # sources subtracted later remove half of what is left in S
    ann_m = np.full(n, 80.0) + rng.normal(0, 0.5, n)
    res = at.cell_attenuation(
        at.RegionLayers(m=m, l=bg, sub=sub), at.RegionLayers(m=ann_m, l=np.full(n, 80.0), sub=np.zeros(n)), FS
    )
    assert res.n_events == 3 and res.n_negative_events == 0
    assert res.retained_s == pytest.approx(0.75, abs=0.05)
    assert res.retained_r == pytest.approx(0.25, abs=0.05)
    assert res.neuropil_share == pytest.approx(0.0, abs=0.05)
    assert res.amp_m == pytest.approx(np.median(at.event_amplitudes(m, res.peaks, FS)))


def test_cell_attenuation_reports_a_neuropil_wide_event_as_shared_with_the_annulus():
    n = 1500
    event = transient(n, [300, 900], amp=20)
    noise = np.random.default_rng(9).normal(0, 0.5, (2, n))
    cell = at.RegionLayers(m=200 + event + 0.5 * event + noise[0], l=200 + event, sub=np.zeros(n))
    ann = at.RegionLayers(m=150 + event + noise[1], l=150 + event, sub=np.zeros(n))
    res = at.cell_attenuation(cell, ann, FS, neuropil_coeff=0.7)
    assert res.n_events == 2
    assert res.neuropil_share == pytest.approx(1 / 1.5, abs=0.05)


def test_cell_attenuation_with_no_events_reports_nan_ratios():
    n = 1000
    flat = at.RegionLayers(m=np.random.default_rng(10).normal(0, 1, n), l=np.zeros(n), sub=np.zeros(n))
    res = at.cell_attenuation(flat, flat, FS, z=8.0)
    assert res.n_events == 0
    assert np.isnan(res.retained_s) and np.isnan(res.amp_m)
