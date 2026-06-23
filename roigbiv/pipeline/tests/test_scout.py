"""Scout mode (Cellpose-only triage) — Vcorr-on-movie unit coverage.

Scout replaces the residual-based Cellpose channel 2 with a correlation map
computed directly on the registered movie (no SVD/L+S/residual). The arithmetic
is shared with the production summary pass via ``_accumulate_summaries``; these
tests pin that the movie path matches the residual path on identical data and
that the tunable knobs (stride, neighbors) behave.
"""
import numpy as np
import pytest

from roigbiv.pipeline.foundation import (
    _accumulate_summaries,
    generate_summary_images,
    vcorr_on_movie,
)
from roigbiv.pipeline.residual import ResidualView
from roigbiv.pipeline.types import PipelineConfig


def _write_data_bin(tmp_path, movie_int16):
    binp = tmp_path / "data.bin"
    movie_int16.tofile(str(binp))
    return binp


def _synthetic_movie(seed=0, T=60, Ly=12, Lx=10):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 50, (T, Ly, Lx)) + 1000).astype(np.int16)


def _reference_vcorr(movie, neighbors=8):
    """Direct, slow Pearson reference for small synthetic movies."""
    if neighbors == 4:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    elif neighbors == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError("neighbors must be 4 or 8")

    movie = movie.astype(np.float64)
    _, Ly, Lx = movie.shape
    out = np.zeros((Ly, Lx), dtype=np.float64)
    count = np.zeros((Ly, Lx), dtype=np.int32)
    eps = 1e-12
    for y in range(Ly):
        for x in range(Lx):
            trace = movie[:, y, x]
            trace_centered = trace - trace.mean()
            trace_norm = np.sqrt(np.sum(trace_centered ** 2))
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= Ly or xx < 0 or xx >= Lx:
                    continue
                neighbor = movie[:, yy, xx]
                neighbor_centered = neighbor - neighbor.mean()
                neighbor_norm = np.sqrt(np.sum(neighbor_centered ** 2))
                den = trace_norm * neighbor_norm
                if den > eps:
                    out[y, x] += np.sum(trace_centered * neighbor_centered) / (den + eps)
                count[y, x] += 1
    return (out / np.maximum(count, 1)).astype(np.float32)


def test_vcorr_on_movie_matches_residual_accumulator(tmp_path):
    """Scout's movie pass equals the production summary pass on identical data."""
    movie = _synthetic_movie()
    T, Ly, Lx = movie.shape
    binp = _write_data_bin(tmp_path, movie)

    scout = vcorr_on_movie(binp, Ly, Lx, T, stride=1, neighbors=8)
    ref = generate_summary_images(
        ResidualView.from_dense(movie.astype(np.float32)), chunk=16,
    )

    assert np.allclose(scout["mean"], ref["mean"], atol=1e-3)
    assert np.allclose(scout["max"], ref["max"], atol=1e-3)
    assert np.allclose(scout["vcorr"], ref["vcorr"], atol=1e-4)


@pytest.mark.parametrize("neighbors", [4, 8])
def test_accumulate_summaries_vcorr_matches_direct_reference(neighbors):
    """Vcorr must match direct Pearson, including boundaries and flat pixels."""
    movie = _synthetic_movie(seed=5, T=24, Ly=5, Lx=6).astype(np.float32)
    movie[:, 0, 0] = 7.0      # zero-variance corner pixel
    movie[:, 2, 3] = -3.0     # zero-variance interior pixel

    out = _accumulate_summaries(iter([(0, 11, movie[:11]), (11, 24, movie[11:])]),
                                5, 6, neighbors=neighbors)
    ref = _reference_vcorr(movie, neighbors=neighbors)
    assert np.allclose(out["vcorr"], ref, atol=1e-5)


def test_scout_mean_equals_movie_mean(tmp_path):
    """The scout 'mean' channel is exactly mean_M (raw registered-movie mean)."""
    movie = _synthetic_movie()
    T, Ly, Lx = movie.shape
    binp = _write_data_bin(tmp_path, movie)

    scout = vcorr_on_movie(binp, Ly, Lx, T)
    assert np.allclose(scout["mean"], movie.astype(np.float64).mean(axis=0),
                       atol=1e-3)


def test_neighbors_4_differs_from_8(tmp_path):
    movie = _synthetic_movie()
    T, Ly, Lx = movie.shape
    binp = _write_data_bin(tmp_path, movie)

    v8 = vcorr_on_movie(binp, Ly, Lx, T, neighbors=8)["vcorr"]
    v4 = vcorr_on_movie(binp, Ly, Lx, T, neighbors=4)["vcorr"]
    assert v4.shape == (Ly, Lx)
    assert not np.allclose(v4, v8)


def test_stride_decimation_runs_and_is_finite(tmp_path):
    movie = _synthetic_movie()
    T, Ly, Lx = movie.shape
    binp = _write_data_bin(tmp_path, movie)

    v = vcorr_on_movie(binp, Ly, Lx, T, stride=3)["vcorr"]
    assert np.isfinite(v).all()


def test_accumulate_summaries_rejects_bad_neighbors():
    def _empty():
        return iter(())

    with pytest.raises(ValueError):
        _accumulate_summaries(_empty(), 4, 4, neighbors=6)


def test_accumulate_summaries_handles_empty_chunks():
    """Decimated chunks that yield zero frames must not corrupt accumulators."""
    movie = _synthetic_movie(T=20, Ly=6, Lx=6).astype(np.float32)

    def _iter():
        yield 0, 10, movie[:10]
        yield 10, 20, movie[10:10]   # empty slab (cs == 0)
        yield 10, 20, movie[10:20]

    out = _accumulate_summaries(_iter(), 6, 6, neighbors=8)
    ref = _accumulate_summaries(iter([(0, 20, movie)]), 6, 6, neighbors=8)
    assert np.allclose(out["mean"], ref["mean"], atol=1e-4)
    assert np.allclose(out["vcorr"], ref["vcorr"], atol=1e-4)


def test_pipelineconfig_scout_defaults():
    cfg = PipelineConfig(fs=7.5)
    assert cfg.scout_mode is False
    assert cfg.scout_vcorr_stride == 1
    assert cfg.scout_vcorr_neighbors == 8
