"""Unit tests for the median/mode chunked-extraction primitives in
:mod:`roigbiv.pipeline.traces`.

Mean's existing matmul-based extraction is already exercised indirectly by
:mod:`test_reextract`; these tests cover only the new, non-linear statistics,
which cannot reuse that path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import mode as scipy_mode  # test oracle only, not in the hot path

from roigbiv.pipeline.traces import (
    _estimate_value_range,
    _mode_via_bincount,
    extract_median_mode_traces_chunked,
)


def _write_memmap(path: Path, arr: np.ndarray) -> None:
    mm = np.memmap(str(path), dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


def test_mode_via_bincount_matches_scipy():
    # A tie-heavy sample (37 draws over 10 distinct values) exercises the
    # smallest-value-on-tie convention both implementations share.
    rng = np.random.default_rng(1)
    sub = rng.integers(-5, 5, size=(20, 37)).astype(np.int32)
    vmin, vmax = int(sub.min()), int(sub.max())
    nbins = vmax - vmin + 1
    got = _mode_via_bincount(sub, vmin, nbins)
    want = np.asarray(scipy_mode(sub, axis=1).mode).reshape(-1).astype(np.float32)
    assert np.array_equal(got, want)


def test_mode_via_bincount_empty_pixels_returns_zeros():
    sub = np.empty((5, 0), dtype=np.int32)
    got = _mode_via_bincount(sub, 0, 1)
    assert np.array_equal(got, np.zeros(5, dtype=np.float32))


def test_extract_median_mode_known_values(tmp_path: Path):
    # A 3-pixel mask: two pixels share a planted value (the mode and also
    # the median of the 3), one differs — unambiguous for both statistics.
    T, H, W = 5, 4, 4
    movie = np.zeros((T, H, W), dtype=np.int16)
    for t in range(T):
        movie[t, 0, 0] = 10 + t
        movie[t, 0, 1] = 10 + t
        movie[t, 0, 2] = 99
    path = tmp_path / "data.bin"
    _write_memmap(path, movie)

    mask = np.zeros((H, W), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[0, 2] = True

    out = extract_median_mode_traces_chunked(
        path, (T, H, W), np.int16, [mask], chunk=2)
    assert set(out.keys()) == {"median", "mode"}
    for t in range(T):
        assert out["mode"][0, t] == 10 + t
        assert out["median"][0, t] == 10 + t


def test_extract_median_mode_subset_of_stats(tmp_path: Path):
    T, H, W = 4, 4, 4
    movie = np.zeros((T, H, W), dtype=np.int16)
    path = tmp_path / "data.bin"
    _write_memmap(path, movie)
    mask = np.zeros((H, W), dtype=bool)
    mask[0, 0] = True

    only_median = extract_median_mode_traces_chunked(
        path, (T, H, W), np.int16, [mask], stats=("median",))
    assert set(only_median.keys()) == {"median"}


def test_extract_median_mode_chunk_boundary_not_divisible(tmp_path: Path):
    T, H, W = 7, 4, 4
    rng = np.random.default_rng(2)
    movie = rng.integers(-100, 100, size=(T, H, W)).astype(np.int16)
    path = tmp_path / "data.bin"
    _write_memmap(path, movie)
    mask = np.zeros((H, W), dtype=bool)
    mask[0:2, 0:2] = True

    chunked = extract_median_mode_traces_chunked(
        path, (T, H, W), np.int16, [mask], chunk=3)
    whole = extract_median_mode_traces_chunked(
        path, (T, H, W), np.int16, [mask], chunk=100)
    assert np.array_equal(chunked["median"], whole["median"])
    assert np.array_equal(chunked["mode"], whole["mode"])


def test_extract_median_mode_empty_mask_is_zero(tmp_path: Path):
    T, H, W = 3, 4, 4
    movie = np.full((T, H, W), 5, dtype=np.int16)
    path = tmp_path / "data.bin"
    _write_memmap(path, movie)
    empty_mask = np.zeros((H, W), dtype=bool)

    out = extract_median_mode_traces_chunked(
        path, (T, H, W), np.int16, [empty_mask])
    assert np.all(out["median"] == 0.0)
    assert np.all(out["mode"] == 0.0)


def test_estimate_value_range_pads_and_covers_sample(tmp_path: Path):
    T, H, W = 10, 4, 4
    movie = np.zeros((T, H, W), dtype=np.int16)
    movie[0] = -50
    movie[-1] = 200
    path = tmp_path / "data.bin"
    _write_memmap(path, movie)

    vmin, vmax = _estimate_value_range(
        path, (T, H, W), np.int16, sample_stride=1, pad=2)
    assert vmin == -52
    assert vmax == 202
