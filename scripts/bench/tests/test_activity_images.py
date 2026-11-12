"""Accept tests for U4 (``bench.activity_images``). Written by the contract; workers do not edit them."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.activity_images import activity_images, iter_frames  # noqa: E402


def pixel_movie(values):
    """A 1 x 1 movie, one frame per value."""
    return [np.full((1, 1), v, dtype=np.uint16) for v in values]


def test_worked_example():
    # 8 bins of 1 frame; k = 4, stride = 2; blocks start 0, 2, 4.
    r = activity_images(pixel_movie([0, 0, 0, 0, 10, 0, 0, 0]), fs=1.0, bin_s=1.0, block_s=4.0)
    assert (r.n_frames, r.n_bins, r.n_blocks) == (8, 8, 3)
    assert r.activity_max[0, 0] == pytest.approx(10.0)
    assert r.temporal_sd[0, 0] == pytest.approx(math.sqrt(100 / 8 - (10 / 8) ** 2), rel=1e-6)


def test_constant_movie_is_zero():
    r = activity_images(pixel_movie([5] * 9), fs=1.0, bin_s=1.0, block_s=4.0)
    assert r.activity_max[0, 0] == 0.0 and r.temporal_sd[0, 0] == 0.0


def test_binning_and_even_median():
    # fs 2 -> 2 frames per bin; 10 frames -> bins [1, 0, 0, 3, 0].
    # k = 4, stride 2: block [0, 4), then the end block [1, 5) -> 2 blocks.
    # Block 0 bins [1, 0, 0, 3]: median 0.5 -> max 2.5. Block 1 [0, 0, 3, 0]: median 0 -> 3.
    r = activity_images(pixel_movie([0, 2, 0, 0, 0, 0, 6, 0, 0, 0]), fs=2.0, bin_s=1.0, block_s=4.0)
    assert (r.n_frames, r.n_bins, r.n_blocks) == (10, 5, 2)
    assert r.activity_max[0, 0] == pytest.approx(3.0)
    assert r.temporal_sd[0, 0] == pytest.approx(math.sqrt(3.36), rel=1e-6)


def test_end_block_covers_trailing_bins():
    # 5 bins, k = 4: block [0, 4) misses bin 4; the end block [1, 5) catches the spike.
    r = activity_images(pixel_movie([0, 0, 0, 0, 9]), fs=1.0, bin_s=1.0, block_s=4.0)
    assert r.n_blocks == 2 and r.activity_max[0, 0] == pytest.approx(9.0)


def test_short_movie_is_one_block():
    r = activity_images(pixel_movie([0, 5, 0]), fs=1.0, bin_s=1.0, block_s=30.0)
    assert (r.n_bins, r.n_blocks) == (3, 1) and r.activity_max[0, 0] == pytest.approx(5.0)


def test_trailing_frames_skip_bins_but_count_in_sd():
    # 2 frames per bin: bins [0, 0]; the trailing 8 is in no bin but is in the SD.
    r = activity_images(pixel_movie([0, 0, 0, 0, 8]), fs=2.0, bin_s=1.0, block_s=30.0)
    assert (r.n_frames, r.n_bins) == (5, 2)
    assert r.activity_max[0, 0] == 0.0
    assert r.temporal_sd[0, 0] == pytest.approx(3.2, rel=1e-6)


def test_frames_per_bin_rounds():
    # fs 7.5, bin 1 s -> round(7.5) = 8 frames per bin; 16 frames -> 2 bins.
    r = activity_images(pixel_movie([1] * 16 + [0]), fs=7.5, bin_s=1.0, block_s=30.0)
    assert (r.n_frames, r.n_bins) == (17, 2)


def test_per_pixel_and_dtype():
    frames = [np.array([[0, 7]], dtype=np.uint16), np.array([[10, 7]], dtype=np.uint16),
              np.array([[0, 7]], dtype=np.uint16), np.array([[0, 7]], dtype=np.uint16)]
    r = activity_images(frames, fs=1.0, bin_s=1.0, block_s=4.0)
    assert r.activity_max.shape == (1, 2) and r.activity_max.dtype == np.float32
    assert r.temporal_sd.dtype == np.float32
    np.testing.assert_allclose(r.activity_max, [[10.0, 0.0]])


def test_accepts_a_generator():
    gen = (f for f in pixel_movie([0, 0, 4, 0]))
    r = activity_images(gen, fs=1.0, bin_s=1.0, block_s=4.0)
    assert r.n_frames == 4 and r.activity_max[0, 0] == pytest.approx(4.0)


@pytest.mark.parametrize("kwargs", [
    {"fs": 0.0}, {"fs": -1.0}, {"fs": 1.0, "bin_s": 0.0}, {"fs": 1.0, "block_s": -30.0},
])
def test_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        activity_images(pixel_movie([0, 1, 2, 3]), **kwargs)


def test_too_short():
    with pytest.raises(ValueError):
        activity_images([], fs=1.0)
    with pytest.raises(ValueError):
        activity_images(pixel_movie([3]), fs=2.0)


def test_iter_frames_tif(tmp_path):
    stack = np.arange(5 * 4 * 6, dtype=np.uint16).reshape(5, 4, 6)
    p = tmp_path / "movie.tif"
    tifffile.imwrite(str(p), stack)
    frames = list(iter_frames(p, "tif"))
    assert len(frames) == 5
    for t in range(5):
        np.testing.assert_array_equal(frames[t], stack[t])


def test_iter_frames_dir_sorted(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for t in (2, 0, 1):
        tifffile.imwrite(str(d / f"frame_{t:03d}.tif"), np.full((3, 3), t, dtype=np.uint16))
    (d / "notes.txt").write_text("not a frame")
    frames = list(iter_frames(d, "frame_dir"))
    assert [int(f[0, 0]) for f in frames] == [0, 1, 2]


def test_iter_frames_bad_kind(tmp_path):
    with pytest.raises(ValueError):
        list(iter_frames(tmp_path, "avi"))
