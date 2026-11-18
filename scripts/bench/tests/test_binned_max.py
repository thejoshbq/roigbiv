"""Tests for ``bench.activity_images.binned_max``."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.activity_images import binned_max  # noqa: E402


def pixel_movie(values):
    return [np.full((1, 1), v, dtype=np.uint16) for v in values]


def test_max_of_bin_means():
    # bins of 2: means 1, 5, 2; trailing frame 100 fills no bin.
    r = binned_max(pixel_movie([0, 2, 4, 6, 1, 3, 100]), frames_per_bin=2)
    assert r.dtype == np.float32
    assert r[0, 0] == pytest.approx(5.0)


def test_a_one_frame_spike_is_diluted_by_its_bin():
    r = binned_max(pixel_movie([0, 0, 0, 0, 0, 70, 0]), frames_per_bin=7)
    assert r[0, 0] == pytest.approx(10.0)


def test_per_pixel():
    a = np.array([[1, 9]], np.uint16)
    b = np.array([[3, 1]], np.uint16)
    r = binned_max([a, a, b, b], frames_per_bin=2)
    assert r.shape == (1, 2)
    assert r.tolist() == [[3.0, 9.0]]


def test_accepts_a_generator():
    r = binned_max((f for f in pixel_movie([1, 1, 4, 4])), frames_per_bin=2)
    assert r[0, 0] == pytest.approx(4.0)


@pytest.mark.parametrize("n", [0, -1])
def test_bad_frames_per_bin(n):
    with pytest.raises(ValueError, match="frames_per_bin"):
        binned_max(pixel_movie([1, 2]), frames_per_bin=n)


def test_no_complete_bin():
    with pytest.raises(ValueError):
        binned_max(pixel_movie([1, 2, 3]), frames_per_bin=7)
