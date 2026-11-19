"""Validity mask: which pixels a rigid registration shift filled with wrapped content (Suite2p 0.14.5 np.roll)."""

import numpy as np
import pytest

from roigbiv.workbench import validity as vd

SHAPE = (20, 30)


def brute_wrapped(yoff, xoff, shape):
    """Per frame, pixel (r, c) after np.roll(frame, (-dy, -dx)) holds source (r + dy, c + dx), wrapped if outside."""
    ly, lx = shape
    r = np.arange(ly)[:, None]
    c = np.arange(lx)[None, :]
    out = []
    for dy, dx in zip(yoff, xoff):
        out.append(((r + dy) < 0) | ((r + dy) >= ly) | ((c + dx) < 0) | ((c + dx) >= lx))
    return np.stack(out)


def test_brute_force_agrees_with_np_roll_semantics():
    ly, lx = SHAPE
    idx = np.arange(ly * lx).reshape(SHAPE)
    rolled = np.roll(idx, (-3, 2), axis=(0, 1))  # dy=3, dx=-2
    src_r, src_c = np.divmod(rolled, lx)
    moved = (src_r != (np.arange(ly)[:, None] + 3)) | (src_c != (np.arange(lx)[None, :] - 2))
    np.testing.assert_array_equal(moved, brute_wrapped([3], [-2], SHAPE)[0])


def test_wrapped_fraction_matches_brute_force_on_random_shifts():
    rng = np.random.default_rng(0)
    yoff = rng.integers(-4, 5, 300).astype(np.float32)
    xoff = rng.integers(-6, 3, 300).astype(np.float32)
    want = brute_wrapped(yoff.astype(int), xoff.astype(int), SHAPE).mean(axis=0)
    np.testing.assert_allclose(vd.wrapped_fraction(yoff, xoff, SHAPE), want)


def test_wrapped_frames_matches_brute_force_at_chosen_pixels():
    rng = np.random.default_rng(1)
    yoff = rng.integers(-3, 4, 120)
    xoff = rng.integers(-3, 4, 120)
    ys, xs = np.array([0, 1, 19, 10, 5]), np.array([0, 29, 3, 15, 28])
    want = brute_wrapped(yoff, xoff, SHAPE)[:, ys, xs]
    np.testing.assert_array_equal(vd.wrapped_frames(yoff, xoff, SHAPE, ys, xs), want)


def test_shifts_must_be_whole_pixels():
    with pytest.raises(ValueError):
        vd.wrapped_fraction(np.array([0.5]), np.array([0.0]), SHAPE)


def test_crop_reproduces_suite2p_compute_crop_symmetric_band():
    yoff = np.array([-3, 0, 2, 1])
    xoff = np.array([0, 4, -1, 0])
    crop = vd.suite2p_crop(yoff, xoff, SHAPE)
    assert crop.yrange == (3, 17) and crop.xrange == (4, 26)


def test_crop_margin_widens_every_side_and_never_inverts():
    crop = vd.suite2p_crop(np.array([1]), np.array([2]), SHAPE, margin=3)
    assert crop.yrange == (4, 16) and crop.xrange == (5, 25)
    tiny = vd.suite2p_crop(np.array([9]), np.array([0]), (12, 12), margin=5)
    assert tiny.yrange[0] <= tiny.yrange[1]


def test_crop_excludes_bad_frames_like_suite2p():
    crop = vd.suite2p_crop(np.array([1, 9, -2]), np.array([0, 0, 0]), SHAPE, badframes=np.array([False, True, False]))
    assert crop.yrange == (2, 18)


def test_valid_mask_is_true_exactly_inside_the_crop():
    crop = vd.Crop(yrange=(2, 18), xrange=(3, 27))
    m = vd.valid_mask(crop, SHAPE)
    assert m.shape == SHAPE and m.sum() == 16 * 24
    assert m[2, 3] and m[17, 26] and not m[1, 3] and not m[18, 26] and not m[2, 2] and not m[2, 27]


def test_outside_share_of_a_footprint():
    m = vd.valid_mask(vd.Crop(yrange=(2, 18), xrange=(0, 30)), SHAPE)
    fp = np.ravel_multi_index((np.array([0, 1, 2, 3]), np.array([5, 5, 5, 5])), SHAPE)
    assert vd.outside_share(fp, m) == pytest.approx(0.5)
    assert vd.outside_share(fp[:3], m) == pytest.approx(2 / 3)  # rows 0 and 1 outside, row 2 inside
    assert np.isnan(vd.outside_share(np.array([], dtype=np.int64), m))


def test_negative_margin_is_refused():
    with pytest.raises(ValueError):
        vd.suite2p_crop(np.array([1]), np.array([1]), SHAPE, margin=-1)


def test_pixels_outside_the_frame_are_refused():
    with pytest.raises(ValueError):
        vd.wrapped_frames(np.array([0]), np.array([0]), SHAPE, np.array([20]), np.array([0]))
    with pytest.raises(ValueError):
        vd.wrapped_frames(np.array([0]), np.array([0]), SHAPE, np.array([0]), np.array([-1]))


def test_no_offsets_means_nothing_wrapped_and_no_crop():
    assert not vd.wrapped_fraction(np.array([]), np.array([]), SHAPE).any()
    assert vd.suite2p_crop(np.array([]), np.array([]), SHAPE) == vd.Crop(yrange=(0, 20), xrange=(0, 30))
