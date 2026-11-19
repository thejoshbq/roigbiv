"""Annotation footprints and the registration shift into a run's movie frame."""

import numpy as np

from roigbiv.workbench import annotations as an


def blobs(shape, pts, r=3):
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    img = np.zeros(shape)
    for y, x in pts:
        img += np.exp(-((ys - y) ** 2 + (xs - x) ** 2) / (2 * r**2))
    return img


def test_best_shift_recovers_the_offset_of_a_crop():
    rng = np.random.default_rng(1)
    pts = rng.uniform(5, 95, size=(40, 2))
    big = blobs((100, 100), pts) + 0.01 * rng.standard_normal((100, 100))
    small = big[6:96, 3:93]  # small[y, x] == big[y + 6, x + 3]
    dy, dx, c = an.best_shift(big, small, search=8)
    assert (dy, dx) == (6, 3) and c > 0.99


def test_to_movie_frame_adds_the_shift_and_drops_cells_outside_the_movie():
    bench_shape, movie_shape = (20, 20), (24, 22)
    fp = np.ravel_multi_index(([5, 5, 6], [5, 6, 5]), bench_shape)
    edge = np.ravel_multi_index(([19], [19]), bench_shape)
    anns = [an.BenchCell(0, (5.3, 5.3), fp, 3), an.BenchCell(1, (19.0, 19.0), edge, 1)]
    out = an.to_movie_frame(anns, bench_shape, movie_shape, shift=(2, 4))
    assert [a.ann_id for a in out] == [0]
    (a,) = out
    assert a.centroid == (7.3, 9.3)
    assert sorted(zip(*np.unravel_index(a.footprint, movie_shape))) == [(7, 9), (7, 10), (8, 9)]


def test_a_point_roi_keeps_no_footprint():
    out = an.to_movie_frame([an.BenchCell(0, (5.0, 5.0), None, None)], (20, 20), (20, 20), shift=(0, 0))
    assert out[0].footprint is None
