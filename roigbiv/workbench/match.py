"""Matcher: hand-drawn annotations to pipeline candidates, outside the pipeline's write path.

Every annotation gets one link row: the candidate it was assigned (one-to-one, minimum total centroid distance
within ``radius``), the distance, the footprint overlap (share of the annotation covered by the candidate) and
the area ratio (candidate over annotation). Two measures come from the links:

- localized: some candidate centroid lies within ``radius`` of the annotation (not one-to-one);
- recoverable: assigned one-to-one with overlap >= ``MIN_OVERLAP`` and area ratio within ``AREA_RATIO``.
  It is None (unknown) when either footprint is missing, and never counted as a hit or a miss.

A collision is an annotation with a candidate in reach that is also in reach of another annotation. The chance
baseline repeats the join against the annotations rotated 180 degrees and shifted by a fixed offset (wrapping, so
no annotation leaves the frame); a sound matcher scores near zero there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHER_VERSION = 1
MIN_OVERLAP = 0.5
AREA_RATIO = (0.5, 2.0)
_FAR = 1e9


@dataclass(frozen=True)
class Annotation:
    ann_id: int
    centroid: tuple  # (y, x) in the movie frame
    footprint: np.ndarray | None  # flat pixel indices; None for a point ROI


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    centroid: tuple
    footprint: np.ndarray | None


@dataclass(frozen=True)
class Link:
    ann_id: int
    candidate_id: str | None
    distance: float | None
    overlap: float | None
    area_ratio: float | None
    localized: bool
    recoverable: bool | None
    collision: bool


def _distances(anns: Sequence[Annotation], cands: Sequence[Candidate]) -> np.ndarray:
    if not anns or not cands:
        return np.zeros((len(anns), len(cands)))
    a = np.array([x.centroid for x in anns], float)
    c = np.array([x.centroid for x in cands], float)
    return np.hypot(a[:, None, 0] - c[None, :, 0], a[:, None, 1] - c[None, :, 1])


def _shape_terms(a: Annotation, c: Candidate) -> tuple[float | None, float | None, bool | None]:
    if a.footprint is None or c.footprint is None or len(a.footprint) == 0:
        return None, None, None
    overlap = len(np.intersect1d(a.footprint, c.footprint, assume_unique=False)) / len(a.footprint)
    ratio = len(c.footprint) / len(a.footprint)
    ok = overlap >= MIN_OVERLAP and AREA_RATIO[0] <= ratio <= AREA_RATIO[1]
    return float(overlap), float(ratio), bool(ok)


def match(anns: Sequence[Annotation], cands: Sequence[Candidate], *, radius: float, shape: tuple) -> list[Link]:
    """One link per annotation, in annotation order."""
    d = _distances(anns, cands)
    near = d <= radius
    assigned: dict[int, int] = {}
    if near.any():
        rows, cols = linear_sum_assignment(np.where(near, d, _FAR))
        assigned = {int(r): int(c) for r, c in zip(rows, cols) if near[r, c]}
    reach = near.sum(axis=0)  # annotations in reach of each candidate
    links = []
    for i, a in enumerate(anns):
        collision = bool((near[i] & (reach > 1)).any()) if len(cands) else False
        j = assigned.get(i)
        if j is None:
            links.append(
                Link(a.ann_id, None, None, None, None, bool(near[i].any()) if len(cands) else False, False, collision)
            )
            continue
        overlap, ratio, ok = _shape_terms(a, cands[j])
        links.append(Link(a.ann_id, cands[j].candidate_id, float(d[i, j]), overlap, ratio, True, ok, collision))
    return links


def _flat_map(anns: Sequence[Annotation], shape: tuple, fn) -> list[Annotation]:
    out = []
    for a in anns:
        fp = None
        if a.footprint is not None:
            ys, xs = np.unravel_index(a.footprint, shape)
            ys, xs = fn(ys, xs)
            fp = np.sort(np.ravel_multi_index((ys, xs), shape))
        cy, cx = fn(np.array([a.centroid[0]]), np.array([a.centroid[1]]))
        out.append(Annotation(a.ann_id, (float(cy[0]), float(cx[0])), fp))
    return out


def rotate_180(anns: Sequence[Annotation], shape: tuple) -> list[Annotation]:
    h, w = shape
    return _flat_map(anns, shape, lambda ys, xs: (h - 1 - ys, w - 1 - xs))


def shift_wrap(anns: Sequence[Annotation], shape: tuple, offset: tuple) -> list[Annotation]:
    h, w = shape
    dy, dx = offset
    return _flat_map(anns, shape, lambda ys, xs: ((ys + dy) % h, (xs + dx) % w))


@dataclass(frozen=True)
class Summary:
    n_annotations: int
    localized: int
    recoverable: int
    recoverable_unknown: int
    collisions: int
    localization_coverage: float
    recoverable_recall: float
    baseline_rotate: float  # recoverable recall against the rotated annotations
    baseline_shift: float
    loc_baseline_rotate: float  # localization coverage against the rotated annotations
    loc_baseline_shift: float
    matcher_version: int = MATCHER_VERSION

    @property
    def baseline(self) -> float:
        return max(self.baseline_rotate, self.baseline_shift)

    @property
    def corrected_recoverable_recall(self) -> float:
        return self.recoverable_recall - self.baseline

    @property
    def corrected_localization_coverage(self) -> float:
        return self.localization_coverage - max(self.loc_baseline_rotate, self.loc_baseline_shift)


def _rates(links: Sequence[Link], n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    return sum(lk.localized for lk in links) / n, sum(lk.recoverable is True for lk in links) / n


def summarize(
    anns: Sequence[Annotation], cands: Sequence[Candidate], *, radius: float, shape: tuple, shift: tuple
) -> Summary:
    links = match(anns, cands, radius=radius, shape=shape)
    n = len(anns)
    loc, rec = _rates(links, n)
    loc_r, rec_r = _rates(match(rotate_180(anns, shape), cands, radius=radius, shape=shape), n)
    loc_s, rec_s = _rates(match(shift_wrap(anns, shape, shift), cands, radius=radius, shape=shape), n)
    return Summary(
        n_annotations=n,
        localized=sum(lk.localized for lk in links),
        recoverable=sum(lk.recoverable is True for lk in links),
        recoverable_unknown=sum(lk.recoverable is None and lk.candidate_id is not None for lk in links),
        collisions=sum(lk.collision for lk in links),
        localization_coverage=loc,
        recoverable_recall=rec,
        baseline_rotate=rec_r,
        baseline_shift=rec_s,
        loc_baseline_rotate=loc_r,
        loc_baseline_shift=loc_s,
    )
