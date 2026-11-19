"""Matcher: annotations to candidates, one-to-one, with collisions and a chance baseline."""

import numpy as np
import pytest

from roigbiv.workbench import match as mt

SHAPE = (60, 80)


def disk(cy, cx, r, shape=SHAPE):
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return np.flatnonzero(((ys - cy) ** 2 + (xs - cx) ** 2) <= r**2)


def ann(i, cy, cx, r=4):
    return mt.Annotation(ann_id=i, centroid=(cy, cx), footprint=disk(cy, cx, r))


def cand(cid, cy, cx, r=4, footprint=True):
    return mt.Candidate(candidate_id=cid, centroid=(cy, cx), footprint=disk(cy, cx, r) if footprint else None)


def test_exact_cells_link_one_to_one_with_full_overlap():
    links = mt.match([ann(0, 10, 10), ann(1, 30, 40)], [cand("b", 30, 40), cand("a", 10, 10)], radius=5.0, shape=SHAPE)
    by = {lk.ann_id: lk for lk in links}
    assert by[0].candidate_id == "a" and by[1].candidate_id == "b"
    assert by[0].distance == 0.0 and by[0].overlap == 1.0 and by[0].area_ratio == 1.0
    assert by[0].recoverable and not by[0].collision


def test_an_offset_candidate_within_radius_is_localized_and_overlap_falls_with_the_offset():
    (lk,) = mt.match([ann(0, 20, 20)], [cand("a", 20, 23)], radius=5.0, shape=SHAPE)
    assert lk.candidate_id == "a" and lk.distance == pytest.approx(3.0)
    assert 0 < lk.overlap < 1
    assert lk.localized


def test_a_candidate_beyond_the_radius_is_not_linked():
    (lk,) = mt.match([ann(0, 20, 20)], [cand("a", 20, 27)], radius=5.0, shape=SHAPE)
    assert lk.candidate_id is None and not lk.localized and not lk.recoverable


def test_one_candidate_between_two_annotations_is_a_collision_and_links_only_one():
    links = mt.match([ann(0, 20, 20), ann(1, 20, 26)], [cand("a", 20, 23)], radius=5.0, shape=SHAPE)
    linked = [lk for lk in links if lk.candidate_id == "a"]
    assert len(linked) == 1
    assert all(lk.collision for lk in links)  # both annotations had the same candidate as their only option
    other = next(lk for lk in links if lk.candidate_id is None)
    assert other.localized  # a candidate was within reach, but it went to the neighbour


def test_assignment_is_one_to_one_and_minimises_total_distance():
    links = mt.match([ann(0, 20, 20), ann(1, 20, 28)], [cand("a", 20, 24), cand("b", 20, 31)], radius=5.0, shape=SHAPE)
    by = {lk.ann_id: lk.candidate_id for lk in links}
    assert by == {0: "a", 1: "b"}


def test_a_much_larger_candidate_is_localized_but_not_recoverable():
    (lk,) = mt.match([ann(0, 30, 40, r=3)], [cand("big", 30, 40, r=12)], radius=5.0, shape=SHAPE)
    assert lk.localized and lk.overlap == 1.0
    assert lk.area_ratio > 2 and not lk.recoverable


def test_a_candidate_without_a_footprint_is_localized_with_overlap_unknown():
    (lk,) = mt.match([ann(0, 30, 40)], [cand("c", 30, 41, footprint=False)], radius=5.0, shape=SHAPE)
    assert lk.localized and lk.overlap is None and lk.area_ratio is None
    assert lk.recoverable is None  # unknown, never counted as a miss or a hit


def test_rotate_180_maps_centroids_and_footprints_into_the_frame():
    a = ann(0, 10, 15)
    (r,) = mt.rotate_180([a], SHAPE)
    assert r.centroid == (49.0, 64.0)
    assert np.array_equal(np.sort(r.footprint), np.sort(disk(49, 64, 4)))


def test_shift_wraps_so_no_annotation_leaves_the_frame():
    (s,) = mt.shift_wrap([ann(0, 55, 75)], SHAPE, (10, 10))
    assert s.centroid == (5.0, 5.0)
    assert len(s.footprint) == len(disk(55, 75, 4))


def test_summary_reports_recall_minus_the_worse_baseline():
    anns = [ann(i, 10 + 12 * (i // 5), 10 + 14 * (i % 5)) for i in range(15)]
    cands = [cand(f"c{i}", a.centroid[0], a.centroid[1]) for i, a in enumerate(anns[:12])]
    s = mt.summarize(anns, cands, radius=5.0, shape=SHAPE, shift=(17, 23))
    assert s.n_annotations == 15
    assert s.localized == 12 and s.recoverable == 12 and s.recoverable_unknown == 0
    assert s.localization_coverage == pytest.approx(0.8)
    assert 0 <= s.baseline_rotate <= 1 and 0 <= s.baseline_shift <= 1
    assert s.corrected_recoverable_recall == pytest.approx(0.8 - max(s.baseline_rotate, s.baseline_shift))


def test_chance_baseline_of_a_dense_random_field_is_high():
    rng = np.random.default_rng(0)
    pts = rng.uniform([5, 5], [55, 75], size=(200, 2))
    anns = [ann(i, *p) for i, p in enumerate(pts[:40])]
    cands = [cand(f"c{i}", *p) for i, p in enumerate(pts[40:])]
    s = mt.summarize(anns, cands, radius=5.0, shape=SHAPE, shift=(17, 23))
    assert s.baseline_rotate > 0.3  # the stop rule's threshold: a matcher this loose must be tightened


def test_matcher_has_a_version():
    assert mt.MATCHER_VERSION == 1


def test_the_assignment_never_links_a_pair_beyond_the_radius():
    links = mt.match([ann(0, 20, 20), ann(1, 50, 70)], [cand("c1", 20, 22), cand("far", 5, 5)], radius=5.0, shape=SHAPE)
    by = {lk.ann_id: lk for lk in links}
    assert by[0].candidate_id == "c1"
    assert by[1].candidate_id is None and not by[1].localized


def test_a_same_size_candidate_offset_past_half_overlap_is_localized_but_not_recoverable():
    (lk,) = mt.match([ann(0, 20, 20)], [cand("a", 20, 25)], radius=6.0, shape=SHAPE)
    assert lk.localized and lk.area_ratio == 1.0
    assert lk.overlap < mt.MIN_OVERLAP and lk.recoverable is False


def test_the_worse_of_the_two_baselines_is_the_one_subtracted():
    # The pair is 180-degree symmetric, so the rotated annotations land exactly on the candidates.
    anns = [ann(0, 10, 10), ann(1, 49, 69)]
    cands = [cand("a", 10, 10), cand("b", 49, 69)]
    s = mt.summarize(anns, cands, radius=5.0, shape=SHAPE, shift=(17, 23))
    assert s.baseline_rotate == 1.0 and s.baseline_shift < 1.0
    assert s.baseline == 1.0 and s.corrected_recoverable_recall == 0.0
