"""Accept tests for U2 (``bench.score``). Written by the contract; workers do not edit them."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.schema import FovScore  # noqa: E402
from bench.score import (  # noqa: E402
    aggregate, inter_rater, inter_rater_ceiling, max_distance_px, score_fov, soma_diameter_px,
)


def test_soma_diameter():
    assert soma_diameter_px([81, 81, 81]) == pytest.approx(2 * math.sqrt(81 / math.pi))
    # nan (POINT ROI) dropped; median of 64, 81, 100 is 81
    assert soma_diameter_px([81, float("nan"), 100, 64]) == pytest.approx(2 * math.sqrt(81 / math.pi))
    assert soma_diameter_px([50, 100]) == pytest.approx(2 * math.sqrt(75 / math.pi))


@pytest.mark.parametrize("areas", [[], [float("nan")], [81, 0], [-4]])
def test_soma_diameter_rejects(areas):
    with pytest.raises(ValueError):
        soma_diameter_px(areas)


def test_max_distance():
    assert max_distance_px(10.0) == pytest.approx(5.0)
    assert max_distance_px(10.0, 0.3) == pytest.approx(3.0)
    for d, f in [(0.0, 0.5), (10.0, 0.0), (-1.0, 0.5)]:
        with pytest.raises(ValueError):
            max_distance_px(d, f)


def test_score_fov_counts():
    gt = np.array([[10, 10], [30, 30], [50, 50]], dtype=float)
    pred = np.array([[10, 12], [30, 30], [50, 56], [80, 80]], dtype=float)
    s = score_fov("f", "GRIN", gt, pred, 5.0)
    assert (s.fov_id, s.lens, s.n_gt, s.n_pred, s.n_tp, s.max_distance) == ("f", "GRIN", 3, 4, 2, 5.0)
    assert s.precision == 0.5 and s.recall == pytest.approx(2 / 3)


def test_distance_rule_is_inclusive_and_strict_beyond():
    gt = np.array([[0.0, 0.0]])
    assert score_fov("f", "GRIN", gt, np.array([[3.0, 4.0]]), 5.0).n_tp == 1
    assert score_fov("f", "GRIN", gt, np.array([[3.0, 4.01]]), 5.0).n_tp == 0


def test_one_to_one():
    gt = np.array([[10.0, 10.0]])
    s = score_fov("f", "GRIN", gt, np.array([[10.0, 11.0], [11.0, 10.0]]), 5.0)
    assert s.n_tp == 1 and s.n_pred == 2
    s = score_fov("f", "GRIN", np.array([[10.0, 10.0], [10.0, 14.0]]), np.array([[10.0, 12.0]]), 5.0)
    assert s.n_tp == 1 and s.n_gt == 2


def test_optimal_not_greedy():
    # Greedy nearest-first pairs gt1 with pred0 (distance 1) and leaves gt0 unmatched.
    gt = np.array([[0.0, 0.0], [0.0, 4.0]])
    pred = np.array([[0.0, 3.0], [0.0, 7.0]])
    assert score_fov("f", "GRIN", gt, pred, 4.0).n_tp == 2


def test_empty_inputs():
    s = score_fov("f", "GRIN", np.zeros((0, 2)), np.array([[1.0, 1.0]]), 5.0)
    assert (s.n_gt, s.n_pred, s.n_tp) == (0, 1, 0) and s.recall is None
    s = score_fov("f", "GRIN", np.array([[1.0, 1.0]]), np.zeros((0, 2)), 5.0)
    assert (s.n_gt, s.n_pred, s.n_tp) == (1, 0, 0) and s.precision is None


def _scores():
    return [
        FovScore("a", "GRIN", 4, 4, 4, 5.0),
        FovScore("b", "GRIN", 4, 4, 3, 5.0),
        FovScore("c", "prism", 3, 3, 3, 5.0),
    ]


def test_aggregate_by_lens():
    g = aggregate(_scores())
    assert set(g) == {"GRIN", "prism"}
    grin = g["GRIN"]
    assert (grin.group, grin.n_fovs, grin.n_gt, grin.n_pred, grin.n_tp) == ("GRIN", 2, 8, 8, 7)
    assert grin.macro_recall == pytest.approx(0.875) and grin.recall == pytest.approx(0.875)
    assert g["prism"].n_fovs == 1 and g["prism"].macro_recall == pytest.approx(1.0)


def test_aggregate_all():
    g = aggregate(_scores(), by="all")
    assert list(g) == ["all"]
    a = g["all"]
    assert (a.n_fovs, a.n_gt, a.n_pred, a.n_tp) == (3, 11, 11, 10)
    assert a.macro_recall == pytest.approx((1.0 + 0.75 + 1.0) / 3)


def test_aggregate_macro_skips_fovs_without_gt():
    g = aggregate(_scores() + [FovScore("d", "GRIN", 0, 2, 0, 5.0)])
    grin = g["GRIN"]
    assert grin.n_fovs == 3 and grin.n_pred == 10 and grin.macro_recall == pytest.approx(0.875)
    only_empty = aggregate([FovScore("d", "GRIN", 0, 2, 0, 5.0)])
    assert only_empty["GRIN"].macro_recall is None


def test_aggregate_edges():
    assert aggregate([]) == {}
    with pytest.raises(ValueError):
        aggregate(_scores(), by="animal")


def test_inter_rater_pair():
    r1 = np.array([[12, 12], [12, 44], [44, 12], [44, 44]], dtype=float)
    r2 = np.array([[12, 12], [12, 44], [44, 12], [24, 28]], dtype=float)
    pairs = inter_rater("syn-a2", {"r2": r2, "r1": r1}, 5.0)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.fov_id, p.rater_a, p.rater_b, p.n_a, p.n_b, p.n_tp) == ("syn-a2", "r1", "r2", 4, 4, 3)
    assert p.f1 == pytest.approx(0.75) and p.max_distance == 5.0


def test_inter_rater_order_and_singletons():
    pts = np.array([[1.0, 1.0]])
    pairs = inter_rater("f", {"c": pts, "a": pts, "b": pts}, 5.0)
    assert [(p.rater_a, p.rater_b) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert inter_rater("f", {"a": pts}, 5.0) == []


def test_inter_rater_ceiling():
    from bench.schema import RaterPair

    pairs = [
        RaterPair("f", "a", "b", 4, 4, 3, 5.0),   # f1 0.75
        RaterPair("f", "a", "c", 2, 2, 1, 5.0),   # f1 0.5
        RaterPair("f", "b", "c", 0, 0, 0, 5.0),   # f1 None, skipped
    ]
    assert inter_rater_ceiling(pairs) == pytest.approx(0.625)
    assert inter_rater_ceiling([]) is None
