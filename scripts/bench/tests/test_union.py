"""Accept tests for U5 (``bench.union``). Written by the contract; workers do not edit them."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.union import aggregate_union, union_score  # noqa: E402

GT = np.array([[10, 10], [30, 30], [50, 50], [70, 70]], dtype=float)


def three_sources():
    return {
        "cp": np.array([[10, 10], [30, 31]], dtype=float),            # gt0, gt1
        "s2p": np.array([[30, 30], [50, 51], [90, 90]], dtype=float),  # gt1, gt2, one FP
        "act": np.array([[70, 70], [10, 11]], dtype=float),            # gt3, gt0
    }


def test_union_counts():
    u = union_score("f", GT, three_sources(), 3.0)
    assert u.fov_id == "f" and u.n_gt == 4 and u.max_distance == 3.0
    assert u.n_candidates == {"cp": 2, "s2p": 3, "act": 2} and u.n_candidates_total == 7
    assert u.pool_n_tp == 4 and u.pool_recall == 1.0 and u.pool_precision == pytest.approx(4 / 7)
    assert u.covered == {"cp": 2, "s2p": 2, "act": 2}
    assert u.unique == {"cp": 0, "s2p": 1, "act": 1}


def test_dict_order_follows_sources():
    u = union_score("f", GT, three_sources(), 3.0)
    assert list(u.n_candidates) == list(u.covered) == list(u.unique) == ["cp", "s2p", "act"]


def test_empty_source():
    srcs = three_sources()
    srcs["none"] = np.zeros((0, 2))
    u = union_score("f", GT, srcs, 3.0)
    assert u.n_candidates["none"] == 0 and u.covered["none"] == 0 and u.unique["none"] == 0
    assert u.unique["s2p"] == 1


def test_pool_is_one_to_one():
    u = union_score("f", np.array([[10.0, 10.0], [10.0, 12.0]]), {"a": np.array([[10.0, 11.0]])}, 3.0)
    assert u.pool_n_tp == 1 and u.covered == {"a": 1} and u.unique == {"a": 1}


def test_distance_gate():
    u = union_score("f", np.array([[0.0, 0.0]]), {"a": np.array([[0.0, 3.5]])}, 3.0)
    assert u.pool_n_tp == 0 and u.covered == {"a": 0}


def test_no_gt():
    u = union_score("f", np.zeros((0, 2)), {"a": np.array([[1.0, 1.0]])}, 3.0)
    assert u.n_gt == 0 and u.pool_recall is None and u.pool_precision == 0.0


def test_aggregate():
    s1 = union_score("f1", GT, three_sources(), 3.0)
    s2 = union_score(
        "f2", np.array([[5.0, 5.0], [20.0, 20.0]]),
        {"cp": np.array([[5.0, 5.0]]), "s2p": np.zeros((0, 2))}, 3.0,
    )
    a = aggregate_union([s1, s2])
    assert a["n_fovs"] == 2 and a["n_gt"] == 6
    assert a["pool_recall"] == pytest.approx(5 / 6)
    assert a["pool_precision"] == pytest.approx(5 / 8)
    assert a["candidates_per_fov"] == pytest.approx(4.0)
    assert a["recall"] == pytest.approx({"cp": 3 / 6, "s2p": 2 / 6, "act": 2 / 6})
    assert a["unique_recall"] == pytest.approx({"cp": 1 / 6, "s2p": 1 / 6, "act": 1 / 6})


def test_aggregate_empty():
    assert aggregate_union([]) == {
        "n_fovs": 0, "n_gt": 0, "pool_recall": None, "pool_precision": None,
        "candidates_per_fov": None, "recall": {}, "unique_recall": {},
    }
