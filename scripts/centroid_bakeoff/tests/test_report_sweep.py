"""Unit tests for report.py's sweep-reporting additions (no matplotlib rendering)."""
from __future__ import annotations

from dataclasses import dataclass

from centroid_bakeoff.report import _pareto_frontier


@dataclass
class _FakeMatch:
    recall: float
    precision: float


class _FakePoint:
    def __init__(self, recall, precision):
        self.match = _FakeMatch(recall, precision)


def test_pareto_frontier_drops_dominated_points():
    # (0.1, 0.5) is dominated by (0.2, 0.9): worse on both axes -> dropped.
    pts = [
        _FakePoint(0.1, 0.5),
        _FakePoint(0.2, 0.9),
        _FakePoint(0.5, 0.7),
        _FakePoint(0.8, 0.3),
    ]
    frontier = _pareto_frontier(pts)
    recalls = [p.match.recall for p in frontier]
    assert recalls == sorted(recalls)          # ascending recall
    assert 0.1 not in recalls                   # dominated point excluded
    assert {0.2, 0.5, 0.8}.issubset(set(recalls))


def test_pareto_frontier_ties_keep_first_seen_max_precision():
    pts = [_FakePoint(0.3, 0.5), _FakePoint(0.3, 0.9), _FakePoint(0.3, 0.2)]
    frontier = _pareto_frontier(pts)
    assert len(frontier) == 1
    assert frontier[0].match.precision == 0.9


def test_pareto_frontier_empty():
    assert _pareto_frontier([]) == []
