"""Cross-session cell correspondence derived without ROICaT.

The matcher under test cannot also be the judge of the matcher under test, so
correspondence here is established from the centroids alone:

1. propose a translation between two sessions three ways — identity, phase
   correlation on the mean projections, and a vote over centroid difference
   vectors;
2. score each proposal by how many centroids it brings into correspondence;
3. keep the winner and Hungarian-assign under it.

Step 2 is the part that matters. Phase correlation on its own picked a peak
237 px off on one pair of the reference prism FOV — right method, wrong
maximum — while identity was correct for that pair and wrong for the next.
Scoring the proposals catches that; trusting any single estimator does not.

Translation only, deliberately. If a rotation or scale change between sessions
were large enough to matter, the inlier counts reported here would stay low and
say so, which is more useful than silently absorbing it into a richer model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from scripts.centroid_bakeoff.point_match import PointMatchResult, match_points


@dataclass
class SessionPairing:
    """Correspondence between two sessions' centroids, plus how it was found."""

    session_a: str
    session_b: str
    shift_yx: tuple[float, float]
    shift_source: str               # "identity" | "phase_correlation" | "centroid_vote"
    match: PointMatchResult
    # Inlier count for every proposal considered, so a near-tie is visible
    # rather than hidden behind the winner.
    proposal_scores: dict[str, int]

    @property
    def n_pairs(self) -> int:
        return self.match.n_tp

    @property
    def median_residual(self) -> Optional[float]:
        if not self.match.tp:
            return None
        return float(np.median([d for _, _, d in self.match.tp]))

    def to_dict(self) -> dict:
        return {
            "session_a": self.session_a,
            "session_b": self.session_b,
            "shift_yx": [float(self.shift_yx[0]), float(self.shift_yx[1])],
            "shift_source": self.shift_source,
            "n_pairs": self.n_pairs,
            "n_a": self.match.n_fn + self.n_pairs,
            "n_b": self.match.n_fp + self.n_pairs,
            "median_residual_px": self.median_residual,
            "proposal_scores": dict(self.proposal_scores),
        }


def pair_sessions(
    name_a: str,
    centroids_a: np.ndarray,
    mean_a: np.ndarray,
    name_b: str,
    centroids_b: np.ndarray,
    mean_b: np.ndarray,
    *,
    max_distance: float = 25.0,
) -> SessionPairing:
    """Best-scoring translation from *b* into *a*'s frame, and the pairing under it."""
    a = np.asarray(centroids_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(centroids_b, dtype=np.float64).reshape(-1, 2)

    proposals: dict[str, tuple[float, float]] = {"identity": (0.0, 0.0)}
    pc = _phase_correlation_shift(mean_a, mean_b)
    if pc is not None:
        proposals["phase_correlation"] = pc
    vote = _centroid_vote_shift(a, b, max_distance=max_distance)
    if vote is not None:
        proposals["centroid_vote"] = vote

    scores: dict[str, int] = {}
    best_name, best_shift, best_match = "identity", (0.0, 0.0), None
    for name, shift in proposals.items():
        m = match_points(a, b + np.asarray(shift), max_distance)
        scores[name] = m.n_tp
        better = best_match is None or m.n_tp > best_match.n_tp or (
            m.n_tp == best_match.n_tp
            and (m.mean_localization_error or np.inf)
            < (best_match.mean_localization_error or np.inf)
        )
        if better:
            best_name, best_shift, best_match = name, shift, m

    assert best_match is not None
    return SessionPairing(
        session_a=name_a, session_b=name_b,
        shift_yx=best_shift, shift_source=best_name,
        match=best_match, proposal_scores=scores,
    )


def transitive_cells(
    session_names: list[str], pairings: dict[tuple[int, int], SessionPairing]
) -> list[dict[int, int]]:
    """Merge pairwise correspondences into cells spanning three or more sessions.

    Union-find over ``(session_idx, roi_idx)`` nodes. A cell that picks up two
    ROIs from one session — which pairwise matching alone permits — is split
    back apart and reported as inconsistent rather than quietly kept, since
    such a cell is evidence the pairwise result is wrong somewhere.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (i, j), pairing in pairings.items():
        for a_idx, b_idx, _ in pairing.match.tp:
            union((i, a_idx), (j, b_idx))

    groups: dict[tuple[int, int], dict[int, int]] = {}
    inconsistent: set[tuple[int, int]] = set()
    for node in list(parent):
        root = find(node)
        sess, roi = node
        members = groups.setdefault(root, {})
        if sess in members and members[sess] != roi:
            inconsistent.add(root)
        members[sess] = roi

    return [g for root, g in groups.items() if root not in inconsistent]


# ── shift proposals ────────────────────────────────────────────────────────


def _phase_correlation_shift(mean_a, mean_b) -> Optional[tuple[float, float]]:
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        return None

    def z(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std() + 1e-9)

    shift, _, _ = phase_cross_correlation(z(mean_a), z(mean_b), upsample_factor=10)
    return float(shift[0]), float(shift[1])


def _centroid_vote_shift(
    a: np.ndarray, b: np.ndarray, *, max_distance: float
) -> Optional[tuple[float, float]]:
    """The translation most pairs of centroids agree on.

    Every (a, b) pair casts one vote for the shift that would superimpose them.
    True pairs all vote for nearly the same vector; false pairs scatter. Take
    the densest bin, then refine to the mean of the votes inside it — the bin
    fixes the mode, the mean recovers sub-bin precision.
    """
    if len(a) == 0 or len(b) == 0:
        return None
    votes = (a[:, None, :] - b[None, :, :]).reshape(-1, 2)
    bin_px = max(4.0, max_distance / 2.0)
    keys = np.round(votes / bin_px).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                      return_counts=True)
    winner = int(np.argmax(counts))
    if counts[winner] < 2:
        return None
    inliers = votes[inverse == winner]
    return float(inliers[:, 0].mean()), float(inliers[:, 1].mean())
