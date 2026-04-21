"""Pure FOV + cell matching logic.

Two-pass FOV match:
  1. Phase correlation on mean_M (downsampled) → translation (dy, dx) + peak.
  2. Cell-distribution Hungarian match → fov_sim.

Cell match (within a matched FOV): Hungarian over a cost matrix combining
centroid distance (after translation) and morphology distance.

All functions operate on numpy arrays and CellFeature dataclasses; no DB
or filesystem coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from roigbiv.registry.fingerprint import CellFeature

AUTO_ACCEPT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60
MAX_CENTROID_PX = 8.0


@dataclass
class FOVMatchResult:
    matched: bool
    fov_sim: float
    peak_correlation: float
    translation_yx: tuple[float, float]
    decision: str


@dataclass
class CellMatch:
    query_label_id: int
    candidate_label_id: Optional[int]
    score: float


@dataclass
class CellMatchResult:
    matches: list[CellMatch] = field(default_factory=list)
    new_query_labels: list[int] = field(default_factory=list)
    missing_candidate_labels: list[int] = field(default_factory=list)


def phase_correlate(query: np.ndarray, candidate: np.ndarray) -> tuple[tuple[float, float], float]:
    """Return ((dy, dx), peak_correlation) aligning `query` onto `candidate`.

    Uses skimage's phase_cross_correlation on equally-shaped images. If shapes
    differ, the larger is center-cropped to match.
    """
    from skimage.registration import phase_cross_correlation

    q, c = _crop_to_common(query, candidate)
    shift, _, _ = phase_cross_correlation(c, q, normalization=None, upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    peak = _normalized_peak(q, c, dy, dx)
    return (dy, dx), peak


def match_fov(
    query_mean_m: np.ndarray,
    query_cells: list[CellFeature],
    candidate_mean_m: np.ndarray,
    candidate_cells: list[CellFeature],
) -> FOVMatchResult:
    (dy, dx), peak = phase_correlate(query_mean_m, candidate_mean_m)

    translated = [
        CellFeature(
            local_label_id=c.local_label_id,
            centroid_y=c.centroid_y + dy,
            centroid_x=c.centroid_x + dx,
            area=c.area,
            solidity=c.solidity,
            eccentricity=c.eccentricity,
            nuclear_shadow_score=c.nuclear_shadow_score,
            soma_surround_contrast=c.soma_surround_contrast,
        )
        for c in query_cells
    ]
    cell_result = match_cells(translated, candidate_cells)

    denom = max(len(query_cells), len(candidate_cells), 1)
    n_matched = len(cell_result.matches)
    match_fraction = n_matched / denom

    if n_matched > 0:
        mean_score = float(np.mean([m.score for m in cell_result.matches]))
    else:
        mean_score = 0.0

    fov_sim = 0.5 * match_fraction + 0.5 * mean_score

    if fov_sim >= AUTO_ACCEPT_THRESHOLD:
        decision = "auto_match"
    elif fov_sim >= REVIEW_THRESHOLD:
        decision = "review"
    else:
        decision = "reject"

    return FOVMatchResult(
        matched=decision == "auto_match",
        fov_sim=float(fov_sim),
        peak_correlation=float(peak),
        translation_yx=(dy, dx),
        decision=decision,
    )


def match_cells(
    query: list[CellFeature],
    candidates: list[CellFeature],
    max_px: float = MAX_CENTROID_PX,
) -> CellMatchResult:
    """Hungarian cell matching.

    `query` is expected to be pre-translated to the candidate coordinate
    frame. Pairs with centroid distance > `max_px` are considered infeasible
    and excluded from the matching; leftovers become new / missing.
    """
    from scipy.optimize import linear_sum_assignment

    if not query or not candidates:
        return CellMatchResult(
            matches=[],
            new_query_labels=[q.local_label_id for q in query],
            missing_candidate_labels=[c.local_label_id for c in candidates],
        )

    Q = len(query)
    C = len(candidates)
    big = 1e6
    cost = np.full((Q, C), big, dtype=np.float64)
    dist_mat = np.zeros((Q, C), dtype=np.float64)

    for i, q in enumerate(query):
        for j, c in enumerate(candidates):
            d = float(np.hypot(q.centroid_y - c.centroid_y, q.centroid_x - c.centroid_x))
            dist_mat[i, j] = d
            if d > max_px:
                continue
            morph = _morph_distance(q, c)
            cost[i, j] = d + 2.0 * morph

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[CellMatch] = []
    matched_rows = set()
    matched_cols = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= big:
            continue
        score = _similarity_from_cost(dist_mat[i, j], _morph_distance(query[i], candidates[j]))
        matches.append(CellMatch(
            query_label_id=int(query[i].local_label_id),
            candidate_label_id=int(candidates[j].local_label_id),
            score=float(score),
        ))
        matched_rows.add(i)
        matched_cols.add(j)

    new_labels = [
        int(query[i].local_label_id) for i in range(Q) if i not in matched_rows
    ]
    missing_labels = [
        int(candidates[j].local_label_id) for j in range(C) if j not in matched_cols
    ]
    return CellMatchResult(
        matches=matches,
        new_query_labels=new_labels,
        missing_candidate_labels=missing_labels,
    )


def _morph_distance(a: CellFeature, b: CellFeature) -> float:
    area_rel = abs(a.area - b.area) / max(a.area, b.area, 1)
    sol_diff = abs(a.solidity - b.solidity)
    ecc_diff = abs(a.eccentricity - b.eccentricity)
    nuc_diff = abs(a.nuclear_shadow_score - b.nuclear_shadow_score)
    return float(area_rel + sol_diff + ecc_diff + 0.5 * nuc_diff)


def _similarity_from_cost(dist_px: float, morph: float) -> float:
    dist_sim = max(0.0, 1.0 - dist_px / MAX_CENTROID_PX)
    morph_sim = max(0.0, 1.0 - morph)
    return 0.5 * dist_sim + 0.5 * morph_sim


def _normalized_peak(q: np.ndarray, c: np.ndarray, dy: float, dx: float) -> float:
    """Pearson correlation after applying the recovered translation to q."""
    from scipy.ndimage import shift as ndi_shift

    shifted = ndi_shift(q, shift=(dy, dx), order=1, mode="nearest")
    a = shifted.ravel() - shifted.mean()
    b = c.ravel() - c.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def _crop_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])

    def _center_crop(x):
        dy = (x.shape[0] - H) // 2
        dx = (x.shape[1] - W) // 2
        return x[dy:dy + H, dx:dx + W]

    return _center_crop(a), _center_crop(b)
