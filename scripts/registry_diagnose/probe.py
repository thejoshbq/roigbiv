"""Run the real matcher and read out what each stage did to the true pairs.

Every measurement here is keyed to the ground-truth pairing from
``ground_truth.py``, so each stage answers one question: *are the cells that
belong together still together by the time this stage is done with them?* A
stage where true-pair scores stop separating from random-pair scores is the
stage that lost the correspondence.

The stage objects come from ``cluster_sessions(..., trace=...)`` rather than a
reimplementation, so what is measured is what actually runs in production.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse

from roigbiv.registry.calibration import CalibrationModel
from roigbiv.registry.match import compute_fov_features
from roigbiv.registry.roicat_adapter import (
    AdapterConfig,
    SessionInput,
    cluster_sessions,
)

# How many random non-pairs to sample as the contrast for each channel. The
# true-pair count is ~15 per session pair, so a few hundred gives a stable
# reference percentile without making the report unreadable.
_N_RANDOM = 500


@dataclass
class ChannelStats:
    """One similarity channel's scores on true pairs vs. everything else."""

    name: str
    true_pairs: list[float] = field(default_factory=list)
    random_pairs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        t = np.asarray(self.true_pairs, dtype=np.float64)
        r = np.asarray(self.random_pairs, dtype=np.float64)
        return {
            "channel": self.name,
            "n_true": int(t.size),
            "true_median": _f(np.median(t)) if t.size else None,
            "true_p10": _f(np.percentile(t, 10)) if t.size else None,
            "random_median": _f(np.median(r)) if r.size else None,
            "random_p90": _f(np.percentile(r, 90)) if r.size else None,
            "separation": _f(np.median(t) - np.percentile(r, 90))
            if t.size and r.size else None,
            "auc": _auc(t, r),
        }


@dataclass
class ProbeResult:
    """Everything one configuration of the matcher revealed."""

    config_name: str
    session_names: list[str]
    n_rois: list[int]
    elapsed_s: float
    alignment_method: str
    alignment_inlier_rate: float
    implied_shift_yx: list[Optional[tuple[float, float]]]
    centroid_residual_before: dict
    centroid_residual_after: dict
    channels: list[ChannelStats]
    d_cutoff: Optional[float]
    pruned_survival: dict
    labels: np.ndarray
    clusters_recovered: dict
    posterior: float
    decision: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "sessions": self.session_names,
            "n_rois": self.n_rois,
            "elapsed_s": round(self.elapsed_s, 1),
            "alignment": {
                "method": self.alignment_method,
                "inlier_rate": _f(self.alignment_inlier_rate),
                "implied_shift_yx": [
                    None if s is None else [_f(s[0]), _f(s[1])]
                    for s in self.implied_shift_yx
                ],
                "true_pair_residual_before": self.centroid_residual_before,
                "true_pair_residual_after": self.centroid_residual_after,
            },
            "channels": [c.to_dict() for c in self.channels],
            "pruning": {"d_cutoff": _f(self.d_cutoff), **self.pruned_survival},
            "clustering": self.clusters_recovered,
            "posterior": _f(self.posterior),
            "decision": self.decision,
            "error": self.error,
        }


def probe(
    config_name: str,
    sessions: list[SessionInput],
    cfg: AdapterConfig,
    *,
    truth: list[dict[int, int]],
    raw_centroids: list[np.ndarray],
    calibration: Optional[CalibrationModel] = None,
    accept_threshold: float = 0.8,
    review_threshold: float = 0.5,
) -> ProbeResult:
    """Cluster *sessions* under *cfg* and score every stage against *truth*.

    ``truth`` is the list of cells from
    :func:`ground_truth.transitive_cells` — each a ``{session_idx: roi_idx}``
    mapping, with ``roi_idx`` indexing ROIs in ascending-``label_id`` order,
    the same order ``cluster_sessions`` concatenates them in.
    """
    offsets = np.cumsum([0] + [len(c) for c in raw_centroids])
    true_pairs = _true_pair_indices(truth, offsets)

    trace: dict = {}
    started = time.time()
    error = None
    try:
        result = cluster_sessions(sessions, cfg, trace=trace)
    except Exception as exc:  # noqa: BLE001 — a crash is itself a finding
        elapsed = time.time() - started
        return _failed_probe(config_name, sessions, raw_centroids, elapsed,
                             cfg, f"{type(exc).__name__}: {exc}")
    elapsed = time.time() - started

    aligner = trace.get("aligner")
    sim = trace.get("sim")
    clusterer = trace.get("clusterer")

    aligned_centroids = _aligned_centroids(aligner, [len(c) for c in raw_centroids])
    residual_before = _residual_stats(truth, raw_centroids)
    residual_after = (
        _residual_stats(truth, aligned_centroids)
        if aligned_centroids is not None else {"note": "ROIs_aligned unavailable"}
    )

    total = int(offsets[-1])
    random_pairs = _random_pair_indices(offsets, exclude=set(true_pairs), n=_N_RANDOM)
    channels = [
        _channel(name, getattr(sim, attr, None), true_pairs, random_pairs)
        for name, attr in (("s_sf", "s_sf"), ("s_NN_z", "s_NN_z"),
                           ("s_SWT_z", "s_SWT_z"))
    ]

    d_cutoff = _f(getattr(clusterer, "d_cutoff", None))
    pruned = _pruned_survival(clusterer, true_pairs, random_pairs)

    features = compute_fov_features(result, query_session_idx=len(sessions) - 1)
    posterior = float((calibration or CalibrationModel()).p_same_fov(features))
    decision = ("auto_match" if posterior >= accept_threshold
                else "review" if posterior >= review_threshold else "reject")

    return ProbeResult(
        config_name=config_name,
        session_names=[s.session_key for s in sessions],
        n_rois=[len(c) for c in raw_centroids],
        elapsed_s=elapsed,
        alignment_method=result.alignment_method,
        alignment_inlier_rate=result.alignment_inlier_rate,
        implied_shift_yx=_implied_shifts(aligner, len(sessions)),
        centroid_residual_before=residual_before,
        centroid_residual_after=residual_after,
        channels=channels,
        d_cutoff=d_cutoff,
        pruned_survival=pruned,
        labels=result.labels,
        clusters_recovered=_score_clusters(result.labels, truth, offsets, total),
        posterior=posterior,
        decision=decision,
        error=error,
    )


# ── stage measurements ─────────────────────────────────────────────────────


def _true_pair_indices(truth, offsets) -> list[tuple[int, int]]:
    """Global ROI index pairs that ground truth says are the same cell."""
    pairs: list[tuple[int, int]] = []
    for cell in truth:
        members = [int(offsets[s] + r) for s, r in sorted(cell.items())]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


def _random_pair_indices(offsets, *, exclude, n) -> list[tuple[int, int]]:
    """Cross-session ROI pairs ground truth does *not* pair.

    Cross-session only: same-session pairs can never be the same cell, so
    including them would make any channel look more discriminating than it is.
    """
    rng = np.random.default_rng(0)
    n_sessions = len(offsets) - 1
    if n_sessions < 2:
        return []
    out: list[tuple[int, int]] = []
    seen = set(exclude) | {(b, a) for a, b in exclude}
    for _ in range(n * 20):
        if len(out) >= n:
            break
        si, sj = rng.choice(n_sessions, size=2, replace=False)
        ci, cj = offsets[si + 1] - offsets[si], offsets[sj + 1] - offsets[sj]
        if ci == 0 or cj == 0:
            continue
        a = int(offsets[si] + rng.integers(ci))
        b = int(offsets[sj] + rng.integers(cj))
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _channel(name, matrix, true_pairs, random_pairs) -> ChannelStats:
    stats = ChannelStats(name=name)
    if matrix is None:
        return stats
    m = sparse.csr_matrix(matrix) if sparse.issparse(matrix) else np.asarray(matrix)
    stats.true_pairs = [_at(m, i, j) for i, j in true_pairs]
    stats.random_pairs = [_at(m, i, j) for i, j in random_pairs]
    return stats


def _pruned_survival(clusterer, true_pairs, random_pairs) -> dict:
    """How many true pairs still have an edge after ROICaT prunes the graph.

    A true pair with no surviving edge cannot be clustered together no matter
    what the Hungarian step does, so this separates "similarity was too weak"
    from "the cutoff threw it away".
    """
    d = getattr(clusterer, "dConj_pruned", None)
    if d is None:
        return {"note": "dConj_pruned unavailable"}
    m = sparse.csr_matrix(d) if sparse.issparse(d) else np.asarray(d)
    true_d = [_at(m, i, j, missing=np.nan) for i, j in true_pairs]
    rand_d = [_at(m, i, j, missing=np.nan) for i, j in random_pairs]
    n_true_edge = int(np.sum(~np.isnan(true_d)))
    return {
        "true_pairs_with_edge": n_true_edge,
        "true_pairs_total": len(true_pairs),
        "true_edge_median_distance": _f(np.nanmedian(true_d))
        if n_true_edge else None,
        "random_pairs_with_edge": int(np.sum(~np.isnan(rand_d))),
        "random_pairs_total": len(random_pairs),
        "random_edge_median_distance": _f(np.nanmedian(rand_d))
        if np.any(~np.isnan(rand_d)) else None,
    }


def _score_clusters(labels, truth, offsets, total) -> dict:
    """Did the final labels put each ground-truth cell in one cluster?"""
    labels = np.asarray(labels).reshape(-1)
    recovered = 0
    split = 0
    for cell in truth:
        idx = [int(offsets[s] + r) for s, r in cell.items()]
        vals = {int(labels[i]) for i in idx if i < labels.size}
        if len(vals) == 1 and -1 not in vals:
            recovered += 1
        else:
            split += 1
    clustered = int(np.sum(labels != -1))
    return {
        "truth_cells": len(truth),
        "recovered_whole": recovered,
        "split_or_unclustered": split,
        "rois_clustered": clustered,
        "rois_total": int(total),
        "distinct_clusters": int(len({int(v) for v in labels if v != -1})),
    }


def _implied_shifts(aligner, n_sessions) -> list[Optional[tuple[float, float]]]:
    """Translation each session's fitted warp applies at the frame centre.

    Read off the remapping grid rather than the transform matrix so it reports
    what was actually applied, including any method-specific convention.
    """
    remap = getattr(aligner, "remappingIdx_geo", None)
    if remap is None:
        return [None] * n_sessions
    out: list[Optional[tuple[float, float]]] = []
    for i in range(n_sessions):
        try:
            grid = np.asarray(remap[i])
            h, w = grid.shape[0], grid.shape[1]
            cy, cx = h // 2, w // 2
            out.append((float(grid[cy, cx, 1] - cy), float(grid[cy, cx, 0] - cx)))
        except Exception:  # noqa: BLE001 — shape varies by ROICaT version
            out.append(None)
    return out


def _aligned_centroids(aligner, counts) -> Optional[list[np.ndarray]]:
    rois = getattr(aligner, "ROIs_aligned", None)
    if rois is None:
        return None
    out: list[np.ndarray] = []
    for k, fps in enumerate(rois):
        m = sparse.csr_matrix(fps)
        n_rois = m.shape[0]
        side = int(round(np.sqrt(m.shape[1])))
        cents = np.zeros((n_rois, 2), dtype=np.float64)
        for r in range(n_rois):
            row = m.getrow(r)
            if row.nnz == 0:
                cents[r] = (np.nan, np.nan)
                continue
            ys, xs = np.divmod(row.indices, side)
            wgt = np.abs(row.data)
            wsum = wgt.sum() or 1.0
            cents[r] = ((ys * wgt).sum() / wsum, (xs * wgt).sum() / wsum)
        out.append(cents)
        if k >= len(counts):
            break
    return out


def _residual_stats(truth, centroids_per_session) -> dict:
    """Distance between ground-truth partners, in whatever frame they're given."""
    dists: list[float] = []
    for cell in truth:
        pts = [centroids_per_session[s][r] for s, r in cell.items()
               if s < len(centroids_per_session)
               and r < len(centroids_per_session[s])]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(np.asarray(pts[i]) - np.asarray(pts[j])))
                if not np.isnan(d):
                    dists.append(d)
    if not dists:
        return {"n": 0}
    arr = np.asarray(dists)
    return {
        "n": int(arr.size),
        "median_px": _f(np.median(arr)),
        "p90_px": _f(np.percentile(arr, 90)),
        "max_px": _f(arr.max()),
    }


def _failed_probe(config_name, sessions, raw_centroids, elapsed, cfg, error):
    return ProbeResult(
        config_name=config_name,
        session_names=[s.session_key for s in sessions],
        n_rois=[len(c) for c in raw_centroids],
        elapsed_s=elapsed,
        alignment_method=cfg.alignment_method,
        alignment_inlier_rate=float("nan"),
        implied_shift_yx=[None] * len(sessions),
        centroid_residual_before=_residual_stats([], raw_centroids),
        centroid_residual_after={},
        channels=[],
        d_cutoff=None,
        pruned_survival={},
        labels=np.zeros(0, dtype=np.int32),
        clusters_recovered={},
        posterior=float("nan"),
        decision="error",
        error=error,
    )


# ── small helpers ──────────────────────────────────────────────────────────


def _at(m, i, j, missing: float = 0.0) -> float:
    """Element ``(i, j)``, mapping a structurally absent entry to *missing*.

    Absent and zero are indistinguishable when reading a sparse matrix, and for
    a pruned graph the difference is the whole question — "no edge" is not
    "an edge of length zero". Callers asking about edges pass ``missing=nan``.
    """
    try:
        v = m[i, j]
    except (IndexError, ValueError):
        return missing
    v = float(v.toarray().ravel()[0]) if sparse.issparse(v) else float(v)
    return missing if v == 0.0 else v


def _f(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def _auc(true_scores: np.ndarray, random_scores: np.ndarray) -> Optional[float]:
    """P(a true pair scores above a random pair) — 0.5 means the channel is blind."""
    if true_scores.size == 0 or random_scores.size == 0:
        return None
    gt = (true_scores[:, None] > random_scores[None, :]).sum()
    eq = (true_scores[:, None] == random_scores[None, :]).sum()
    denom = true_scores.size * random_scores.size
    return round(float((gt + 0.5 * eq) / denom), 4)
