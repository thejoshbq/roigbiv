"""Residual-retention diagnostics for L+S background calibration (§3.3).

The sequential pipeline detects Stage 1 on ``mean_M`` (raw movie mean) but
**subtraction (§5) and Stages 2–4 consume the residual ``S = M − L``**. If the
top-*k* SVD background ``L`` absorbs a soma's brightness (the documented
*k-too-high* regime, spec §3.3 / Blindspot 4), that cell vanishes from ``S`` and
the downstream stages can never recover it — even when Stage 1 detected it fine.

This module quantifies that absorption per ground-truth soma, replacing the
spec's eyeball "soma shapes clearly visible in mean(S), absent from mean(L)"
check with a number that can gate Phase A and drive Phase-K k-selection.

Identity used: because ``M = L + S`` (per frame, hence per-pixel time-mean),
``mean_M = mean_L + mean_S``. So the retained-brightness fraction is

    r_S(mask) = Σ_mask mean_S / Σ_mask (mean_S + mean_L)
              = Σ_mask mean_S / Σ_mask mean_M

``r_S → 1`` ⇒ the soma survives into the residual (good); ``r_S → 0`` ⇒ it was
absorbed into the low-rank background (bad). Values can fall slightly outside
[0, 1] because ``mean_S`` is a signed residual; callers threshold on ``τ_retain``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["mask_retention", "retention_summary", "count_vcorr_maxima"]


def mask_retention(mean_S: np.ndarray, mean_L: np.ndarray,
                   mask: np.ndarray) -> float:
    """Fraction of one GT soma's mean brightness retained in residual ``S``.

    Parameters
    ----------
    mean_S, mean_L : (H, W) float — Foundation residual / background mean projections
        (``summary/mean_S.tif`` and ``mean_L.tif``).
    mask : (H, W) bool — a single ground-truth soma footprint.

    Returns
    -------
    float — ``r_S`` for this soma (see module docstring). NaN if the mask is
    empty or the denominator (total soma brightness) is ~0.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    s = float(np.asarray(mean_S, dtype=np.float64)[m].sum())
    denom = s + float(np.asarray(mean_L, dtype=np.float64)[m].sum())
    if abs(denom) < 1e-9:
        return float("nan")
    return s / denom


def retention_summary(mean_S: np.ndarray, mean_L: np.ndarray,
                      masks: list[np.ndarray],
                      tau_retain: float = 0.5) -> dict:
    """Aggregate per-soma retention across a GT mask set.

    Returns a dict with median/mean/min ``r_S`` and the fraction of somata
    clearing ``tau_retain`` — the quantity that gates Phase A (A1) and Phase-K
    gate 3, and that the k-sweep maximizes (without re-introducing background).
    """
    vals = np.array(
        [mask_retention(mean_S, mean_L, mk) for mk in masks], dtype=np.float64
    )
    finite = vals[np.isfinite(vals)]
    n = int(finite.size)
    return {
        "n_masks": len(masks),
        "n_scored": n,
        "r_S_median": float(np.median(finite)) if n else float("nan"),
        "r_S_mean": float(np.mean(finite)) if n else float("nan"),
        "r_S_min": float(np.min(finite)) if n else float("nan"),
        "tau_retain": tau_retain,
        "frac_pass": float(np.mean(finite >= tau_retain)) if n else float("nan"),
        "per_mask": vals.tolist(),
    }


def count_vcorr_maxima(vcorr_S: np.ndarray, *, min_distance: int = 8,
                       threshold_rel: float = 0.3) -> int:
    """Count localized hotspots in a Vcorr(S) map — the §3.3 k-plateau backbone.

    The spec's k-selection heuristic (spec:297-308, Blindspot 3/4) inspects where
    ``Vcorr(S)`` shows localized soma hotspots. Counting local maxima vs ``k`` and
    taking the **plateau** (where extra rank stops surfacing new hotspots) gives an
    automated k-selector to pair with :func:`retention_summary`. Returns the number
    of peaks; falls back to a simple threshold-count if scikit-image is absent.
    """
    arr = np.asarray(vcorr_S, dtype=np.float32)
    try:
        from skimage.feature import peak_local_max
    except ImportError:
        thr = float(arr.min() + threshold_rel * (arr.max() - arr.min()))
        return int((arr > thr).sum())
    peaks = peak_local_max(
        arr, min_distance=min_distance, threshold_rel=threshold_rel,
        exclude_border=False,
    )
    return int(peaks.shape[0])
