"""Motion-correction quality metrics for a temporal-mean image.

A well-registered movie has a *sharp* temporal mean; residual motion blurs it.
Per-row jitter (the rowwise-pcc failure mode) shows up as horizontal banding,
which :func:`banding_score` isolates. Shared by ``scripts/bench_motion_correction.py``
(offline A/B bench across backends) and the Pipeline page's per-FOV metrics
panel (live quality readout after each foundation run).
"""
from __future__ import annotations

import numpy as np


def _znorm(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-8)


def lap_var(img: np.ndarray) -> float:
    """Variance of the Laplacian — classic focus/sharpness measure."""
    from scipy.ndimage import laplace
    return float(laplace(_znorm(img)).var())


def lap_var_smooth(img: np.ndarray, sigma: float = 1.0) -> float:
    """Laplacian variance after a light Gaussian blur.

    Pre-smoothing suppresses per-pixel shot/scan noise (which inflates the raw
    Laplacian on unregistered means) so the metric tracks *cell-edge* sharpness
    rather than noise. This is the headline sharpness number for the gate.
    """
    from scipy.ndimage import laplace, gaussian_filter
    return float(laplace(gaussian_filter(_znorm(img), sigma)).var())


def grad_energy(img: np.ndarray) -> float:
    a = _znorm(img)
    gy, gx = np.gradient(a)
    return float((gx ** 2 + gy ** 2).mean())


def tenengrad(img: np.ndarray) -> float:
    from scipy.ndimage import sobel
    a = _znorm(img)
    sx, sy = sobel(a, axis=0), sobel(a, axis=1)
    return float((sx ** 2 + sy ** 2).mean())


def grad_anisotropy(img: np.ndarray) -> float:
    """Horizontal/vertical gradient-energy ratio.

    Horizontal (x) jitter blur suppresses horizontal gradients, pulling this
    below 1.0. ~1.0 means isotropic sharpness (healthy).
    """
    a = _znorm(img)
    gy, gx = np.gradient(a)
    ex = float((gx ** 2).mean())
    ey = float((gy ** 2).mean())
    return ex / (ey + 1e-12)


def banding_score(img: np.ndarray) -> float:
    """Row-to-row streak energy. Higher = more horizontal banding.

    Collapse to a per-row mean profile, high-pass it (remove smooth cellular
    structure), and report the residual variance. Pure horizontal bands survive
    the column-average; isotropic cell texture largely cancels.
    """
    from scipy.ndimage import gaussian_filter1d
    a = _znorm(img)
    row = a.mean(axis=1)
    hp = row - gaussian_filter1d(row, 4.0)
    return float(hp.var())


def contrast_rms(img: np.ndarray) -> float:
    """Michelson-ish dynamic range on a robust percentile span."""
    a = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(a, [1, 99])
    return float(hi - lo)


def compute_metrics(img: np.ndarray) -> dict:
    return {
        "lap_var_smooth": lap_var_smooth(img),
        "lap_var": lap_var(img),
        "grad_energy": grad_energy(img),
        "tenengrad": tenengrad(img),
        "grad_anisotropy_xy": grad_anisotropy(img),
        "banding_score": banding_score(img),
        "contrast_rms": contrast_rms(img),
    }
