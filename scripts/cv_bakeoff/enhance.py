"""Summary-image enrichment transforms (Part 2 of the CV-only refactor).

Foundation writes raw float32 summaries with **no** contrast normalization
applied. These composable transforms squeeze more signal out of those images
before a detector sees them. They operate purely in image space (a loaded
``summary`` dict), so iteration is fast and env-free — ideal for tuning against
the classical detector, which has no learned priors and exposes summary-image
quality directly.

Temporal-statistic summaries (PNR, multi-radius correlation, per-pixel
skew/kurtosis) need the residual movie and belong in ``foundation.py``'s summary
generation, not here; once a transform proves out it graduates there.

Usage::

    from cv_bakeoff.enhance import apply_enhancements
    summary = apply_enhancements(summary, "tophat,clahe")
    summary = apply_enhancements(summary, "dog:2:6,stretch")   # colon args

A transform chain is applied in order to **every** channel in the dict; pick the
channel a detector consumes via ``--background``. Each transform normalizes its
own input as needed and returns float32.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, restoration
from skimage.filters import difference_of_gaussians, unsharp_mask


def _stretch01(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


# ── transforms ───────────────────────────────────────────────────────────────
# Each takes (img, *float_args) and returns a float32 array.

def t_stretch(img, lo_pct: float = 1.0, hi_pct: float = 99.5):
    """Percentile clip + linear stretch to [0, 1]."""
    return _stretch01(img, lo_pct, hi_pct)


def t_clahe(img, clip: float = 0.01, kernel: float = 0):
    """Contrast-limited adaptive histogram equalization (local contrast)."""
    norm = _stretch01(img)
    k = int(kernel) if kernel else None
    return exposure.equalize_adapthist(
        norm, clip_limit=clip, kernel_size=k,
    ).astype(np.float32)


def t_tophat(img, radius: float = 15.0):
    """White top-hat: subtract a morphological opening to flatten background."""
    norm = _stretch01(img)
    footprint = _disk(int(round(radius)))
    return ndi.white_tophat(norm, footprint=footprint).astype(np.float32)


def t_unsharp(img, radius: float = 3.0, amount: float = 1.0):
    """Unsharp masking — sharpen soma edges."""
    return unsharp_mask(_stretch01(img), radius=radius, amount=amount).astype(np.float32)


def t_tvdenoise(img, weight: float = 0.1):
    """Total-variation denoising (edge-preserving smooth)."""
    return restoration.denoise_tv_chambolle(
        _stretch01(img), weight=weight,
    ).astype(np.float32)


def t_dog(img, sigma_low: float = 2.0, sigma_high: float = 6.0):
    """Difference-of-Gaussians band-pass at the soma spatial-frequency scale."""
    return difference_of_gaussians(
        _stretch01(img), low_sigma=sigma_low, high_sigma=sigma_high,
    ).astype(np.float32)


def t_log(img, sigma: float = 3.0):
    """Laplacian-of-Gaussian blob response (negated so blobs are bright)."""
    norm = _stretch01(img)
    return (-ndi.gaussian_laplace(norm, sigma=sigma)).astype(np.float32)


def t_gamma(img, gamma: float = 0.5):
    """Gamma correction on the stretched image (gamma<1 lifts dim cells)."""
    return exposure.adjust_gamma(_stretch01(img), gamma).astype(np.float32)


def t_log1p(img):
    """Log compression of dynamic range."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr - arr.min()
    return np.log1p(arr).astype(np.float32)


def _disk(radius: int) -> np.ndarray:
    r = max(1, radius)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx <= r * r)


_REGISTRY: dict[str, Callable] = {
    "stretch": t_stretch,
    "clahe": t_clahe,
    "tophat": t_tophat,
    "unsharp": t_unsharp,
    "tvdenoise": t_tvdenoise,
    "dog": t_dog,
    "log": t_log,
    "gamma": t_gamma,
    "log1p": t_log1p,
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def _parse_token(token: str) -> tuple[Callable, list[float]]:
    parts = token.split(":")
    name = parts[0].strip()
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown enhancement {name!r}; available: {', '.join(available())}"
        )
    args = [float(a) for a in parts[1:] if a != ""]
    return _REGISTRY[name], args


def apply_enhancements(summary: dict[str, np.ndarray], spec: str) -> dict[str, np.ndarray]:
    """Apply a comma-list transform chain to every channel; return a new dict."""
    chain = [_parse_token(tok) for tok in spec.split(",") if tok.strip()]
    out: dict[str, np.ndarray] = {}
    for name, arr in summary.items():
        cur = np.asarray(arr, dtype=np.float32)
        for fn, args in chain:
            cur = fn(cur, *args)
        out[name] = cur
    return out
