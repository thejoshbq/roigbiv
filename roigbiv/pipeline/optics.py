"""Optics-agnostic auto-adaptation: classify the FOV, derive the parameters.

The pipeline ships profiles (``profiles.py``) tuned per lens — GRIN (bright,
round, ~12 px somata, 512²) vs PRISM (dim, diffuse, ~56 px, 1024²). Asking a
user to *know* and *select* the lens is overhead we want to remove: the goal is
"upload FOVs → pipeline adapts itself → user only tightens ROIs in review".

This module supplies the two halves of that adaptation, split by *when* the
signal becomes available:

1. **Pre-foundation prior** (``classify_optics_prior``) — from cheap signals
   known at ingest (frame dimensions, optional pixel-size metadata). Picks the
   *categorical* profile (channel layout, model, denoise, foundation strip
   height). Deliberately coarse: a wide ambiguous band routes to ``generic`` +
   low confidence so the run pauses for the user rather than guessing.

2. **Post-foundation scale measurement** (``measure_soma_scale`` +
   ``derive_scale_params``) — from the registered ``mean_M`` once foundation has
   produced it. Measures the FOV's own soma scale and *derives* every numeric
   gate (areas, separations, pool radii) from it, so the profile's hardcoded
   area bounds become fallbacks, not authorities. This is what generalizes to
   lens types we have not tuned a profile for.

Precedence the callers enforce::

    dataclass defaults  <  categorical profile  <  scale-derived numerics  <  explicit/confirmed

Everything here is **total**: bad/missing input yields a low-confidence or
``ok=False`` result, never an exception — auto-adaptation must never be able to
crash a run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "OpticsPrior",
    "classify_optics_prior",
    "SomaScale",
    "measure_soma_scale",
    "derive_scale_params",
    "scale_plausible",
    "auto_scale_active",
    "resolved_config_payload",
    "SCALE_DERIVED_FIELDS",
    "GRIN_MAX_DIM",
    "PRISM_MIN_DIM",
]

# Profiles for which post-foundation scale derivation runs. GRIN (512²) is the
# validated baseline and is left byte-identical: a user on the GRIN profile has
# asserted the optics, so we don't second-guess its tuned gates.
_AUTO_SCALE_PROFILES = ("prism", "generic")

# The numeric gate fields ``derive_scale_params`` writes. Kept here so resume's
# fingerprint can exclude the auto-derived ones (a deterministic function of the
# on-disk mean_M + profile) without re-listing them.
SCALE_DERIVED_FIELDS = frozenset({
    "diameter", "min_area", "max_area", "gate1_merge_peak_min_separation",
    "spatial_pool_radius", "cluster_distance", "stage4_min_area",
    "stage4_max_area", "tile_norm_blocksize",
})


def resolved_config_payload(cfg) -> "dict | None":
    """Serializable optics-config snapshot to persist on the FOV registry row.

    Returns ``None`` for non-auto runs (no ``auto_adapt`` provenance), so the
    registry only remembers configs that were actually auto-resolved. The blob
    lets a repeat FOV reuse what worked instead of re-discovering.
    """
    aa = getattr(cfg, "auto_adapt", None) or {}
    if not aa:
        return None
    return {
        "profile": getattr(cfg, "profile", "grin"),
        "auto_scale": bool(getattr(cfg, "auto_scale", False)),
        "diameter": getattr(cfg, "diameter", None),
        "min_area": getattr(cfg, "min_area", None),
        "max_area": getattr(cfg, "max_area", None),
        "tile_norm_blocksize": getattr(cfg, "tile_norm_blocksize", None),
        "auto_adapt": aa,
    }


def auto_scale_active(cfg) -> bool:
    """True if post-foundation scale derivation applies to *cfg*.

    Single source of truth shared by the derivation hook (``run.py``) and the
    resume fingerprint (``resume.py``) so they never disagree about which fields
    are auto-derived.
    """
    return bool(getattr(cfg, "auto_scale", False)) and \
        getattr(cfg, "profile", "grin") in _AUTO_SCALE_PROFILES

# Frame-size decision band. 512² GRIN sits well below GRIN_MAX_DIM; 1024² PRISM
# well above PRISM_MIN_DIM. The gap between them is the *ambiguous* band that
# routes to ``generic`` + low confidence (→ pause-to-confirm), so an unusual or
# unseen frame size asks the user rather than silently mis-classifying.
GRIN_MAX_DIM = 640
PRISM_MIN_DIM = 896

# Pixel-size (µm/px) sanity envelopes, used only when metadata is present, as a
# tiebreaker/confidence booster — never the sole signal (most inputs lack it).
_GRIN_PIXEL_UM_MAX = 1.4   # GRIN tends to finer sampling
_PRISM_PIXEL_UM_MIN = 1.4  # PRISM coarser


@dataclass(frozen=True)
class OpticsPrior:
    """Pre-foundation categorical classification from cheap ingest signals."""

    profile_name: str            # "grin" | "prism" | "generic"
    confidence: str              # "high" | "low"
    reasons: list[str]           # human-readable; surfaced to run log + UI
    max_dim: int                 # max(H, W)
    pixel_size_um: "float | None" = None


def _max_dim(shape) -> int:
    """``max(H, W)`` from a ``(T, H, W)`` or ``(H, W)`` shape. ``0`` if unusable."""
    try:
        if shape is None or len(shape) < 2:
            return 0
        return int(max(int(shape[-2]), int(shape[-1])))
    except (TypeError, ValueError):
        return 0


def classify_optics_prior(shape, tiff_metadata: "dict | None" = None) -> OpticsPrior:
    """Classify the categorical profile from frame size (+ pixel size if present).

    Decision rule (intentionally conservative — see module docstring)::

        max_dim <= GRIN_MAX_DIM   -> grin    (high)
        max_dim >= PRISM_MIN_DIM  -> prism    (high)
        otherwise                 -> generic  (low; triggers pause-to-confirm)

    A ``pixel_size_um`` from metadata, when available, can *demote* a size-based
    pick to low confidence if it contradicts the size class, but never overrides
    the size pick on its own (metadata is frequently absent or wrong).

    Always returns an ``OpticsPrior`` — never raises.
    """
    md = tiff_metadata or {}
    pixel_um = None
    raw_px = md.get("pixel_size_um")
    if isinstance(raw_px, (int, float)) and raw_px > 0:
        pixel_um = float(raw_px)

    max_dim = _max_dim(shape)
    reasons: list[str] = []

    if max_dim == 0:
        reasons.append("frame shape unavailable; deferring to safe generic profile")
        return OpticsPrior("generic", "low", reasons, max_dim, pixel_um)

    if max_dim <= GRIN_MAX_DIM:
        profile, confidence = "grin", "high"
        reasons.append(f"max_dim={max_dim} <= {GRIN_MAX_DIM} → GRIN-class (512²) optics")
    elif max_dim >= PRISM_MIN_DIM:
        profile, confidence = "prism", "high"
        reasons.append(f"max_dim={max_dim} >= {PRISM_MIN_DIM} → PRISM-class (1024²) optics")
    else:
        profile, confidence = "generic", "low"
        reasons.append(
            f"max_dim={max_dim} in ambiguous band "
            f"({GRIN_MAX_DIM}, {PRISM_MIN_DIM}) → generic; please confirm optics"
        )

    # Pixel-size cross-check (only when present): demote to low confidence on
    # contradiction so the user is asked rather than silently trusted.
    if pixel_um is not None:
        if profile == "grin" and pixel_um >= _PRISM_PIXEL_UM_MIN:
            confidence = "low"
            reasons.append(
                f"pixel_size={pixel_um:.2f} µm/px is coarse for GRIN — confirm optics"
            )
        elif profile == "prism" and pixel_um <= _GRIN_PIXEL_UM_MAX:
            confidence = "low"
            reasons.append(
                f"pixel_size={pixel_um:.2f} µm/px is fine for PRISM — confirm optics"
            )

    return OpticsPrior(profile, confidence, reasons, max_dim, pixel_um)


# ---------------------------------------------------------------------------
# Post-foundation soma-scale measurement (Phase 2)
# ---------------------------------------------------------------------------

# Minimum number of cleanly-sized somata before a measurement is trusted. Below
# this the FOV is too sparse/dim to size reliably → ok=False → pause-to-confirm.
MIN_SCALE_SUPPORT = 5


@dataclass(frozen=True)
class SomaScale:
    """Measured soma-size statistics on a registered ``mean_M`` (pixels)."""

    diameter_med: float = 0.0
    diameter_p5: float = 0.0
    diameter_p95: float = 0.0
    area_med: float = 0.0
    area_p5: float = 0.0
    area_p95: float = 0.0
    n_somata: int = 0
    ok: bool = False


def measure_soma_scale(mean_M, dog_map=None, *, n_peaks: int = 40,
                       box_radius: int = 40) -> SomaScale:
    """DoG-peak + per-peak-Otsu soma sizing on the registered mean image.

    Promoted from ``scripts/measure_prism_scale.py`` (the empirical grounding for
    the PRISM preset) into the pipeline so it is the single source of truth for
    soma scale (``stage1._estimate_diameter_px`` also calls through here).

    Parameters
    ----------
    mean_M : 2-D array (the foundation ``mean_M`` summary image).
    dog_map : optional precomputed difference-of-Gaussians map (foundation
        already computes one); when ``None`` it is computed here.

    Returns a ``SomaScale``. ``ok`` is False (and stats are zero) on any failure
    or when fewer than ``MIN_SCALE_SUPPORT`` somata could be sized — never raises.
    """
    try:
        import numpy as np
        from skimage.feature import peak_local_max
        from skimage.filters import difference_of_gaussians, threshold_otsu
        from skimage.measure import label, regionprops

        img = np.asarray(mean_M, dtype=np.float32)
        if img.ndim != 2 or img.size == 0:
            return SomaScale()

        dog = (np.asarray(dog_map, dtype=np.float32)
               if dog_map is not None
               else difference_of_gaussians(img, low_sigma=3.0, high_sigma=15.0))

        peaks = peak_local_max(
            dog, min_distance=25, threshold_rel=0.15,
            num_peaks=n_peaks, exclude_border=box_radius,
        )

        H, W = img.shape
        diameters: list[float] = []
        areas: list[float] = []
        for (y, x) in peaks:
            y0, y1 = max(0, y - box_radius), min(H, y + box_radius)
            x0, x1 = max(0, x - box_radius), min(W, x + box_radius)
            crop = img[y0:y1, x0:x1]
            if crop.size < 100:
                continue
            try:
                t = threshold_otsu(crop)
            except Exception:
                continue
            lab = label(crop > t)
            cy, cx = y - y0, x - x0
            target = lab[cy, cx]
            if target == 0:
                continue
            for r in regionprops((lab == target).astype(np.uint8)):
                if r.area < 30 or r.area > 8000:
                    continue
                diameters.append(float(r.equivalent_diameter))
                areas.append(float(r.area))
                break

        n = len(diameters)
        if n < MIN_SCALE_SUPPORT:
            return SomaScale(n_somata=n, ok=False)

        d = np.asarray(diameters)
        a = np.asarray(areas)
        return SomaScale(
            diameter_med=float(np.median(d)),
            diameter_p5=float(np.percentile(d, 5)),
            diameter_p95=float(np.percentile(d, 95)),
            area_med=float(np.median(a)),
            area_p5=float(np.percentile(a, 5)),
            area_p95=float(np.percentile(a, 95)),
            n_somata=n,
            ok=True,
        )
    except Exception:
        return SomaScale()


def derive_scale_params(scale: SomaScale) -> dict:
    """Derive the *continuous* numeric gate params from a measured soma scale.

    Returns a flat dict keyed by ``PipelineConfig`` field names (overlays the
    profile's hardcoded numerics). Grounded so GRIN (d≈12) and PRISM (d≈56) both
    fall out of one formula:

    - ``d=12``  → min_area≈45,  max_area≈340   (near the tuned 80/600, conservative)
    - ``d=56``  → min_area≈985, max_area≈7389  (inside the validated 900..9000 band)

    The ``0.40``/``1.5`` multipliers are provisional — tune against PRISM ground
    truth before treating as load-bearing (see ``profiles.py`` provenance note).
    """
    d = max(1.0, float(scale.diameter_med))
    a_circ = math.pi * (d / 2.0) ** 2

    # Percentile-anchored bounds (matches measure_prism_scale.py's own suggestion)
    # with a circular-area floor/ceiling so a thin measurement cannot collapse
    # or explode the gate. The hard floor of 30 px² matches measure_soma_scale's
    # own minimum sized-component area — below that a "soma" is noise, so the gate
    # must not open wider than that even on a sparse small-soma FOV.
    min_area = max(30, int(round(min(0.40 * a_circ, 0.6 * scale.area_p5))))
    max_area = int(round(max(1.5 * scale.area_p95, 3.0 * a_circ)))

    return {
        "diameter": int(round(d)),
        "min_area": min_area,
        "max_area": max_area,
        "gate1_merge_peak_min_separation": int(round(d / 2.0)),
        "spatial_pool_radius": max(4, int(round(d / 2.0))),
        "cluster_distance": max(6, int(round(d * 0.6))),
        "stage4_min_area": int(round(0.6 * min_area)),
        "stage4_max_area": int(round(0.6 * max_area)),
        "tile_norm_blocksize": 256 if d > 28 else 128,
    }


def scale_plausible(scale: SomaScale, profile_name: "str | None" = None) -> bool:
    """Cross-check a measured scale against the resolved profile.

    Rejects measurements that contradict the profile badly enough to suggest the
    measurement (not the profile) is wrong — too sparse, bimodal junk, or a soma
    size that flatly disagrees with the lens class. On rejection the caller keeps
    the profile fallback and routes to pause-to-confirm. Keying on the resolved
    profile name (not only the auto prior) means the check also guards explicit
    ``--profile prism`` runs whose FOV turns out GRIN-sized.
    """
    if not scale.ok or scale.n_somata < MIN_SCALE_SUPPORT:
        return False
    # Bimodal / junk measurement: a huge spread between p5 and p95 diameters.
    if scale.diameter_p5 > 0 and (scale.diameter_p95 / scale.diameter_p5) > 6.0:
        return False
    # PRISM-class profile but GRIN-sized somata (or vice versa) → distrust.
    if profile_name == "prism" and scale.diameter_med < 18.0:
        return False
    if profile_name == "grin" and scale.diameter_med > 40.0:
        return False
    return True
