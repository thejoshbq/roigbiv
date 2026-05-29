"""
ROI G. Biv pipeline — Gate 1: Morphological Validation (spec §6).

Validates Stage 1 ROIs before source subtraction:
  - Computes per-ROI morphological features (area, solidity, eccentricity)
    via skimage.measure.regionprops
  - Computes soma-surround contrast via annulus construction that excludes
    other ROI pixels
  - Looks up mean DoG score across mask pixels (more robust than single-centroid
    sample per Plan agent D1)
  - Applies the accept/flag/reject decision from spec §6 with per-criterion
    absolute margins for "marginal" flagging (Plan agent B7)

Decision rules (spec §6):
  REJECT if area ∉ [80, 350] OR solidity < 0.55 OR eccentricity > 0.90
          OR (nuclear_shadow_score strongly negative AND contrast ≤ 0.1)
  FLAG   if exactly one criterion fails within its per-criterion margin
  ACCEPT otherwise
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops

from roigbiv.pipeline.types import ROI, PipelineConfig


def compute_annulus(
    mask: np.ndarray,
    other_masks_union: np.ndarray,
    inner_buffer: int,
    outer_radius: int,
) -> np.ndarray:
    """Annulus around `mask`, excluding `other_masks_union` pixels.

    ring = dilate(mask, outer_radius) AND NOT dilate(mask, inner_buffer)
           AND NOT other_masks_union
    """
    outer = binary_dilation(mask, iterations=outer_radius)
    inner = binary_dilation(mask, iterations=inner_buffer)
    ring = outer & ~inner & ~other_masks_union
    return ring


def compute_soma_surround_contrast(
    mask: np.ndarray,
    annulus: np.ndarray,
    mean_S: np.ndarray,
) -> float:
    """(mean[mask] − mean[annulus]) / mean[annulus]. Positive = brighter than surround."""
    if not mask.any() or not annulus.any():
        return 0.0
    m = float(mean_S[mask].mean())
    a = float(mean_S[annulus].mean())
    # Avoid division by values near zero (post-subtraction mean can be tiny)
    denom = a if abs(a) > 1e-6 else (1e-6 if a >= 0 else -1e-6)
    return (m - a) / denom


def evaluate_gate1(
    candidates: list[np.ndarray],
    probs: list[float],
    mean_S: np.ndarray,
    vcorr_S: np.ndarray,
    dog_map: np.ndarray,
    cfg: PipelineConfig,
    starting_label_id: int = 1,
) -> list[ROI]:
    """Apply Gate 1 to a list of candidate Stage 1 masks.

    Parameters
    ----------
    candidates        : list of (H, W) bool arrays
    probs             : list of float, same length — Cellpose probabilities
    mean_S            : (H, W) float32 — for contrast computation
    vcorr_S           : (H, W) float32 — saved in features for later inspection
    dog_map           : (H, W) float32 — for nuclear shadow score
    cfg               : PipelineConfig
    starting_label_id : int — first ROI label to assign

    Returns
    -------
    list of ROI with all spatial features, gate_outcome, confidence populated.
    Rejected ROIs are included (gate_outcome == "reject") so the full Stage 1
    report can record them. Caller should filter to accept+flag before
    source subtraction.
    """
    H, W = mean_S.shape

    # Union of all candidate masks for annulus exclusion
    all_union = np.zeros((H, W), dtype=bool)
    for m in candidates:
        all_union |= m

    # DoG threshold for "strongly negative" — 10th percentile of the map
    dog_thresh_strong_neg = float(
        np.percentile(dog_map, cfg.dog_strong_negative_percentile)
    )

    rois: list[ROI] = []
    next_label = starting_label_id

    for i, (mask, prob) in enumerate(zip(candidates, probs)):
        area = int(mask.sum())
        if area == 0:
            continue

        # regionprops: use the mask as a label image with a single nonzero label
        lbl = mask.astype(np.uint8)
        rp_list = regionprops(lbl)
        if not rp_list:
            continue
        rp = rp_list[0]
        solidity = float(rp.solidity) if rp.solidity is not None else 0.0
        eccentricity = float(rp.eccentricity) if rp.eccentricity is not None else 0.0

        # Soma-surround contrast
        others = all_union & ~mask
        annulus = compute_annulus(
            mask, others,
            inner_buffer=cfg.annulus_inner_buffer,
            outer_radius=cfg.annulus_outer_radius,
        )
        contrast = compute_soma_surround_contrast(mask, annulus, mean_S)

        # Nuclear shadow score: mean DoG over mask (more robust than centroid sample)
        nuclear_shadow_score = float(dog_map[mask].mean())

        # ── Decision rules ────────────────────────────────────────────────
        failures = []
        if area < cfg.min_area:
            failures.append(("area_low", area - cfg.min_area))
        elif area > cfg.max_area:
            failures.append(("area_high", area - cfg.max_area))
        if solidity < cfg.min_solidity:
            failures.append(("solidity", cfg.min_solidity - solidity))
        if eccentricity > cfg.max_eccentricity:
            failures.append(("eccentricity", eccentricity - cfg.max_eccentricity))
        if contrast <= cfg.min_contrast:
            failures.append(("contrast", cfg.min_contrast - contrast))

        # Conjunctive DoG rule: reject only if strongly negative AND contrast fails
        strong_neg_dog = (nuclear_shadow_score < dog_thresh_strong_neg)
        contrast_fails = (contrast <= cfg.min_contrast)
        dog_reject = strong_neg_dog and contrast_fails

        # Determine outcome
        if dog_reject or len([f for f in failures if f[0] != "contrast"]) >= 2:
            # Definite reject: DoG-contrast conjunction, or ≥2 hard failures
            outcome = "reject"
            confidence = "requires_review"
        elif len(failures) == 0:
            outcome = "accept"
            confidence = "high"
        elif len(failures) == 1:
            # Check if the single failure is within its per-criterion margin → flag
            name, delta = failures[0]
            margin_ok = False
            d = abs(delta)
            if name in ("area_low", "area_high"):
                margin_ok = d <= cfg.flag_area_margin
            elif name == "solidity":
                margin_ok = d <= cfg.flag_solidity_margin
            elif name == "eccentricity":
                margin_ok = d <= cfg.flag_eccentricity_margin
            elif name == "contrast":
                margin_ok = d <= cfg.flag_contrast_margin
            if margin_ok:
                outcome = "flag"
                confidence = "moderate"
            else:
                outcome = "reject"
                confidence = "requires_review"
        else:
            # 2+ failures but not dog_reject — still reject
            outcome = "reject"
            confidence = "requires_review"

        gate_reasons = [f"{name}:{delta:+.3f}" for name, delta in failures]
        if dog_reject:
            gate_reasons.append(f"dog_contrast_conjunction:score={nuclear_shadow_score:.3f}")

        roi = ROI(
            mask=mask,
            label_id=next_label,
            source_stage=1,
            confidence=confidence,
            gate_outcome=outcome,
            area=area,
            solidity=solidity,
            eccentricity=eccentricity,
            nuclear_shadow_score=nuclear_shadow_score,
            soma_surround_contrast=contrast,
            cellpose_prob=float(prob),
            gate_reasons=gate_reasons,
            features={
                "vcorr_mean": float(vcorr_S[mask].mean()),
                "mean_S_mean": float(mean_S[mask].mean()),
                "dog_thresh_strong_neg": dog_thresh_strong_neg,
            },
        )
        rois.append(roi)
        next_label += 1

    return rois
