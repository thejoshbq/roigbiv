"""
ROI G. Biv --- Step 10: Quality control and activity-type classification.

Stage A: Automated cell/not-cell rejection based on spatial and temporal features.
Stage B: Activity-type classification (phasic, tonic, sparse, ambiguous).

All thresholds are configurable via pipeline.yaml [classify] section.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("classify")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_qc_features(
    F: np.ndarray,
    Fneu: np.ndarray,
    dFF: np.ndarray,
    roi_mask: np.ndarray,
    records: list[dict],
    fs: float = 30.0,
) -> pd.DataFrame:
    """Compute per-ROI quality control features.

    Features:
      std           — standard deviation of dF/F trace
      skew          — skewness of dF/F trace
      pct_range     — 95th - 5th percentile of dF/F
      area_px       — ROI area in pixels
      compact       — 4*pi*area / perimeter^2 (circularity)
      snr           — max(dF/F) / std(dF/F)
      mean_F        — mean raw fluorescence
      cv            — coefficient of variation (std/mean of raw F)
      n_transients  — number of threshold crossings in dF/F
      source_branch — bitmask: A=1, B=2, C=4

    Parameters
    ----------
    F          : (n_rois, T) raw fluorescence
    Fneu       : (n_rois, T) neuropil fluorescence
    dFF        : (n_rois, T) dF/F traces
    roi_mask   : (Ly, Lx) uint16 merged mask
    records    : merge records with source_branches info
    fs         : frame rate in Hz

    Returns
    -------
    features : pd.DataFrame indexed by roi_id (1-based)
    """
    n_rois = F.shape[0]
    labels = np.unique(roi_mask[roi_mask > 0])

    rows = []
    for i in range(n_rois):
        roi_id = i + 1
        trace = dFF[i]
        f_raw = F[i]

        # Temporal features
        std_val = float(np.std(trace))
        mean_trace = float(np.mean(trace))
        skew_val = 0.0
        if std_val > 1e-10:
            skew_val = float(np.mean(((trace - mean_trace) / std_val) ** 3))

        pct_range = float(np.percentile(trace, 95) - np.percentile(trace, 5))
        snr = float(np.max(trace) / std_val) if std_val > 1e-10 else 0.0
        mean_f = float(np.mean(f_raw))
        cv = float(np.std(f_raw) / mean_f) if mean_f > 1e-10 else 0.0

        # Transient counting: threshold crossings above 2*std
        threshold = 2.0 * std_val
        above = trace > threshold
        # Count rising edges
        transitions = np.diff(above.astype(np.int8))
        n_transients = int(np.sum(transitions == 1))

        # Spatial features
        area_px = 0
        compact = 0.0
        if roi_id in labels or roi_id <= roi_mask.max():
            roi_pixels = roi_mask == roi_id
            area_px = int(roi_pixels.sum())

            if area_px > 0:
                # Compute perimeter: count boundary pixels (pixels with at
                # least one 4-connected neighbor outside the ROI)
                padded = np.pad(roi_pixels, 1, mode="constant", constant_values=False)
                interior = (
                    padded[1:-1, 1:-1]
                    & padded[:-2, 1:-1]
                    & padded[2:, 1:-1]
                    & padded[1:-1, :-2]
                    & padded[1:-1, 2:]
                )
                perimeter = int(roi_pixels.sum() - interior.sum())
                if perimeter > 0:
                    compact = float(4.0 * np.pi * area_px / (perimeter ** 2))

        # Source branch encoding
        source_branch = 0
        if i < len(records):
            branches = records[i].get("source_branches", "")
            if "A" in branches:
                source_branch |= 1
            if "B" in branches:
                source_branch |= 2
            if "C" in branches:
                source_branch |= 4

        rows.append({
            "roi_id": roi_id,
            "std": round(std_val, 6),
            "skew": round(skew_val, 4),
            "pct_range": round(pct_range, 6),
            "area_px": area_px,
            "compact": round(compact, 4),
            "snr": round(snr, 4),
            "mean_F": round(mean_f, 4),
            "cv": round(cv, 4),
            "n_transients": n_transients,
            "source_branch": source_branch,
        })

    return pd.DataFrame(rows).set_index("roi_id")


# ---------------------------------------------------------------------------
# Stage A: Cell/not-cell rejection
# ---------------------------------------------------------------------------

def classify_cell_nocell(
    features: pd.DataFrame,
    snr_min: float = 2.0,
    area_min: int = 30,
    area_max: int = 500,
    compact_min: float = 0.15,
) -> pd.Series:
    """Stage A: Binary cell/not-cell classification.

    An ROI is rejected (is_cell=False) if ANY of:
      - SNR < snr_min
      - area outside [area_min, area_max]
      - compactness < compact_min

    Parameters
    ----------
    features    : DataFrame from compute_qc_features
    snr_min     : minimum signal-to-noise ratio
    area_min    : minimum ROI area in pixels
    area_max    : maximum ROI area in pixels
    compact_min : minimum circularity (0-1)

    Returns
    -------
    is_cell : pd.Series of bool, indexed by roi_id
    """
    is_cell = (
        (features["snr"] >= snr_min)
        & (features["area_px"] >= area_min)
        & (features["area_px"] <= area_max)
        & (features["compact"] >= compact_min)
    )
    return is_cell


# ---------------------------------------------------------------------------
# Stage B: Activity-type classification
# ---------------------------------------------------------------------------

def classify_activity_type(
    features: pd.DataFrame,
    is_cell: pd.Series,
    skew_phasic: float = 1.5,
    cv_tonic: float = 0.3,
    min_transients_sparse: int = 5,
) -> pd.Series:
    """Stage B: Activity-type classification.

    For each ROI passing Stage A:
      phasic    — skewness >= skew_phasic
      tonic     — source_branch includes C (bit 4) OR cv < cv_tonic
      sparse    — n_transients < min_transients_sparse
      ambiguous — none of the above

    ROIs failing Stage A are labeled 'rejected'.

    Returns
    -------
    activity_type : pd.Series of str, indexed by roi_id
    """
    result = pd.Series("rejected", index=features.index, dtype=str)

    for roi_id in features.index:
        if not is_cell.loc[roi_id]:
            continue

        row = features.loc[roi_id]
        branch_c = bool(int(row["source_branch"]) & 4)

        if row["skew"] >= skew_phasic:
            result.loc[roi_id] = "phasic"
        elif branch_c or row["cv"] < cv_tonic:
            result.loc[roi_id] = "tonic"
        elif row["n_transients"] < min_transients_sparse:
            result.loc[roi_id] = "sparse"
        else:
            result.loc[roi_id] = "ambiguous"

    return result


# ---------------------------------------------------------------------------
# Per-FOV pipeline
# ---------------------------------------------------------------------------

def classify_fov(
    stem: str,
    traces_dir: Path,
    merged_mask_dir: Path,
    merge_records_path: Path,
    out_dir: Path,
    fs: float = 30.0,
    snr_min: float = 2.0,
    area_min: int = 30,
    area_max: int = 500,
    compact_min: float = 0.15,
    skew_phasic: float = 1.5,
    cv_tonic: float = 0.3,
    min_transients_sparse: int = 5,
) -> pd.DataFrame:
    """Full QC + classification pipeline for one FOV.

    Loads traces and merge records, computes features, runs Stage A + B,
    writes {stem}_classification.csv to out_dir.

    Returns DataFrame with all features + is_cell + activity_type.
    """
    import tifffile

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_classification.csv"

    # Load traces
    f_path = traces_dir / f"{stem}_F.npy"
    fneu_path = traces_dir / f"{stem}_Fneu.npy"
    dff_path = traces_dir / f"{stem}_dFF.npy"

    if not f_path.exists():
        log.warning("  %s: traces not found", stem)
        return pd.DataFrame()

    F = np.load(str(f_path))
    Fneu = np.load(str(fneu_path))
    dFF = np.load(str(dff_path))

    # Load merged mask
    mask_path = merged_mask_dir / f"{stem}_merged_masks.tif"
    if not mask_path.exists():
        log.warning("  %s: merged mask not found", stem)
        return pd.DataFrame()
    roi_mask = tifffile.imread(str(mask_path))

    # Load merge records
    records = []
    if merge_records_path.exists():
        df_rec = pd.read_csv(str(merge_records_path))
        records = df_rec.to_dict("records")

    # Compute features
    features = compute_qc_features(F, Fneu, dFF, roi_mask, records, fs=fs)

    # Stage A
    is_cell = classify_cell_nocell(
        features, snr_min=snr_min, area_min=area_min,
        area_max=area_max, compact_min=compact_min,
    )

    # Stage B
    activity_type = classify_activity_type(
        features, is_cell, skew_phasic=skew_phasic,
        cv_tonic=cv_tonic, min_transients_sparse=min_transients_sparse,
    )

    # Combine
    features["is_cell"] = is_cell
    features["activity_type"] = activity_type
    features["fov"] = stem

    features.to_csv(str(out_path))
    log.info("  %s: %d/%d cells, types: %s",
             stem,
             int(is_cell.sum()), len(is_cell),
             activity_type[is_cell].value_counts().to_dict())

    return features.reset_index()
