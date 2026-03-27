"""
ROI G. Biv -- Napari viewer for consensus ROIs with per-tier label layers.

Opens seven layers for a single FOV:
  1. mean       -- grayscale mean projection
  2. cellpose   -- Cellpose probability heatmap (hot colormap, additive)
  3. suite2p    -- ALL Suite2p ROIs (hidden by default)
  4. gold       -- GOLD tier ROIs only
  5. silver     -- SILVER tier ROIs only
  6. bronze     -- BRONZE tier ROIs only (hidden by default)
  7. outlines   -- Refined 2px outlines, tier-colored (watershed on mean gradient)

Launch via Streamlit button or CLI:
  python -m roigbiv.napari_viewer --stem STEM --results-dir PATH --projections-dir PATH
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# Tier display colors — matches viz.py _TIER_COLORS (as RGBA float tuples)
_TIER_RGBA = {
    "GOLD":   (1.0, 0.843, 0.0, 0.7),    # #FFD700
    "SILVER": (0.0, 0.749, 1.0, 0.7),     # #00BFFF
    "BRONZE": (1.0, 0.388, 0.278, 0.7),   # #FF6347
}
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

# Tiers hidden by default in Napari layer panel
_TIER_VISIBLE = {"GOLD": True, "SILVER": True, "BRONZE": False}


def _resolve_mean_path(projections_dir: Path, stem: str) -> Path:
    """Return the mean-projection path for *stem*, or raise FileNotFoundError."""
    for name in (f"{stem}_mean.tif", f"{stem}_mc_mean.tif"):
        p = projections_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No mean projection found for '{stem}' in {projections_dir}.\n"
        f"Expected: {stem}_mean.tif or {stem}_mc_mean.tif"
    )


def _build_tier_masks(
    all_masks: np.ndarray, csv_path: Path, stem: str,
) -> dict[str, np.ndarray]:
    """Build per-tier label arrays from the union mask + scored CSV.

    Returns {"GOLD": array, "SILVER": array, "BRONZE": array} where each
    array retains the original label IDs for that tier's ROIs (0 elsewhere).
    """
    empty = {tier: np.zeros_like(all_masks) for tier in _TIER_RGBA}

    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found -- tier layers will be empty.")
        return empty

    df = pd.read_csv(str(csv_path))
    df["tier"] = df["tier"].str.upper()
    fov_df = df[df["fov"] == stem]

    if fov_df.empty:
        print(f"WARNING: No rows for stem '{stem}' in CSV -- tier layers will be empty.")
        return empty

    tier_masks = {}
    for tier in _TIER_RGBA:
        tier_ids = fov_df.loc[fov_df["tier"] == tier, "roi_id"].astype(int).values
        if len(tier_ids) > 0:
            tier_masks[tier] = np.where(
                np.isin(all_masks, tier_ids), all_masks, np.uint16(0),
            )
        else:
            tier_masks[tier] = np.zeros_like(all_masks)

    return tier_masks


def _make_tier_colormap(rgba_tuple, label_ids):
    """Create a direct colormap mapping all *label_ids* to a single color."""
    from napari.utils.colormaps import direct_colormap

    color_dict = {0: np.array(_TRANSPARENT, dtype=np.float32)}
    color_arr = np.array(rgba_tuple, dtype=np.float32)
    for lid in label_ids:
        color_dict[int(lid)] = color_arr
    return direct_colormap(color_dict)


def _build_refined_outlines(
    mean_img: np.ndarray,
    all_masks: np.ndarray,
    csv_path: Path,
    stem: str,
    tiers: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Refine Suite2p mask boundaries via watershed on mean image gradient.

    Uses the Sobel gradient of the (smoothed) mean projection as the landscape
    for watershed segmentation, seeded by eroded Suite2p masks.  Boundaries
    snap to real intensity edges in the image rather than the raw pixel-level
    Suite2p boundaries.

    Parameters
    ----------
    mean_img : (Ly, Lx) float32 mean projection
    all_masks : (Ly, Lx) uint16 union label image (0 = background)
    csv_path : path to scored_rois_summary.csv
    stem : FOV identifier
    tiers : tier names to include (default None = all tiers)

    Returns
    -------
    refined_mask : uint16 label image filtered to requested tiers
    color_dict : {label_id: RGBA array} for ``direct_colormap``
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import watershed

    # -- Determine which ROI IDs belong to which tier -----------------------
    roi_tier_map: dict[int, str] = {}  # roi_id → tier name
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        df["tier"] = df["tier"].str.upper()
        fov_df = df[df["fov"] == stem]
        for _, row in fov_df.iterrows():
            roi_tier_map[int(row["roi_id"])] = row["tier"]

    # Filter to requested tiers
    requested = (
        {t.upper() for t in tiers} if tiers is not None
        else set(_TIER_RGBA.keys())
    )
    valid_ids = {rid for rid, t in roi_tier_map.items() if t in requested}

    if not valid_ids:
        empty = np.zeros_like(all_masks)
        color_dict = {0: np.array(_TRANSPARENT, dtype=np.float32)}
        return empty, color_dict

    # -- Edge map from smoothed mean image ----------------------------------
    smoothed = gaussian(mean_img, sigma=1.0, preserve_range=True)
    edges = sobel(smoothed)

    # -- Build seed markers via erosion -------------------------------------
    any_roi = all_masks > 0
    eroded_binary = binary_erosion(any_roi, iterations=2)
    markers = np.where(eroded_binary, all_masks, np.uint16(0))

    # Restore centroids for ROIs lost to erosion
    original_ids = set(np.unique(all_masks)) - {0}
    surviving_ids = set(np.unique(markers)) - {0}
    for lost_id in original_ids - surviving_ids:
        ys, xs = np.where(all_masks == lost_id)
        if len(ys) > 0:
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            markers[cy, cx] = lost_id

    # Background marker: regions far from any ROI
    bg_id = int(all_masks.max()) + 1
    roi_neighborhood = binary_dilation(any_roi, iterations=5)
    markers[~roi_neighborhood] = bg_id

    # -- Watershed ----------------------------------------------------------
    refined = watershed(edges, markers=markers)

    # Remove background label
    refined[refined == bg_id] = 0
    refined = refined.astype(np.uint16)

    # -- Filter to requested tiers ------------------------------------------
    keep = np.isin(refined, np.array(list(valid_ids), dtype=np.uint16))
    refined[~keep] = 0

    # -- Build tier-colored colormap dict -----------------------------------
    color_dict = {0: np.array(_TRANSPARENT, dtype=np.float32)}
    for rid in valid_ids:
        tier = roi_tier_map[rid]
        color_dict[rid] = np.array(_TIER_RGBA[tier], dtype=np.float32)

    return refined, color_dict


def open_fov(stem: str, results_dir: str, projections_dir: str,
             outline_tiers: list[str] | None = None) -> None:
    """Open a Napari viewer for a single FOV with seven layers.

    Parameters
    ----------
    outline_tiers : list of tier names for the outline layer (default: all).

    Blocks until the viewer window is closed.
    """
    import napari

    results_dir = Path(results_dir)
    projections_dir = Path(projections_dir)

    # -- Resolve file paths -------------------------------------------------
    mean_path = _resolve_mean_path(projections_dir, stem)
    masks_path = results_dir / f"{stem}_all_s2p_masks.tif"
    cellprob_path = results_dir / f"{stem}_roi_cellprob.tif"
    csv_path = results_dir / "scored_rois_summary.csv"

    for p in (masks_path, cellprob_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run the pipeline first to generate scored outputs."
            )

    # -- Load data ----------------------------------------------------------
    mean_img = tifffile.imread(str(mean_path)).astype(np.float32)
    cellprob_map = tifffile.imread(str(cellprob_path)).astype(np.float32)
    all_masks = tifffile.imread(str(masks_path))
    tier_masks = _build_tier_masks(all_masks, csv_path, stem)

    # -- Build viewer -------------------------------------------------------
    viewer = napari.Viewer(title=f"ROI G. Biv -- {stem}")

    viewer.add_image(mean_img, name="mean", colormap="gray")
    viewer.add_image(
        cellprob_map, name="cellpose",
        colormap="hot", opacity=0.4, blending="additive",
    )
    viewer.add_labels(all_masks, name="suite2p", visible=False, opacity=0.5)

    for tier, rgba in _TIER_RGBA.items():
        mask = tier_masks[tier]
        label_ids = np.unique(mask[mask > 0])
        cmap = _make_tier_colormap(rgba, label_ids)
        viewer.add_labels(
            mask,
            name=tier.lower(),
            colormap=cmap,
            opacity=0.6,
            visible=_TIER_VISIBLE[tier],
        )

    # -- Refined outline layer (watershed on mean gradient) -----------------
    refined_mask, outline_colors = _build_refined_outlines(
        mean_img, all_masks, csv_path, stem, tiers=outline_tiers,
    )
    if refined_mask.any():
        from napari.utils.colormaps import direct_colormap

        outline_cmap = direct_colormap(outline_colors)
        outline_layer = viewer.add_labels(
            refined_mask, name="outlines",
            colormap=outline_cmap,
            opacity=0.9,
        )
        outline_layer.contour = 2

    napari.run()


def main():
    ap = argparse.ArgumentParser(
        description="ROI G. Biv -- Napari viewer with per-tier ROI layers",
    )
    ap.add_argument("--stem", required=True, help="FOV stem name")
    ap.add_argument("--results-dir", required=True, help="Directory with scored TIFs + CSV")
    ap.add_argument("--projections-dir", required=True, help="Directory with mean projections")
    ap.add_argument("--outline-tiers", nargs="*", default=None,
                    help="Tiers to include in outline layer (default: all)")
    args = ap.parse_args()

    open_fov(args.stem, args.results_dir, args.projections_dir,
             outline_tiers=args.outline_tiers)


if __name__ == "__main__":
    main()
