"""
ROI G. Biv — Union ROI building (Stage 6).

Merges the two Suite2p detection passes (activity + anatomy) via Hungarian
IoU matching, assigns GOLD/SILVER/BRONZE confidence tiers, and scores every
union ROI with the Cellpose cell-probability heatmap.

Public API
----------
build_union_batch() — process all FOVs; save TIFFs + CSV; return DataFrame
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stat_to_mask(stat, Ly: int, Lx: int) -> np.ndarray:
    """Convert Suite2p stat.npy (all ROIs, no iscell filter) to uint16 label image."""
    mask = np.zeros((Ly, Lx), dtype=np.uint16)
    for i, s in enumerate(stat):
        ypix = s["ypix"]
        xpix = s["xpix"]
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = i + 1  # 1-indexed
    return mask


def _get_cellprob(flows, Ly: int, Lx: int) -> np.ndarray:
    """
    Extract per-pixel cell probability from Cellpose flows output.

    Tries flows[2] then flows[1]. Applies sigmoid if values are outside [0, 1]
    (raw logits from some Cellpose versions).
    """
    for idx in (2, 1):
        if idx >= len(flows):
            continue
        arr = np.asarray(flows[idx])
        if arr.ndim == 2 and arr.shape == (Ly, Lx):
            prob = arr.astype(np.float32)
            if prob.min() < 0 or prob.max() > 1:
                prob = 1.0 / (1.0 + np.exp(-prob))
            return prob
    raise RuntimeError(
        f"Cannot find a ({Ly}, {Lx}) 2D probability array in Cellpose flows. "
        f"Array shapes: {[np.asarray(f).shape for f in flows]}"
    )


# ---------------------------------------------------------------------------
# Core per-FOV function
# ---------------------------------------------------------------------------

def build_union(activity_dir, anatomy_dir, projections_dir, out_dir,
                model_path, diameter: float, use_vcorr: bool,
                iou_threshold: float) -> list:
    """
    Build union ROI masks for all FOVs present in both Suite2p output directories.

    For each FOV:
    1. Converts stat.npy → uint16 label images for both passes.
    2. Runs Hungarian IoU matching; assigns GOLD / SILVER / BRONZE tiers.
    3. Builds a union mask (anatomy boundaries for GOLD/SILVER; activity pixels
       for BRONZE — anatomy contours from Suite2p's mean-image pass are
       generally smoother and more anatomically precise).
    4. Runs the Cellpose model in probability-extraction mode
       (cellprob_threshold=-6 keeps all candidates; consensus handles filtering).
    5. Scores each union ROI by its mean Cellpose probability.

    Parameters
    ----------
    activity_dir    : path-like — Suite2p activity-pass output root
    anatomy_dir     : path-like — Suite2p anatomy-pass output root
    projections_dir : path-like — directory with {stem}_mean.tif + {stem}_vcorr.tif
    out_dir         : path-like — where to write per-FOV TIFFs
    model_path      : path-like — Cellpose checkpoint (CellposeModel pretrained_model=)
    diameter        : float     — expected cell diameter in pixels
    use_vcorr       : bool      — stack Vcorr as 2nd channel alongside mean image
    iou_threshold   : float     — minimum IoU for GOLD tier (default 0.3)

    Returns
    -------
    list of dicts — one dict per ROI (all FOVs combined);
                    suitable for writing to scored_rois_summary.csv.
    """
    from cellpose import models
    import torch
    from roigbiv.match import match_and_tier, build_consensus_mask

    activity_dir = Path(activity_dir)
    anatomy_dir  = Path(anatomy_dir)
    projections_dir = Path(projections_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [1, 2] if use_vcorr else [0, 0]
    _use_gpu = torch.cuda.is_available()
    if not _use_gpu:
        print("WARNING: No GPU detected. Cellpose will run on CPU and may be very slow.")
    model = models.CellposeModel(gpu=_use_gpu, pretrained_model=str(model_path))

    activity_fovs = {d.name for d in activity_dir.iterdir() if d.is_dir()}
    anatomy_fovs  = {d.name for d in anatomy_dir.iterdir()  if d.is_dir()}
    common_fovs   = sorted(activity_fovs & anatomy_fovs)

    if not common_fovs:
        print(f"WARNING: No FOVs found in both:\n  {activity_dir}\n  {anatomy_dir}")
        return []

    all_rows = []

    for stem in common_fovs:
        # ── Resumability ──────────────────────────────────────────────────
        out_mask_path = out_dir / f"{stem}_all_s2p_masks.tif"
        if out_mask_path.exists():
            print(f"  {stem}: skipped (already exists)")
            continue

        act_plane = activity_dir / stem / "suite2p" / "plane0"
        ana_plane = anatomy_dir  / stem / "suite2p" / "plane0"
        required  = ("stat.npy", "iscell.npy", "ops.npy")

        if not all((act_plane / f).exists() for f in required):
            print(f"  {stem}: missing activity Suite2p outputs, skipping")
            continue
        if not all((ana_plane / f).exists() for f in required):
            print(f"  {stem}: missing anatomy Suite2p outputs, skipping")
            continue

        # Locate mean image
        mean_path = projections_dir / f"{stem}_mean.tif"
        if not mean_path.exists():
            mean_path = projections_dir / f"{stem}_mc_mean.tif"
        if not mean_path.exists():
            print(f"  {stem}: no mean.tif in {projections_dir}, skipping")
            continue

        t0 = time.time()
        print(f"  {stem} ... ", end="", flush=True)

        # ── Load Suite2p outputs ─────────────────────────────────────────
        act_stat   = np.load(str(act_plane / "stat.npy"),   allow_pickle=True)
        act_iscell = np.load(str(act_plane / "iscell.npy"))
        act_ops    = np.load(str(act_plane / "ops.npy"),    allow_pickle=True).item()
        ana_stat   = np.load(str(ana_plane / "stat.npy"),   allow_pickle=True)
        ana_iscell = np.load(str(ana_plane / "iscell.npy"))
        ana_ops    = np.load(str(ana_plane / "ops.npy"),    allow_pickle=True).item()

        Ly, Lx = act_ops["Ly"], act_ops["Lx"]
        if (ana_ops["Ly"], ana_ops["Lx"]) != (Ly, Lx):
            print(f"dimension mismatch, skipping")
            continue

        # ── Convert stat → uint16 masks ──────────────────────────────────
        act_mask = _stat_to_mask(act_stat, Ly, Lx)
        ana_mask = _stat_to_mask(ana_stat, Ly, Lx)

        # anatomy plays the 'cp' role; activity plays the 's2p' role
        records = match_and_tier(ana_mask, act_mask,
                                 iou_threshold=iou_threshold,
                                 s2p_iscell=act_iscell)

        union_mask = build_consensus_mask(ana_mask, act_mask, records,
                                          tiers=("GOLD", "SILVER", "BRONZE"))

        # ── Cellpose probability scoring ─────────────────────────────────
        img = tifffile.imread(str(mean_path)).astype(np.float32)
        if use_vcorr:
            vcorr_path = projections_dir / f"{stem}_vcorr.tif"
            if not vcorr_path.exists():
                vcorr_path = projections_dir / f"{stem}_mc_vcorr.tif"
            if vcorr_path.exists():
                vcorr = tifffile.imread(str(vcorr_path)).astype(np.float32)
                img = np.stack([img, vcorr], axis=-1)
            else:
                print(f"\n  WARNING: no Vcorr for {stem}, using zeros as 2nd channel")
                img = np.stack([img, np.zeros_like(img)], axis=-1)

        _, flows, _ = model.eval(
            img, diameter=diameter, channels=channels,
            cellprob_threshold=-6, normalize=True,
        )
        cellprob_map = _get_cellprob(flows, Ly, Lx)

        # ── Per-ROI probability scoring ──────────────────────────────────
        ana_prob_by_label = {i + 1: float(ana_iscell[i, 1]) for i in range(len(ana_iscell))}
        act_prob_by_label = {i + 1: float(act_iscell[i, 1]) for i in range(len(act_iscell))}
        act_cell_by_label = {i + 1: int(act_iscell[i, 0])   for i in range(len(act_iscell))}

        prob_img = np.zeros((Ly, Lx), dtype=np.float32)

        for new_id, rec in enumerate(records, start=1):
            ypix, xpix = np.where(union_mask == new_id)
            mean_prob = float(cellprob_map[ypix, xpix].mean()) if len(ypix) > 0 else 0.0
            prob_img[ypix, xpix] = mean_prob

            ana_lbl = rec["cellpose_label"]
            act_lbl = rec["s2p_label"]
            all_rows.append({
                "fov":                stem,
                "roi_id":             new_id,
                "tier":               rec["tier"],
                "iou_score":          rec["iou_score"],
                "activity_iscell":    act_cell_by_label.get(act_lbl, -1),
                "activity_s2p_prob":  act_prob_by_label.get(act_lbl, -1.0),
                "anatomy_s2p_prob":   ana_prob_by_label.get(ana_lbl, -1.0),
                "cellpose_mean_prob": round(mean_prob, 5),
            })

        tifffile.imwrite(str(out_dir / f"{stem}_all_s2p_masks.tif"), union_mask)
        tifffile.imwrite(str(out_dir / f"{stem}_roi_cellprob.tif"),  prob_img)

        elapsed = time.time() - t0
        n_g = sum(1 for r in records if r["tier"] == "GOLD")
        n_s = sum(1 for r in records if r["tier"] == "SILVER")
        n_b = sum(1 for r in records if r["tier"] == "BRONZE")
        print(f"GOLD={n_g} SILVER={n_s} BRONZE={n_b}  ({elapsed:.0f}s)")

    return all_rows


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def build_union_batch(tif_files, activity_dir, anatomy_dir, projections_dir,
                      model_path, output_dir, diameter: float = 30,
                      iou_threshold: float = 0.3,
                      tiers=("gold", "silver"),
                      use_vcorr: bool = True) -> pd.DataFrame:
    """
    Run union ROI building for all FOVs and write results to *output_dir*.

    Outputs
    -------
    ``{output_dir}/{stem}_all_s2p_masks.tif``
        uint16 label image — each unique integer identifies one consensus ROI.
    ``{output_dir}/{stem}_roi_cellprob.tif``
        float32 per-pixel Cellpose probability heatmap (for downstream thresholding).
    ``{output_dir}/scored_rois_summary.csv``
        Per-ROI table: fov, roi_id, tier, iou_score, activity_iscell,
        activity_s2p_prob, anatomy_s2p_prob, cellpose_mean_prob.

    Parameters
    ----------
    tif_files       : list of path-like — original TIF paths (used only for stem names)
    activity_dir    : path-like — Suite2p activity-pass output root
    anatomy_dir     : path-like — Suite2p anatomy-pass output root
    projections_dir : path-like — directory with {stem}_mean.tif + {stem}_vcorr.tif
    model_path      : path-like — Cellpose checkpoint path
    output_dir      : path-like — where to write results
    diameter        : float     — Cellpose expected cell diameter in pixels
    iou_threshold   : float     — minimum IoU for GOLD matching
    tiers           : sequence  — tiers to include in default output mask
    use_vcorr       : bool      — stack Vcorr as 2nd input channel

    Returns
    -------
    pd.DataFrame — scored_rois_summary (empty DataFrame if nothing was processed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUnion ROI building  [diameter={diameter}px  iou_threshold={iou_threshold}]")
    print(f"  Model:         {model_path}")
    print(f"  Output tiers:  {list(tiers)}")
    print(f"  Vcorr channel: {use_vcorr}")

    all_rows = build_union(
        activity_dir, anatomy_dir, projections_dir, output_dir,
        model_path, diameter, use_vcorr, iou_threshold,
    )

    csv_path = output_dir / "scored_rois_summary.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(str(csv_path), index=False)
        n_fovs = df["fov"].nunique()
        print(f"\nSummary: {len(df)} ROIs across {n_fovs} FOVs  →  {csv_path}")
        return df

    print("No new ROIs processed (all FOVs may have been skipped or no common FOVs found).")
    return pd.DataFrame()
