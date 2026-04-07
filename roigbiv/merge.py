"""
ROI G. Biv --- Step 8: Three-branch ROI merge.

Merges masks from Branch A (Cellpose), Branch B (Suite2p), and Branch C
(tonic detection) via pairwise IoU matching with union-find for transitive
grouping.

Confidence tiers (by source branches present):
  ABC — all three agree (highest confidence)
  AB  — Cellpose + Suite2p (high confidence; Cellpose boundaries)
  AC  — Cellpose + tonic
  BC  — Suite2p + tonic
  A   — Cellpose only (moderate; low-variance neuron)
  B   — Suite2p only (morphologically ambiguous, temporally active)
  C   — Tonic only (candidate tonic neuron; flag for manual review)

Pixel priority when building the merged mask:
  Cellpose (A) > Suite2p (B) > Tonic (C)
"""

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("merge")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stat_to_mask(
    stat: np.ndarray,
    Ly: int,
    Lx: int,
    iscell: np.ndarray | None = None,
    min_prob: float = 0.0,
) -> np.ndarray:
    """Convert Suite2p stat.npy to uint16 label image.

    Parameters
    ----------
    stat     : array of dicts from Suite2p stat.npy
    Ly, Lx   : spatial dimensions
    iscell   : (n_rois, 2) array; if provided, filter by iscell[:,1] >= min_prob
    min_prob : minimum iscell probability to include (default 0.0 = keep all)

    Returns
    -------
    mask : (Ly, Lx) uint16 — 0 = background, 1-indexed labels
    """
    mask = np.zeros((Ly, Lx), dtype=np.uint16)
    label_id = 0
    for i, s in enumerate(stat):
        if iscell is not None and i < len(iscell):
            if iscell[i, 1] < min_prob:
                continue
        label_id += 1
        ypix = s["ypix"]
        xpix = s["xpix"]
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = label_id
    return mask


def _centroid(mask: np.ndarray, label: int) -> tuple[int, int]:
    """Centroid (y, x) of a labeled region."""
    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        return -1, -1
    return int(np.mean(ys)), int(np.mean(xs))


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def compute_iou_matrix(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise IoU between all labeled ROIs in two uint16 masks.

    Returns
    -------
    iou      : (n_a, n_b) float32
    labels_a : unique non-zero labels in mask_a
    labels_b : unique non-zero labels in mask_b
    """
    labels_a = np.unique(mask_a[mask_a > 0])
    labels_b = np.unique(mask_b[mask_b > 0])

    if len(labels_a) == 0 or len(labels_b) == 0:
        return (np.zeros((len(labels_a), len(labels_b)), dtype=np.float32),
                labels_a, labels_b)

    iou = np.zeros((len(labels_a), len(labels_b)), dtype=np.float32)
    for i, la in enumerate(labels_a):
        pixels_a = mask_a == la
        for j, lb in enumerate(labels_b):
            pixels_b = mask_b == lb
            intersection = np.logical_and(pixels_a, pixels_b).sum()
            if intersection == 0:
                continue
            union = np.logical_or(pixels_a, pixels_b).sum()
            iou[i, j] = intersection / union

    return iou, labels_a, labels_b


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Lightweight union-find for transitive ROI grouping."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Core 3-way merge
# ---------------------------------------------------------------------------

def merge_three_branches(
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    mask_c: np.ndarray | None,
    iou_threshold: float = 0.3,
    s2p_iscell: np.ndarray | None = None,
    mean_image: np.ndarray | None = None,
    require_spatial_support: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Three-way ROI merge with branch-priority pixel assignment.

    Any branch mask may be None (missing). At least one must be provided.

    Algorithm:
      1. Compute pairwise IoU matrices for all present branch pairs.
      2. Hungarian matching on each pair; edges for IoU >= threshold.
      3. Union-find groups transitively matched ROIs into clusters.
      4. Tier assignment from which branches contribute to each cluster.
      5. Pixel assignment from highest-priority branch present.

    Parameters
    ----------
    mask_a                  : uint16 labeled mask from Branch A (Cellpose), or None
    mask_b                  : uint16 labeled mask from Branch B (Suite2p), or None
    mask_c                  : uint16 labeled mask from Branch C (tonic), or None
    iou_threshold           : minimum IoU for a match (default 0.3)
    s2p_iscell              : Suite2p iscell (n_rois, 2) for metadata
    mean_image              : (Ly, Lx) float32 mean projection; used by spatial gate
    require_spatial_support : if True, discard C-only ROIs whose centroid falls below
                              the FOV 25th-percentile intensity in mean_image

    Returns
    -------
    merged_mask : (Ly, Lx) uint16
    records     : list of dicts per merged ROI
    """
    # Determine spatial dimensions
    shape = None
    for m in (mask_a, mask_b, mask_c):
        if m is not None:
            shape = m.shape
            break
    if shape is None:
        raise ValueError("At least one branch mask must be provided")

    # Replace None with empty masks
    empty = np.zeros(shape, dtype=np.uint16)
    if mask_a is None:
        mask_a = empty
    if mask_b is None:
        mask_b = empty
    if mask_c is None:
        mask_c = empty

    # Extract labels per branch (keyed as "A_label", "B_label", "C_label")
    labels_a = np.unique(mask_a[mask_a > 0])
    labels_b = np.unique(mask_b[mask_b > 0])
    labels_c = np.unique(mask_c[mask_c > 0])

    # Build union-find over nodes: ("A", label), ("B", label), ("C", label)
    uf = _UnionFind()
    iou_edges = {}  # store best IoU for each pair type

    def _match_pair(m1, m2, tag1, tag2):
        """Run Hungarian matching on a pair and merge matched nodes in UF."""
        iou_mat, labs1, labs2 = compute_iou_matrix(m1, m2)
        if iou_mat.size == 0:
            return {}
        row_idx, col_idx = linear_sum_assignment(-iou_mat)
        pair_ious = {}
        for r, c in zip(row_idx, col_idx):
            score = float(iou_mat[r, c])
            if score >= iou_threshold:
                n1 = (tag1, int(labs1[r]))
                n2 = (tag2, int(labs2[c]))
                uf.union(n1, n2)
                pair_ious[(n1, n2)] = score
        return pair_ious

    iou_ab = _match_pair(mask_a, mask_b, "A", "B")
    iou_ac = _match_pair(mask_a, mask_c, "A", "C")
    iou_bc = _match_pair(mask_b, mask_c, "B", "C")

    # Collect all edges for IoU reporting
    all_edges = {}
    all_edges.update(iou_ab)
    all_edges.update(iou_ac)
    all_edges.update(iou_bc)

    # Group nodes by cluster root
    clusters = {}
    all_nodes = (
        [("A", int(l)) for l in labels_a]
        + [("B", int(l)) for l in labels_b]
        + [("C", int(l)) for l in labels_c]
    )
    for node in all_nodes:
        root = uf.find(node)
        clusters.setdefault(root, []).append(node)

    # Build merged mask and records
    merged_mask = np.zeros(shape, dtype=np.uint16)
    records = []
    roi_id = 0
    branch_masks = {"A": mask_a, "B": mask_b, "C": mask_c}

    for _root, members in sorted(clusters.items(), key=lambda x: str(x[0])):
        roi_id += 1
        branches_present = set()
        branch_labels = {"A": -1, "B": -1, "C": -1}

        for branch, label in members:
            branches_present.add(branch)
            branch_labels[branch] = label

        # Tier from branch set
        tier_key = "".join(sorted(branches_present))
        tier = tier_key  # "A", "B", "C", "AB", "AC", "BC", "ABC"

        # Assign pixels from highest-priority branch
        for priority_branch in ("A", "B", "C"):
            if priority_branch in branches_present:
                lbl = branch_labels[priority_branch]
                pixels = branch_masks[priority_branch] == lbl
                merged_mask[pixels] = roi_id
                break

        # Centroid and area from merged mask
        cy, cx = _centroid(merged_mask, roi_id)
        area = int((merged_mask == roi_id).sum())

        # Best IoU per pair involving this cluster
        best_iou_ab = 0.0
        best_iou_ac = 0.0
        best_iou_bc = 0.0
        for (n1, n2), score in all_edges.items():
            if uf.find(n1) == uf.find(members[0]):
                if {n1[0], n2[0]} == {"A", "B"}:
                    best_iou_ab = max(best_iou_ab, score)
                elif {n1[0], n2[0]} == {"A", "C"}:
                    best_iou_ac = max(best_iou_ac, score)
                elif {n1[0], n2[0]} == {"B", "C"}:
                    best_iou_bc = max(best_iou_bc, score)

        # Suite2p iscell probability
        s2p_prob = -1.0
        if s2p_iscell is not None and branch_labels["B"] > 0:
            idx = branch_labels["B"] - 1
            if idx < len(s2p_iscell):
                s2p_prob = float(s2p_iscell[idx, 1])

        records.append({
            "roi_id": roi_id,
            "tier": tier,
            "source_branches": tier_key,
            "iou_ab": round(best_iou_ab, 4),
            "iou_ac": round(best_iou_ac, 4),
            "iou_bc": round(best_iou_bc, 4),
            "a_label": branch_labels["A"],
            "b_label": branch_labels["B"],
            "c_label": branch_labels["C"],
            "centroid_y": cy,
            "centroid_x": cx,
            "area_px": area,
            "s2p_iscell_prob": round(s2p_prob, 4),
            "review_flag": tier_key == "C",
        })

    # Spatial support gate: discard C-only ROIs in dark regions of the FOV
    if require_spatial_support and mean_image is not None:
        p25 = float(np.percentile(mean_image, 25))
        surviving = []
        for r in records:
            if r["tier"] == "C":
                cy, cx = r["centroid_y"], r["centroid_x"]
                Lmy, Lmx = mean_image.shape
                if (0 <= cy < Lmy and 0 <= cx < Lmx
                        and float(mean_image[cy, cx]) <= p25):
                    merged_mask[merged_mask == r["roi_id"]] = 0
                    log.debug("Rejected C-only ROI %d: centroid below FOV p25 (%.1f)",
                              r["roi_id"], p25)
                    continue
            surviving.append(r)
        records = surviving

    return merged_mask, records


# ---------------------------------------------------------------------------
# Per-FOV orchestration
# ---------------------------------------------------------------------------

def merge_fov(
    stem: str,
    branch_a_dir: Path,
    s2p_dir: Path,
    branch_c_dir: Path,
    out_dir: Path,
    iou_threshold: float = 0.3,
    s2p_min_prob: float = 0.0,
    mean_image_dir: Path | None = None,
    require_spatial_support: bool = True,
) -> list[dict]:
    """Run 3-way merge for a single FOV.

    Loads masks from standard paths, runs merge_three_branches, writes
    merged mask TIF and per-ROI records CSV.

    Resumability: skips if {stem}_merged_masks.tif already exists.

    Returns list of record dicts (empty if skipped or no inputs found).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{stem}_merged_masks.tif"
    if mask_path.exists():
        log.info("  %s: skipped (merged mask exists)", stem)
        return []

    mean_image = None

    # Load Branch A (Cellpose)
    mask_a = None
    a_path = branch_a_dir / f"{stem}_masks.tif"
    if a_path.exists():
        mask_a = tifffile.imread(str(a_path))
        if mask_a.max() == 0:
            mask_a = None

    # Load Branch B (Suite2p)
    mask_b = None
    s2p_iscell = None
    plane_dir = s2p_dir / stem / "suite2p" / "plane0"
    stat_path = plane_dir / "stat.npy"
    iscell_path = plane_dir / "iscell.npy"
    ops_path = plane_dir / "ops.npy"
    if stat_path.exists() and ops_path.exists():
        stat = np.load(str(stat_path), allow_pickle=True)
        ops = np.load(str(ops_path), allow_pickle=True).item()
        Ly, Lx = ops["Ly"], ops["Lx"]
        iscell = None
        if iscell_path.exists():
            iscell = np.load(str(iscell_path))
            s2p_iscell = iscell
        mask_b = stat_to_mask(stat, Ly, Lx, iscell=iscell, min_prob=s2p_min_prob)
        if mask_b.max() == 0:
            mask_b = None
        if require_spatial_support and "meanImg" in ops:
            mean_image = np.array(ops["meanImg"], dtype=np.float32)

    # Override mean image from dedicated directory if provided
    if mean_image_dir is not None:
        mi_path = mean_image_dir / f"{stem}_mean.tif"
        if mi_path.exists():
            mean_image = tifffile.imread(str(mi_path)).astype(np.float32)

    # Load Branch C (tonic)
    mask_c = None
    c_path = branch_c_dir / f"{stem}_tonic_masks.tif"
    if c_path.exists():
        mask_c = tifffile.imread(str(c_path))
        if mask_c.max() == 0:
            mask_c = None

    # Check we have at least one branch
    if mask_a is None and mask_b is None and mask_c is None:
        log.warning("  %s: no branch masks found, skipping", stem)
        return []

    n_a = int(mask_a.max()) if mask_a is not None else 0
    n_b = int(mask_b.max()) if mask_b is not None else 0
    n_c = int(mask_c.max()) if mask_c is not None else 0
    log.info("  %s: A=%d, B=%d, C=%d ROIs", stem, n_a, n_b, n_c)

    merged_mask, records = merge_three_branches(
        mask_a, mask_b, mask_c,
        iou_threshold=iou_threshold,
        s2p_iscell=s2p_iscell,
        mean_image=mean_image,
        require_spatial_support=require_spatial_support,
    )

    # Write outputs
    tifffile.imwrite(str(mask_path), merged_mask)

    records_path = out_dir / f"{stem}_merge_records.csv"
    if records:
        with open(records_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys())
            w.writeheader()
            w.writerows(records)

    # Log tier summary
    tier_counts = {}
    for r in records:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    tier_str = "  ".join(f"{t}={c}" for t, c in sorted(tier_counts.items()))
    log.info("    -> %d merged ROIs: %s", len(records), tier_str)

    # Add FOV stem to each record for batch summary
    for r in records:
        r["fov"] = stem

    return records


def merge_batch(
    stems: list[str],
    branch_a_dir: Path,
    s2p_dir: Path,
    branch_c_dir: Path,
    out_dir: Path,
    iou_threshold: float = 0.3,
    s2p_min_prob: float = 0.0,
    mean_image_dir: Path | None = None,
    require_spatial_support: bool = True,
) -> pd.DataFrame:
    """Run 3-way merge for all FOV stems. Returns summary DataFrame."""
    all_records = []
    n_done = n_skip = n_err = 0

    for stem in stems:
        try:
            recs = merge_fov(
                stem, branch_a_dir, s2p_dir, branch_c_dir, out_dir,
                iou_threshold=iou_threshold,
                s2p_min_prob=s2p_min_prob,
                mean_image_dir=mean_image_dir,
                require_spatial_support=require_spatial_support,
            )
            if recs:
                all_records.extend(recs)
                n_done += 1
            else:
                n_skip += 1
        except Exception as exc:
            log.error("  %s: ERROR — %s", stem, exc)
            n_err += 1

    # Write combined summary
    if all_records:
        df = pd.DataFrame(all_records)
        summary_path = out_dir / "merge_summary.csv"
        df.to_csv(str(summary_path), index=False)
        log.info("Summary: %d ROIs across %d FOVs -> %s",
                 len(df), df["fov"].nunique(), summary_path)
        return df

    log.info("Complete: %d processed, %d skipped, %d errors", n_done, n_skip, n_err)
    return pd.DataFrame()
