"""Diagnostic helpers for the detection regression investigation.

Load a labeled mask TIFF, compute regionprops, do IoU matching between two
mask sets, and render simple overlay figures. Used by all Phase 2 experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops


def load_label_image(path: str | Path) -> np.ndarray:
    """Load a labeled TIFF; returns uint16 (H, W)."""
    arr = tifffile.imread(str(path))
    return np.asarray(arr, dtype=np.uint32)


def label_props(label_img: np.ndarray, intensity: np.ndarray | None = None) -> list[dict]:
    """Per-ROI regionprops summary."""
    props = regionprops(label_img, intensity_image=intensity)
    out = []
    for p in props:
        d = {
            "label": int(p.label),
            "area": int(p.area),
            "solidity": float(p.solidity),
            "eccentricity": float(p.eccentricity),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "bbox": [int(b) for b in p.bbox],
        }
        if intensity is not None:
            d["mean_intensity"] = float(p.mean_intensity)
        out.append(d)
    return out


def iou_match(
    labels_a: np.ndarray, labels_b: np.ndarray, min_iou: float = 0.3
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy IoU matching between two label images.

    Returns (matches, unmatched_a, unmatched_b).
    matches = list of (label_a, label_b, iou).
    """
    ids_a = np.unique(labels_a)
    ids_a = ids_a[ids_a != 0]
    ids_b = np.unique(labels_b)
    ids_b = ids_b[ids_b != 0]

    # Precompute per-label masks and areas
    masks_a = {int(i): (labels_a == i) for i in ids_a}
    masks_b = {int(i): (labels_b == i) for i in ids_b}
    area_a = {i: int(m.sum()) for i, m in masks_a.items()}
    area_b = {i: int(m.sum()) for i, m in masks_b.items()}

    pairs: list[tuple[int, int, float]] = []
    for ia, ma in masks_a.items():
        best_iou = 0.0
        best_ib = None
        for ib, mb in masks_b.items():
            inter = int((ma & mb).sum())
            if inter == 0:
                continue
            union = area_a[ia] + area_b[ib] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_ib = ib
        if best_ib is not None and best_iou >= min_iou:
            pairs.append((ia, best_ib, best_iou))

    # Greedy: prefer highest-IoU when a_b is claimed twice
    pairs.sort(key=lambda p: -p[2])
    used_a, used_b = set(), set()
    matches: list[tuple[int, int, float]] = []
    for ia, ib, iou in pairs:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matches.append((ia, ib, iou))

    unmatched_a = [int(i) for i in ids_a if int(i) not in used_a]
    unmatched_b = [int(i) for i in ids_b if int(i) not in used_b]
    return matches, unmatched_a, unmatched_b


def boundary_image(labels: np.ndarray) -> np.ndarray:
    """Return a boolean boundary image for labeled regions."""
    from skimage.segmentation import find_boundaries

    return find_boundaries(labels, mode="outer")


def render_overlay(
    image: np.ndarray,
    label_groups: dict[str, tuple[np.ndarray, tuple[int, int, int]]],
    out_path: str | Path,
    title: str = "",
    gamma: float = 0.7,
) -> None:
    """Render a matplotlib PNG with `image` as grayscale background and each
    entry in `label_groups` drawn as boundary outlines in the given RGB color.

    label_groups is {name: (label_image, (R, G, B))}.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    lo, hi = np.quantile(image, [0.01, 0.995])
    norm = np.clip((image - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    norm = norm ** gamma

    H, W = norm.shape
    rgb = np.stack([norm, norm, norm], axis=-1)  # (H, W, 3)

    for name, (labels, color) in label_groups.items():
        b = find_boundaries(labels, mode="outer")
        col = np.array(color, dtype=np.float32) / 255.0
        rgb[b] = col

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    # Legend
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=np.array(color) / 255.0, label=name)
        for name, (_, color) in label_groups.items()
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def annotate_labels(
    image: np.ndarray,
    labels: np.ndarray,
    annotations: dict[int, str],
    out_path: str | Path,
    title: str = "",
    color: tuple[int, int, int] = (255, 255, 0),
    gamma: float = 0.7,
) -> None:
    """Render `image` with label boundaries and per-ROI text annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries

    lo, hi = np.quantile(image, [0.01, 0.995])
    norm = np.clip((image - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    norm = norm ** gamma

    rgb = np.stack([norm, norm, norm], axis=-1)
    b = find_boundaries(labels, mode="outer")
    col = np.array(color, dtype=np.float32) / 255.0
    rgb[b] = col

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    props = regionprops(labels)
    for p in props:
        lid = int(p.label)
        if lid not in annotations:
            continue
        y, x = p.centroid
        ax.text(x, y, annotations[lid], color="#ffff00", fontsize=7,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
