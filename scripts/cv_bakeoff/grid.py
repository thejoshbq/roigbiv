"""Side-by-side overlay grid for the CV bake-off.

Renders one matplotlib panel per detector — the chosen background image
(percentile-stretched) with that method's ROI contours overlaid — plus a bare
background panel for reference. Borrows overlay.py's proven percentile-stretch
and ``find_contours`` approach (the data models genuinely differ: overlay.py
colours by gate outcome over a live ``FOVData``; here we compare methods with no
ground truth, so the shapes are deliberately parallel rather than shared).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.measure import find_contours, regionprops  # noqa: E402

from cv_bakeoff.detector import DetectorResult  # noqa: E402


def _stretch_to_uint8(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5):
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _draw_contours(ax, label_mask: np.ndarray, color: str = "cyan") -> None:
    for region in regionprops(label_mask):
        sub = (label_mask == region.label)
        for contour in find_contours(sub.astype(np.float32), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.6)


def render_grid(
    background: np.ndarray,
    results: list[DetectorResult],
    *,
    fov_stem: str,
    background_name: str,
    out_path: Path,
    enhance_label: str | None = None,
) -> Path:
    """Write a comparison grid PNG. Returns the path written."""
    bg = _stretch_to_uint8(background)
    n_panels = len(results) + 1
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 5.0 * nrows), squeeze=False,
    )
    flat = axes.ravel()

    flat[0].imshow(bg, cmap="gray")
    flat[0].set_title(f"{background_name} (background)", fontsize=10)
    flat[0].axis("off")

    for ax, res in zip(flat[1:], results):
        ax.imshow(bg, cmap="gray")
        _draw_contours(ax, res.label_mask)
        rt = res.meta.get("runtime_s", "?")
        ax.set_title(
            f"{res.meta.get('method', '?')} · {res.n_rois} ROIs · {rt}s",
            fontsize=10,
        )
        ax.axis("off")

    for ax in flat[n_panels:]:
        ax.axis("off")

    suptitle = f"{fov_stem}  —  CV bake-off"
    if enhance_label:
        suptitle += f"  (enhance: {enhance_label})"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
