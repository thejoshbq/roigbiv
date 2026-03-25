"""
ROI G. Biv — Interactive Colab viewer.

create_colab_viewer() launches an ipywidgets UI for exploring processed FOVs.
Requires: ipywidgets, matplotlib, scipy, pandas, tifffile.
No napari dependency — uses matplotlib scatter for contour overlays.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# Tier → display color (colorblind-aware)
_TIER_COLORS = {
    "GOLD":   "#FFD700",   # gold
    "SILVER": "#00BFFF",   # deep sky blue
    "BRONZE": "#FF6347",   # tomato
}


def _make_contour(mask: np.ndarray, label: int):
    """
    Return (ys, xs) arrays of contour pixels for ROI *label* in *mask*.
    Uses 1-iteration binary dilation: dilated_roi XOR roi = boundary ring.
    """
    from scipy.ndimage import binary_dilation
    roi = mask == label
    boundary = binary_dilation(roi, iterations=1) & ~roi
    return np.where(boundary)


def create_colab_viewer(output_dir) -> None:
    """
    Launch an interactive ipywidgets viewer for processed FOVs.

    Controls
    --------
    FOV dropdown
        Selects which field of view to display.  Populated automatically from
        ``*_all_s2p_masks.tif`` files in *output_dir*.
    Tier selector (multi-select)
        Toggle GOLD / SILVER / BRONZE tier visibility.
    Probability slider
        Minimum ``cellpose_mean_prob`` threshold — ROIs below this value are
        hidden.  Range: -6.0 (show all) to 6.0 (show only highest confidence).

    Display
    -------
    - Background: mean projection (1–99 percentile normalized, grayscale).
    - Overlay: ROI boundary contours colored by tier.
      GOLD = gold  ·  SILVER = cyan  ·  BRONZE = tomato
    - Status line: total ROIs shown, broken down by tier.

    Mean images are looked for in the following order:
      1. ``{output_dir}/../projections/{stem}_mean.tif``
      2. ``{output_dir}/../projections/{stem}_mc_mean.tif``
      3. ``{output_dir}/{stem}_mean.tif``

    Parameters
    ----------
    output_dir : path-like — directory written by build_union_batch()
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display as ipy_display
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    mask_files = sorted(output_dir.glob("*_all_s2p_masks.tif"))

    if not mask_files:
        print(f"No processed FOVs found in: {output_dir}")
        return

    stems = [f.name.replace("_all_s2p_masks.tif", "") for f in mask_files]

    # Load summary CSV (optional — needed for per-ROI tier + prob info)
    csv_path = output_dir / "scored_rois_summary.csv"
    summary_df = pd.read_csv(str(csv_path)) if csv_path.exists() else None
    if summary_df is not None:
        summary_df["tier"] = summary_df["tier"].str.upper()

    # ── Widgets ────────────────────────────────────────────────────────────
    fov_dropdown = widgets.Dropdown(
        options=stems,
        value=stems[0],
        description="FOV:",
        layout=widgets.Layout(width="65%"),
    )
    tier_select = widgets.SelectMultiple(
        options=["GOLD", "SILVER", "BRONZE"],
        value=["GOLD", "SILVER"],
        description="Tiers:",
        rows=3,
        layout=widgets.Layout(width="180px"),
    )
    prob_slider = widgets.FloatSlider(
        value=0.0,
        min=-6.0,
        max=6.0,
        step=0.1,
        description="Min prob:",
        continuous_update=False,
        layout=widgets.Layout(width="65%"),
    )
    out_widget = widgets.Output()

    # ── Render function ────────────────────────────────────────────────────
    def _render(change=None):
        stem            = fov_dropdown.value
        selected_tiers  = set(tier_select.value)
        min_prob        = prob_slider.value

        mask_path = output_dir / f"{stem}_all_s2p_masks.tif"
        mean_path = _find_mean(output_dir, stem)

        with out_widget:
            out_widget.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

            # ── Background ────────────────────────────────────────────────
            mask = tifffile.imread(str(mask_path))
            if mean_path:
                bg = tifffile.imread(str(mean_path)).astype(np.float32)
                p1, p99 = np.percentile(bg, [1, 99])
                bg = np.clip((bg - p1) / max(p99 - p1, 1e-6), 0, 1)
                ax.imshow(bg, cmap="gray", interpolation="none")
            else:
                ax.imshow(np.zeros(mask.shape, dtype=np.float32),
                          cmap="gray", interpolation="none")

            # ── ROI overlays ──────────────────────────────────────────────
            n_shown = {t: 0 for t in ("GOLD", "SILVER", "BRONZE")}
            n_total = {t: 0 for t in ("GOLD", "SILVER", "BRONZE")}

            if summary_df is not None:
                fov_df = summary_df[summary_df["fov"] == stem]
                for _, row in fov_df.iterrows():
                    tier = str(row["tier"]).upper()
                    if tier not in _TIER_COLORS:
                        continue
                    n_total[tier] += 1
                    if tier not in selected_tiers:
                        continue
                    if row.get("cellpose_mean_prob", 0.0) < min_prob:
                        continue
                    ys, xs = _make_contour(mask, int(row["roi_id"]))
                    if len(ys) > 0:
                        ax.scatter(xs, ys, c=_TIER_COLORS[tier],
                                   s=0.4, alpha=0.85, marker=".", linewidths=0)
                        n_shown[tier] += 1

            # ── Title / status ────────────────────────────────────────────
            total_shown = sum(n_shown.values())
            total_all   = sum(n_total.values())
            status = (
                f"Showing {total_shown}/{total_all} ROIs  ·  "
                f"{n_shown['GOLD']} GOLD  "
                f"{n_shown['SILVER']} SILVER  "
                f"{n_shown['BRONZE']} BRONZE"
            )
            ax.set_title(f"{stem}\n{status}", fontsize=9)
            ax.axis("off")

            # ── Legend ────────────────────────────────────────────────────
            legend_elems = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_TIER_COLORS[t], markersize=9, label=t)
                for t in ("GOLD", "SILVER", "BRONZE")
                if t in selected_tiers
            ]
            if legend_elems:
                ax.legend(handles=legend_elems, loc="upper right",
                          fontsize=8, framealpha=0.6)

            plt.tight_layout()
            plt.show()

    # ── Wire callbacks ─────────────────────────────────────────────────────
    fov_dropdown.observe(_render, names="value")
    tier_select.observe(_render, names="value")
    prob_slider.observe(_render, names="value")

    ipy_display(widgets.VBox([
        widgets.HBox([fov_dropdown, tier_select]),
        prob_slider,
        out_widget,
    ]))

    _render()


def _find_mean(output_dir: Path, stem: str):
    """Return the first existing mean-projection path for *stem*, or None."""
    candidates = [
        output_dir.parent / "projections" / f"{stem}_mean.tif",
        output_dir.parent / "projections" / f"{stem}_mc_mean.tif",
        output_dir / f"{stem}_mean.tif",
    ]
    return next((p for p in candidates if p.exists()), None)
