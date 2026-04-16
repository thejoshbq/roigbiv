"""
ROI G. Biv pipeline — Napari viewer for sequential pipeline outputs.

Opens up to 8 layers per FOV:

  IMAGE LAYERS (bottom to top):
    1. Mean (L)              gray     — background component summary
    2. Max (S)               gray     — peak fluorescence on residual
    3. Std (S)               gray     — variance projection on residual
    4. Vcorr (S)             magma    — local correlation on residual
    5. Nuclear Shadow (DoG)  hot add. — soma shadow score
    6. Mean (S)              gray     — denoised mean of residual (primary view)

  LABEL LAYERS:
    7. Stage 1 ROIs          — accept=green, flag=yellow (via direct_colormap)
    8. All ROIs              — colored by source_stage (Stage 1=cyan in this phase)

Only 'Mean (S)', 'Stage 1 ROIs', 'All ROIs' are visible by default; the rest
are present-but-hidden and can be toggled via napari's native layer panel.

No widgets, no keybindings, no callbacks (spec + plan rule 5).
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.types import FOVData


# Colors for Stage 1 ROIs by gate_outcome (RGBA, 0-1)
_STAGE1_COLORS = {
    "accept": (0.00, 0.85, 0.00, 0.65),  # green
    "flag":   (1.00, 0.90, 0.00, 0.65),  # yellow
    "reject": (0.50, 0.50, 0.50, 0.30),  # gray (hidden: rejects aren't shown)
}
_STAGE2_COLORS = {
    "accept": (1.00, 0.50, 0.00, 0.65),  # orange
    "flag":   (1.00, 0.80, 0.20, 0.65),  # amber
    "reject": (0.50, 0.50, 0.50, 0.30),
}
_STAGE3_COLORS = {
    "accept": (0.85, 0.00, 0.85, 0.65),  # magenta
    "flag":   (1.00, 0.60, 1.00, 0.65),  # pink
    "reject": (0.50, 0.50, 0.50, 0.30),
}
# Stage 4 has no "accept" tier — every surviving candidate is "flag" with
# confidence="requires_review". One color for the flag tier only.
_STAGE4_COLORS = {
    "accept": (1.00, 0.70, 0.00, 0.65),  # gold (unused, but kept for symmetry)
    "flag":   (1.00, 0.70, 0.00, 0.65),  # gold
    "reject": (0.50, 0.50, 0.50, 0.30),
}

# Colors for unified ROIs by source_stage (RGBA)
_STAGE_COLORS = {
    1: (0.00, 0.70, 1.00, 0.65),  # cyan — Stage 1
    2: (1.00, 0.50, 0.00, 0.65),  # orange — Stage 2 (future)
    3: (0.85, 0.00, 0.85, 0.65),  # magenta — Stage 3 (future)
    4: (1.00, 1.00, 0.00, 0.65),  # yellow — Stage 4 (future)
}

# Colors by activity type (spec §13.3; used by the primary "All ROIs" layer)
_ACTIVITY_COLORS = {
    "phasic":    (0.00, 0.85, 0.00, 0.65),  # green
    "sparse":    (0.00, 0.85, 0.85, 0.65),  # cyan
    "tonic":     (1.00, 0.60, 0.00, 0.65),  # orange
    "silent":    (0.55, 0.55, 0.55, 0.55),  # gray
    "ambiguous": (1.00, 0.90, 0.00, 0.65),  # yellow
}

# Colors by HITL review priority
_PRIORITY_COLORS = {
    1: (0.95, 0.10, 0.10, 0.75),  # red — Stage 4 tonic review
    2: (1.00, 0.85, 0.00, 0.75),  # yellow — flagged
    3: (1.00, 0.55, 0.00, 0.75),  # orange — single-event
}

_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def _build_label_image(
    shape_hw: tuple,
    rois_filter,
) -> tuple[np.ndarray, dict]:
    """Build a (H, W) uint16 label image from a filtered subset of ROIs.

    `rois_filter` is a list of (ROI, label_id) tuples so callers can override
    the label used in the image (e.g., for gate_outcome coloring).

    Returns (label_image, {label_id: ROI}).
    """
    H, W = shape_hw
    img = np.zeros((H, W), dtype=np.uint16)
    roi_by_id = {}
    for roi, lid in rois_filter:
        img[roi.mask] = lid
        roi_by_id[int(lid)] = roi
    return img, roi_by_id


def _make_direct_colormap(color_dict):
    """Build a napari direct colormap from {label_id: rgba}."""
    from napari.utils.colormaps import direct_colormap
    full = {0: np.array(_TRANSPARENT, dtype=np.float32)}
    for lid, rgba in color_dict.items():
        full[int(lid)] = np.array(rgba, dtype=np.float32)
    return direct_colormap(full)


def display_pipeline_results(
    fov: FOVData,
    review_queue: list | None = None,
) -> "napari.Viewer":
    """Open a napari viewer with all pipeline layers.

    Blocks until the viewer is closed (calls napari.run).

    Parameters
    ----------
    fov : FOVData — populated through at least Foundation + Stage 1 + Gate 1.
          Later stages (Stage 2-4) are stubbed: layers are added conditionally
          based on whether any ROIs of that source_stage exist.
    review_queue : optional HITL review queue (list of dicts from
          hitl.build_review_queue). When provided, a "Review Queue" layer is
          added colored by priority.
    """
    import napari

    stem = fov.raw_path.stem.replace("_mc", "")
    viewer = napari.Viewer(title=f"ROI G. Biv (pipeline) — {stem}")

    # Determine (H, W) from whichever summary is available
    first_img = fov.mean_M if fov.mean_M is not None else fov.mean_S
    H, W = first_img.shape

    # ── Image layers (all hidden except Mean (M)) ──────────────────────────
    if fov.mean_L is not None:
        viewer.add_image(fov.mean_L, name="Mean (L)", colormap="gray", visible=False)
    if fov.mean_S is not None:
        viewer.add_image(fov.mean_S, name="Mean (S)", colormap="gray", visible=False)
    if fov.max_S is not None:
        viewer.add_image(fov.max_S, name="Max (S)", colormap="gray", visible=False)
    if fov.std_S is not None:
        viewer.add_image(fov.std_S, name="Std (S)", colormap="gray", visible=False)
    if fov.vcorr_S is not None:
        viewer.add_image(fov.vcorr_S, name="Vcorr (S)", colormap="magma", visible=False)
    if fov.dog_map is not None:
        viewer.add_image(
            fov.dog_map, name="Nuclear Shadow (DoG)",
            colormap="hot", opacity=0.6, blending="additive", visible=False,
        )
    # Mean (M) — raw morphological image — is the primary view by default
    if fov.mean_M is not None:
        viewer.add_image(fov.mean_M, name="Mean (M)", colormap="gray", visible=True)

    # ── Stage 4 diagnostic image layers: per-window correlation contrast ───
    # Hidden by default — useful for HITL review of Stage 4 candidates.
    for window_name in ("fast", "medium", "slow"):
        cmap_img = fov.corr_contrast_maps.get(window_name) if fov.corr_contrast_maps else None
        if cmap_img is not None:
            viewer.add_image(
                cmap_img,
                name=f"Correlation Contrast ({window_name})",
                colormap="magma",
                visible=False,
            )

    # ── Stage 1 label layer (accept=green, flag=yellow) ─────────────────────
    stage1_rois = [r for r in fov.rois
                   if r.source_stage == 1 and r.gate_outcome in ("accept", "flag")]
    if stage1_rois:
        stage1_img, stage1_lookup = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in stage1_rois],
        )
        color_map = {lid: _STAGE1_COLORS[roi.gate_outcome]
                     for lid, roi in stage1_lookup.items()}
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            stage1_img,
            name="Stage 1 ROIs",
            colormap=cmap,
            opacity=0.65,
            visible=True,
        )

    # ── Stage 2 label layer (accept=orange, flag=amber) ────────────────────
    stage2_rois = [r for r in fov.rois
                   if r.source_stage == 2 and r.gate_outcome in ("accept", "flag")]
    if stage2_rois:
        stage2_img, stage2_lookup = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in stage2_rois],
        )
        color_map = {lid: _STAGE2_COLORS[roi.gate_outcome]
                     for lid, roi in stage2_lookup.items()}
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            stage2_img,
            name="Stage 2 ROIs",
            colormap=cmap,
            opacity=0.65,
            visible=False,
        )

    # ── Stage 3 label layer (accept=magenta, flag=pink) ────────────────────
    stage3_rois = [r for r in fov.rois
                   if r.source_stage == 3 and r.gate_outcome in ("accept", "flag")]
    if stage3_rois:
        stage3_img, stage3_lookup = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in stage3_rois],
        )
        color_map = {lid: _STAGE3_COLORS[roi.gate_outcome]
                     for lid, roi in stage3_lookup.items()}
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            stage3_img,
            name="Stage 3 ROIs",
            colormap=cmap,
            opacity=0.65,
            visible=False,
        )

    # ── Stage 4 label layer (flag=gold; no accept tier for Stage 4) ────────
    stage4_rois = [r for r in fov.rois
                   if r.source_stage == 4 and r.gate_outcome in ("accept", "flag")]
    if stage4_rois:
        stage4_img, stage4_lookup = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in stage4_rois],
        )
        color_map = {lid: _STAGE4_COLORS[roi.gate_outcome]
                     for lid, roi in stage4_lookup.items()}
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            stage4_img,
            name="Stage 4 ROIs",
            colormap=cmap,
            opacity=0.65,
            visible=False,
        )

    # ── Diagnostic "All ROIs (by stage)" layer — colored by source_stage ───
    all_rois = [r for r in fov.rois if r.gate_outcome in ("accept", "flag")]
    if all_rois:
        all_img_stage, all_lookup_stage = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in all_rois],
        )
        color_map = {lid: _STAGE_COLORS.get(roi.source_stage, (0.5, 0.5, 0.5, 0.5))
                     for lid, roi in all_lookup_stage.items()}
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            all_img_stage,
            name="All ROIs (by stage)",
            colormap=cmap,
            opacity=0.65,
            visible=False,
        )

    # ── Primary "All ROIs" layer — colored by activity type (spec §13.3) ───
    # Only available after classification runs (every ROI has activity_type).
    classified = [r for r in all_rois if r.activity_type is not None]
    if classified:
        all_img_act, all_lookup_act = _build_label_image(
            (H, W),
            [(r, r.label_id) for r in classified],
        )
        color_map = {
            lid: _ACTIVITY_COLORS.get(roi.activity_type or "ambiguous",
                                      _ACTIVITY_COLORS["ambiguous"])
            for lid, roi in all_lookup_act.items()
        }
        cmap = _make_direct_colormap(color_map)
        viewer.add_labels(
            all_img_act,
            name="All ROIs",
            colormap=cmap,
            opacity=0.65,
            visible=True,
        )

    # ── HITL Review Queue layer — only Priority 1/2/3 ROIs ─────────────────
    if review_queue:
        priority_by_label = {
            int(entry["label_id"]): int(entry["priority"])
            for entry in review_queue
            if int(entry.get("priority", 4)) in (1, 2, 3)
        }
        review_rois = [r for r in all_rois if int(r.label_id) in priority_by_label]
        if review_rois:
            review_img, review_lookup = _build_label_image(
                (H, W),
                [(r, r.label_id) for r in review_rois],
            )
            color_map = {
                lid: _PRIORITY_COLORS.get(priority_by_label[int(lid)],
                                          (0.5, 0.5, 0.5, 0.5))
                for lid, _ in review_lookup.items()
            }
            cmap = _make_direct_colormap(color_map)
            viewer.add_labels(
                review_img,
                name="Review Queue",
                colormap=cmap,
                opacity=0.75,
                visible=False,
            )

    napari.run()
    return viewer
