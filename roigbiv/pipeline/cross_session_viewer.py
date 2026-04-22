"""Cross-session napari viewer for FOV-level cell tracking.

Opens one napari window showing every session registered under a given FOV,
with three layers per session:

    1. ``S{N}: {date} Mean``     — image, gray, visible
    2. ``S{N}: {date} ROIs``     — labels, per-cell colored so the same
                                   ``global_cell_id`` has the same color across
                                   sessions. Orphans render in neutral gray.
    3. ``S{N}: {date} Cell IDs`` — points + text at centroids, hidden by
                                   default. Text shows first 8 chars of
                                   ``global_cell_id``.

Sessions are ordered chronologically (oldest first). All sessions in one FOV
share ``(H, W)`` because the registry's fingerprint hash incorporates shape,
so no per-session padding is required.
"""
from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

import numpy as np

from roigbiv.registry import build_store
from roigbiv.registry.roicat_adapter import (
    centroids_from_merged_masks,
    load_session_input,
)


_GRAY_ORPHAN: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 0.40)
_TRANSPARENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _rgba_for_global_cell_id(gid: str) -> tuple[float, float, float, float]:
    """Deterministic hash → HSV → RGBA. Stable across processes."""
    h = hashlib.md5(gid.encode()).digest()
    hue = ((h[0] << 8) | h[1]) / 65535.0
    saturation = 0.55 + (h[2] / 255.0) * 0.40
    value = 0.70 + (h[3] / 255.0) * 0.30
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, 0.65)


def _make_direct_colormap(color_dict: dict[int, tuple[float, float, float, float]]):
    """Build a napari direct colormap from ``{label_id: rgba}``.

    Label ``0`` is forced transparent so the background doesn't shadow the
    underlying mean projection.
    """
    from napari.utils.colormaps import direct_colormap

    full = {0: np.array(_TRANSPARENT, dtype=np.float32)}
    for lid, rgba in color_dict.items():
        full[int(lid)] = np.array(rgba, dtype=np.float32)
    return direct_colormap(full)


def display_cross_session_fov(fov_id: str) -> None:
    """Open a napari viewer for every session of the given FOV.

    Blocks until the viewer is closed.
    """
    store = build_store()
    store.ensure_schema()

    fov = store.get_fov(fov_id)
    if fov is None:
        raise ValueError(f"FOV not found in registry: {fov_id}")

    sessions = sorted(store.list_sessions(fov_id), key=lambda s: s.session_date)
    if not sessions:
        raise ValueError(f"FOV {fov_id} has no registered sessions")

    obs_by_session = {
        s.session_id: store.list_observations_for_session(s.session_id)
        for s in sessions
    }

    unique_gids: set[str] = {
        obs.global_cell_id
        for observations in obs_by_session.values()
        for obs in observations
    }
    palette: dict[str, tuple[float, float, float, float]] = {
        gid: _rgba_for_global_cell_id(gid) for gid in unique_gids
    }

    import napari

    viewer = napari.Viewer(
        title=f"Cross-session FOV {fov_id[:8]} — {fov.animal_id}/{fov.region}",
    )

    for idx, session in enumerate(sessions, start=1):
        si = load_session_input(Path(session.output_dir))

        # Map local_label_id → global_cell_id for this session.
        gid_by_label: dict[int, str] = {
            int(obs.local_label_id): obs.global_cell_id
            for obs in obs_by_session[session.session_id]
        }

        # Build a colormap covering every label actually present in the mask —
        # orphans (no observation) fall back to neutral gray. Including every
        # unique label in the dict prevents KeyErrors inside direct_colormap.
        labels_in_mask = np.unique(si.merged_masks)
        labels_in_mask = labels_in_mask[labels_in_mask != 0]
        label_to_rgba: dict[int, tuple[float, float, float, float]] = {}
        for lid in labels_in_mask.tolist():
            gid = gid_by_label.get(int(lid))
            label_to_rgba[int(lid)] = palette[gid] if gid is not None else _GRAY_ORPHAN

        date_str = session.session_date.isoformat()
        base_name = f"S{idx}: {date_str}"

        viewer.add_image(
            si.mean_m,
            name=f"{base_name} Mean",
            colormap="gray",
            visible=True,
        )

        viewer.add_labels(
            si.merged_masks,
            name=f"{base_name} ROIs",
            opacity=0.5,
            visible=True,
            colormap=_make_direct_colormap(label_to_rgba),
        )

        centroids = centroids_from_merged_masks(si.merged_masks)
        if centroids.shape[0] > 0:
            # centroids_from_merged_masks orders by ascending label_id — mirror
            # that order when building the text list.
            ordered_labels = labels_in_mask.tolist()
            texts = [
                (gid_by_label[int(lid)][:8] if int(lid) in gid_by_label else "?")
                for lid in ordered_labels
            ]
            viewer.add_points(
                centroids,
                name=f"{base_name} Cell IDs",
                visible=False,
                size=1,
                face_color="transparent",
                text={"string": texts, "color": "white", "size": 9},
            )

    napari.run()
