"""Unified data loading for the UI.

:class:`FOVBundle` is the Viewer/Review page's view of one pipeline output
directory: mean projection + ROIs (with pipeline + user corrections applied)
+ per-ROI polygon contours + optional cross-session global_cell_id map.

:class:`CrossSessionBundle` groups several FOVBundles sharing a single
``fov_id`` and a ``(session_id, local_label_id) → global_cell_id`` table, so
the viewer can color ROIs consistently across days.

Both bundles are cache-friendly — computed once per output_dir and stored in
:class:`roigbiv.ui.services.app_state.AppState`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.corrections import (
    apply_corrections,
    load_corrections,
)
from roigbiv.pipeline.loaders import load_fov_from_output_dir
from roigbiv.pipeline.types import ROI


@dataclass
class ROIRender:
    """Viewer-ready per-ROI geometry.

    ``contours`` is a list of ``(y[], x[])`` tuples — one per topologically-
    distinct ring in the mask. Most ROIs have one ring; the list handles
    holes / disconnected components if they ever appear.
    """

    label_id: int
    source_stage: int
    gate_outcome: str
    activity_type: Optional[str]
    area: int
    centroid_yx: tuple[float, float]
    contours: list[tuple[list[float], list[float]]]
    global_cell_id: Optional[str] = None
    is_user: bool = False
    features: dict = field(default_factory=dict)


@dataclass
class FOVBundle:
    """One session / FOV output directory, decoded for the UI."""

    output_dir: Path
    stem: str
    mean_M: Optional[np.ndarray]
    shape: tuple[int, int]
    rois: list[ROIRender]
    registry: Optional[dict]
    session_id: Optional[str]
    fov_id: Optional[str]

    def roi_by_label(self, label_id: int) -> Optional[ROIRender]:
        for r in self.rois:
            if r.label_id == label_id:
                return r
        return None


@dataclass
class SessionRef:
    session_id: str
    session_date: Optional[date]
    output_dir: Path
    fov_posterior: Optional[float]


@dataclass
class CrossSessionBundle:
    fov_id: str
    animal_id: Optional[str]
    region: Optional[str]
    sessions: list[SessionRef]
    bundles: dict[str, FOVBundle]     # keyed by session_id


# ── FOV bundle ─────────────────────────────────────────────────────────────


def load_fov_bundle(output_dir: Path) -> FOVBundle:
    """Load a :class:`FOVBundle`, replaying any HITL corrections."""
    output_dir = Path(output_dir)
    fov, _review_queue = load_fov_from_output_dir(output_dir)
    shape_hw = _hw_shape(fov)

    ops = load_corrections(output_dir)
    rois = apply_corrections(fov.rois, ops, shape_hw) if ops else fov.rois

    registry = _maybe_json(output_dir / "registry_match.json")
    gcid_by_label = _gcid_by_label_from_registry(registry)

    rendered = [
        _render_roi(roi, gcid_by_label.get(int(roi.label_id)))
        for roi in rois
    ]

    return FOVBundle(
        output_dir=output_dir,
        stem=output_dir.name,
        mean_M=fov.mean_M,
        shape=shape_hw,
        rois=rendered,
        registry=registry,
        session_id=(registry or {}).get("session_id"),
        fov_id=(registry or {}).get("fov_id"),
    )


def _hw_shape(fov) -> tuple[int, int]:
    if fov.mean_M is not None:
        H, W = int(fov.mean_M.shape[0]), int(fov.mean_M.shape[1])
        return (H, W)
    if isinstance(fov.shape, tuple) and len(fov.shape) >= 3:
        return (int(fov.shape[1]), int(fov.shape[2]))
    raise ValueError("cannot infer FOV shape from loaded FOVData")


def render_roi(roi: ROI, gcid: Optional[str] = None) -> ROIRender:
    """Public renderer — used by both :func:`load_fov_bundle` and the Review page."""
    return _render_roi(roi, gcid)


def _render_roi(roi: ROI, gcid: Optional[str]) -> ROIRender:
    from skimage.measure import find_contours

    mask = roi.mask
    centroid = _centroid_yx(mask)
    contours: list[tuple[list[float], list[float]]] = []
    if mask is not None and mask.any():
        for ring in find_contours(mask.astype(float), 0.5):
            # find_contours returns (row, col) — keep as (y, x) for Plotly.
            ys = ring[:, 0].tolist()
            xs = ring[:, 1].tolist()
            contours.append((ys, xs))

    features = {}
    # Keep only JSON-native scalars — big arrays stay out of the UI payload.
    for k, v in (roi.features or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            features[k] = v

    return ROIRender(
        label_id=int(roi.label_id),
        source_stage=int(roi.source_stage),
        gate_outcome=str(roi.gate_outcome),
        activity_type=roi.activity_type,
        area=int(roi.area),
        centroid_yx=centroid,
        contours=contours,
        global_cell_id=gcid,
        is_user=bool(features.pop("user_added", False)),
        features=features,
    )


def _centroid_yx(mask: Optional[np.ndarray]) -> tuple[float, float]:
    if mask is None or not mask.any():
        return (0.0, 0.0)
    ys, xs = np.where(mask)
    return (float(ys.mean()), float(xs.mean()))


def _maybe_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _gcid_by_label_from_registry(registry: Optional[dict]) -> dict[int, str]:
    if not registry:
        return {}
    out: dict[int, str] = {}
    for entry in registry.get("cell_assignments", []):
        try:
            lid = int(entry.get("local_label_id"))
            gid = entry.get("global_cell_id")
        except (TypeError, ValueError):
            continue
        if gid:
            out[lid] = str(gid)
    return out


# ── Cross-session bundle ───────────────────────────────────────────────────


def load_cross_session_bundle(fov_id: str) -> CrossSessionBundle:
    """Build a :class:`CrossSessionBundle` for every session tied to ``fov_id``.

    The store is opened fresh from the current env (so the workspace
    ``ROIGBIV_REGISTRY_DSN`` is honored). Sessions with missing output
    directories are silently dropped — same behavior as the napari viewer.
    """
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()

    fov = store.get_fov(fov_id)
    sessions_rows = sorted(
        store.list_sessions(fov_id), key=lambda s: s.session_date or date.min,
    )

    session_refs: list[SessionRef] = []
    bundles: dict[str, FOVBundle] = {}
    for row in sessions_rows:
        out_dir = Path(row.output_dir)
        if not out_dir.exists():
            continue
        try:
            bundle = load_fov_bundle(out_dir)
        except Exception:  # noqa: BLE001
            continue
        # Replace registry-derived gcids with the authoritative DB observations,
        # in case the on-disk registry_match.json is stale after a rematch.
        gcids_by_label = {
            int(obs.local_label_id): obs.global_cell_id
            for obs in store.list_observations_for_session(row.session_id)
        }
        for rr in bundle.rois:
            rr.global_cell_id = gcids_by_label.get(rr.label_id, rr.global_cell_id)

        session_refs.append(SessionRef(
            session_id=row.session_id,
            session_date=row.session_date,
            output_dir=out_dir,
            fov_posterior=row.fov_posterior,
        ))
        bundles[row.session_id] = bundle

    return CrossSessionBundle(
        fov_id=fov_id,
        animal_id=getattr(fov, "animal_id", None),
        region=getattr(fov, "region", None),
        sessions=session_refs,
        bundles=bundles,
    )
